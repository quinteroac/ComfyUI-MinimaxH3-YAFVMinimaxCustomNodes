from fractions import Fraction
import uuid

import comfy.model_management as mm
import comfy.samplers
from comfy_api.latest import InputImpl, Types
from comfy_extras.nodes_audio import vae_decode_audio
from comfy_extras.nodes_lt import LTXVConcatAVLatent, LTXVSeparateAVLatent
from comfy_extras.nodes_video import save_video_preview
from nodes import VAEDecode, common_ksampler
from server import PromptServer

from .h3_upscaler import H3Upscaler, model_names
from .project_continuity import apply_context
from .temporal_sampling import refine_audio, sample_temporal, window_sizes


def without_keyframes(conditioning):
    return [[embedding, {key: value for key, value in metadata.items()
                         if key != "minimax_keyframes"}]
            for embedding, metadata in conditioning]


class MiniMaxH3TwoPassSampler:
    CATEGORY = "MiniMax H3/sampling"
    FUNCTION = "sample"
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("latent", "latent_pass1")
    DESCRIPTION = "Two-pass audiovisual sampling with optional 3D upscale and an immediate Pass 1 preview."

    def __init__(self):
        self.upscaler = H3Upscaler()

    @classmethod
    def INPUT_TYPES(cls):
        upscale_models = model_names()
        default_upscaler = next((name for name in upscale_models if "minimax_h3_latent_upscaler_3d" in name), upscale_models[0])
        required = {
            "model_pass1": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "latent": ("LATENT",),
            "enable_latent_upscale": ("BOOLEAN", {"default": True}),
            "enable_pass2": ("BOOLEAN", {"default": True}),
            "enable_pass1_preview": ("BOOLEAN", {"default": False}),
            "total_steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
            "split_step": ("INT", {"default": 4, "min": 1, "max": 10000}),
        }
        for stage in (1, 2):
            required.update({
                f"seed_pass{stage}": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                             "control_after_generate": True}),
                f"cfg_pass{stage}": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                f"sampler_pass{stage}": (comfy.samplers.KSampler.SAMPLERS, {"default": "lcm"}),
                f"scheduler_pass{stage}": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
            })
        required.update({
            "pass1_return_with_leftover_noise": ("BOOLEAN", {"default": False}),
            "pass2_add_noise": ("BOOLEAN", {"default": True}),
            "upscale_model": (upscale_models, {"default": default_upscaler}),
            "upscale_mode": (["scale by multiplier", "target dimensions", "megapixels"],),
            "upscale_scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.05}),
            "upscale_width": ("INT", {"default": 1280, "min": 64, "max": 8192, "step": 8}),
            "upscale_height": ("INT", {"default": 704, "min": 64, "max": 8192, "step": 8}),
            "upscale_megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 16.0, "step": 0.1}),
            "upscale_align": ("INT", {"default": 32, "min": 1, "max": 512}),
            "upscale_temporal_chunking": ("BOOLEAN", {"default": True}),
            "upscale_force_unload": ("BOOLEAN", {"default": True}),
            "upscale_device": (["cuda", "rocm", "cpu"],),
            "upscale_precision": (["fp16", "fp32", "bf16"],),
            "preview_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 1.0}),
            "preview_audio": ("BOOLEAN", {"default": False}),
        })
        return {
            "required": required,
            "optional": {
                "model_pass2": ("MODEL", {"tooltip": "Unconnected: use model_pass1."}),
                "video_vae": ("VAE", {"tooltip": "Required only for Pass 1 preview."}),
                "audio_vae": ("VAE", {"tooltip": "Required only when preview audio is enabled."}),
                "context_latent_pass2": ("LATENT", {"tooltip": "Approved previous clip at Pass 2 resolution, from Project Context."}),
                "pass2_sampling_mode": (["full", "temporal"], {"default": "full",
                    "tooltip": "Temporal refines overlapping video windows. Choose whether to preserve or refine audio separately."}),
                "pass2_chunking_mode": (["auto (chunk count)", "manual (frames)"], {"default": "auto (chunk count)",
                    "tooltip": "Choose either an automatic chunk count or explicit temporal window values."}),
                "pass2_chunk_count": ("INT", {"default": 2, "min": 1, "max": 32,
                    "tooltip": "Automatic mode: desired number of temporal chunks for the input duration."}),
                "pass2_chunk_frames": ("INT", {"default": 73, "min": 22, "max": 3600, "step": 17,
                    "tooltip": "Window length in video frames at 24 fps. Rounded up to H3's 17k+5 grid."}),
                "pass2_overlap_frames": ("INT", {"default": 22, "min": 5, "max": 3600, "step": 17,
                    "tooltip": "Shared video frames between windows. Must be shorter than the window."}),
                "pass2_audio_mode": (["preserve", "refine"], {"default": "preserve",
                    "tooltip": "Temporal mode only. Refine the full audio before upscale, with Pass 1 video frozen."}),
                "pass2_audio_steps": ("INT", {"default": 8, "min": 1, "max": 10000,
                    "tooltip": "Total audio sampling schedule, independent of the video step distribution."}),
                "pass2_audio_start_step": ("INT", {"default": 4, "min": 0, "max": 9999,
                    "tooltip": "Start of audio refinement. 4 of 8 executes the last 4 steps. Lower starts change audio more."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def VALIDATE_INPUTS(cls, enable_latent_upscale, upscale_model):
        if enable_latent_upscale and upscale_model not in model_names():
            return "Select an installed latent upscale model."
        if enable_latent_upscale and upscale_model.startswith("("):
            return "Place the H3 3D checkpoint in models/latent_upscale_models."
        return True

    def sample(self, model_pass1, positive, negative, latent, enable_latent_upscale,
               enable_pass2, enable_pass1_preview, total_steps, split_step,
               seed_pass1, cfg_pass1, sampler_pass1, scheduler_pass1,
               seed_pass2, cfg_pass2, sampler_pass2, scheduler_pass2,
               pass1_return_with_leftover_noise, pass2_add_noise,
               upscale_model, upscale_mode, upscale_scale, upscale_width, upscale_height,
               upscale_megapixels, upscale_align, upscale_temporal_chunking,
               upscale_force_unload, upscale_device, upscale_precision,
               preview_fps, preview_audio, model_pass2=None, video_vae=None,
               audio_vae=None, unique_id=None, pass2_sampling_mode="full",
               pass2_chunking_mode="auto (chunk count)", pass2_chunk_count=2,
               pass2_chunk_frames=73, pass2_overlap_frames=22, pass2_audio_mode="preserve",
               pass2_audio_steps=8, pass2_audio_start_step=4, context_latent_pass2=None):
        if enable_pass2 and not 1 <= split_step < total_steps:
            raise ValueError("With Pass 2 enabled, split_step must be between 1 and total_steps - 1.")
        if enable_pass2 and pass2_sampling_mode == "temporal":
            if pass2_chunking_mode == "manual (frames)":
                window_sizes(pass2_chunk_frames, pass2_overlap_frames)
            elif pass2_chunking_mode != "auto (chunk count)":
                raise ValueError("Choose automatic chunk count or manual temporal frames, not both.")
            elif not isinstance(pass2_chunk_count, int) or pass2_chunk_count < 1:
                raise ValueError("Automatic temporal chunking requires at least one chunk.")
            if pass1_return_with_leftover_noise:
                raise ValueError("Temporal Pass 2 needs clean Pass 1 latents. Disable pass1_return_with_leftover_noise.")
            if pass2_audio_mode == "refine" and not 0 <= pass2_audio_start_step < pass2_audio_steps:
                raise ValueError("Audio start step must be between 0 and pass2_audio_steps - 1.")
        if enable_pass1_preview and video_vae is None:
            raise ValueError("Connect video_vae or disable Pass 1 preview.")
        if enable_pass1_preview and preview_audio:
            if audio_vae is None or not latent["samples"].is_nested:
                raise ValueError("Audio preview requires audio_vae and an audiovisual latent.")

        server = PromptServer.instance
        client_id = server.client_id
        run_id = uuid.uuid4().hex

        def notify(stage, **data):
            server.send_sync("yafv-h3-stage", {
                "node": str(unique_id), "run_id": run_id, "stage": stage, **data,
            }, client_id)

        preview = None
        notify("pass1", clear=True)
        try:
            mm.throw_exception_if_processing_interrupted()
            result = common_ksampler(
                model_pass1, seed_pass1, total_steps, cfg_pass1, sampler_pass1,
                scheduler_pass1, positive, negative, latent, start_step=0,
                last_step=split_step if enable_pass2 else total_steps,
                force_full_denoise=not (enable_pass2 and pass1_return_with_leftover_noise),
            )[0]
            pass1 = dict(result, yafv_clean=not (enable_pass2 and pass1_return_with_leftover_noise))

            if enable_pass1_preview:
                mm.throw_exception_if_processing_interrupted()
                notify("preview")
                images = VAEDecode().decode(video_vae, result)[0]
                mm.throw_exception_if_processing_interrupted()
                audio = vae_decode_audio(audio_vae, result) if preview_audio else None
                video = InputImpl.VideoFromComponents(Types.VideoComponents(
                    images=images, audio=audio, frame_rate=Fraction(str(preview_fps)),
                ))
                preview = save_video_preview(video).as_dict()["images"][0]
                del images, audio, video
                notify("preview_ready", preview=preview)

            mm.throw_exception_if_processing_interrupted()
            if enable_pass2 and pass2_sampling_mode == "temporal" and pass2_audio_mode == "refine":
                notify("audio_refine")
                result = refine_audio(
                    model_pass1 if model_pass2 is None else model_pass2,
                    seed_pass2, pass2_audio_steps, pass2_audio_start_step, cfg_pass2,
                    sampler_pass2, scheduler_pass2, positive, negative, result, pass2_add_noise,
                )

            mm.throw_exception_if_processing_interrupted()
            if enable_latent_upscale:
                notify("upscale")
                av = result["samples"].is_nested
                if av:
                    video_latent, audio_latent = LTXVSeparateAVLatent.execute(result).result
                else:
                    video_latent = result
                video_latent = self.upscaler.upscale(
                    video_latent, upscale_model, upscale_mode, upscale_scale,
                    upscale_width, upscale_height, upscale_megapixels, upscale_align,
                    upscale_temporal_chunking, upscale_force_unload, upscale_device,
                    upscale_precision,
                )
                result = (LTXVConcatAVLatent.execute(video_latent, audio_latent).result[0]
                          if av else video_latent)
                del video_latent
                if av:
                    del audio_latent

            mm.throw_exception_if_processing_interrupted()
            if enable_pass2:
                notify("pass2")
                second_model = model_pass1 if model_pass2 is None else model_pass2
                positive_pass2 = without_keyframes(positive)
                negative_pass2 = without_keyframes(negative)
                if context_latent_pass2 is not None:
                    count = latent.get("yafv_context_length")
                    if count is None:
                        raise ValueError("Pass 2 context requires Project Motion Context on the Pass 1 latent")
                    positive_pass2, _, result = apply_context(
                        positive_pass2, result, context_latent_pass2, count,
                        video_transition_steps=0, audio_transition_steps=0, video_anchor_only=False,
                    )
                if pass2_sampling_mode == "temporal":
                    def chunk_callback(chunk, chunks, frame_start, frame_end):
                        notify("pass2", chunk=chunk, chunks=chunks,
                               frame_start=frame_start, frame_end=frame_end)

                    result = sample_temporal(
                        second_model, seed_pass2, total_steps, cfg_pass2, sampler_pass2,
                        scheduler_pass2, positive_pass2, negative_pass2, result, split_step,
                        pass2_add_noise,
                        pass2_chunk_frames if pass2_chunking_mode == "manual (frames)" else None,
                        pass2_overlap_frames if pass2_chunking_mode == "manual (frames)" else None,
                        chunk_count=pass2_chunk_count if pass2_chunking_mode == "auto (chunk count)" else None,
                        chunk_callback=chunk_callback,
                    )
                else:
                    result = common_ksampler(
                        second_model, seed_pass2, total_steps, cfg_pass2, sampler_pass2, scheduler_pass2,
                        positive_pass2, negative_pass2, result, disable_noise=not pass2_add_noise,
                        start_step=split_step, last_step=total_steps, force_full_denoise=True,
                    )[0]
            mm.throw_exception_if_processing_interrupted()
            notify("complete")
            return {"ui": {"h3_preview": [preview] if preview else []}, "result": (result, pass1)}
        except mm.InterruptProcessingException:
            notify("cancelled")
            raise
        except Exception:
            notify("error")
            raise
