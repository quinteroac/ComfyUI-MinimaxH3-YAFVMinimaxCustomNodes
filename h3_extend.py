"""H3 continuation using native conditioning, without global model patches.

Reference preparation adapted from kat3ri/ComfyUI-MiniMax-H3-Extend
(MIT, nodes.py) and ComfyUI's native MiniMax H3 nodes.
"""
import math

import node_helpers
import comfy_extras.nodes_minimax_h3 as native
from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE
from comfy.nested_tensor import NestedTensor


def _context_keyframes(context_latent, context_frames):
    samples = context_latent["samples"]
    video, audio = samples.unbind() if samples.is_nested else (samples, None)
    if video.ndim != 5 or video.shape[:2] != (1, 24) or video.shape[2] < 1:
        raise ValueError("H3 context requires video latents [1,24,T,H,W].")
    if context_frames < 1:
        raise ValueError("context_frames must be at least 1.")
    count = min(context_frames, video.shape[2])
    # Preserve Extend's backwards cadence and final context anchor at zero.
    indices = [0]
    for i in range(1, count):
        indices.append(indices[-1] - FRAME_PER_TOKEN[-i % len(FRAME_PER_TOKEN)])
    indices.reverse()
    tail = video[:, :, -count:]
    guides = [{"resolved_frame_index": index, "latent": tail[:, :, i:i + 1].clone(),
               "yafv_context": True} for i, index in enumerate(indices)]
    if audio is not None:
        if audio.ndim != 4 or audio.shape[:3] != (1, 32, 2):
            raise ValueError("H3 context requires audio latents [1,32,2,T].")
        span = sum(FRAME_PER_TOKEN[k % len(FRAME_PER_TOKEN)] for k in range(-count, 0))
        audio_count = min(round(span * FRAME_RESCALE), audio.shape[-1])
        if audio_count:
            guides.append({"resolved_frame_index": -audio_count / FRAME_RESCALE,
                           "audio_latent": audio[..., -audio_count:].clone(),
                           "yafv_context": True})
    return video.shape[-1] * 16, video.shape[-2] * 16, guides


def _pin_last_context_frame(vae, context_latent, width, height, resize_fn):
    samples = context_latent["samples"]
    video = samples.unbind()[0] if samples.is_nested else samples
    decoded = vae.decode(video[:, :, -6:])
    # H3 VAE output is [B,T,H,W,C].
    image = resize_fn(decoded[:, -1], width, height, "disabled")
    return {"resolved_frame_index": 0, "image": image}


def _build_ref_blocks(vae, audio_vae, width, height, frame_count, ref_image_size,
                       ref_images, ref_videos, ref_video_audios, ref_audios):

    CANVAS_MULTIPLE = 32
    REF_IMAGE_SHORT_EDGE = 2048
    FPS = 24
    encode_ref_audio = native._encode_ref_audio

    ref_items = []
    ref_blocks = []

    for img in (ref_images or {}).values():
        if img is None:
            continue
        h, w = img.shape[1], img.shape[2]
        if ref_image_size == "match":
            scale = min(1.0, math.sqrt((width * height) / (w * h)))
        else:
            scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
        tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        resized = native._resize(img[:1], tw, th, "disabled")
        z = vae.encode(resized)
        ref_items.append({"type": "image", "data": resized})
        ref_blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": z})

    ref_video_audios = ref_video_audios or {}
    for name, video_frames in (ref_videos or {}).items():
        if video_frames is None:
            continue
        soundtrack = ref_video_audios.get("ref_video_audio_" + name.rsplit("_", 1)[-1])
        vh, vw = video_frames.shape[1], video_frames.shape[2]
        cw, ch = native.adapt_canvas(vw, vh)
        if vw * vh < cw * ch:
            cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        frames = native._resize(video_frames, cw, ch, "disabled")
        if frames.shape[0] > frame_count:
            frames = frames[:frame_count]
        n = frames.shape[0]
        if n < 5:
            raise ValueError("MiniMax H3 reference videos need at least 5 frames (~0.2s at 24 fps)")
        while n % 17 != 5:
            n -= 1
        frames = frames[:n]
        z = vae.encode(frames)
        audio_latent, ref_audio_t = (None, 0)
        if soundtrack is not None:
            audio_latent, ref_audio_t = encode_ref_audio(audio_vae, soundtrack)
            ref_items.append({"type": "audio"})
        sample_idx = list(range(0, frames.shape[0], FPS // 2))
        qwen_frames = frames[sample_idx]
        ref_items.append({"type": "video", "data": qwen_frames,
                          "timestamps": [i / 2.0 for i in range(len(sample_idx))]})
        ref_blocks.append({"kind": "video_audio" if ref_audio_t else "video",
                           "latent_t": z.shape[2], "latent_h": ch // 16, "latent_w": cw // 16,
                           "ref_audio_t": ref_audio_t, "latent": z, "audio_latent": audio_latent})

    for audio in (ref_audios or {}).values():
        if audio is None:
            continue
        audio_latent, ref_audio_t = encode_ref_audio(audio_vae, audio)
        ref_items.append({"type": "audio"})
        ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent})

    return ref_items, ref_blocks


def _execute(clip, vae, context_latent, prompt, length, context_frames=2, pin_last_frame=True,
             first_frame=None, last_frame=None, audio_vae=None, ref_image_size="match", ref_images=None,
             ref_videos=None, ref_video_audios=None, ref_audios=None):

    width, height, keyframes = _context_keyframes(context_latent, context_frames)
    if first_frame is not None:
        img = native._resize(first_frame[:1], width, height, "disabled")
        keyframes.append({"resolved_frame_index": 0, "image": img})
    elif pin_last_frame:
        keyframes.append(_pin_last_context_frame(vae, context_latent, width, height, native._resize))
    latent, frame_count = native._empty_av_latent(width, height, length)
    if last_frame is not None:
        # aspect-preserving cover-crop ("follower"), same convention stock's
        # own MiniMaxH3ImageToVideo uses for its last_frame -- distinct from
        # first_frame's plain stretch ("geometry anchor")
        img = native._resize(last_frame[:1], width, height, "center")
        keyframes.append({"resolved_frame_index": frame_count - 1, "image": img})

    ref_items, ref_blocks = ([], [])
    if any((ref_images, ref_videos, ref_audios)):
        if audio_vae is None and (ref_video_audios or ref_audios):
            raise ValueError("audio_vae is required when ref_video_audios or ref_audios are supplied")
        ref_items, ref_blocks = _build_ref_blocks(vae, audio_vae, width, height, frame_count, ref_image_size,
                                                   ref_images, ref_videos, ref_video_audios, ref_audios)

    tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
    cond = clip.encode_from_tokens_scheduled(tokens)

    for kf in keyframes:
        if "image" in kf:
            kf["latent"] = vae.encode(kf.pop("image"))

    values = {"minimax_keyframes": keyframes, "minimax_frame_count": frame_count}
    if ref_blocks:
        values["minimax_refs"] = ref_blocks
    cond = node_helpers.conditioning_set_values(cond, values)
    return cond, latent


class YAFVH3VideoExtend:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "context_latent": ("LATENT", {"tooltip": "AV latent output from a prior MiniMax H3 generation to continue from"}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "length": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17,
                                   "tooltip": "New frame count at 24 fps for the continuation only (excludes context_frames)"}),
                "context_frames": ("INT", {"default": 2, "min": 1, "max": 64,
                                           "tooltip": "Trailing latent frames of context_latent carried over as context"}),
                "pin_last_frame": ("BOOLEAN", {"default": True,
                                               "tooltip": "Decode context_latent's true trailing pixel frame and pin it as this call's frame 0. Ignored if first_frame is connected."}),
            },
            "optional": {
                "audio_vae": ("VAE",),
                "first_frame": ("IMAGE", {"tooltip": "Hard-pin this call's frame 0 to an exact image (e.g. the prior clip's real last output frame) instead of pin_last_frame's decode"}),
                "last_frame": ("IMAGE", {"tooltip": "Pin this continuation segment's own final frame to an exact image -- e.g. to land precisely on a known next shot/keyframe instead of leaving the ending fully generated. "}),
                "ref_image_size": (["match", "max"], {"default": "match"}),
                "ref_images": ("IMAGE", {"tooltip": "Reference image(s) -- connect a batch (e.g. via ImageBatch) for more than one; each frame becomes its own <Picture i> reference. Not the native node's per-slot Autogrow inputs -- this is a single batched socket."}),
                "ref_audio": ("AUDIO", {"tooltip": "One standalone reference audio clip"}),
                "ref_video": ("IMAGE", {"tooltip": "Reference video frames at 24 fps."}),
                "ref_video_audio": ("AUDIO", {"tooltip": "Soundtrack for ref_video; requires audio_vae."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/conditioning"
    DESCRIPTION = "Continue a MiniMax H3 clip using native video and audio keyframes."

    def run(self, clip, vae, context_latent, prompt, length, context_frames=2, pin_last_frame=True,
            audio_vae=None, first_frame=None, last_frame=None, ref_image_size="match", ref_images=None, ref_audio=None,
            ref_video=None, ref_video_audio=None):
        ref_images_dict = None
        if ref_images is not None:
            ref_images_dict = {f"ref_image_{i + 1}": ref_images[i:i + 1] for i in range(ref_images.shape[0])}
        ref_audios_dict = {"ref_audio_1": ref_audio} if ref_audio is not None else None

        cond, latent = _execute(
            clip, vae, context_latent, prompt, length, context_frames=context_frames,
            pin_last_frame=pin_last_frame, first_frame=first_frame, last_frame=last_frame, audio_vae=audio_vae,
            ref_image_size=ref_image_size, ref_images=ref_images_dict, ref_audios=ref_audios_dict,
            ref_videos={"ref_video_1": ref_video} if ref_video is not None else None,
            ref_video_audios={"ref_video_audio_1": ref_video_audio} if ref_video_audio is not None else None,
        )
        return (cond, latent)


class YAFVH3EncodeAV:
    CATEGORY = "MiniMax H3/latent"
    FUNCTION = "run"
    RETURN_TYPES = ("LATENT",)
    DESCRIPTION = "Encode source video and optional audio for H3 continuation."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae": ("VAE",), "images": ("IMAGE",)},
                "optional": {"audio_vae": ("VAE",), "audio": ("AUDIO",)}}

    def run(self, vae, images, audio_vae=None, audio=None):
        if audio is not None and audio_vae is None:
            raise ValueError("Connect audio_vae when supplying audio.")
        video = vae.encode(images[..., :3])
        if audio is None:
            return ({"samples": video},)
        encoded_audio, _ = native._encode_ref_audio(audio_vae, audio)
        return ({"samples": NestedTensor((video, encoded_audio))},)
