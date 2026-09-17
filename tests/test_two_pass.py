"""Run with ComfyUI's Python: python -m unittest discover -s tests -v."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
COMFY = ROOT.parents[1]
sys.path.insert(0, str(COMFY))
import comfy.options
comfy.options.enable_args_parsing()
sys.argv = [sys.argv[0], "--cpu"]

import torch
import folder_paths
from comfy.nested_tensor import NestedTensor

spec = importlib.util.spec_from_file_location("yafv_h3", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from yafv_h3 import two_pass as sampling
from yafv_h3 import h3_upscaler as upscale


def defaults():
    inputs = sampling.MiniMaxH3TwoPassSampler.INPUT_TYPES()["required"]
    values = {}
    for name, definition in inputs.items():
        kind = definition[0]
        if len(definition) > 1 and "default" in definition[1]:
            values[name] = definition[1]["default"]
        elif isinstance(kind, list):
            values[name] = kind[0]
    return values


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        server = types.SimpleNamespace(client_id="test", send_sync=lambda event, data, sid: self.events.append(data))
        self.server_patch = patch.object(sampling.PromptServer, "instance", server, create=True)
        self.server_patch.start()
        self.addCleanup(self.server_patch.stop)
        self.video = torch.zeros(1, 24, 2, 2, 2)
        self.audio = torch.ones(1, 32, 2, 8)
        self.latent = {"samples": NestedTensor((self.video, self.audio)), "batch_index": [0]}
        self.args = dict(defaults(), model_pass1=object(), positive=[], negative=[], latent=self.latent)

    def test_all_switches_and_fallback(self):
        for enabled_upscale in (False, True):
            for enabled_pass2 in (False, True):
                with self.subTest(upscale=enabled_upscale, pass2=enabled_pass2):
                    self.args.update(enable_latent_upscale=enabled_upscale, enable_pass2=enabled_pass2)
                    with patch.object(sampling, "common_ksampler", side_effect=lambda *a, **kw: (a[8],)) as sampler, \
                         patch.object(sampling.H3Upscaler, "upscale", side_effect=lambda latent, *a: latent) as scaler:
                        result = sampling.MiniMaxH3TwoPassSampler().sample(**self.args)
                    self.assertEqual(sampler.call_count, 2 if enabled_pass2 else 1)
                    self.assertEqual(scaler.call_count, int(enabled_upscale))
                    self.assertIs(sampler.call_args_list[-1].args[0], self.args["model_pass1"])
                    first = sampler.call_args_list[0].kwargs
                    self.assertEqual(first["last_step"], 4 if enabled_pass2 else 8)
                    self.assertTrue(first["force_full_denoise"])
                    output = result["result"][0]
                    self.assertIs(output["samples"].unbind()[1], self.audio)
                    self.assertEqual(output["batch_index"], [0])

    def test_separate_second_model(self):
        second_model = object()
        with patch.object(sampling, "common_ksampler", side_effect=lambda *a, **kw: (a[8],)) as sampler:
            sampling.MiniMaxH3TwoPassSampler().sample(**dict(self.args, model_pass2=second_model, enable_latent_upscale=False))
        self.assertIs(sampler.call_args_list[1].args[0], second_model)
        self.assertEqual(sampler.call_args_list[1].kwargs["start_step"], 4)
        self.assertFalse(sampler.call_args_list[1].kwargs["disable_noise"])

    def test_temporal_mode_routes_only_second_pass(self):
        second_model = object()
        with patch.object(sampling, "common_ksampler", return_value=(self.latent,)) as sampler, \
             patch.object(sampling, "sample_temporal", return_value=self.latent) as temporal:
            sampling.MiniMaxH3TwoPassSampler().sample(**dict(
                self.args, model_pass2=second_model, enable_latent_upscale=False,
                pass2_sampling_mode="temporal", pass2_chunking_mode="manual (frames)",
                pass2_chunk_frames=73, pass2_overlap_frames=22,
            ))
        self.assertEqual(sampler.call_count, 1)
        self.assertEqual(temporal.call_count, 1)
        self.assertIs(temporal.call_args.args[0], second_model)
        self.assertEqual(temporal.call_args.args[9:], (4, True, 73, 22))

    def test_audio_refinement_precedes_upscale_and_uses_second_model(self):
        for second_model in (None, object()):
            order = []
            refined = dict(self.latent, samples=NestedTensor((self.video, self.audio + 2)))
            def refine(*args):
                order.append("audio")
                self.assertIs(args[0], self.args["model_pass1"] if second_model is None else second_model)
                self.assertEqual(args[2:4], (10, 6))
                return refined
            def upscale(latent, *args):
                order.append("upscale")
                return latent
            def chunks(*args, **kwargs):
                order.append("chunks")
                torch.testing.assert_close(args[8]["samples"].unbind()[1], self.audio + 2)
                return args[8]
            with patch.object(sampling, "common_ksampler", return_value=(self.latent,)), \
                 patch.object(sampling, "refine_audio", side_effect=refine), \
                 patch.object(sampling.H3Upscaler, "upscale", side_effect=upscale), \
                 patch.object(sampling, "sample_temporal", side_effect=chunks):
                sampling.MiniMaxH3TwoPassSampler().sample(**dict(
                    self.args, model_pass2=second_model, pass2_sampling_mode="temporal",
                    pass2_audio_mode="refine", pass2_audio_steps=10, pass2_audio_start_step=6,
                ))
            self.assertEqual(order, ["audio", "upscale", "chunks"])

    def test_audio_refinement_validation_and_cancellation(self):
        args = dict(self.args, pass2_sampling_mode="temporal", pass2_audio_mode="refine")
        with patch.object(sampling, "common_ksampler") as sampler:
            with self.assertRaises(ValueError):
                sampling.MiniMaxH3TwoPassSampler().sample(**dict(args, pass2_audio_start_step=8))
            sampler.assert_not_called()
        with patch.object(sampling, "common_ksampler", return_value=(self.latent,)), \
             patch.object(sampling, "refine_audio", side_effect=sampling.mm.InterruptProcessingException), \
             patch.object(sampling.H3Upscaler, "upscale") as scaler, \
             patch.object(sampling, "sample_temporal") as chunks:
            with self.assertRaises(sampling.mm.InterruptProcessingException):
                sampling.MiniMaxH3TwoPassSampler().sample(**args)
            scaler.assert_not_called()
            chunks.assert_not_called()

    def test_audio_refinement_skipped_outside_active_mode(self):
        for mode, enabled, audio_mode in (("full", True, "refine"), ("temporal", False, "refine"), ("temporal", True, "preserve")):
            with patch.object(sampling, "common_ksampler", return_value=(self.latent,)), \
                 patch.object(sampling, "sample_temporal", return_value=self.latent), \
                 patch.object(sampling, "refine_audio") as refine:
                sampling.MiniMaxH3TwoPassSampler().sample(**dict(
                    self.args, enable_latent_upscale=False, enable_pass2=enabled,
                    pass2_sampling_mode=mode, pass2_audio_mode=audio_mode, pass2_audio_start_step=999,
                ))
            refine.assert_not_called()

    def test_temporal_settings_ignored_when_second_pass_disabled(self):
        with patch.object(sampling, "common_ksampler", return_value=(self.latent,)) as sampler, \
             patch.object(sampling, "sample_temporal") as temporal:
            sampling.MiniMaxH3TwoPassSampler().sample(**dict(
                self.args, enable_pass2=False, enable_latent_upscale=False,
                pass2_sampling_mode="temporal", pass2_chunk_frames=22, pass2_overlap_frames=73,
            ))
        self.assertEqual(sampler.call_count, 1)
        self.assertEqual(sampler.call_args.kwargs["last_step"], 8)
        temporal.assert_not_called()

    def test_preview_precedes_upscale_and_cancel_stops_work(self):
        preview = {"filename": "test.mp4", "subfolder": "", "type": "temp"}
        fake_preview = types.SimpleNamespace(as_dict=lambda: {"images": [preview]})
        def interrupt_after_preview():
            if any(event["stage"] == "preview_ready" for event in self.events):
                raise sampling.mm.InterruptProcessingException()
        with patch.object(sampling, "common_ksampler", return_value=(self.latent,)) as sampler, \
             patch.object(sampling.VAEDecode, "decode", return_value=(torch.zeros(2, 8, 8, 3),)), \
             patch.object(sampling, "save_video_preview", return_value=fake_preview), \
             patch.object(sampling.mm, "throw_exception_if_processing_interrupted", side_effect=interrupt_after_preview), \
             patch.object(sampling.H3Upscaler, "upscale") as scaler:
            with self.assertRaises(sampling.mm.InterruptProcessingException):
                sampling.MiniMaxH3TwoPassSampler().sample(**dict(self.args, enable_pass1_preview=True, video_vae=object()))
        self.assertEqual(sampler.call_count, 1)
        scaler.assert_not_called()
        self.assertEqual(self.events[-2]["stage"], "preview_ready")
        self.assertEqual(self.events[-1]["stage"], "cancelled")

    def test_preview_off_never_decodes(self):
        with patch.object(sampling, "common_ksampler", return_value=(self.latent,)), \
             patch.object(sampling.VAEDecode, "decode") as decode, \
             patch.object(sampling, "vae_decode_audio") as audio:
            sampling.MiniMaxH3TwoPassSampler().sample(**dict(self.args, enable_latent_upscale=False))
        decode.assert_not_called()
        audio.assert_not_called()

    def test_missing_vae_fails_before_sampling(self):
        with patch.object(sampling, "common_ksampler") as sampler:
            with self.assertRaisesRegex(ValueError, "video_vae"):
                sampling.MiniMaxH3TwoPassSampler().sample(**dict(self.args, enable_pass1_preview=True))
        sampler.assert_not_called()

    def test_disabled_upscale_does_not_require_checkpoint(self):
        self.assertTrue(sampling.MiniMaxH3TwoPassSampler.VALIDATE_INPUTS(False, "missing.safetensors"))

    def test_preview_encodes_a_real_video(self):
        images = torch.zeros(3, 16, 16, 3)
        images[1, :, :, 0] = 1
        audio = {"waveform": torch.zeros(1, 2, 6000), "sample_rate": 48000}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(folder_paths, "get_temp_directory", return_value=directory), \
             patch.object(sampling, "common_ksampler", return_value=(self.latent,)), \
             patch.object(sampling.VAEDecode, "decode", return_value=(images,)), \
             patch.object(sampling, "vae_decode_audio", return_value=audio):
            result = sampling.MiniMaxH3TwoPassSampler().sample(**dict(
                self.args, enable_latent_upscale=False, enable_pass2=False,
                enable_pass1_preview=True, preview_audio=True, video_vae=object(), audio_vae=object(),
            ))
            preview = result["ui"]["h3_preview"][0]
            self.assertTrue((Path(directory) / preview["subfolder"] / preview["filename"]).is_file())
            stages = [event["stage"] for event in self.events]
            self.assertLess(stages.index("preview_ready"), stages.index("complete"))


class UpscaleTests(unittest.TestCase):
    def test_sizing_modes(self):
        self.assertEqual(upscale.target_size(20, 10, "scale by multiplier", 2, 0, 0, 0, 32), (40, 20, 2))
        self.assertEqual(upscale.target_size(20, 10, "target dimensions", 1, 640, 640, 0, 32), (40, 40, 3))
        self.assertEqual(upscale.target_size(16, 16, "megapixels", 1, 0, 0, 1, 32), (64, 64, 4))

    def test_checkpoint_loading_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            model = upscale.LatentResizer3D(channels=32, in_blocks=1, out_blocks=1, temporal_every=0)
            torch.save({"model": {"upscaler." + k: v for k, v in model.state_dict().items()}}, Path(directory) / "tiny.pth")
            latent = {"samples": torch.rand(1, 24, 2, 2, 2),
                      "noise_mask": torch.ones(1, 1, 2, 2, 2), "batch_index": [3]}
            with patch.dict(folder_paths.folder_names_and_paths, {upscale.MODEL_FOLDER: ([directory], {".pth"})}), \
                 torch.inference_mode():
                result = upscale.H3Upscaler().upscale(latent, "tiny.pth", "scale by multiplier", 2, 0, 0, 0,
                                                    32, True, True, "cpu", "fp32")
            self.assertEqual(result["samples"].shape, (1, 24, 2, 4, 4))
            self.assertEqual(result["noise_mask"].shape[-3:], (2, 4, 4))
            self.assertEqual(result["batch_index"], [3])
            self.assertEqual(latent["samples"].shape[-2:], (2, 2))

    def test_cache_reuses_one_checkpoint_and_invalidates_precision(self):
        with tempfile.TemporaryDirectory() as directory:
            model = upscale.LatentResizer3D(channels=32, in_blocks=1, out_blocks=1, temporal_every=0)
            torch.save(model.state_dict(), Path(directory) / "tiny.pth")
            owner = upscale.H3Upscaler()
            with patch.dict(folder_paths.folder_names_and_paths, {upscale.MODEL_FOLDER: ([directory], {".pth"})}):
                first = owner.load("tiny.pth", torch.device("cpu"), "fp32")
                self.assertIs(owner.load("tiny.pth", torch.device("cpu"), "fp32"), first)
                self.assertIsNot(owner.load("tiny.pth", torch.device("cpu"), "fp16"), first)

    def test_original_network_parity(self):
        source = ROOT.parent / "Comfyui_Minimax_h3_latent_Upscaler/nodes/minimax_h3_latent_upscaler_3d.py"
        if not source.exists():
            self.skipTest("Original installed upscaler is not available")
        spec = importlib.util.spec_from_file_location("original_h3_upscaler", source)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        torch.manual_seed(7)
        reference = original.LatentResizer3D(channels=32, in_blocks=1, out_blocks=1).eval()
        port = upscale.LatentResizer3D(channels=32, in_blocks=1, out_blocks=1)
        port.load_state_dict(reference.state_dict(), strict=True)
        with torch.inference_mode():
            for frames, chunking in ((3, False), (35, True)):
                x = torch.rand(1, 24, frames, 2, 2)
                kwargs = dict(scale=2, target_size=(frames, 4, 4), enable_chunking=chunking)
                torch.testing.assert_close(port(x, **kwargs), reference(x, **kwargs), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
