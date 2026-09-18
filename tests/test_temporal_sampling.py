import unittest
from unittest.mock import patch

from test_two_pass import sampling
import torch
from comfy.nested_tensor import NestedTensor
from comfy.model_patcher import ModelPatcher
from comfy.supported_models import MiniMaxH3
from yafv_h3 import temporal_sampling as temporal


class WindowTests(unittest.TestCase):
    def test_automatic_window_size_matches_ten_second_example(self):
        self.assertEqual(temporal.automatic_window_sizes(72, 2), (141, 22))

    def test_automatic_window_size_covers_requested_count(self):
        for chunks in range(1, 7):
            chunk_frames, overlap_frames = temporal.automatic_window_sizes(107, chunks)
            windows = temporal.temporal_windows(107, chunk_frames, overlap_frames, exact_tail=True)
            self.assertEqual(len(windows), chunks)

    def test_coverage_cadence_and_tail(self):
        for tokens in range(2, 208, 5):
            for chunk, overlap in ((22, 5), (73, 22), (73, 56), (124, 39)):
                windows = temporal.temporal_windows(tokens, chunk, overlap)
                self.assertEqual(windows[0][0], 0)
                self.assertEqual(windows[-1][1], tokens)
                counts = [0] * tokens
                for index, (start, stop) in enumerate(windows):
                    self.assertEqual(start % 5, 0)
                    self.assertEqual((stop - start - 2) % 5, 0)
                    self.assertLessEqual(temporal.frame_boundary(stop) - temporal.frame_boundary(start), chunk)
                    if index:
                        self.assertGreater(start, windows[index - 1][0])
                        self.assertLess(start, windows[index - 1][1])
                    for position in range(start, stop):
                        counts[position] += 1
                self.assertTrue(all(counts))

    def test_fifteen_second_window_plan(self):
        self.assertEqual(temporal.frame_boundary(107), 362)
        self.assertEqual(temporal.temporal_windows(107, 73, 22),
                         [(0, 22), (15, 37), (30, 52), (45, 67), (60, 82), (75, 97), (85, 107)])

    def test_invalid_overlap_and_native_shape(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            temporal.window_sizes(73, 73)
        with self.assertRaisesRegex(ValueError, "5k"):
            temporal.temporal_windows(23, 73, 22)


class ConditioningTests(unittest.TestCase):
    def test_keyframes_and_independent_references(self):
        embedding = torch.ones(1, 2, 3)
        guide = torch.arange(24 * 22 * 4, dtype=torch.float32).reshape(1, 24, 22, 2, 2)
        refs = [{"kind": "image", "latent": torch.ones(1, 24, 1, 2, 2)}]
        metadata = {"minimax_refs": refs, "minimax_token_tags": [1], "minimax_keyframes": [
            {"resolved_frame_index": 0, "latent": guide},
            {"resolved_frame_index": 5, "latent": torch.ones(1, 24, 1, 2, 2)},
            {"resolved_frame_index": 72, "latent": torch.full((1, 24, 1, 2, 2), 7.0)},
        ]}
        result = temporal.window_conditioning([[embedding, metadata]], 51, 124, 4, 4)
        self.assertIs(result[0][0], embedding)
        self.assertIs(result[0][1]["minimax_refs"], refs)
        guides = result[0][1]["minimax_keyframes"]
        self.assertEqual([g["resolved_frame_index"] for g in guides], [0, 21])
        self.assertEqual(guides[0]["latent"].shape, (1, 24, 7, 4, 4))
        torch.testing.assert_close(guides[0]["latent"][..., 0, 0], guide[:, :, 15:, 0, 0])
        self.assertEqual(metadata["minimax_keyframes"][0]["latent"].shape[2], 22)
        self.assertEqual(metadata["minimax_keyframes"][2]["resolved_frame_index"], 72)

    def test_video_audio_guides_keep_their_timing(self):
        guide = {"resolved_frame_index": 0,
                 "latent": torch.ones(1, 24, 22, 2, 2),
                 "audio_latent": torch.arange(244, dtype=torch.float32).reshape(1, 1, 1, 244)}
        result = temporal.window_conditioning([[None, {"minimax_keyframes": [guide]}]], 51, 124, 2, 2)
        visual, audio = result[0][1]["minimax_keyframes"]
        self.assertEqual(visual["resolved_frame_index"], 0)
        self.assertNotIn("audio_latent", visual)
        self.assertNotIn("latent", audio)
        self.assertEqual(audio["audio_latent"].shape[-1], 122)
        self.assertEqual(audio["audio_latent"].flatten()[0].item(), 85)
        self.assertAlmostEqual(audio["resolved_frame_index"], 0)

    def test_non_aligned_guide_crop_preserves_five_token_phase(self):
        guide = {"resolved_frame_index": 10, "latent": torch.ones(1, 24, 42, 2, 2)}
        result = temporal.window_conditioning([[None, {"minimax_keyframes": [guide]}]], 51, 124, 2, 2)
        cropped = result[0][1]["minimax_keyframes"][0]
        self.assertEqual(cropped["resolved_frame_index"], -7)


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.video = torch.linspace(0, 1, 24 * 107 * 4).reshape(1, 24, 107, 2, 2)
        self.audio = torch.arange(32 * 2 * 603, dtype=torch.float32).reshape(1, 32, 2, 603)
        self.latent = {"samples": NestedTensor((self.video, self.audio)), "batch_index": [0], "custom_metadata": "keep"}

    def run_sample(self, sampler, add_noise=True, **kwargs):
        with patch.object(temporal.comfy.sample, "sample", side_effect=sampler):
            return temporal.sample_temporal(object(), 123, 8, 1, "lcm", "simple", [], [],
                                            self.latent, 4, add_noise, 73, 22, **kwargs)

    def test_bounded_windows_shared_noise_audio_and_metadata(self):
        calls = []
        stages = []
        def sampler(*args, **kwargs):
            calls.append((args, kwargs))
            return args[8]
        output = self.run_sample(sampler, chunk_callback=lambda *data: stages.append(data))
        self.assertEqual(output["samples"].unbind()[0].shape, self.video.shape)
        torch.testing.assert_close(output["samples"].unbind()[0][:, :, :22], self.video[:, :, :22])
        torch.testing.assert_close(output["samples"].unbind()[1], self.audio, rtol=0, atol=0)
        self.assertEqual(output["custom_metadata"], "keep")
        plan = temporal.serial_chunk_plan(107, 73, 22)
        self.assertEqual(len(calls), len(plan))
        self.assertEqual(stages[0], (1, len(plan), 0, 73))
        self.assertEqual(stages[-1][1], len(plan))
        for (args, kwargs), (start, stop, prefix, prefix_audio, *_rest) in zip(calls, plan):
            video, audio = args[8].unbind()
            self.assertEqual(video.shape[2], prefix + stop - start)
            self.assertEqual(video.device.type, "cpu")
            a0 = round(temporal.frame_boundary(start) * temporal.FRAME_RESCALE)
            source_audio = self.audio[..., a0:a0 + audio.shape[-1] - prefix_audio]
            torch.testing.assert_close(audio[..., prefix_audio:], source_audio)
            self.assertEqual(torch.count_nonzero(kwargs["noise_mask"].unbind()[1]), 0)
            self.assertEqual(kwargs["start_step"], 4)
            self.assertEqual(kwargs["last_step"], 8)
            self.assertTrue(kwargs["force_full_denoise"])

    def test_noise_disabled_uses_full_video_denoising_for_warm_start(self):
        mask = torch.ones(1, 1, 107, 2, 2)
        mask[:, :, 20:40] = 0
        self.latent["noise_mask"] = NestedTensor((mask, torch.ones_like(self.audio)))
        masks = []
        def sampler(*args, **kwargs):
            self.assertTrue(kwargs["disable_noise"])
            self.assertEqual(torch.count_nonzero(args[1].unbind()[0]), 0)
            masks.append(kwargs["noise_mask"].unbind()[0])
            return args[8]
        result = self.run_sample(sampler, add_noise=False)
        self.assertTrue(masks[1].all())
        self.assertIs(result["noise_mask"], self.latent["noise_mask"])
        self.assertEqual(result["samples"].unbind()[0].shape, self.video.shape)
        torch.testing.assert_close(result["samples"].unbind()[0][:, :, :22], self.video[:, :, :22])

    def test_cancel_after_first_window(self):
        calls = []
        def sampler(*args, **kwargs):
            calls.append(True)
            return args[8]
        def interrupt():
            if calls:
                raise temporal.mm.InterruptProcessingException()
        with patch.object(temporal.mm, "throw_exception_if_processing_interrupted", side_effect=interrupt):
            with self.assertRaises(temporal.mm.InterruptProcessingException):
                self.run_sample(sampler)
        self.assertEqual(len(calls), 1)

    def test_single_window_and_batch(self):
        self.video = self.video[:, :, :7].repeat(2, 1, 1, 1, 1)
        self.audio = self.audio[..., :37].repeat(2, 1, 1, 1)
        self.latent = {"samples": NestedTensor((self.video, self.audio)), "batch_index": [0, 0]}
        def sampler(*args, **kwargs):
            torch.testing.assert_close(args[1].unbind()[0][0], args[1].unbind()[0][1])
            return args[8]
        result = self.run_sample(sampler)
        torch.testing.assert_close(result["samples"].unbind()[0], self.video, rtol=0, atol=0)
        torch.testing.assert_close(result["samples"].unbind()[1], self.audio, rtol=0, atol=0)


class AudioRefinementTests(unittest.TestCase):
    def test_full_audio_mask_schedule_and_metadata(self):
        video = torch.ones(1, 24, 12, 2, 2)
        audio = torch.ones(1, 32, 2, 65)
        for existing_mask in (False, True):
            latent = {"samples": NestedTensor((video, audio)), "custom_metadata": "keep"}
            if existing_mask:
                latent["noise_mask"] = NestedTensor((torch.ones_like(video), torch.zeros_like(audio)))
            with patch.object(temporal.comfy.sample, "sample", return_value=NestedTensor((video + 9, audio + 2))) as sampler:
                result = temporal.refine_audio(object(), 9, 8, 4, 1, "lcm", "simple", [], [], latent, False)
            args, kwargs = sampler.call_args
            self.assertEqual(args[2], 8)
            self.assertEqual(kwargs["start_step"], 4)
            self.assertEqual(kwargs["last_step"], 8)
            self.assertTrue(kwargs["disable_noise"])
            self.assertEqual(args[8].unbind()[1].shape, audio.shape)
            video_mask, audio_mask = kwargs["noise_mask"].unbind()
            self.assertFalse(video_mask.any())
            self.assertEqual(bool(audio_mask.any()), not existing_mask)
            torch.testing.assert_close(result["samples"].unbind()[0], video, rtol=0, atol=0)
            torch.testing.assert_close(result["samples"].unbind()[1], audio + 2)
            self.assertEqual(result["custom_metadata"], "keep")
            if existing_mask:
                self.assertIs(result["noise_mask"], latent["noise_mask"])


class NativeSamplerTests(unittest.TestCase):
    def test_native_lcm_with_small_h3_model(self):
        config = MiniMaxH3({
            "hidden_size": 32, "num_layers": 0, "token_refiner_num_layers": 0,
            "num_attention_heads": 1, "attention_head_dim": 32, "ffn_hidden_size": 64,
            "text_dim": 32, "timestep_input_dim": 16, "time_embed_hidden_size": 32,
            "time_embed_dim": 32, "rope_inv_freq_len": 4, "dtype": torch.float32,
        })
        model = config.get_model({}, device=torch.device("cpu"))
        patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
        video = torch.ones(1, 24, 12, 2, 2)
        audio = torch.rand(1, 32, 2, 65)
        cond = [[torch.zeros(1, 2, 32), {}]]
        with torch.inference_mode():
            for tensor in model.diffusion_model.state_dict().values():
                tensor.zero_()
            refined = temporal.refine_audio(
                patcher, 9, 4, 2, 1, "lcm", "simple", cond, cond,
                {"samples": NestedTensor((video, audio))}, True,
            )
            torch.testing.assert_close(refined["samples"].unbind()[0], video, rtol=0, atol=0)
            refined_audio = refined["samples"].unbind()[1]
            self.assertTrue(torch.isfinite(refined_audio).all())
            self.assertFalse(torch.equal(refined_audio, audio))
            result = temporal.sample_temporal(
                patcher, 9, 4, 1, "lcm", "simple", cond, cond,
                refined, 2, True, 22, 5,
            )
        out_video, out_audio = result["samples"].unbind()
        self.assertEqual(out_video.shape, video.shape)
        self.assertTrue(torch.isfinite(out_video).all())
        torch.testing.assert_close(out_audio, refined_audio, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
