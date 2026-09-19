import unittest
from unittest.mock import Mock, patch

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

    def test_56_22_windows_keep_exact_overlap_without_redundant_tail(self):
        windows = temporal.temporal_windows(107, 56, 22, exact_tail=True)
        self.assertEqual(windows, [(start, start + 17) for start in range(0, 91, 10)])
        self.assertEqual(temporal.temporal_windows(22, 56, 22, exact_tail=True),
                         [(0, 17), (10, 22)])


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
        self.latent = {"samples": NestedTensor((self.video, self.audio)), "custom_metadata": "keep"}
        self.model = Mock()

    def test_one_solver_run_with_full_state_and_frozen_audio(self):
        with patch.object(temporal.comfy.sample, "sample", return_value=self.latent["samples"]) as sampler:
            output = temporal.sample_temporal(
                self.model, 123, 8, 2, "er_sde", "simple", [], [],
                self.latent, 4, True, 56, 22,
            )
        sampler.assert_called_once()
        args, kwargs = sampler.call_args
        self.assertIs(args[0], self.model.clone.return_value)
        self.assertEqual(args[4], "er_sde")
        torch.testing.assert_close(args[8].unbind()[0], self.video)
        video_mask, audio_mask = kwargs["noise_mask"].unbind()
        self.assertTrue(video_mask.all())
        self.assertFalse(audio_mask.any())
        self.assertEqual(kwargs["start_step"], 4)
        self.assertEqual(kwargs["last_step"], 8)
        self.assertTrue(kwargs["force_full_denoise"])
        self.model.add_wrapper_with_key.assert_not_called()
        wrapper = self.model.clone.return_value.add_wrapper_with_key.call_args.args[2]
        self.assertEqual(len(wrapper.windows), 10)
        torch.testing.assert_close(output["samples"].unbind()[1], self.audio, rtol=0, atol=0)
        self.assertEqual(output["custom_metadata"], "keep")

    def test_no_noise_and_single_window(self):
        latent = {"samples": NestedTensor((self.video[:, :, :7], self.audio[..., :37]))}
        with patch.object(temporal.comfy.sample, "sample", return_value=latent["samples"]) as sampler:
            output = temporal.sample_temporal(
                self.model, 123, 8, 1, "lcm", "simple", [], [], latent, 4, False, 56, 22,
            )
        self.model.clone.return_value.add_wrapper_with_key.assert_not_called()
        self.assertFalse(sampler.call_args.args[1].unbind()[0].any())
        self.assertTrue(sampler.call_args.kwargs["disable_noise"])
        torch.testing.assert_close(output["samples"].unbind()[0], latent["samples"].unbind()[0])

    def test_conditioning_cache_is_released_on_sampler_failure(self):
        def fail(*args, **kwargs):
            wrapper = self.model.clone.return_value.add_wrapper_with_key.call_args.args[2]
            wrapper.conditionings["test"] = torch.ones(1)
            raise RuntimeError("sampling failed")
        with patch.object(temporal.comfy.sample, "sample", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "sampling failed"):
                temporal.sample_temporal(
                    self.model, 123, 8, 1, "lcm", "simple", [], [], self.latent, 4, True, 56, 22,
                )
        wrapper = self.model.clone.return_value.add_wrapper_with_key.call_args.args[2]
        self.assertFalse(wrapper.conditionings)

    def test_shared_state_at_every_evaluation_and_no_window_feedback(self):
        windows = temporal.temporal_windows(107, 56, 22, exact_tail=True)
        stages = []
        wrapper = temporal.TemporalDenoising(
            [self.video.shape, self.audio.shape], windows, 123, lambda *args: stages.append(args),
        )
        state, shapes = temporal.comfy.utils.pack_latents((self.video, self.audio))
        conds = [[], None]
        for step in range(3):
            calls = []
            current_video, current_audio = temporal.comfy.utils.unpack_latents(state, shapes)
            def executor(model, local_conds, window, sigma, options):
                index = len(calls)
                start, stop = windows[index]
                a0 = round(temporal.frame_boundary(start) * temporal.FRAME_RESCALE)
                a1 = round(temporal.frame_boundary(stop) * temporal.FRAME_RESCALE)
                local_shapes = [(1, 24, stop - start, 2, 2), (1, 32, 2, a1 - a0)]
                video, audio = temporal.comfy.utils.unpack_latents(window, local_shapes)
                torch.testing.assert_close(video, current_video[:, :, start:stop], rtol=0, atol=0)
                torch.testing.assert_close(audio, current_audio[..., a0:a1], rtol=0, atol=0)
                self.assertEqual(float(sigma), 1.0 / (step + 1))
                self.assertIsNone(local_conds[1])
                calls.append(True)
                prediction, _ = temporal.comfy.utils.pack_latents((video + index + 1, audio))
                return [prediction, torch.zeros_like(prediction)]
            predicted = wrapper(executor, Mock(), conds, state, torch.tensor(1.0 / (step + 1), dtype=torch.float64), {})
            self.assertEqual(len(calls), 10)
            out_video = temporal.comfy.utils.unpack_latents(predicted[0], shapes)[0]
            # Both windows predict the same positions from identical x at this step.
            expected = current_video[:, :, 10:17] + 1 + torch.arange(1, 8).reshape(1, 1, 7, 1, 1) / 8
            torch.testing.assert_close(out_video[:, :, 10:17], expected)
            state, _ = temporal.comfy.utils.pack_latents((out_video, self.audio))
        self.assertEqual(len(stages), 30)
        self.assertEqual(stages[0], (1, 10, 0, 56))

    def test_conditioning_rebuilt_once_per_window_with_local_shapes(self):
        windows = temporal.temporal_windows(107, 56, 22, exact_tail=True)
        wrapper = temporal.TemporalDenoising([self.video.shape, self.audio.shape], windows, 123, None)
        state, _ = temporal.comfy.utils.pack_latents((self.video, self.audio))
        conds = [[{"model_conds": {}, "minimax_keyframes": [
            {"resolved_frame_index": 40, "latent": torch.ones(1, 24, 1, 2, 2)},
        ]}], [{"model_conds": {}}]]
        model = Mock()
        model.extra_conds.return_value = {}
        def executor(model, local_conds, window, sigma, options):
            return [window, window]
        for _ in range(2):
            wrapper(executor, model, conds, state, torch.tensor(0.5), {})
        self.assertEqual(model.extra_conds.call_count, 2 * len(windows))
        params = model.extra_conds.call_args_list[2].kwargs
        self.assertEqual(params["latent_shapes"][0][2], 17)
        self.assertEqual(params["minimax_keyframes"][0]["resolved_frame_index"], 6)
        mask_video, mask_audio = temporal.comfy.utils.unpack_latents(params["denoise_mask"], params["latent_shapes"])
        self.assertTrue(mask_video.all())
        self.assertFalse(mask_audio.any())
        self.assertEqual(conds[0][0]["minimax_keyframes"][0]["resolved_frame_index"], 40)

    def test_large_overlaps_and_short_tail_preserve_identity(self):
        for frames, overlap in ((56, 39), (73, 56), (22, 5)):
            windows = temporal.temporal_windows(107, frames, overlap, exact_tail=True)
            wrapper = temporal.TemporalDenoising([self.video.shape, self.audio.shape], windows, 123, None)
            state, shapes = temporal.comfy.utils.pack_latents((self.video, self.audio))
            result = wrapper(lambda model, conds, x, sigma, options: [x], Mock(), [[]], state, torch.tensor(0.5), {})
            video = temporal.comfy.utils.unpack_latents(result[0], shapes)[0]
            torch.testing.assert_close(video, self.video)

    def test_cancel_between_windows(self):
        windows = temporal.temporal_windows(107, 56, 22, exact_tail=True)
        wrapper = temporal.TemporalDenoising([self.video.shape, self.audio.shape], windows, 123, None)
        state, _ = temporal.comfy.utils.pack_latents((self.video, self.audio))
        calls = []
        def executor(model, conds, x, sigma, options):
            calls.append(True)
            return [x]
        def interrupt():
            if calls:
                raise temporal.mm.InterruptProcessingException()
        with patch.object(temporal.mm, "throw_exception_if_processing_interrupted", side_effect=interrupt):
            with self.assertRaises(temporal.mm.InterruptProcessingException):
                wrapper(executor, Mock(), [[]], state, torch.tensor(0.5), {})
        self.assertEqual(len(calls), 1)


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
    def test_native_solvers_keep_global_history_with_small_h3_model(self):
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
            for sampler in ("lcm", "er_sde", "heun"):
                for cfg in (1, 2):
                    with self.subTest(sampler=sampler, cfg=cfg):
                        full = temporal.sample_temporal(
                            patcher, 9, 5, cfg, sampler, "simple", cond, cond,
                            refined, 2, True, 39, 5,
                        )
                        with patch.object(model, "apply_model", wraps=model.apply_model) as apply:
                            result = temporal.sample_temporal(
                                patcher, 9, 5, cfg, sampler, "simple", cond, cond,
                                refined, 2, True, 22, 5,
                            )
                        out_video, out_audio = result["samples"].unbind()
                        self.assertTrue(torch.isfinite(out_video).all())
                        torch.testing.assert_close(out_video, full["samples"].unbind()[0])
                        torch.testing.assert_close(out_audio, refined_audio, rtol=0, atol=0)
                        self.assertGreaterEqual(apply.call_count, 6)
                        for call in apply.call_args_list:
                            self.assertEqual(call.kwargs["latent_shapes"][0][2], 7)
                        self.assertFalse(patcher.get_all_wrappers(temporal.WrappersMP.CALC_COND_BATCH))


if __name__ == "__main__":
    unittest.main()
