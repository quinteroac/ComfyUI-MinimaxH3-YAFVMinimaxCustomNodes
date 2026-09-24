from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from test_two_pass import sampling, defaults
import torch
from comfy.nested_tensor import NestedTensor
from yafv_h3 import project_nodes as project
from yafv_h3.project_continuity import apply_context, project_lengths, trim_media


def av(frames=124, size=2, value=1.0):
    tokens = 2 + (frames - 5) // 17 * 5
    return {"samples": NestedTensor((
        torch.full((1, 24, tokens, size, size), value),
        torch.full((1, 32, 2, round(frames * 5 / 3)), value),
    ))}


class ContinuityTests(unittest.TestCase):
    def test_trim_ends_at_last_generated_frame(self):
        self.assertEqual(project_lengths(124), (124, 124))
        self.assertEqual(project_lengths(124, 22), (158, 136))
        for context in (5, 22, 39, 56):
            for requested in (5, 17, 124, 250):
                generated, output = project_lengths(requested, context)
                self.assertEqual(generated % 17, 5)
                self.assertEqual(output % 17, 0)
                self.assertGreaterEqual(output, requested)
                images = torch.arange(generated)
                audio = {"waveform": torch.arange(generated * 100).reshape(1, 1, -1), "sample_rate": 2400}
                frames, sound = trim_media(images, audio, context, output)
                self.assertEqual(frames[0].item(), context)
                self.assertEqual(frames[-1].item(), generated - 1)
                self.assertEqual(sound["waveform"][0, 0, -1].item(), generated * 100 - 1)

    def test_second_pass_restores_high_resolution_context(self):
        low = av(39, 2, 3.0)
        high = av(39, 4, 4.0)
        context = av(124, 4, 9.0)
        latent = dict(av(39), yafv_context_length=22)
        args = dict(defaults(), model_pass1=Mock(), positive=[], negative=[], latent=latent,
                    context_latent_pass2=context, enable_latent_upscale=True)
        server = Mock(client_id="test")
        for mode in ("full", "temporal"):
            with self.subTest(mode=mode), \
                 patch.object(sampling.PromptServer, "instance", server, create=True), \
                 patch.object(sampling, "common_ksampler", side_effect=[(low,), (high,)]) as sampler, \
                 patch.object(sampling, "sample_temporal", return_value=high) as temporal, \
                 patch.object(sampling.H3Upscaler, "upscale", return_value={"samples": high["samples"].unbind()[0]}):
                result = sampling.MiniMaxH3TwoPassSampler().sample(**dict(args, pass2_sampling_mode=mode))
            call = sampler.call_args_list[1] if mode == "full" else temporal.call_args
            video, audio = call.args[8]["samples"].unbind()
            vm, am = call.args[8]["noise_mask"].unbind()
            self.assertTrue((video[:, :, :7] == 9).all())
            self.assertTrue((video[:, :, 7:] == 4).all())
            self.assertTrue((audio[..., :37] == 9).all())
            self.assertFalse(vm[:, :, :7].any())
            self.assertFalse(am[..., :37].any())
            self.assertTrue(vm[:, :, 7:].all())
            self.assertTrue(result["result"][1]["yafv_clean"])
            self.assertIs(result["result"][1]["samples"], low["samples"])
            self.assertTrue((low["samples"].unbind()[0] == 3).all())

    def test_restored_guides_and_resolution_check(self):
        embedding = torch.zeros(1)
        cond = [[embedding, {"minimax_refs": ["keep"]}]]
        out, trim, latent = apply_context(cond, av(39, 4), av(124, 4, 9), 22, 0, 0, False)
        self.assertEqual(trim, 22)
        self.assertEqual(out[0][1]["minimax_refs"], ["keep"])
        self.assertEqual([k["resolved_frame_index"] for k in out[0][1]["minimax_keyframes"][:-1]],
                         [0, 1, 5, 9, 13, 17, 18])
        with self.assertRaisesRegex(ValueError, "resolution"):
            apply_context(cond, av(39, 4), av(124, 2), 22)


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=64x64:r=24",
                        "-frames:v", "124", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(self.source)], check=True)
        for target, name, value in (
            (project.folder_paths, "get_output_directory", lambda: str(self.root)),
            (project, "_local_video_source", lambda video: (str(self.source), [])),
            (project, "_project_output", lambda name, manifest: manifest),
        ):
            patcher = patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_approved_generation_roundtrip_and_no_vae_encode(self):
        for value in (1.0, 7.0):
            project.YAFVProjectCommit.execute("test", 0, None, latent_pass1=av(value=value),
                                              latent_final=av(size=4, value=value))
        manifest = project._load_manifest("test")
        manifest["segments"]["0"]["active_generation"] = 0
        project._save_manifest("test", manifest)
        directory = project._project_dir("test")
        record = project._active_generation(manifest, 0, approved_only=True)
        self.assertTrue((directory / record["context_latents"]).is_file())
        low, high = project._load_context_latents(directory, record)
        for a, b in zip(low["samples"].unbind(), av()["samples"].unbind()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        vae = Mock()
        with patch.object(project, "_extract_frame", return_value=torch.zeros(32, 32, 3)):
            output = project.YAFVProjectContext.execute("test", 1, target_width=32, target_height=32,
                                                        vae=vae, audio_vae=vae).result
        vae.encode.assert_not_called()
        self.assertEqual(output[6:8], (158, 136))
        self.assertTrue((output[5]["samples"].unbind()[0] == 1).all())
        self.assertEqual(output[8]["samples"].unbind()[0].shape[-2:], (4, 4))
        self.assertNotIn("noise_mask", output[5])
        with patch.object(project, "_extract_frame", return_value=torch.zeros(64, 64, 3)):
            with self.assertRaisesRegex(ValueError, "different resolution"):
                project.YAFVProjectContext.execute("test", 1, target_width=64, target_height=64)
        output = project.YAFVProjectContext.execute("test", 1, scene_mode="New scene").result
        self.assertIsNone(output[5])
        self.assertIsNone(output[8])
        self.assertEqual(output[6:8], (124, 124))

    def test_reject_invisible_tail_and_noisy_context_before_commit(self):
        with self.assertRaisesRegex(ValueError, "final frame"):
            project.YAFVProjectCommit.execute("bad", 0, None, latent_pass1=av(158),
                                              latent_final=av(158, 4), trim_frames=22)
        self.assertFalse(project._manifest_path("bad").exists())
        noisy = dict(av(), yafv_clean=False)
        with self.assertRaisesRegex(ValueError, "leftover_noise"):
            project.YAFVProjectCommit.execute("bad", 0, None, latent_pass1=noisy, latent_final=av(size=4))
        self.assertFalse(project._manifest_path("bad").exists())

    def test_continuation_saves_the_visible_endpoint(self):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=s=64x64:r=24",
                        "-frames:v", "136", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(self.source)], check=True)
        project.YAFVProjectCommit.execute("continued", 1, None, latent_pass1=av(158),
                                          latent_final=av(158, 4), trim_frames=22)
        record = project._active_generation(project._load_manifest("continued"), 1, True)
        self.assertEqual((record["trim_frames"], record["output_frames"]), (22, 136))
        low, high = project._load_context_latents(project._project_dir("continued"), record)
        for saved, original in ((low, av(158)), (high, av(158, 4))):
            for a, b in zip(saved["samples"].unbind(), original["samples"].unbind()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_legacy_mp4_fallback_and_path_containment(self):
        project.YAFVProjectCommit.execute("legacy", 0, None)
        vae, audio_vae = Mock(), Mock()
        vae.encode.return_value = av(22)["samples"].unbind()[0]
        with patch.object(project, "_extract_frame", return_value=torch.zeros(32, 32, 3)), \
             patch.object(project, "_encode_context_av", return_value=av(22)):
            output = project.YAFVProjectContext.execute("legacy", 1, target_width=32, target_height=32,
                                                        vae=vae, audio_vae=audio_vae).result
        vae.encode.assert_called_once()
        self.assertIsNotNone(output[5])
        self.assertIsNone(output[8])
        with self.assertRaisesRegex(ValueError, "Invalid project context path"):
            project._load_context_latents(project._project_dir("legacy"), {"context_latents": "../../escape"})


if __name__ == "__main__":
    unittest.main()
