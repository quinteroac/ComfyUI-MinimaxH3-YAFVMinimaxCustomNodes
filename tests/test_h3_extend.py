import unittest
from unittest.mock import Mock, patch

from test_two_pass import sampling
import torch
from comfy.ldm.minimax.model import PackedLayout, FRAME_RESCALE
from comfy.nested_tensor import NestedTensor
from yafv_h3 import h3_extend as extend, temporal_sampling


class ExtendTests(unittest.TestCase):
    def setUp(self):
        self.video = torch.arange(24 * 7 * 4).reshape(1, 24, 7, 2, 2).float()
        self.audio = torch.ones(1, 32, 2, 60)
        self.latent = {"samples": NestedTensor((self.video, self.audio))}

    def test_native_context_positions_and_reference_origin(self):
        for count in (1, 2, 6, 64):
            _, _, guides = extend._context_keyframes(self.latent, count)
            refs = [{"kind": "audio", "ref_audio_t": 8}]
            layout = PackedLayout(3, 7, 2, 2, 37, keyframes=guides, refs=refs)
            origin = float(layout.position_ids[layout.segments[-1][0], 0])
            self.assertEqual(origin, 11)
            positions = [float(layout.position_ids[a, 0]) - origin
                         for a, b, kind in layout.segments if kind == "cond"]
            self.assertAlmostEqual(positions[-1], 0)
            self.assertEqual(len(positions), min(count, 7))
            if count == 6:
                torch.testing.assert_close(torch.tensor(positions),
                                           torch.tensor([-17, -16, -12, -8, -4, 0]) * FRAME_RESCALE)
            a, b, _ = next(s for s in layout.segments if s[2] == "cond_audio")
            audio_start = float(layout.position_ids[a, 0]) - origin
            self.assertAlmostEqual(audio_start + (b - a) / 2, 0)
            self.assertNotEqual(guides[0]["latent"].untyped_storage().data_ptr(),
                                self.video.untyped_storage().data_ptr())

    def test_temporal_context_survives_and_resizes(self):
        _, _, guides = extend._context_keyframes(self.latent, 2)
        metadata = {"minimax_keyframes": guides}
        cropped = temporal_sampling.window_conditioning([[None, metadata]], 51, 124, 4, 4)[0][1]
        result = cropped["minimax_keyframes"]
        self.assertEqual([g["resolved_frame_index"] for g in result],
                         [g["resolved_frame_index"] for g in guides])
        self.assertEqual(result[0]["latent"].shape[-2:], (4, 4))
        self.assertEqual(guides[0]["latent"].shape[-2:], (2, 2))
        PackedLayout(3, 22, 4, 4, 120, keyframes=result)

    def test_node_with_endpoints_refs_and_native_layout(self):
        clip = Mock()
        clip.encode_from_tokens_scheduled.return_value = [[torch.zeros(1, 3, 4), {}]]
        vae = Mock()
        vae.encode.return_value = torch.ones(1, 24, 1, 2, 2)
        image = torch.ones(1, 32, 32, 3)
        original = PackedLayout.__init__
        positive, latent = extend.YAFVH3VideoExtend().run(
            clip, vae, self.latent, "continue", 22, first_frame=image,
            last_frame=image, ref_images=image.repeat(2, 1, 1, 1))
        metadata = positive[0][1]
        self.assertEqual(len(metadata["minimax_refs"]), 2)
        self.assertEqual([g["resolved_frame_index"] for g in metadata["minimax_keyframes"][-2:]], [0, 21])
        PackedLayout(3, 7, 2, 2, 37, keyframes=metadata["minimax_keyframes"],
                     refs=metadata["minimax_refs"])
        self.assertIs(PackedLayout.__init__, original)
        vae.decode.assert_not_called()
        self.assertTrue(latent["samples"].is_nested)

    def test_pin_decodes_last_pixel_frame(self):
        vae = Mock()
        decoded = torch.arange(1 * 5 * 32 * 32 * 3).reshape(1, 5, 32, 32, 3).float()
        vae.decode.return_value = decoded
        result = extend._pin_last_context_frame(vae, self.latent, 32, 32, lambda x, *a: x)
        torch.testing.assert_close(result["image"], decoded[:, -1])
        self.assertEqual(vae.decode.call_args.args[0].shape[2], 6)

    def test_encode_audio_and_missing_vae(self):
        vae = Mock()
        vae.encode.return_value = self.video
        node = extend.YAFVH3EncodeAV()
        images = torch.zeros(22, 32, 32, 4)
        with self.assertRaisesRegex(ValueError, "audio_vae"):
            node.run(vae, images, audio={})
        with patch.object(extend.native, "_encode_ref_audio", return_value=(self.audio, 60)):
            result = node.run(vae, images, audio_vae=object(), audio={})[0]
        self.assertEqual(vae.encode.call_args.args[0].shape[-1], 3)
        self.assertTrue(result["samples"].is_nested)
        self.assertIs(node.run(vae, images)[0]["samples"], self.video)
