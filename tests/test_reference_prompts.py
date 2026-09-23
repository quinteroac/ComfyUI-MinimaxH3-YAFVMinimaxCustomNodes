import asyncio
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer
import av
import torch

from test_video_prompts import ROOT, module as prompts, FakeClip, png, server

package = types.ModuleType("reference_prompts_test_package")
package.__path__ = [str(ROOT)]
package.video_prompts = prompts
sys.modules[package.__name__] = package
spec = importlib.util.spec_from_file_location(package.__name__ + ".reference_prompts", ROOT / "reference_prompts.py")
reference = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reference
with patch.dict(sys.modules, {"server": types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=server))}):
    spec.loader.exec_module(reference)


class ReferencePromptTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.video = Path(cls.directory.name) / "video.mp4"
        cls.silent = Path(cls.directory.name) / "silent.mp4"
        cls.audio = Path(cls.directory.name) / "audio.wav"
        cls.run_ffmpeg("-f", "lavfi", "-i", "testsrc2=size=64x32:rate=12:duration=1", "-f", "lavfi", "-i",
                       "sine=frequency=440:sample_rate=24000:duration=1", "-c:v", "libx264", "-c:a", "aac", "-shortest", cls.video)
        cls.run_ffmpeg("-i", cls.video, "-an", "-c:v", "copy", cls.silent)
        cls.run_ffmpeg("-f", "lavfi", "-i", "sine=frequency=880:sample_rate=16000:duration=0.5", cls.audio)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    @staticmethod
    def run_ffmpeg(*args):
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *map(str, args)], check=True)

    async def asyncSetUp(self):
        prompts.library = prompts.PromptLibrary()
        self.pending = []
        server.prompt_queue = types.SimpleNamespace(get_current_queue_volatile=lambda: ([], self.pending))
        app = web.Application(middlewares=[prompts.protect_prompt_submission])
        app.add_routes(prompts.routes)
        app.add_routes(reference.routes)
        self.validation_started = asyncio.Event()
        self.validation_continue = asyncio.Event()
        async def submit(request):
            data = await request.json()
            self.validation_started.set()
            await self.validation_continue.wait()
            self.pending.append((1, "queued", data["prompt"]))
            return web.json_response({})
        app.router.add_post("/prompt", submit)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        prompts.library.collections.clear()
        prompts.library.leases.clear()
        prompts.library.collect(set())

    async def save(self, uploads=None, actions=None, entry=None, prompt="Scene", collection="reference:1"):
        form = FormData()
        fields = {"collection": collection, "prompt": prompt}
        if entry:
            fields.update(entry=entry["id"], base=entry["revision"])
        for name in reference.MEDIA_TYPES:
            fields[name + "_action"] = (actions or {}).get(name, "upload" if name in (uploads or {}) else "keep")
        for key, value in fields.items():
            form.add_field(key, value)
        for name, value in (uploads or {}).items():
            form.add_field(name, value.read_bytes() if isinstance(value, Path) else value, filename=name)
        form._is_multipart = True
        return await self.client.post("/yafv/prompts/reference-entry", data=form)

    async def saved(self, **kwargs):
        response = await self.save(**kwargs)
        self.assertEqual(response.status, 200, await response.text())
        return (await response.json())["entries"][-1]

    def execute(self, entry, **kwargs):
        return reference.YAFVReferenceVideoPrompts().execute("reference:1", entry["revision"], **kwargs)

    async def test_outputs_native_types_video_rate_and_extracted_audio(self):
        entry = await self.saved(uploads={"ref_image_0": png(), "ref_image_1": png((30, 90)),
                                         "ref_video_0": self.video, "ref_audio_0": self.audio,
                                         "ref_audio_1": self.audio, "ref_audio_2": self.audio})
        self.assertEqual(entry["media_names"]["ref_video_0"], "ref_video_0")
        self.assertEqual(entry["media_names"]["ref_video_audio_0"], "ref_video_0 · soundtrack")
        outputs = self.execute(entry)["result"]
        self.assertEqual(len(outputs), 16)
        self.assertEqual(outputs[0], "Scene")
        self.assertEqual(tuple(outputs[1].shape), (1, 32, 64, 3))
        self.assertEqual(tuple(outputs[2].shape), (1, 90, 30, 3))
        self.assertEqual(tuple(outputs[3].shape), (24, 32, 64, 3))
        self.assertFalse(torch.equal(outputs[3][0], outputs[3][-1]))
        self.assertEqual(outputs[4]["sample_rate"], 24000)
        self.assertEqual(tuple(outputs[5]["waveform"].shape), (1, 1, 8000))
        self.assertTrue(all(output["waveform"].dtype == torch.float32 for output in outputs[4:8]))
        path = prompts.library.revisions[entry["revision"]].media["ref_video_0"]
        with av.open(path) as container:
            self.assertEqual(container.streams.video[0].average_rate, 24)
        response = await self.client.get(f'/yafv/prompts/media/{entry["revision"]}/ref_video_0?collection=reference:1',
                                         headers={"Range": "bytes=0-31"})
        self.assertEqual(response.status, 206)
        self.assertEqual(len(await response.read()), 32)

    async def test_missing_media_and_text_passthrough(self):
        entry = await self.saved(prompt="  Original prompt.\n")
        for kwargs in ({}, {"clip": FakeClip()}, {"text": "Expand"}, {"clip": FakeClip(), "text": " "}):
            self.assertEqual(self.execute(entry, **kwargs)["result"], ("  Original prompt.\n",) + (None,) * 15)
        inputs = reference.YAFVReferenceVideoPrompts.INPUT_TYPES()
        self.assertEqual(inputs["optional"].pop("context_video")[0], "VIDEO")
        self.assertEqual(inputs, prompts.YAFVVideoPrompts.INPUT_TYPES())

    async def test_visual_generation_tag_order_and_sampling(self):
        entry = await self.saved(uploads={"ref_image_1": png(), "ref_video_0": self.video, "ref_audio_2": self.audio})
        clip = FakeClip()
        result = self.execute(entry, clip=clip, text="Expand precisely", max_length=800, seed=19, sampling_mode="on")
        self.assertEqual(result["result"][0], "A generated video prompt.")
        prompt = clip.calls[0][1]
        self.assertIn("ref_image_1 = <Picture 1>", prompt)
        self.assertIn("ref_video_audio_0 = <Audio 1>", prompt)
        self.assertIn("ref_audio_2 = <Audio 2>", prompt)
        self.assertIn("not analyzed", prompt)
        self.assertIsNone(clip.calls[0][2]["audio"])
        self.assertEqual(tuple(clip.calls[0][2]["image"].shape), (1, 1260, 1536, 3))
        self.assertEqual(clip.calls[1][2]["seed"], 19)
        draw = Mock(wraps=reference.ImageDraw.Draw(reference.Image.new("RGB", (1536, 1260))))
        with patch.object(reference.ImageDraw, "Draw", return_value=draw):
            reference.reference_sheet(dict(zip(reference.MEDIA_TYPES, result["result"][1:])))
        labels = [call.args[1] for call in draw.text.call_args_list]
        self.assertEqual(len(labels), 9)
        self.assertEqual(labels[1], "<Video 1> 0.00s")
        self.assertEqual(labels[-1], "<Video 1> 0.96s")
        clip.tokenize = lambda *args, **kwargs: {"tokens": [1, 2]}
        with self.assertRaisesRegex(ValueError, "vision-capable model"):
            self.execute(entry, clip=clip, text="Expand")

    async def test_audio_replacement_removal_and_silent_video(self):
        entry = await self.saved(uploads={"ref_video_0": self.video, "ref_video_audio_0": self.audio})
        self.assertEqual(self.execute(entry)["result"][4]["sample_rate"], 16000)
        entry = await self.saved(entry=entry, actions={"ref_video_audio_0": "remove"})
        self.assertIsNone(self.execute(entry)["result"][4])
        entry = await self.saved(entry=entry, uploads={"ref_video_0": self.video})
        self.assertEqual(self.execute(entry)["result"][4]["sample_rate"], 24000)
        entry = await self.saved(entry=entry, uploads={"ref_video_0": self.silent})
        self.assertIsNone(self.execute(entry)["result"][4])
        entry = await self.saved(entry=entry, uploads={"ref_video_0": self.video}, actions={"ref_video_audio_0": "remove"})
        self.assertIsNone(self.execute(entry)["result"][4])

    async def test_eight_images_two_videos_and_stable_output_indices(self):
        node = reference.YAFVReferenceVideoPrompts
        self.assertEqual(node.RETURN_NAMES[:8], ("generated_prompt", "ref_image_0", "ref_image_1", "ref_video_0",
                                               "ref_video_audio_0", "ref_audio_0", "ref_audio_1", "ref_audio_2"))
        uploads = {f"ref_image_{i}": png((32 + i, 24)) for i in range(8)}
        uploads.update(ref_video_0=self.video, ref_video_1=self.video, ref_video_audio_1=self.audio,
                       ref_audio_0=self.audio, ref_audio_1=self.audio, ref_audio_2=self.audio)
        entry = await self.saved(uploads=uploads)
        outputs = dict(zip(node.RETURN_NAMES, self.execute(entry)["result"]))
        self.assertEqual(len(outputs), 16)
        for i in range(8):
            self.assertEqual(tuple(outputs[f"ref_image_{i}"].shape), (1, 24, 32 + i, 3))
        for i in range(2):
            self.assertEqual(tuple(outputs[f"ref_video_{i}"].shape), (24, 32, 64, 3))
        self.assertEqual(outputs["ref_video_audio_0"]["sample_rate"], 24000)
        self.assertEqual(outputs["ref_video_audio_1"]["sample_rate"], 16000)
        sheet, description = reference.reference_sheet(outputs)
        self.assertEqual(sheet.size, (1536, 3360))
        self.assertIn("ref_image_7 = <Picture 8>", description)
        self.assertIn("ref_video_1 = <Video 2>", description)
        self.assertIn("ref_video_audio_1 = <Audio 2>", description)
        self.assertIn("ref_audio_2 = <Audio 5>", description)
        entry = await self.saved(entry=entry, actions={"ref_video_1": "remove", "ref_image_3": "remove"})
        outputs = dict(zip(node.RETURN_NAMES, self.execute(entry)["result"]))
        self.assertIsNone(outputs["ref_image_3"])
        self.assertEqual(tuple(outputs["ref_image_4"].shape), (1, 24, 36, 3))
        self.assertIsNone(outputs["ref_video_1"])
        self.assertIsNone(outputs["ref_video_audio_1"])
        self.assertEqual(outputs["ref_video_audio_0"]["sample_rate"], 24000)

    async def test_second_video_audio_actions_and_sparse_tags(self):
        entry = await self.saved(uploads={"ref_image_7": png(), "ref_video_1": self.video, "ref_audio_2": self.audio})
        clip = FakeClip()
        self.execute(entry, clip=clip, text="Expand")
        text = clip.calls[0][1]
        self.assertIn("ref_image_7 = <Picture 1>", text)
        self.assertIn("ref_video_1 = <Video 1>", text)
        self.assertIn("ref_video_audio_1 = <Audio 1>", text)
        self.assertIn("ref_audio_2 = <Audio 2>", text)
        entry = await self.saved(entry=entry, actions={"ref_video_audio_1": "remove"})
        self.assertIsNone(self.execute(entry)["result"][-1])
        entry = await self.saved(entry=entry, uploads={"ref_video_1": self.video})
        self.assertEqual(self.execute(entry)["result"][-1]["sample_rate"], 24000)
        entry = await self.saved(entry=entry, uploads={"ref_video_1": self.silent})
        self.assertIsNone(self.execute(entry)["result"][-1])

    async def test_invalid_upload_is_atomic_and_cleans_files(self):
        entry = await self.saved(uploads={"ref_image_0": png()})
        before = set(Path(reference.media_directory.name).iterdir())
        response = await self.save(entry=entry, uploads={"ref_image_0": png(), "ref_audio_0": b"broken audio"})
        self.assertEqual(response.status, 400)
        self.assertIn("ref_audio_0", (await response.json())["error"])
        self.assertEqual(prompts.library.view("reference:1")["entries"][0]["revision"], entry["revision"])
        self.assertEqual(set(Path(reference.media_directory.name).iterdir()), before)
        self.assertFalse(prompts.library.leases)

    async def test_revision_files_survive_queue_edit_delete_and_then_release(self):
        entry = await self.saved(uploads={"ref_video_0": self.video})
        old = prompts.library.revisions[entry["revision"]]
        submission = {"prompt": {"1": {"class_type": "YAFVReferenceVideoPrompts", "inputs": {"revision_id": old.id}}}}
        pending = asyncio.create_task(self.client.post("/prompt", json=submission))
        await self.validation_started.wait()
        newer = await self.saved(entry=entry, uploads={"ref_video_0": self.silent})
        self.assertTrue(all(Path(p).exists() for p in old.media.values() if p))
        self.validation_continue.set()
        self.assertEqual((await pending).status, 200)
        await self.client.delete(f'/yafv/prompts/entry/{newer["id"]}?collection=reference:1')
        self.assertIsNotNone(self.execute(entry)["result"][4])
        self.pending.clear()
        prompts.collect_unused()
        self.assertTrue(all(not Path(p).exists() for p in old.media.values() if p))
        self.assertFalse(prompts.library.revisions)

    async def test_concurrent_media_edit_keeps_files_until_conflict_resolves(self):
        entry = await self.saved(uploads={"ref_image_0": png()})
        old_path = prompts.library.revisions[entry["revision"]].media["ref_image_0"]
        started, resume = asyncio.Event(), asyncio.Event()
        original = reference.ffmpeg
        async def paused_ffmpeg(*args):
            started.set()
            await resume.wait()
            await original(*args)
        with patch.object(reference, "ffmpeg", side_effect=paused_ffmpeg):
            slow = asyncio.create_task(self.save(entry=entry, uploads={"ref_video_0": self.silent}))
            await started.wait()
            await self.saved(entry=entry, actions={"ref_image_0": "remove"})
            self.assertTrue(Path(old_path).exists())
            resume.set()
            response = await slow
        self.assertEqual(response.status, 400)
        self.assertIn("another window", (await response.json())["error"])
        self.assertFalse(Path(old_path).exists())
        self.assertFalse(prompts.library.leases)
        self.assertFalse(list(Path(reference.media_directory.name).iterdir()))

    async def test_shared_files_and_collection_isolation(self):
        entry = await self.saved(uploads={"ref_image_0": png()})
        path = prompts.library.revisions[entry["revision"]].media["ref_image_0"]
        newer = await self.saved(entry=entry, prompt="Edited")
        self.assertTrue(Path(path).exists())
        self.assertNotIn(entry["revision"], prompts.library.revisions)
        response = await self.client.get(f'/yafv/prompts/media/{newer["revision"]}/ref_image_0?collection=another')
        self.assertEqual(response.status, 404)
        response = await self.save(entry=entry, prompt="Stale")
        self.assertEqual(response.status, 400)
        await self.client.delete(f'/yafv/prompts/entry/{newer["id"]}?collection=reference:1')
        self.assertFalse(Path(path).exists())


if __name__ == "__main__":
    unittest.main()
