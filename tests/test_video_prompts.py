import asyncio
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
from comfy_extras.nodes_textgen import TextGenerate

server = types.SimpleNamespace(routes=web.RouteTableDef(), app=web.Application())
spec = importlib.util.spec_from_file_location("video_prompts_under_test", ROOT / "video_prompts.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
with patch.dict(sys.modules, {"server": types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=server))}):
    spec.loader.exec_module(module)


def png(size=(64, 32), color="red"):
    result = io.BytesIO()
    Image.new("RGB", size, color).save(result, "PNG")
    return result.getvalue()


class FakeClip:
    def __init__(self):
        self.calls = []

    def tokenize(self, text, **kwargs):
        self.calls.append(("tokenize", text, kwargs))
        return {"tokens": [1, {"type": "image", "data": kwargs["image"]}, 2]} if kwargs.get("image") is not None else {"tokens": [1, 2]}

    def generate(self, tokens, **kwargs):
        self.calls.append(("generate", tokens, kwargs))
        return [3, 4]

    def decode(self, tokens):
        self.calls.append(("decode", tokens))
        return "A generated video prompt."


class VideoPromptTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        module.library = module.PromptLibrary()
        self.pending = []
        server.prompt_queue = types.SimpleNamespace(get_current_queue_volatile=lambda: ([], self.pending))
        self.app = web.Application(middlewares=[module.protect_prompt_submission])
        self.app.add_routes(module.routes)
        self.validation_started = asyncio.Event()
        self.validation_continue = asyncio.Event()
        async def submit(request):
            data = await request.json()
            self.validation_started.set()
            await self.validation_continue.wait()
            self.pending.append((1, "queued", data["prompt"]))
            return web.json_response({})
        self.app.router.add_post("/prompt", submit)
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def save(self, prompt="A quiet scene", first=None, last=None, collection="graph-a:1", entry=None, base=None, first_action=None):
        form = FormData()
        for key, value in {"collection": collection, "prompt": prompt, "entry": entry, "base": base}.items():
            if value is not None:
                form.add_field(key, value)
        for key, raw in [("first", first), ("last", last)]:
            form.add_field(key + "_action", "upload" if raw else (first_action if key == "first" and first_action else "keep"))
            if raw:
                form.add_field(key, raw, filename=key + ".png", content_type="image/png")
        # Force multipart even for a text-only entry, matching browser FormData.
        form._is_multipart = True
        return await self.client.post("/yafv/prompts/entry", data=form)

    async def test_text_passthrough_all_missing_input_combinations(self):
        item = module.library.save("scope", "  Preserve my exact prompt.\n", None, None)
        for clip, instructions in [(None, None), (FakeClip(), None), (None, "Instructions"), (FakeClip(), "   ")]:
            result = module.YAFVVideoPrompts().execute("scope", item.id, clip=clip, text=instructions)
            self.assertEqual(result["result"], (item.prompt, None, None))
            if clip:
                self.assertFalse(clip.calls)

    async def test_media_browser_sources_filtering_and_containment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, outputs = root / "input", root / "output"
            inputs.mkdir()
            outputs.mkdir()
            (inputs / "scenes").mkdir()
            (inputs / "frame.png").write_bytes(png())
            (inputs / "private.txt").write_text("not media")
            (outputs / "result.png").write_bytes(png())
            (root / "outside.png").write_bytes(png())
            (inputs / "escape.png").symlink_to(root / "outside.png")
            with patch.object(module.folder_paths, "get_directory_by_type", side_effect=lambda kind: str(root / kind)):
                response = await self.client.get("/yafv/prompts/browse", params={"source": "input", "kind": "image"})
                self.assertEqual(response.status, 200)
                entries = (await response.json())["entries"]
                self.assertEqual([entry["name"] for entry in entries], ["scenes", "frame.png"])
                response = await self.client.get("/yafv/prompts/browse", params={"source": "output", "search": "RESULT"})
                self.assertEqual((await response.json())["entries"][0]["name"], "result.png")
                response = await self.client.get("/yafv/prompts/browse-file", params={"source": "input", "path": "frame.png"})
                self.assertEqual(await response.read(), png())
                for path in ("../outside.png", str(root / "outside.png"), "escape.png", "private.txt"):
                    response = await self.client.get("/yafv/prompts/browse-file", params={"source": "input", "path": path})
                    self.assertEqual(response.status, 404, path)
                for params in ({"source": "temp"}, {"kind": "invalid"}, {"path": ".."}, {"page": "bad"}):
                    response = await self.client.get("/yafv/prompts/browse", params=params)
                    self.assertEqual(response.status, 400, params)

    async def test_frame_names_survive_save_keep_and_remove(self):
        response = await self.save(first=png(), last=png())
        entry = (await response.json())["entries"][0]
        self.assertEqual(entry["media_names"], {"first": "first.png", "last": "last.png"})
        response = await self.save(entry=entry["id"], base=entry["revision"], first_action="remove")
        entry = (await response.json())["entries"][0]
        self.assertEqual(entry["media_names"], {"last": "last.png"})

    async def test_native_generate_text_parameters_and_original_frames(self):
        first, last = png((64, 32), "red"), png((30, 90), "blue")
        item = module.library.save("scope", "A bird takes flight.", first, last)
        clip = FakeClip()
        with patch.object(module.TextGenerate, "execute", wraps=TextGenerate.execute) as generate:
            result = module.YAFVVideoPrompts().execute("scope", item.id, clip=clip, text="Write a precise cinematic prompt.",
                                                      max_length=800, sampling_mode="on", temperature=.4, seed=19, mtp="off")
        self.assertEqual(generate.call_count, 1)
        prompt, first_output, last_output = result["result"]
        self.assertEqual(prompt, "A generated video prompt.")
        self.assertEqual(tuple(first_output.shape), (1, 32, 64, 3))
        self.assertEqual(tuple(last_output.shape), (1, 90, 30, 3))
        self.assertTrue(torch.all(first_output[..., 0] == 1))
        self.assertTrue(torch.all(last_output[..., 2] == 1))
        tokenize = clip.calls[0]
        self.assertIn("Write a precise cinematic prompt.", tokenize[1])
        self.assertIn("A bird takes flight.", tokenize[1])
        self.assertIn("FIRST FRAME on the left", tokenize[1])
        self.assertEqual(tuple(tokenize[2]["image"].shape), (1, 808, 1536, 3))
        self.assertFalse(tokenize[2]["skip_template"])
        parameters = clip.calls[1][2]
        self.assertEqual(parameters["seed"], 19)
        self.assertEqual(parameters["temperature"], .4)
        self.assertEqual(parameters["max_length"], 800)
        self.assertFalse(parameters["mtp"])

    async def test_single_last_frame_is_labeled_and_missing_first_is_none(self):
        item = module.library.save("scope", "Arrive here", None, png())
        clip = FakeClip()
        result = module.YAFVVideoPrompts().execute("scope", item.id, clip=clip, text="Expand")
        self.assertIn("LAST FRAME", clip.calls[0][1])
        self.assertEqual(tuple(clip.calls[0][2]["image"].shape), (1, 32, 64, 3))
        self.assertIsNone(result["result"][1])

    async def test_generation_error_is_not_silent_passthrough(self):
        item = module.library.save("scope", "Prompt", None, None)
        with self.assertRaisesRegex(RuntimeError, "Generate Text"):
            module.YAFVVideoPrompts().execute("scope", item.id, clip=object(), text="Expand")

    async def test_model_cannot_silently_ignore_visual_reference(self):
        item = module.library.save("scope", "Prompt", png(), None)
        clip = FakeClip()
        clip.tokenize = lambda *args, **kwargs: {"tokens": [1, 2]}
        with self.assertRaisesRegex(ValueError, "vision-capable model"):
            module.YAFVVideoPrompts().execute("scope", item.id, clip=clip, text="Expand")
        self.assertFalse(clip.calls)

    async def test_upload_edit_remove_and_isolation(self):
        response = await self.save(first=png(), last=png((80, 20)))
        self.assertEqual(response.status, 200, await response.text())
        data = await response.json()
        entry = data["entries"][0]
        response = await self.client.get(f'/yafv/prompts/image/{entry["revision"]}/first?collection=graph-a:1')
        self.assertEqual(response.status, 200)
        self.assertEqual(Image.open(io.BytesIO(await response.read())).size, (64, 32))
        response = await self.save(prompt="Edited", entry=entry["id"], base=entry["revision"], first_action="remove")
        updated = (await response.json())["entries"][0]
        self.assertFalse(updated["first"])
        self.assertTrue(updated["last"])
        self.assertNotEqual(updated["revision"], entry["revision"])
        self.assertNotIn(entry["revision"], module.library.revisions)
        response = await self.client.get("/yafv/prompts/library?collection=graph-b:1")
        self.assertEqual((await response.json())["entries"], [])
        response = await self.client.delete(f'/yafv/prompts/entry/{entry["id"]}?collection=graph-a:1')
        self.assertEqual((await response.json())["entries"], [])
        self.assertFalse(module.library.revisions)

    async def test_concurrent_edit_conflict(self):
        response = await self.save()
        entry = (await response.json())["entries"][0]
        await self.save(prompt="Second", entry=entry["id"], base=entry["revision"])
        response = await self.save(prompt="Stale", entry=entry["id"], base=entry["revision"])
        self.assertEqual(response.status, 400)
        self.assertEqual(module.library.view("graph-a:1")["entries"][0]["prompt"], "Second")

    async def test_queue_snapshot_survives_validation_edit_and_delete_then_releases(self):
        item = module.library.save("graph-a:1", "Original", png(), None)
        submission = {"prompt": {"1": {"class_type": "YAFVVideoPrompts", "inputs": {"revision_id": item.id}}}}
        pending = asyncio.create_task(self.client.post("/prompt", json=submission))
        await self.validation_started.wait()
        module.library.save("graph-a:1", "Modified", None, None, entry=item.entry, base=item.id)
        module.collect_unused()
        self.assertIn(item.id, module.library.revisions)
        self.validation_continue.set()
        self.assertEqual((await pending).status, 200)
        module.library.delete("graph-a:1", item.entry)
        module.collect_unused()
        result = module.YAFVVideoPrompts().execute("graph-a:1", item.id)
        self.assertEqual(result["result"][0], "Original")
        self.assertEqual(tuple(result["result"][1].shape), (1, 32, 64, 3))
        self.pending.clear()
        module.collect_unused()
        self.assertFalse(module.library.revisions)
        self.assertFalse(module.library.results)
        self.assertFalse(module.library.leases)

    async def test_session_reload_and_restart_invalidation(self):
        item = module.library.save("scope", "Keep during reload", None, None)
        response = await self.client.get("/yafv/prompts/library?collection=scope")
        self.assertEqual((await response.json())["selected"], item.entry)
        before = module.YAFVVideoPrompts.IS_CHANGED("scope", item.id)
        module.library = module.PromptLibrary()
        after = module.YAFVVideoPrompts.IS_CHANGED("scope", item.id)
        self.assertNotEqual(before, after)
        self.assertIsInstance(module.YAFVVideoPrompts.VALIDATE_INPUTS("scope", item.id), str)
        response = await self.client.get("/yafv/prompts/library?collection=scope")
        self.assertEqual((await response.json())["entries"], [])

    async def test_selection_advances_after_delete(self):
        first = module.library.save("scope", "First", None, None)
        second = module.library.save("scope", "Second", None, None)
        third = module.library.save("scope", "Third", None, None)
        module.library.select("scope", second.entry)
        module.library.delete("scope", second.entry)
        self.assertEqual(module.library.view("scope")["selected"], third.entry)
        module.library.delete("scope", third.entry)
        self.assertEqual(module.library.view("scope")["selected"], first.entry)


if __name__ == "__main__":
    unittest.main()
