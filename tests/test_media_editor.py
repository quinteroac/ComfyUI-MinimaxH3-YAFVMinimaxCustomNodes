"""Local HTTP/FFmpeg tests; no model loading or live ComfyUI queue changes."""
import asyncio
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
server = types.SimpleNamespace(routes=web.RouteTableDef(), send_sync=lambda *args: None)
spec = importlib.util.spec_from_file_location("editor_under_test", ROOT / "media_editor.py")
editor = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"folder_paths": types.SimpleNamespace(), "server": types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=server))}):
    spec.loader.exec_module(editor)


class EditorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="yafv-test-")
        self.root = Path(self.temp.name)
        Image.new("RGB", (160, 90), "blue").save(self.root / "still.png")
        await editor.command("ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=blue:s=160x90:r=24:d=2",
                             "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:v", "libx264", "-threads", "1",
                             "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", self.root / "video.mp4")
        self.history = {"job": {"outputs": {"1": {"images": [{"filename": "still.png", "type": "output"}]},
                                                     "2": {"gifs": [{"filename": "video.mp4", "type": "temp"}]}}}}
        server.prompt_queue = types.SimpleNamespace(get_history=lambda: self.history, get_current_queue_volatile=lambda: ([], []))
        editor.folder_paths.get_directory_by_type = lambda kind: str(self.root)
        media = list(editor.history_media(self.history).values())
        self.image_id, self.video_id = media[0]["id"], media[1]["id"]
        self.app = web.Application()
        self.app.add_routes(editor.routes)
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        self.temp.cleanup()

    def stroke(self, start=.25, end=.75):
        return {"color": "#ff0000", "opacity": 1, "width": .15, "points": [[.2, .5], [.8, .5]], "start": start, "end": end}

    def clip(self, media_id=None, **kwargs):
        return {"media_id": media_id or self.video_id, "in": .5, "out": 1.5, "strokes": [], **kwargs}

    async def test_catalog_metadata_and_no_paths(self):
        response = await self.client.get("/yafv/editor/media")
        media = (await response.json())["media"]
        self.assertEqual(len(media), 2)
        self.assertNotIn("fullpath", json.dumps(media))
        response = await self.client.get(f"/yafv/editor/media/{self.video_id}")
        meta = await response.json()
        self.assertEqual((meta["width"], meta["height"]), (160, 90))
        self.assertTrue(meta["audio"])

    async def test_deleted_history_and_traversal_rejected(self):
        self.history.clear()
        response = await self.client.get(f"/yafv/editor/source/{self.image_id}")
        self.assertEqual(response.status, 404)
        self.history["evil"] = {"outputs": {"x": {"images": [{"filename": "../outside.png", "type": "output"}]}}}
        bad_id = next(iter(editor.history_media(self.history)))
        with self.assertRaises(ValueError):
            editor.resolve_media(bad_id)

    async def test_symlink_outside_results_rejected(self):
        (self.root / "escape.png").symlink_to("/etc/passwd")
        self.history["evil"] = {"outputs": {"x": {"images": [{"filename": "escape.png", "type": "output"}]}}}
        bad_id = next(k for k, m in editor.history_media(self.history).items() if m["filename"] == "escape.png")
        with self.assertRaises(ValueError):
            editor.resolve_media(bad_id)

    async def test_frame_drawing_only_in_time_window(self):
        for time, red in [(.1, False), (.5, True), (.9, False)]:
            response = await self.client.post("/yafv/editor/frame", json={"media_id": self.video_id, "time": time, "strokes": [self.stroke()]})
            self.assertEqual(response.status, 200)
            image = Image.open(io.BytesIO(await response.read()))
            pixel = image.getpixel((80, 45))
            self.assertEqual(pixel[0] > 200, red)
            self.assertEqual(image.size, (160, 90))

    async def test_mixed_timeline_audio_duration_and_temporal_ink(self):
        directory = self.root / "render"
        directory.mkdir()
        data = {"width": 160, "height": 90, "fps": 24, "clips": [
            self.clip(**{"in": .5, "out": 1.5, "strokes": [self.stroke(.75, 1.25)]}),
            self.clip(self.image_id, **{"in": 0, "out": .5}),
            self.clip(frame_time=1, **{"in": 0, "out": .5}),
        ]}
        output = await editor.render_timeline(data, directory, lambda value: None)
        meta = await editor.probe(output, "video")
        self.assertTrue(meta["audio"])
        self.assertAlmostEqual(meta["duration"], 2, delta=.1)
        for time, red in [(.1, False), (.5, True), (.9, False), (1.2, False)]:
            image = await editor.frame_image(output, "video", time)
            self.assertEqual(image.getpixel((80, 45))[0] > 200, red)
        audio = await editor.command("ffmpeg", "-v", "error", "-i", output, "-f", "f32le", "-ac", "1", "pipe:1")
        import array
        samples = array.array("f", audio)
        self.assertGreater(max(abs(x) for x in samples[10000:30000]), .01)
        self.assertLess(max(abs(x) for x in samples[60000:70000]), .001)

    async def test_export_response_and_cleanup(self):
        before = set(Path(tempfile.gettempdir()).glob("yafv-editor-*"))
        response = await self.client.post("/yafv/editor/export/test", json={"width": 160, "height": 90, "fps": 24,
                                                                           "clips": [self.clip(self.image_id)]})
        self.assertEqual(response.status, 200, await response.text() if response.status != 200 else "")
        self.assertEqual(response.content_type, "video/mp4")
        self.assertGreater(len(await response.read()), 1000)
        self.assertFalse(editor.exports)
        self.assertEqual(before, set(Path(tempfile.gettempdir()).glob("yafv-editor-*")))

    async def test_cancel_stops_process_and_removes_temporary_files(self):
        started = asyncio.Event()
        async def slow_render(data, directory, progress):
            started.set()
            await editor.command(sys.executable, "-c", "import time; time.sleep(60)")
        before = set(Path(tempfile.gettempdir()).glob("yafv-editor-*"))
        with patch.object(editor, "render_timeline", slow_render):
            request = asyncio.create_task(self.client.post("/yafv/editor/export/cancel-me", json={}))
            await started.wait()
            response = await self.client.delete("/yafv/editor/export/cancel-me")
            self.assertTrue((await response.json())["cancelled"])
            response = await asyncio.wait_for(request, 3)
            self.assertEqual(response.status, 409)
        self.assertFalse(editor.exports)
        self.assertEqual(before, set(Path(tempfile.gettempdir()).glob("yafv-editor-*")))

    async def test_invalid_range_does_not_export(self):
        response = await self.client.post("/yafv/editor/export/invalid", json={"width": 160, "height": 90, "clips": [self.clip(out=100)]})
        self.assertEqual(response.status, 400)
        self.assertFalse(editor.exports)


if __name__ == "__main__":
    unittest.main()
