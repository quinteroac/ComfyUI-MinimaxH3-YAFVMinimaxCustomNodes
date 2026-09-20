"""Optional browser smoke test: python tests/reference_prompts_browser.py (requires Playwright)."""
import asyncio
import json
import os
import types

from aiohttp import web
from aiohttp.test_utils import TestServer
from playwright.async_api import async_playwright

from test_reference_prompts import ROOT, ReferencePromptTests, prompts, reference, server
from test_video_prompts import png


async def main():
    ReferencePromptTests.setUpClass()
    server.prompt_queue = types.SimpleNamespace(get_current_queue_volatile=lambda: ([], []))
    app = web.Application()
    app.add_routes(prompts.routes)
    app.add_routes(reference.routes)

    async def html(request):
        legacy = request.query.get("legacy") == "1"
        node_class = prompts.YAFVVideoPrompts if legacy else reference.YAFVReferenceVideoPrompts
        data = {"name": node_class.__name__, "output": node_class.RETURN_TYPES, "output_name": node_class.RETURN_NAMES}
        parameters = [[name, definition[1]["default"]] for name, definition in node_class.INPUT_TYPES()["required"].items()]
        return web.Response(text='''<!doctype html><html><body style="background:#111"><script type="module">
window.app = {registerExtension(extension) {window.extension = extension}};
const events = new EventTarget();
window.api = {fetchApi: (url, options) => fetch(url, options), apiURL: url => url,
  addEventListener: (...args) => events.addEventListener(...args), removeEventListener: (...args) => events.removeEventListener(...args)};
await import('/panel.js');
const data = NODE_DATA;
class Node {
  constructor() {
    this.id = 1; this.graph = {id: 'test'}; this.inputs = [];
    this.widgets = PARAMETERS.map(([name, value]) => ({name, value}));
    this.outputs = data.output_name.slice(0, 8).map((name, i) => ({name, type: data.output[i], links: [100 + i]}));
  }
  setSize() {} setDirtyCanvas() {}
  addOutput(name, type) {this.outputs.push({name, type, links: null});}
  addDOMWidget(name, type, element) {
    element.style.width = '1020px'; element.style.height = '840px'; document.body.append(element); return {};
  }
}
await extension.beforeRegisterNodeDef(Node, data);
window.node = new Node(); node.onNodeCreated(); node.onConfigure(); node.onConfigure();
</script></body></html>'''.replace("NODE_DATA", json.dumps(data)).replace("PARAMETERS", json.dumps(parameters)), content_type="text/html")

    async def javascript(request):
        source = (ROOT / "web/video_prompts.js").read_text()
        source = source.replace('import { app } from "../../scripts/app.js";', 'const app = window.app;')
        source = source.replace('import { api } from "../../scripts/api.js";', 'const api = window.api;')
        return web.Response(text=source, content_type="text/javascript")

    app.router.add_get("/", html)
    app.router.add_get("/panel.js", javascript)
    app.router.add_get("/video_prompts.css", lambda request: web.FileResponse(ROOT / "web/video_prompts.css"))
    try:
        async with TestServer(app) as http, async_playwright() as playwright:
            executable = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
            browser = await playwright.chromium.launch(executable_path=executable, headless=True, args=["--no-sandbox"])
            page = await browser.new_page(viewport={"width": 1200, "height": 1000})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            await page.goto(str(http.make_url("/")))
            await page.wait_for_function("node.videoPrompts.hydrated")
            assert await page.locator(".frames > .frame").count() == 0
            assert await page.evaluate("node.outputs.length") == 16
            assert await page.evaluate("node.outputs.slice(0, 8).every((slot, i) => slot.links[0] === 100 + i)")

            async def attach(kind, file):
                await page.locator('[data-action="toggleReferences"]').click()
                async with page.expect_file_chooser() as chooser:
                    await page.locator(f'[data-action="addReference"][data-id="{kind}"]').click()
                await (await chooser.value).set_files(file)

            image = {"name": "image.png", "mimeType": "image/png", "buffer": png()}
            await attach("image", [])
            assert await page.locator(".frames > .frame").count() == 0
            for _ in range(8):
                await attach("image", image)
            for _ in range(2):
                await attach("video", str(ReferencePromptTests.video))
            for _ in range(3):
                await attach("audio", str(ReferencePromptTests.audio))
            assert await page.locator(".frames > .frame").count() == 13
            assert await page.locator(".soundtrack").count() == 2
            for kind in ("image", "video", "audio"):
                assert await page.locator(f'[data-action="addReference"][data-id="{kind}"]').is_disabled()
            await page.locator('[data-action="removeImage"][data-id="ref_image_3"]').click()
            assert await page.locator(".ref_image_3").count() == 0
            assert await page.locator(".ref_image_4").count() == 1
            await attach("image", image)
            assert await page.locator(".ref_image_3").count() == 1
            await page.locator(".ref_video_audio_1 input").set_input_files(str(ReferencePromptTests.audio))
            await page.locator(".prompt").fill("Eight images and two videos")
            await page.locator('[data-action="save"]').click()
            await page.wait_for_function("node.videoPrompts.current?.ref_video_audio_1 && !node.videoPrompts.busy")
            for index in range(2):
                await page.wait_for_function(f'document.querySelector(".ref_video_{index} video").readyState >= 2')
                await page.wait_for_function(f'document.querySelector(".ref_video_audio_{index} audio").readyState >= 2')
            await page.locator(".ref_video_1 video").evaluate("element => element.play()")
            await page.wait_for_function('document.querySelector(".ref_video_1 video").currentTime > 0')
            await page.locator(".ref_video_1 video").evaluate("element => element.pause()")
            await page.locator('[data-action="removeImage"][data-id="ref_video_audio_1"]').click()
            assert await page.locator(".ref_video_audio_1 audio").is_hidden()
            assert await page.locator(".ref_video_audio_0 audio").is_visible()
            await page.locator('[data-action="save"]').click()
            await page.wait_for_function("!node.videoPrompts.current.ref_video_audio_1 && !node.videoPrompts.busy")
            await page.locator(".prompt").fill("Unsaved edit")
            await page.locator('[data-action="new"]').click()
            assert await page.locator(".unsaved").is_visible()
            await page.locator('[data-action="discard"]').click()
            assert await page.locator(".frames > .frame").count() == 0
            await page.locator(".prompt").fill("Text only")
            await page.locator('[data-action="save"]').click()
            await page.wait_for_function("node.videoPrompts.entries.length === 2 && !node.videoPrompts.busy")
            await page.locator(".choose").first.click()
            await page.wait_for_function("node.videoPrompts.current.ref_image_7 && !node.videoPrompts.busy")
            assert await page.locator(".frames > .frame").count() == 13
            await page.reload()
            await page.wait_for_function("node.videoPrompts.hydrated && node.videoPrompts.current?.ref_video_1")
            assert await page.locator(".frames > .frame").count() == 13
            await page.locator('[data-action="removeImage"][data-id="ref_video_0"]').click()
            assert await page.locator(".ref_video_0").count() == 0
            assert await page.locator(".ref_video_audio_0").count() == 0
            await page.locator('[data-action="save"]').click()
            await page.wait_for_function("!node.videoPrompts.current.ref_video_0 && !node.videoPrompts.busy")
            assert await page.locator(".ref_video_1").count() == 1
            await page.locator(".form").evaluate("element => element.scrollTop = 0")
            await page.screenshot(path="/tmp/yafv-reference-panel.png", full_page=True)
            await page.locator('.form [data-action="delete"]').click()
            await page.wait_for_function("node.videoPrompts.entries.length === 1 && !node.videoPrompts.busy")
            await page.goto(str(http.make_url("/?legacy=1")))
            await page.wait_for_function("node.videoPrompts.hydrated")
            assert await page.locator(".frames > .frame").count() == 2
            await page.locator(".first input").set_input_files(image)
            await page.locator(".prompt").fill("Original node")
            await page.locator('[data-action="save"]').click()
            await page.wait_for_function("node.videoPrompts.current?.first && !node.videoPrompts.busy")
            assert not errors, errors
            await browser.close()
            print("Browser checks passed: dynamic cards, limits, cancellation, slot reuse, audio pairing, playback, drafts, reload, legacy outputs and original node.")
    finally:
        ReferencePromptTests.tearDownClass()


if __name__ == "__main__":
    asyncio.run(main())
