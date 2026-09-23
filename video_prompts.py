"""Temporary prompt libraries and immutable queue revisions for image-to-video."""
from collections import Counter
from dataclasses import dataclass, field
import io
from pathlib import Path
import threading
import uuid

from aiohttp import web
import numpy as np
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
import torch

from comfy_extras.nodes_textgen import TextGenerate
from server import PromptServer


@dataclass(frozen=True)
class PromptRevision:
    id: str
    collection: str
    entry: str
    prompt: str
    first: bytes | None
    last: bytes | None
    media: dict = field(default_factory=dict)
    kind: str = "video"


class PromptLibrary:
    def __init__(self):
        self.epoch = uuid.uuid4().hex
        self.collections = {}
        self.revisions = {}
        self.results = {}
        self.leases = Counter()
        self.lock = threading.RLock()

    def collection(self, key):
        return self.collections.setdefault(key, {"entries": {}, "selected": None})

    def revision(self, collection, revision):
        with self.lock:
            item = self.revisions.get(revision)
            if item is None or item.collection != collection:
                raise ValueError("The item is no longer available. Select one from the list; restarting ComfyUI clears the session.")
            return item

    def view(self, key):
        with self.lock:
            collection = self.collection(key)
            entries = []
            for entry, revision in collection["entries"].items():
                item = self.revisions[revision]
                entries.append({"id": entry, "revision": revision, "prompt": item.prompt,
                                "first": bool(item.first), "last": bool(item.last),
                                "generated": self.results.get(revision, ""),
                                **{name: bool(path) for name, path in item.media.items()}})
            return {"epoch": self.epoch, "entries": entries, "selected": collection["selected"]}

    def save(self, key, prompt, first, last, entry=None, base=None, *, media=None, kind="video"):
        with self.lock:
            collection = self.collection(key)
            if entry is not None and collection["entries"].get(entry) != base:
                raise ValueError("The item changed or was deleted in another window. Refresh the list before saving.")
            entry = entry or uuid.uuid4().hex
            item = PromptRevision(uuid.uuid4().hex, key, entry, prompt, first, last, dict(media or {}), kind)
            self.revisions[item.id] = item
            collection["entries"][entry] = item.id
            collection["selected"] = entry
            return item

    def select(self, key, entry):
        with self.lock:
            collection = self.collection(key)
            if entry not in collection["entries"]:
                raise ValueError("Select an available item.")
            collection["selected"] = entry

    def delete(self, key, entry):
        with self.lock:
            collection = self.collection(key)
            order = list(collection["entries"])
            if entry not in order:
                raise ValueError("The item was already deleted.")
            position = order.index(entry)
            del collection["entries"][entry]
            if collection["selected"] == entry:
                remaining = list(collection["entries"])
                collection["selected"] = remaining[min(position, len(remaining) - 1)] if remaining else None

    def collect(self, queued):
        with self.lock:
            live = {revision for c in self.collections.values() for revision in c["entries"].values()}
            keep = live | queued | set(self.leases)
            removed_files = set()
            for revision in self.revisions.keys() - keep:
                removed_files.update(path for path in self.revisions[revision].media.values() if path)
                del self.revisions[revision]
                self.results.pop(revision, None)
            live_files = {path for item in self.revisions.values() for path in item.media.values() if path}
            for path in removed_files - live_files:
                Path(path).unlink(missing_ok=True)


library = PromptLibrary()
routes = web.RouteTableDef()


def prompt_revisions(prompt):
    if not isinstance(prompt, dict):
        return set()
    return {node["inputs"]["revision_id"] for node in prompt.values()
            if isinstance(node, dict) and node.get("class_type") in {"YAFVVideoPrompts", "YAFVReferenceVideoPrompts"}
            and isinstance(node.get("inputs", {}).get("revision_id"), str)}


def collect_unused():
    running, pending = PromptServer.instance.prompt_queue.get_current_queue_volatile()
    queued = set()
    for item in running + pending:
        queued.update(prompt_revisions(item[2]))
    library.collect(queued)


@web.middleware
async def protect_prompt_submission(request, handler):
    # Keep revisions alive between HTTP submission, async validation and queue insertion.
    if request.method != "POST" or request.path not in {"/prompt", "/api/prompt"}:
        return await handler(request)
    try:
        data = await request.json()
    except (ValueError, UnicodeDecodeError):
        return await handler(request)
    revisions = prompt_revisions(data.get("prompt")) if isinstance(data, dict) else set()
    with library.lock:
        library.leases.update(revisions)
    try:
        return await handler(request)
    finally:
        with library.lock:
            for revision in revisions:
                library.leases[revision] -= 1
                if not library.leases[revision]:
                    del library.leases[revision]
        collect_unused()


def normalized_png(raw):
    with Image.open(io.BytesIO(raw)) as image:
        image.seek(0)
        image = ImageOps.exif_transpose(image).convert("RGBA")
        background = Image.new("RGBA", image.size, "white")
        image = Image.alpha_composite(background, image).convert("RGB")
        output = io.BytesIO()
        image.save(output, "PNG")
        return output.getvalue()


def load_frame(raw):
    if raw is None:
        return None
    with Image.open(io.BytesIO(raw)) as image:
        return image.convert("RGB")


def image_tensor(image):
    return None if image is None else torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def contains_image(tokens):
    if isinstance(tokens, dict):
        return tokens.get("type") == "image" or any(contains_image(value) for value in tokens.values())
    if isinstance(tokens, (list, tuple)):
        return any(contains_image(value) for value in tokens)
    return False


class VisualPromptClip:
    """Validate native visual tokens without changing Generate Text's generation path."""
    def __init__(self, clip):
        self.clip = clip

    def tokenize(self, *args, **kwargs):
        tokens = self.clip.tokenize(*args, **kwargs)
        if not contains_image(tokens):
            raise ValueError("The connected CLIP did not include the reference image. Use a vision-capable model or remove the frames to generate text.")
        return tokens

    def generate(self, *args, **kwargs):
        return self.clip.generate(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.clip.decode(*args, **kwargs)


def visual_reference(first, last):
    if first is None:
        return last, "The reference image is the LAST FRAME." if last is not None else ""
    if last is None:
        return first, "The reference image is the FIRST FRAME."
    # Only the model's reference is resized. Output frames retain their original dimensions.
    panel_width, panel_height, label_height = 768, 768, 40
    sheet = Image.new("RGB", (panel_width * 2, panel_height + label_height), "#18212b")
    draw = ImageDraw.Draw(sheet)
    for index, (image, label) in enumerate([(first, "FIRST FRAME"), (last, "LAST FRAME")]):
        fitted = ImageOps.contain(image, (panel_width, panel_height), Image.Resampling.LANCZOS)
        x = index * panel_width
        sheet.paste(fitted, (x + (panel_width - fitted.width) // 2, label_height + (panel_height - fitted.height) // 2))
        draw.text((x + 16, 10), label, fill="white", font_size=24)
    return sheet, "The reference contains two labeled panels: FIRST FRAME on the left, LAST FRAME on the right. They are separate moments, not a split-screen scene."


class YAFVVideoPrompts:
    CATEGORY = "YAFV/video"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = ("generated_prompt", "first_frame", "last_frame")
    DESCRIPTION = "Select a temporary prompt and optional first/last frames. With CLIP and text instructions, uses ComfyUI Generate Text."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "collection_id": ("STRING", {"default": ""}),
            "revision_id": ("STRING", {"default": ""}),
            "max_length": ("INT", {"default": 512, "min": 1, "max": 32768}),
            "sampling_mode": (["off", "on"], {"default": "off"}),
            "temperature": ("FLOAT", {"default": .7, "min": .01, "max": 2, "step": .01}),
            "top_k": ("INT", {"default": 64, "min": 0, "max": 1000}),
            "top_p": ("FLOAT", {"default": .95, "min": 0, "max": 1, "step": .01}),
            "min_p": ("FLOAT", {"default": .05, "min": 0, "max": 1, "step": .01}),
            "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0, "max": 5, "step": .01}),
            "presence_penalty": ("FLOAT", {"default": 0, "min": 0, "max": 5, "step": .01}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "thinking": ("BOOLEAN", {"default": False}),
            "use_default_template": ("BOOLEAN", {"default": True}),
            "mtp": (["auto", "off", "2", "3", "4", "5"], {"default": "auto"}),
        }, "optional": {
            "clip": ("CLIP",),
            "text": ("STRING", {"forceInput": True,
                "tooltip": "Generate Text instructions. If disconnected or empty, returns the original prompt."}),
            "context_image": ("IMAGE", {"tooltip": "Project continuity image; replaces the current clip's first frame when connected."}),
        }}

    @classmethod
    def IS_CHANGED(cls, collection_id, revision_id, **kwargs):
        return (library.epoch, revision_id)

    @classmethod
    def VALIDATE_INPUTS(cls, collection_id, revision_id):
        try:
            library.revision(collection_id, revision_id)
            return True
        except ValueError as error:
            return str(error)

    def _context_image(self, context_image):
        if context_image is None:
            return None
        if not torch.is_tensor(context_image) or context_image.ndim < 4:
            raise ValueError("context_image must be a ComfyUI IMAGE batch.")
        return Image.fromarray(
            (context_image[:1].detach().cpu().clamp(0, 1)[0].numpy() * 255).round().astype(np.uint8)
        ).convert("RGB")

    def prepare(self, item, needs_visual, context_image=None, context_video=None):
        first, last = load_frame(item.first), load_frame(item.last)
        if context_image is not None:
            first = self._context_image(context_image)
        reference, description = visual_reference(first, last) if needs_visual else (None, "")
        return (image_tensor(first), image_tensor(last)), reference, description

    def execute(self, collection_id, revision_id, max_length=512, sampling_mode="off", temperature=.7,
                top_k=64, top_p=.95, min_p=.05, repetition_penalty=1.05, presence_penalty=0, seed=0,
                thinking=False, use_default_template=True, mtp="auto", clip=None, text=None,
                context_image=None, context_video=None):
        item = library.revision(collection_id, revision_id)
        needs_visual = clip is not None and text is not None and bool(text.strip())
        media, reference, description = self.prepare(
            item, needs_visual, context_image=context_image, context_video=context_video
        )
        generated = item.prompt
        if needs_visual:
            prompt = f"INSTRUCTIONS\n{text}\n\nUSER PROMPT\n{item.prompt}"
            if description:
                prompt += f"\n\nVISUAL REFERENCE\n{description}"
            sampling = {"sampling_mode": sampling_mode, "temperature": temperature, "top_k": top_k,
                        "top_p": top_p, "min_p": min_p, "repetition_penalty": repetition_penalty,
                        "presence_penalty": presence_penalty, "seed": seed}
            try:
                generator_clip = VisualPromptClip(clip) if reference is not None else clip
                output = TextGenerate.execute(generator_clip, prompt, max_length, sampling, image=image_tensor(reference),
                                              thinking=thinking, use_default_template=use_default_template, mtp=mtp)
            except (AttributeError, NotImplementedError, TypeError) as error:
                raise RuntimeError(f"The connected CLIP could not run Generate Text with these inputs: {error}") from error
            generated = output.args[0]
        with library.lock:
            if revision_id in library.revisions:
                library.results[revision_id] = generated
        return {"ui": {"yafv_prompt": [{"collection": collection_id, "revision": revision_id, "text": generated}]},
                "result": (generated, *media)}


@routes.get("/yafv/prompts/library")
async def get_library(request):
    collect_unused()
    return web.json_response(library.view(request.query["collection"]), headers={"Cache-Control": "no-store"})


@routes.post("/yafv/prompts/entry")
async def save_entry(request):
    try:
        parts = await request.multipart()
        fields, uploads = {}, {}
        async for part in parts:
            if part.name in {"first", "last"}:
                uploads[part.name] = normalized_png(await part.read())
            elif part.name in {"collection", "entry", "base", "prompt", "first_action", "last_action"}:
                fields[part.name] = await part.text()
        key, prompt = fields["collection"], fields["prompt"]
        if not key or not prompt.strip():
            raise ValueError("Write a prompt before adding it.")
        old = library.revision(key, fields["base"]) if fields.get("entry") else None
        images = {}
        for name in ("first", "last"):
            action = fields.get(f"{name}_action", "keep")
            if action not in {"keep", "remove", "upload"}:
                raise ValueError("Invalid image action.")
            if action == "upload" and name not in uploads:
                raise ValueError("Missing image file.")
            images[name] = uploads[name] if action == "upload" else (getattr(old, name) if action == "keep" and old else None)
        library.save(key, prompt, images["first"], images["last"], fields.get("entry") or None, fields.get("base"))
        collect_unused()
        return web.json_response(library.view(key))
    except (ValueError, KeyError, UnidentifiedImageError, OSError) as error:
        return web.json_response({"error": str(error)}, status=400)


@routes.post("/yafv/prompts/select")
async def select_entry(request):
    try:
        data = await request.json()
        library.select(data["collection"], data["entry"])
        return web.json_response(library.view(data["collection"]))
    except (ValueError, KeyError) as error:
        return web.json_response({"error": str(error)}, status=400)


@routes.delete("/yafv/prompts/entry/{entry}")
async def delete_entry(request):
    try:
        key = request.query["collection"]
        library.delete(key, request.match_info["entry"])
        collect_unused()
        return web.json_response(library.view(key))
    except (ValueError, KeyError) as error:
        return web.json_response({"error": str(error)}, status=400)


@routes.get("/yafv/prompts/image/{revision}/{frame}")
async def get_frame(request):
    try:
        item = library.revision(request.query["collection"], request.match_info["revision"])
        frame = request.match_info["frame"]
        if frame not in {"first", "last"}:
            raise ValueError("Invalid frame.")
        raw = getattr(item, frame)
        if raw is None:
            raise ValueError("This item does not have that image.")
        return web.Response(body=raw, content_type="image/png", headers={"Cache-Control": "no-store"})
    except ValueError as error:
        return web.json_response({"error": str(error)}, status=404)


if hasattr(PromptServer, "instance"):
    for route in routes:
        PromptServer.instance.routes.route(route.method, route.path, **route.kwargs)(route.handler)
    PromptServer.instance.app.middlewares.append(protect_prompt_submission)
