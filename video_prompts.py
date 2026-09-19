"""Temporary prompt libraries and immutable queue revisions for image-to-video."""
from collections import Counter
from dataclasses import dataclass
import io
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
                raise ValueError("El elemento ya no está disponible. Selecciona uno de la lista; la sesión se borra al reiniciar ComfyUI.")
            return item

    def view(self, key):
        with self.lock:
            collection = self.collection(key)
            entries = []
            for entry, revision in collection["entries"].items():
                item = self.revisions[revision]
                entries.append({"id": entry, "revision": revision, "prompt": item.prompt,
                                "first": bool(item.first), "last": bool(item.last),
                                "generated": self.results.get(revision, "")})
            return {"epoch": self.epoch, "entries": entries, "selected": collection["selected"]}

    def save(self, key, prompt, first, last, entry=None, base=None):
        with self.lock:
            collection = self.collection(key)
            if entry is not None and collection["entries"].get(entry) != base:
                raise ValueError("El elemento cambió o fue eliminado en otra ventana. Recarga la lista antes de guardarlo.")
            entry = entry or uuid.uuid4().hex
            item = PromptRevision(uuid.uuid4().hex, key, entry, prompt, first, last)
            self.revisions[item.id] = item
            collection["entries"][entry] = item.id
            collection["selected"] = entry
            return item

    def select(self, key, entry):
        with self.lock:
            collection = self.collection(key)
            if entry not in collection["entries"]:
                raise ValueError("Selecciona un elemento disponible.")
            collection["selected"] = entry

    def delete(self, key, entry):
        with self.lock:
            collection = self.collection(key)
            order = list(collection["entries"])
            if entry not in order:
                raise ValueError("El elemento ya fue eliminado.")
            position = order.index(entry)
            del collection["entries"][entry]
            if collection["selected"] == entry:
                remaining = list(collection["entries"])
                collection["selected"] = remaining[min(position, len(remaining) - 1)] if remaining else None

    def collect(self, queued):
        with self.lock:
            live = {revision for c in self.collections.values() for revision in c["entries"].values()}
            keep = live | queued | set(self.leases)
            for revision in self.revisions.keys() - keep:
                del self.revisions[revision]
                self.results.pop(revision, None)


library = PromptLibrary()
routes = web.RouteTableDef()


def prompt_revisions(prompt):
    if not isinstance(prompt, dict):
        return set()
    return {node["inputs"]["revision_id"] for node in prompt.values()
            if isinstance(node, dict) and node.get("class_type") == "YAFVVideoPrompts"
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
            raise ValueError("El CLIP conectado no incorporó la imagen de referencia. Usa un modelo con soporte visual o quita los frames para generar texto.")
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
        }, "optional": {"clip": ("CLIP",), "text": ("STRING", {"forceInput": True,
            "tooltip": "Instrucciones para Generate Text. Si no están conectadas o están vacías, se devuelve el prompt original."})}}

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

    def execute(self, collection_id, revision_id, max_length=512, sampling_mode="off", temperature=.7,
                top_k=64, top_p=.95, min_p=.05, repetition_penalty=1.05, presence_penalty=0, seed=0,
                thinking=False, use_default_template=True, mtp="auto", clip=None, text=None):
        item = library.revision(collection_id, revision_id)
        first, last = load_frame(item.first), load_frame(item.last)
        generated = item.prompt
        if clip is not None and text is not None and text.strip():
            reference, description = visual_reference(first, last)
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
                raise RuntimeError(f"El CLIP conectado no pudo ejecutar Generate Text con estas entradas: {error}") from error
            generated = output.args[0]
        with library.lock:
            if revision_id in library.revisions:
                library.results[revision_id] = generated
        return {"ui": {"yafv_prompt": [{"collection": collection_id, "revision": revision_id, "text": generated}]},
                "result": (generated, image_tensor(first), image_tensor(last))}


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
            raise ValueError("Escribe un prompt antes de agregarlo.")
        old = library.revision(key, fields["base"]) if fields.get("entry") else None
        images = {}
        for name in ("first", "last"):
            action = fields.get(f"{name}_action", "keep")
            if action not in {"keep", "remove", "upload"}:
                raise ValueError("Acción de imagen inválida.")
            if action == "upload" and name not in uploads:
                raise ValueError("Falta el archivo de imagen.")
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
            raise ValueError("Frame inválido.")
        raw = getattr(item, frame)
        if raw is None:
            raise ValueError("Este elemento no tiene esa imagen.")
        return web.Response(body=raw, content_type="image/png", headers={"Cache-Control": "no-store"})
    except ValueError as error:
        return web.json_response({"error": str(error)}, status=404)


if hasattr(PromptServer, "instance"):
    for route in routes:
        PromptServer.instance.routes.route(route.method, route.path, **route.kwargs)(route.handler)
    PromptServer.instance.app.middlewares.append(protect_prompt_submission)
