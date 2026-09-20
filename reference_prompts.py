"""Session-scoped reference media for MiniMax H3 prompt entries."""
import asyncio
from pathlib import Path
import tempfile
import uuid

import av
from aiohttp import web
import numpy as np
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
import torch

from . import video_prompts as prompts
from server import PromptServer


MEDIA_TYPES = {
    "ref_image_0": "image", "ref_image_1": "image", "ref_video_0": "video",
    "ref_video_audio_0": "audio", "ref_audio_0": "audio", "ref_audio_1": "audio", "ref_audio_2": "audio",
    **{f"ref_image_{index}": "image" for index in range(2, 8)},
    "ref_video_1": "video", "ref_video_audio_1": "audio",
}
media_directory = tempfile.TemporaryDirectory(prefix="yafv-reference-prompts-")
routes = web.RouteTableDef()


async def ffmpeg(*args):
    process = await asyncio.create_subprocess_exec(
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *map(str, args),
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    try:
        _, stderr = await process.communicate()
    except (asyncio.CancelledError, OSError):
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise
    if process.returncode:
        raise ValueError(stderr.decode(errors="replace").strip()[-1200:])


def inspect_media(path, kind):
    with av.open(str(path)) as container:
        streams = container.streams.video if kind == "video" else container.streams.audio
        if not streams:
            raise ValueError(f"El archivo no contiene {kind}.")
        if next(container.decode(streams[0]), None) is None:
            raise ValueError("El archivo no contiene frames decodificables.")
        return {"audio": bool(container.streams.audio),
                "pixel_format": "yuv420p" if kind == "video" and streams[0].width % 2 == 0 and streams[0].height % 2 == 0 else "yuv444p"}


def decode_video(path):
    with av.open(str(path)) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    if len(frames) < 5:
        raise ValueError("MiniMax requiere al menos 5 frames a 24 fps.")
    return torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)


def decode_audio(path):
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout=stream.layout, rate=stream.rate)
        frames = []
        for frame in container.decode(stream):
            frames.extend(converted.to_ndarray() for converted in resampler.resample(frame))
        frames.extend(converted.to_ndarray() for converted in resampler.resample(None))
        if not frames:
            raise ValueError("El archivo no contiene audio decodificable.")
        return {"waveform": torch.from_numpy(np.concatenate(frames, axis=1)).unsqueeze(0),
                "sample_rate": stream.rate}


def reference_sheet(media):
    panels, descriptions = [], []
    picture = 0
    for name in (f"ref_image_{index}" for index in range(8)):
        if media[name] is not None:
            picture += 1
            label = f"<Picture {picture}>"
            panels.append((Image.fromarray((media[name][0].numpy() * 255).round().astype(np.uint8)), label))
            descriptions.append(f"{name} = {label}")
    audio, video_number = 0, 0
    for slot in range(2):
        name = f"ref_video_{slot}"
        video = media[name]
        if video is None:
            continue
        video_number += 1
        audio_name = f"ref_video_audio_{slot}"
        if media[audio_name] is not None:
            audio += 1
            descriptions.append(f"{audio_name} = <Audio {audio}> (soundtrack of <Video {video_number}>; not analyzed)")
        descriptions.append(f"{name} = <Video {video_number}>")
        for index in np.linspace(0, len(video) - 1, min(8, len(video)), dtype=int):
            frame = Image.fromarray((video[index].numpy() * 255).round().astype(np.uint8))
            panels.append((frame, f"<Video {video_number}> {index / 24:.2f}s"))
    for name in ("ref_audio_0", "ref_audio_1", "ref_audio_2"):
        if media[name] is not None:
            audio += 1
            descriptions.append(f"{name} = <Audio {audio}> (not analyzed)")
    description = ("Use MiniMax reference tags exactly as mapped below. Only present references are numbered. "
                   "Do not infer audio contents. The visual sheet shows separate references and chronological video "
                   "samples, not a collage scene.\n" + "\n".join(descriptions))
    if not panels:
        return None, description
    columns = min(3, len(panels))
    sheet = Image.new("RGB", (columns * 512, ((len(panels) + columns - 1) // columns) * 420), "#18212b")
    draw = ImageDraw.Draw(sheet)
    for index, (frame, label) in enumerate(panels):
        x, y = (index % columns) * 512, (index // columns) * 420
        fitted = ImageOps.contain(frame, (512, 384), Image.Resampling.LANCZOS)
        sheet.paste(fitted, (x + (512 - fitted.width) // 2, y + 36 + (384 - fitted.height) // 2))
        draw.text((x + 12, y + 6), label, fill="white", font_size=22)
    return sheet, description


class YAFVReferenceVideoPrompts(prompts.YAFVVideoPrompts):
    # Append new slots after the original seven references to preserve workflow links.
    RETURN_TYPES = ("STRING", *("AUDIO" if kind == "audio" else "IMAGE" for kind in MEDIA_TYPES.values()))
    RETURN_NAMES = ("generated_prompt", *MEDIA_TYPES)
    DESCRIPTION = "Prompts y referencias temporales para MiniMax H3 Reference to Video. Video a 24 fps; audio independiente."

    @classmethod
    def VALIDATE_INPUTS(cls, collection_id, revision_id):
        result = super().VALIDATE_INPUTS(collection_id, revision_id)
        if result is True and prompts.library.revision(collection_id, revision_id).kind != "reference":
            return "Selecciona un elemento de prompts para Reference to Video."
        return result

    def prepare(self, item, needs_visual):
        if item.kind != "reference":
            raise ValueError("Selecciona un elemento de prompts para Reference to Video.")
        media = {}
        for name, kind in MEDIA_TYPES.items():
            path = item.media.get(name)
            if path is None:
                media[name] = None
            elif kind == "image":
                with Image.open(path) as image:
                    media[name] = prompts.image_tensor(image.convert("RGB"))
            else:
                media[name] = decode_video(path) if kind == "video" else decode_audio(path)
        reference, description = reference_sheet(media) if needs_visual else (None, "")
        return tuple(media.values()), reference, description


@routes.post("/yafv/prompts/reference-entry")
async def save_reference_entry(request):
    created = []
    committed = False
    name = "elemento"
    leased = None
    def target(suffix):
        path = Path(media_directory.name) / (uuid.uuid4().hex + suffix)
        created.append(path)
        return path
    try:
        fields, uploads = {}, {}
        with tempfile.TemporaryDirectory(prefix="upload-", dir=media_directory.name) as staging:
            parts = await request.multipart()
            async for part in parts:
                if part.name in MEDIA_TYPES:
                    path = Path(staging) / part.name
                    with path.open("wb") as output:
                        while chunk := await part.read_chunk():
                            output.write(chunk)
                    uploads[part.name] = path
                elif part.name in {"collection", "entry", "base", "prompt"} or part.name in {f"{n}_action" for n in MEDIA_TYPES}:
                    fields[part.name] = await part.text()
            key, prompt = fields["collection"], fields["prompt"]
            if not key or not prompt.strip():
                raise ValueError("Escribe un prompt antes de agregarlo.")
            old = prompts.library.revision(key, fields["base"]) if fields.get("entry") else None
            if old:
                with prompts.library.lock:
                    prompts.library.leases[old.id] += 1
                    leased = old.id
            if old and old.kind != "reference":
                raise ValueError("El elemento no pertenece a Reference to Video.")
            media = {n: old.media.get(n) if old else None for n in MEDIA_TYPES}
            for name, kind in MEDIA_TYPES.items():
                action = fields.get(f"{name}_action", "keep")
                if action not in {"keep", "remove", "upload"}:
                    raise ValueError("Acción de archivo inválida.")
                if action == "keep":
                    continue
                media[name] = None
                if kind == "video":
                    audio_name = name.replace("ref_video_", "ref_video_audio_")
                    media[audio_name] = None
                if action == "remove":
                    continue
                if name not in uploads:
                    raise ValueError("Falta el archivo.")
                source = uploads[name]
                if kind == "image":
                    path = target(".png")
                    path.write_bytes(prompts.normalized_png(source.read_bytes()))
                elif kind == "video":
                    metadata = await asyncio.to_thread(inspect_media, source, kind)
                    path = target(".mp4")
                    await ffmpeg("-i", source, "-map", "0:v:0", "-an", "-vf", "fps=24", "-c:v", "libx264",
                                 "-crf", "18", "-pix_fmt", metadata["pixel_format"], "-movflags", "+faststart", path)
                    with av.open(str(path)) as container:
                        if container.streams.video[0].frames < 5:
                            raise ValueError("MiniMax requiere al menos 5 frames a 24 fps.")
                    if metadata["audio"] and fields.get(f"{audio_name}_action", "keep") == "keep":
                        soundtrack = target(".wav")
                        await ffmpeg("-i", source, "-map", "0:a:0", "-vn", "-c:a", "pcm_f32le", soundtrack)
                        await asyncio.to_thread(inspect_media, soundtrack, "audio")
                        media[audio_name] = str(soundtrack)
                else:
                    path = target(".wav")
                    await ffmpeg("-i", source, "-map", "0:a:0", "-vn", "-c:a", "pcm_f32le", path)
                    await asyncio.to_thread(inspect_media, path, kind)
                media[name] = str(path)
            name = "elemento"
            prompts.library.save(key, prompt, None, None, fields.get("entry") or None, fields.get("base"),
                                 media=media, kind="reference")
            committed = True
        prompts.collect_unused()
        return web.json_response(prompts.library.view(key))
    except (ValueError, KeyError, UnidentifiedImageError, OSError, av.error.FFmpegError) as error:
        return web.json_response({"error": f"{name}: {error}"}, status=400)
    finally:
        if leased:
            with prompts.library.lock:
                prompts.library.leases[leased] -= 1
                if not prompts.library.leases[leased]:
                    del prompts.library.leases[leased]
            prompts.collect_unused()
        for path in created:
            if not committed or str(path) not in media.values():
                path.unlink(missing_ok=True)


@routes.get("/yafv/prompts/media/{revision}/{name}")
async def get_media(request):
    try:
        item = prompts.library.revision(request.query["collection"], request.match_info["revision"])
        name = request.match_info["name"]
        if name not in MEDIA_TYPES or not item.media.get(name):
            raise ValueError("Este elemento no tiene esa referencia.")
        return web.FileResponse(item.media[name], headers={"Cache-Control": "no-store"})
    except ValueError as error:
        return web.json_response({"error": str(error)}, status=404)


if hasattr(PromptServer, "instance"):
    for route in routes:
        PromptServer.instance.routes.route(route.method, route.path, **route.kwargs)(route.handler)
