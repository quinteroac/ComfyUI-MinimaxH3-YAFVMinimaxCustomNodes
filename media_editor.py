"""Session-only media editing. Source files must still belong to prompt history."""
import asyncio
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import tempfile

from aiohttp import web
from PIL import Image, ImageDraw, ImageOps
import folder_paths
from server import PromptServer


class YAFVMediaEditor:
    CATEGORY = "YAFV/media"
    FUNCTION = "execute"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    DESCRIPTION = "Queue, image/video timeline and pencil in one panel. Edits last only for this browser session."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def execute(self):
        return ()


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".gif"}
routes = web.RouteTableDef()
exports = {}


def descriptors(value, default_type="output"):
    if isinstance(value, dict):
        if isinstance(value.get("filename"), str):
            suffix = Path(value["filename"]).suffix.lower()
            if suffix in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
                yield {"filename": value["filename"], "subfolder": value.get("subfolder", ""),
                       "type": value.get("type", default_type),
                       "kind": "image" if suffix in IMAGE_EXTENSIONS else "video"}
        else:
            for key, child in value.items():
                yield from descriptors(child, "temp" if key == "h3_preview" else default_type)
    elif isinstance(value, list):
        for child in value:
            yield from descriptors(child, default_type)


def history_media(history):
    result = {}
    for prompt_id, entry in history.items():
        for node_id, output in entry.get("outputs", {}).items():
            for file in descriptors(output):
                if file["type"] not in {"output", "temp"}:
                    continue
                identity = json.dumps([prompt_id, node_id, file], sort_keys=True)
                media_id = hashlib.sha256(identity.encode()).hexdigest()
                result[media_id] = {**file, "id": media_id, "prompt_id": prompt_id, "node_id": node_id}
    return result


def resolve_media(media_id):
    media = history_media(PromptServer.instance.prompt_queue.get_history()).get(media_id)
    if media is None:
        raise ValueError("El medio ya no está en el historial temporal.")
    if media["type"] not in {"output", "temp"}:
        raise ValueError("Solo se admiten resultados output/temp del historial.")
    root = Path(folder_paths.get_directory_by_type(media["type"])).resolve()
    path = (root / media["subfolder"] / media["filename"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("El archivo no está disponible dentro de su carpeta de resultados.")
    return media, path


def number(value, name, minimum=0, maximum=None):
    value = float(value)
    if not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"Valor inválido: {name}.")
    return value


async def command(*args, progress=None, duration=1):
    process = await asyncio.create_subprocess_exec(*map(str, args), stdout=asyncio.subprocess.PIPE,
                                                   stderr=asyncio.subprocess.PIPE)
    async def read_progress():
        async for line in process.stdout:
            if line.startswith(b"out_time_us=") and progress:
                raw = line.strip().split(b"=", 1)[1]
                if raw.isdigit():
                    progress(min(1, int(raw) / 1_000_000 / duration))
    try:
        if progress:
            _, error = await asyncio.gather(read_progress(), process.stderr.read())
            output = b""
            await process.wait()
        else:
            output, error = await process.communicate()
        if process.returncode:
            raise ValueError(error.decode(errors="replace")[-1600:] or "FFmpeg no pudo procesar el medio.")
        return output
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


async def probe(path, kind):
    if kind == "image":
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            return {"width": image.width, "height": image.height, "duration": 3, "fps": 24, "audio": False}
    raw = await command("ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path)
    data = json.loads(raw)
    video = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if video is None:
        raise ValueError("El archivo no contiene video.")
    ratio = video.get("avg_frame_rate", "0/1").split("/")
    fps = float(ratio[0]) / float(ratio[1]) if len(ratio) == 2 and float(ratio[1]) else 24
    width, height = video["width"], video["height"]
    rotation = next((s.get("rotation", 0) for s in video.get("side_data_list", []) if "rotation" in s), 0)
    if abs(rotation) % 180 == 90:
        width, height = height, width
    return {"width": width, "height": height, "duration": float(data["format"].get("duration") or video.get("duration") or 0),
            "fps": fps or 24, "audio": any(s["codec_type"] == "audio" for s in data["streams"])}


def validate_strokes(strokes):
    if not isinstance(strokes, list) or len(strokes) > 1000:
        raise ValueError("Máximo 1000 trazos por clip.")
    result = []
    for stroke in strokes:
        color = stroke["color"]
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            raise ValueError("Color de lápiz inválido.")
        points = stroke["points"]
        if not points or len(points) > 20000:
            raise ValueError("Trazo vacío o demasiado largo.")
        start, end = number(stroke["start"], "inicio del dibujo"), number(stroke["end"], "fin del dibujo")
        if end <= start:
            raise ValueError("El dibujo debe tener duración positiva.")
        result.append({"color": color, "width": number(stroke["width"], "grosor", .00001, 1),
                       "opacity": number(stroke["opacity"], "opacidad", 0, 1), "start": start, "end": end,
                       "points": [(number(p[0], "x", 0, 1), number(p[1], "y", 0, 1)) for p in points]})
    return result


def paint(size, strokes):
    canvas = Image.new("RGBA", size)
    for stroke in strokes:
        layer = Image.new("RGBA", size)
        draw = ImageDraw.Draw(layer)
        color = tuple(bytes.fromhex(stroke["color"][1:])) + (round(255 * stroke["opacity"]),)
        width = max(1, round(stroke["width"] * size[0]))
        points = [(round(x * (size[0] - 1)), round(y * (size[1] - 1))) for x, y in stroke["points"]]
        if len(points) > 1:
            draw.line(points, fill=color, width=width, joint="curve")
        radius = width / 2
        for x, y in points:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        canvas = Image.alpha_composite(canvas, layer)
    return canvas


async def frame_image(path, kind, time):
    if kind == "image":
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert("RGBA")
    raw = await command("ffmpeg", "-v", "error", "-i", path, "-ss", time, "-frames:v", 1,
                        "-f", "image2pipe", "-vcodec", "png", "pipe:1")
    if not raw:
        raise ValueError("No hay frame en ese instante.")
    with Image.open(io.BytesIO(raw)) as image:
        return image.convert("RGBA")


async def render_clip(clip, target, size, fps, directory, progress):
    media, path = resolve_media(clip["media_id"])
    meta = await probe(path, media["kind"])
    still = media["kind"] == "image" or clip.get("frame_time") is not None
    start, end = number(clip["in"], "entrada"), number(clip["out"], "salida")
    if end <= start or (not still and end > meta["duration"] + .05):
        raise ValueError("El tramo está fuera de la duración del video.")
    # Every clip occupies a whole number of output frames, including its audio.
    count = max(1, round((end - start) * fps))
    duration = count / fps
    strokes = validate_strokes(clip.get("strokes", []))
    args = ["ffmpeg", "-v", "error", "-y", "-filter_complex_threads", "1"]
    if still:
        source = directory / "source.png"
        frame_time = number(clip.get("frame_time", 0), "frame")
        (await frame_image(path, media["kind"], frame_time)).save(source)
        args += ["-loop", "1", "-framerate", str(fps), "-i", str(source)]
    else:
        args += ["-ss", str(start), "-i", str(path)]
    args += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
    groups = []
    for stroke in strokes:
        begin, finish = max(start, stroke["start"]), min(end, stroke["end"])
        if finish > begin:
            interval = (begin - start, finish - start)
            if groups and groups[-1][0] == interval:
                groups[-1][1].append(stroke)
            else:
                groups.append((interval, [stroke]))
    filters = ["[0:v]setpts=PTS-STARTPTS,setsar=1[v0]"]
    for index, ((begin, finish), group) in enumerate(groups):
        overlay = directory / f"ink-{index}.png"
        paint((meta["width"], meta["height"]), group).save(overlay)
        args += ["-i", str(overlay)]
        filters.append(f"[v{index}][{index + 2}:v]overlay=eof_action=repeat:enable='gte(t,{begin})*lt(t,{finish})'[v{index + 1}]")
    width, height = size
    filters.append(f"[v{len(groups)}]scale={width}:{height}:force_original_aspect_ratio=decrease,"
                   f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},"
                   f"tpad=stop_mode=clone:stop_duration={duration},trim=duration={duration},format=yuv420p[v]")
    audio = "0:a" if meta["audio"] and not still else "1:a"
    filters.append(f"[{audio}]asetpts=PTS-STARTPTS,aresample=48000,aformat=channel_layouts=stereo,"
                   f"apad,atrim=duration={duration}[a]")
    args += ["-filter_complex", ";".join(filters), "-map", "[v]", "-map", "[a]", "-t", str(duration),
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "pcm_s16le",
             "-threads", "2", "-progress", "pipe:1", "-nostats", str(target)]
    await command(*args, progress=progress, duration=duration)
    resolve_media(clip["media_id"])


async def render_timeline(data, directory, progress):
    clips = data["clips"]
    if not clips:
        raise ValueError("El timeline está vacío.")
    fps = number(data.get("fps", 24), "FPS", 1, 240)
    width = int(number(data["width"], "ancho", 2, 8192))
    height = int(number(data["height"], "alto", 2, 8192))
    size = (width + width % 2, height + height % 2)
    for index, clip in enumerate(clips):
        clip_dir = directory / str(index)
        clip_dir.mkdir()
        await render_clip(clip, directory / f"clip-{index}.mkv", size, fps, clip_dir,
                          lambda value: progress((index + value) / (len(clips) + 1)))
    listing = directory / "clips.txt"
    listing.write_text("\n".join(f"file 'clip-{i}.mkv'" for i in range(len(clips))))
    output = directory / "montaje.mp4"
    await command("ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "1", "-i", listing,
                  "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", output)
    for clip in clips:
        resolve_media(clip["media_id"])
    progress(1)
    return output


@routes.get("/yafv/editor/media")
async def media_list(request):
    queue = PromptServer.instance.prompt_queue
    running, pending = queue.get_current_queue_volatile()
    return web.json_response({"media": list(history_media(queue.get_history()).values()),
                              "running": len(running), "pending": len(pending),
                              "ffmpeg": bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))})


@routes.get("/yafv/editor/media/{media_id}")
async def media_info(request):
    try:
        media, path = resolve_media(request.match_info["media_id"])
        return web.json_response(await probe(path, media["kind"]))
    except (ValueError, OSError) as error:
        return web.json_response({"error": str(error)}, status=400)


@routes.get("/yafv/editor/source/{media_id}")
async def media_source(request):
    try:
        _, path = resolve_media(request.match_info["media_id"])
        return web.FileResponse(path, headers={"Cache-Control": "no-store"})
    except ValueError as error:
        return web.json_response({"error": str(error)}, status=404)


@routes.post("/yafv/editor/frame")
async def export_frame(request):
    try:
        data = await request.json()
        media, path = resolve_media(data["media_id"])
        time = number(data.get("time", 0), "tiempo")
        image = await frame_image(path, media["kind"], time)
        strokes = validate_strokes(data.get("strokes", []))
        image = Image.alpha_composite(image, paint(image.size, [s for s in strokes if s["start"] <= time < s["end"]]))
        output = io.BytesIO()
        image.save(output, format="PNG")
        resolve_media(data["media_id"])
        return web.Response(body=output.getvalue(), content_type="image/png", headers={"Cache-Control": "no-store"})
    except (ValueError, KeyError, TypeError, OSError) as error:
        return web.json_response({"error": str(error)}, status=400)


@routes.post("/yafv/editor/export/{job_id}")
async def export_video(request):
    job_id = request.match_info["job_id"]
    if job_id in exports:
        return web.json_response({"error": "La exportación ya está activa."}, status=409)
    task = asyncio.current_task()
    exports[job_id] = task
    try:
        data = await request.json()
        def progress(value):
            if request.transport is None or request.transport.is_closing():
                raise asyncio.CancelledError
            PromptServer.instance.send_sync("yafv-editor-progress", {"job_id": job_id, "progress": value}, data.get("client_id"))
        with tempfile.TemporaryDirectory(prefix="yafv-editor-") as temporary:
            output = await render_timeline(data, Path(temporary), progress)
            response = web.StreamResponse(headers={"Content-Type": "video/mp4", "Content-Length": str(output.stat().st_size),
                                                   "Content-Disposition": 'attachment; filename="montaje.mp4"', "Cache-Control": "no-store"})
            await response.prepare(request)
            with output.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    await response.write(chunk)
            await response.write_eof()
            return response
    except asyncio.CancelledError:
        return web.json_response({"error": "Exportación cancelada."}, status=409)
    except (ValueError, KeyError, TypeError, OSError) as error:
        return web.json_response({"error": str(error)}, status=400)
    finally:
        exports.pop(job_id, None)


@routes.delete("/yafv/editor/export/{job_id}")
async def cancel_export(request):
    task = exports.get(request.match_info["job_id"])
    if task:
        task.cancel()
    return web.json_response({"cancelled": task is not None})


if hasattr(PromptServer, "instance"):
    for route in routes:
        PromptServer.instance.routes.route(route.method, route.path, **route.kwargs)(route.handler)
