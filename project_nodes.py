"""Interactive MiniMax H3 project nodes.

The project deliberately does not use H3 Video Extend.  Each segment is a
fresh Ref2Vid or FL2V generation; the project stores generations, exposes the
approved clip tail as context, and composes the active timeline after every
generation/review action.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import folder_paths
import comfy.model_management as model_management
from .project_continuity import apply_context, trim_media
import numpy as np
import torch
from PIL import Image
from aiohttp import web

from comfy_api.latest import InputImpl, io
from comfy_execution.graph_utils import ExecutionBlocker
from server import PromptServer


_PROJECT_ROOT = "yafv_projects"
_SAFE_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_ .")
_REVIEW_PENDING: dict[str, dict[str, Any]] = {}


def _safe_name(value: Any) -> str:
    raw = str(value or "default").strip()
    cleaned = "".join(char if char in _SAFE_CHARS else "_" for char in raw)
    return cleaned.strip(" .")[:120] or "default"


def _project_dir(project_name: Any) -> Path:
    root = Path(folder_paths.get_output_directory()).resolve() / _PROJECT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    path = (root / _safe_name(project_name)).resolve()
    if root not in path.parents:
        raise ValueError("Invalid project name")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _manifest_path(project_name: Any) -> Path:
    return _project_dir(project_name) / "project.json"


def _empty_manifest(project_name: Any) -> dict[str, Any]:
    return {
        "version": 1,
        "project_name": _safe_name(project_name),
        "created_at": time.time(),
        "updated_at": time.time(),
        "segments": {},
    }


def _load_manifest(project_name: Any) -> dict[str, Any]:
    path = _manifest_path(project_name)
    if not path.is_file():
        return _empty_manifest(project_name)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read project state: {error}") from error
    if not isinstance(value, dict) or not isinstance(value.get("segments", {}), dict):
        raise ValueError("Invalid YAFV project state")
    return value


def _save_manifest(project_name: Any, manifest: dict[str, Any]) -> None:
    path = _manifest_path(project_name)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    manifest["updated_at"] = time.time()
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _segment(manifest: dict[str, Any], index: int) -> dict[str, Any]:
    key = str(int(index))
    value = manifest["segments"].setdefault(
        key,
        {"index": int(index), "status": "empty", "active_generation": None, "generations": []},
    )
    value.setdefault("generations", [])
    value.setdefault("active_generation", None)
    value.setdefault("status", "empty")
    return value


def _active_generation(manifest: dict[str, Any], index: int, approved_only: bool = False) -> dict[str, Any] | None:
    segment = manifest.get("segments", {}).get(str(int(index)))
    if not isinstance(segment, dict):
        return None
    generations = segment.get("generations", [])
    active = segment.get("active_generation")
    candidates = [
        item for item in generations
        if isinstance(item, dict) and item.get("video")
        and (not approved_only or item.get("status") == "approved")
    ]
    if active is not None:
        selected = next((item for item in candidates if item.get("generation") == active), None)
        if selected is not None:
            return selected
    return max(candidates, key=lambda item: int(item.get("generation", -1)), default=None)


def _local_video_source(video: Any) -> tuple[str, list[str]]:
    temporary: list[str] = []
    if not hasattr(video, "get_stream_source"):
        candidates = []

        def collect_paths(value: Any) -> None:
            if isinstance(value, (str, os.PathLike)):
                candidates.append(os.fspath(value))
            elif isinstance(value, dict):
                for item in value.values():
                    collect_paths(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect_paths(item)

        collect_paths(video)
        video_extensions = {".mp4", ".mkv", ".webm", ".mov", ".avi"}
        source = next(
            (path for path in reversed(candidates)
             if Path(path).suffix.lower() in video_extensions and os.path.isfile(path)),
            None,
        )
        if source is None:
            raise ValueError("VHS_VideoCombine did not return a supported video file")

        # VHS_VideoCombine returns a filename container. Those files are
        # already persistent output files and must not be deleted here.
        return source, temporary

    try:
        source = video.get_stream_source()
    except (AttributeError, NotImplementedError, RuntimeError, TypeError, ValueError):
        source = None
    if isinstance(source, (str, os.PathLike)) and os.path.isfile(source):
        return os.fspath(source), temporary

    fd, path = tempfile.mkstemp(suffix=".mp4", dir=folder_paths.get_temp_directory())
    os.close(fd)
    try:
        video.save_to(path)
    except Exception:
        Path(path).unlink(missing_ok=True)
        raise
    temporary.append(path)
    return path, temporary


def _review_video(video: Any) -> Any:
    if hasattr(video, "get_stream_source"):
        return video
    # A VHS filename container points to a persistent output file. For a
    # generic in-memory Video, keep the materialized temp file alive until
    # the downstream commit has consumed it.
    source, _temporary = _local_video_source(video)
    return InputImpl.VideoFromFile(source)


def _video_data_url(video: Any) -> str:
    source = video.get_stream_source()
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        with open(path, "rb") as stream:
            data = stream.read()
        mime = "video/webm" if Path(path).suffix.lower() == ".webm" else "video/mp4"
    else:
        source.seek(0)
        data = source.read()
        mime = "video/mp4"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _display_node_id(unique_id: Any, dynprompt: Any) -> str:
    if dynprompt is not None:
        try:
            return str(dynprompt.get_display_node_id(str(unique_id)))
        except (AttributeError, TypeError, ValueError):
            pass
    return str(unique_id or "")


async def _wait_review(future: asyncio.Future) -> Any:
    # Match ComfyUI-Media-Review-Checkpoints: polling keeps the execution
    # task cooperative while the HTTP handler resolves the decision future.
    while not future.done():
        model_management.throw_exception_if_processing_interrupted()
        try:
            return await asyncio.wait_for(asyncio.shield(future), 0.25)
        except asyncio.TimeoutError:
            pass
    model_management.throw_exception_if_processing_interrupted()
    return future.result()


def _resolve_review(future, result):
    # Runs on the owning loop; approve/cancel can arrive concurrently.
    if not future.done():
        future.set_result(result)


def _ffprobe(path: str) -> dict[str, Any]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace")[-800:])
    value = json.loads(result.stdout.decode())
    streams = value.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    duration = float(video.get("duration") or value.get("format", {}).get("duration") or 0)
    return {
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "fps": _parse_rate(video.get("avg_frame_rate") or video.get("r_frame_rate") or "24/1"),
        "duration": max(0.0, duration),
    }


def _encode_context_av(path: str, start: float, duration: float, video_latent: Any,
                       audio_vae: Any) -> dict[str, Any]:
    """Encode an approved MP4 tail in native ComfyUI H3 AV format."""
    import comfy.nested_tensor

    sample_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
    sample_count = max(1, round(float(duration) * sample_rate))
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-ss", str(max(0.0, start)),
            "-t", str(max(0.0, duration)), "-i", str(path),
            "-f", "f32le", "-ac", "2", "-ar", str(sample_rate), "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode == 0 and result.stdout:
        audio = np.frombuffer(result.stdout, dtype=np.float32)
        audio = torch.from_numpy(audio.copy())
        audio = audio[: (audio.numel() // 2) * 2].reshape(-1, 2).T
    else:
        audio = torch.zeros((2, sample_count), dtype=torch.float32)
    audio_latent = audio_vae.encode(audio.unsqueeze(0).movedim(1, -1))
    return {"samples": comfy.nested_tensor.NestedTensor((video_latent, audio_latent))}


def _parse_rate(value: str) -> float:
    try:
        numerator, denominator = str(value).split("/", 1)
        return float(numerator) / float(denominator or 1)
    except (ValueError, ZeroDivisionError):
        return 24.0


def _extract_frame(path: str, timestamp: float, width: int, height: int) -> torch.Tensor:
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-ss", f"{max(0.0, timestamp):.6f}", "-i", path,
            "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode or not result.stdout:
        raise RuntimeError(f"Could not extract context frame from {path}")
    with Image.open(__import__("io").BytesIO(result.stdout)) as image:
        image = image.convert("RGB")
        if width and height and image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy())


def _active_video_path(project_name: Any, index: int, approved_only: bool = True) -> Path | None:
    manifest = _load_manifest(project_name)
    generation = _active_generation(manifest, index, approved_only=approved_only)
    if generation is None:
        return None
    path = (_project_dir(project_name) / str(generation["video"])).resolve()
    project_dir = _project_dir(project_name).resolve()
    if project_dir not in path.parents or not path.is_file():
        return None
    return path


def _compose_active(project_name: Any, manifest: dict[str, Any]) -> Path | None:
    sources: list[Path] = []
    for key, segment in sorted(manifest.get("segments", {}).items(), key=lambda item: int(item[0])):
        if not isinstance(segment, dict):
            continue
        generation = _active_generation(manifest, int(key), approved_only=False)
        if generation is None:
            continue
        path = (_project_dir(project_name) / str(generation["video"])).resolve()
        if path.is_file():
            sources.append(path)
    if not sources:
        return None
    output = _project_dir(project_name) / "timeline.mp4"
    list_file = _project_dir(project_name) / f".concat-{uuid.uuid4().hex}.txt"
    try:
        concat_lines = []
        for path in sources:
            escaped_path = str(path).replace("'", "'\\''")
            concat_lines.append(f"file '{escaped_path}'")
        list_file.write_text(
            "\n".join(concat_lines),
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-movflags", "+faststart", str(output),
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors="replace")[-1000:])
    finally:
        list_file.unlink(missing_ok=True)
    return output


def _project_output(project_name: Any, manifest: dict[str, Any]) -> io.NodeOutput:
    timeline = _compose_active(project_name, manifest)
    state = json.dumps(manifest, ensure_ascii=False)
    if timeline is None:
        return io.NodeOutput(None, state)
    return io.NodeOutput(InputImpl.VideoFromFile(str(timeline)), state)


class YAFVProjectContext(io.ComfyNode):
    """Expose the approved previous segment as Ref2Vid/FL2V context."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YAFVProjectContext",
            display_name="YAFV Project Context",
            category="YAFV/Project",
            description="Extract the approved previous clip tail for the next independent generation.",
            inputs=[
                io.String.Input("project_name", default="default"),
                io.Int.Input("segment_index", default=0, min=0),
                io.Int.Input("context_frames", default=22, min=5, max=120),
                io.Image.Input("initial_frame", optional=True),
                io.Vae.Input("vae", optional=True),
                io.Vae.Input("audio_vae", optional=True),
                io.Int.Input("clip_frames", default=124, min=5, max=3600, optional=True),
                io.Combo.Input("scene_mode", options=["Continue scene", "New scene"],
                               default="Continue scene", optional=True,
                               tooltip="New scene skips previous context without changing the project or clip index."),
            ],
            outputs=[
                io.Image.Output("context_images"),
                io.Image.Output("last_frame"),
                io.String.Output("context_video_path"),
                io.String.Output("project_state"),
                io.Video.Output("context_video"),
                io.Latent.Output("context_latent"),
                io.Int.Output("generation_length"),
                io.Int.Output("output_frames"),
            ],
        )

    @classmethod
    def execute(cls, project_name: str, segment_index: int, context_frames: int = 22,
                initial_frame=None, vae=None, audio_vae=None, clip_frames: int = 124,
                scene_mode: str = "Continue scene"):
        scene_mode = {"Continuar escena": "Continue scene", "Nueva escena": "New scene"}.get(scene_mode, scene_mode)
        if scene_mode not in ("Continue scene", "New scene"):
            raise ValueError("Invalid scene_mode")
        clip_frames = max(5, int(clip_frames))
        clip_frames += (5 - clip_frames) % 17
        manifest = _load_manifest(project_name)
        state = json.dumps(manifest, ensure_ascii=False)
        if scene_mode == "New scene":
            return io.NodeOutput(None, None, "", state, None, None,
                                 int(clip_frames), int(clip_frames))
        previous = _active_video_path(project_name, int(segment_index) - 1, approved_only=True)
        if previous is None:
            return io.NodeOutput(None, initial_frame, "", state, None, None,
                                 int(clip_frames), int(clip_frames))
        metadata = _ffprobe(str(previous))
        fps = metadata["fps"] or 24.0
        duration = metadata["duration"]
        count = max(5, int(context_frames))
        count += (5 - count) % 17
        start = max(0.0, duration - (count / fps))
        timestamps = [start + index / fps for index in range(count)]
        frames = torch.stack([
            _extract_frame(str(previous), timestamp, metadata["width"], metadata["height"])
            for timestamp in timestamps
        ], dim=0)
        context_latent = None
        if vae is not None:
            video_latent = vae.encode(frames[..., :3])
            if audio_vae is not None:
                context_latent = _encode_context_av(
                    str(previous), start, count / fps, video_latent, audio_vae,
                )
            else:
                context_latent = {"samples": video_latent}
        return io.NodeOutput(
            frames, frames[-1:].contiguous(), str(previous), state,
            InputImpl.VideoFromFile(str(previous)), context_latent,
            int(clip_frames) + ((count + 16) // 17) * 17, int(clip_frames),
        )


class YAFVProjectMotionContext(io.ComfyNode):
    """Apply native H3 continuity, bypassed for segment zero."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YAFVProjectMotionContext",
            display_name="YAFV Project Motion Context",
            category="YAFV/Project",
            description="Apply native H3 latent continuity to the next project clip.",
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.Vae.Input("vae"),
                io.Latent.Input("latent"),
                io.Latent.Input("context_latent", optional=True),
                io.Combo.Input("context_length", options=["5", "22", "39", "56"], default="22"),
                io.Int.Input("video_transition_steps", default=4, min=0, max=32),
                io.Int.Input("audio_transition_steps", default=4, min=0, max=80),
                io.Boolean.Input("video_anchor_only", default=True),
            ],
            outputs=[
                io.Conditioning.Output("conditioning"),
                io.Int.Output("trim_frames"),
                io.Latent.Output("latent"),
            ],
        )

    @classmethod
    def execute(cls, conditioning, vae, latent, context_latent=None,
                context_length="22", video_transition_steps=4,
                audio_transition_steps=4, video_anchor_only=True):
        if context_latent is None:
            return io.NodeOutput(conditioning, 0, latent)
        return io.NodeOutput(*apply_context(
            conditioning=conditioning,
            latent=latent,
            context_latent=context_latent,
            context_length=context_length,
            video_transition_steps=video_transition_steps,
            audio_transition_steps=audio_transition_steps,
            video_anchor_only=video_anchor_only,
        ))


class YAFVProjectMediaTrim(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YAFVProjectMediaTrim", display_name="YAFV Project Media Trim",
            category="YAFV/Project",
            inputs=[io.Image.Input("images"), io.Audio.Input("audio", optional=True),
                    io.Int.Input("trim_frames", default=0, min=0),
                    io.Int.Input("output_frames", default=124, min=1),
                    io.Float.Input("fps", default=24.0, min=1.0)],
            outputs=[io.Image.Output("images"), io.Audio.Output("audio")],
        )

    @classmethod
    def execute(cls, images, audio=None, trim_frames=0, output_frames=124, fps=24.0):
        return io.NodeOutput(*trim_media(images, audio, trim_frames, output_frames, fps))


class YAFVProjectCommit(io.ComfyNode):
    """Store one generated segment and return the currently composed project timeline."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YAFVProjectCommit",
            display_name="YAFV Project Commit Segment",
            category="YAFV/Project",
            description="Save an approved segment and update the active project timeline.",
            inputs=[
                io.String.Input("project_name", default="default"),
                io.Int.Input("segment_index", default=0, min=0),
                io.AnyType.Input("video"),
                io.String.Input("generator_mode", default="ref2vid", optional=True),
                io.Int.Input("seed", default=0, optional=True),
            ],
            outputs=[io.Video.Output("timeline"), io.String.Output("project_state")],
            is_output_node=True,
            not_idempotent=True,
        )

    @classmethod
    def execute(cls, project_name: str, segment_index: int, video: Any, generator_mode: str = "ref2vid", seed: int = 0):
        manifest = _load_manifest(project_name)
        directory = _project_dir(project_name)
        segment = _segment(manifest, int(segment_index))
        generation = max(
            (int(item.get("generation", -1)) for item in segment["generations"] if isinstance(item, dict)),
            default=-1,
        ) + 1
        segment_dir = directory / f"segment_{int(segment_index):03d}"
        segment_dir.mkdir(parents=True, exist_ok=True)
        target = segment_dir / f"generation_{generation:03d}.mp4"
        source, temporary = _local_video_source(video)
        try:
            shutil.copy2(source, target)
        finally:
            for path in temporary:
                Path(path).unlink(missing_ok=True)
        segment["generations"].append({
            "generation": generation,
            "video": str(target.relative_to(directory)),
            "status": "approved",
            "generator_mode": str(generator_mode or "ref2vid"),
            "seed": int(seed),
            "created_at": time.time(),
        })
        segment["active_generation"] = generation
        segment["status"] = "approved"
        _save_manifest(project_name, manifest)
        return _project_output(project_name, manifest)


class YAFVProjectReview(io.ComfyNode):
    """Pause on a generated candidate and pass it onward only when approved."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YAFVProjectReview",
            display_name="YAFV Project Review",
            category="YAFV/Project",
            description="Review a candidate in the UI; only an approved video reaches Project Commit.",
            inputs=[
                io.String.Input("project_name", default="default"),
                io.Int.Input("segment_index", default=0, min=0),
                io.AnyType.Input("video"),
            ],
            hidden=[io.Hidden.dynprompt, io.Hidden.unique_id],
            outputs=[io.Video.Output("video"), io.String.Output("review_status")],
            not_idempotent=True,
        )

    @classmethod
    async def execute(cls, project_name: str, segment_index: int, video: Any,
                      dynprompt=None, unique_id=None):
        media = _review_video(video)
        unique_id = unique_id or cls.hidden.unique_id
        dynprompt = dynprompt or cls.hidden.dynprompt
        token = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        _REVIEW_PENDING[token] = {
            "future": future,
            "loop": asyncio.get_running_loop(),
        }
        try:
            data = {"token": token, "node_id": _display_node_id(unique_id, dynprompt),
                    "video_preview": await asyncio.to_thread(_video_data_url, media)}
            _REVIEW_PENDING[token]["data"] = data
            PromptServer.instance.send_sync("yafv_project_review", data)
            result = await _wait_review(future)
        finally:
            _REVIEW_PENDING.pop(token, None)
            if not future.done():
                future.cancel()
            PromptServer.instance.send_sync("yafv_project_review_closed", {"token": token})
        action = result.get("action") if isinstance(result, dict) else None
        if action in {"reject", "cancel"}:
            message = (
                "candidate rejected; nothing was committed"
                if action == "reject"
                else "project review cancelled"
            )
            return {"result": (ExecutionBlocker(None), message)}
        if action != "approve":
            raise ValueError("Project review action must be approve or reject")
        return io.NodeOutput(media, "approved")


NODE_CLASS_MAPPINGS = {
    "YAFVProjectMediaTrim": YAFVProjectMediaTrim,
    "YAFVProjectContext": YAFVProjectContext,
    "YAFVProjectCommit": YAFVProjectCommit,
    "YAFVProjectReview": YAFVProjectReview,
    "YAFVProjectMotionContext": YAFVProjectMotionContext,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YAFVProjectMediaTrim": "YAFV · Project Media Trim",
    "YAFVProjectContext": "YAFV · Project Context",
    "YAFVProjectCommit": "YAFV · Project Commit Segment",
    "YAFVProjectReview": "YAFV · Project Review",
    "YAFVProjectMotionContext": "YAFV · Project Motion Context",
}


if getattr(PromptServer, "instance", None) is not None:
    @PromptServer.instance.routes.get("/yafv/project-review/pending")
    async def _project_review_pending(request):
        return web.json_response([p["data"] for p in list(_REVIEW_PENDING.values())
                                  if "data" in p and not p["future"].done()])

    @PromptServer.instance.routes.post("/yafv/project-review/stop")
    async def _project_review_stop(request):
        model_management.interrupt_current_processing()
        return web.json_response({"ok": True})

    @PromptServer.instance.routes.post("/yafv/project-review")
    async def _project_review_decision(request):
        try:
            payload = await request.json()
            token = str(payload.get("token", ""))
            action = str(payload.get("action", ""))
            if action not in {"approve", "reject"}:
                raise ValueError("Invalid project review action")
            pending = _REVIEW_PENDING.get(token)
            if pending is None or pending["future"].done():
                raise ValueError("Project review is no longer pending")
            pending["loop"].call_soon_threadsafe(
                _resolve_review, pending["future"], {"action": action}
            )
            return web.json_response({"ok": True})
        except (ValueError, KeyError, TypeError) as error:
            return web.json_response({"error": str(error)}, status=400)

    @PromptServer.instance.routes.post("/yafv/project-review/cancel")
    async def _project_review_cancel(request):
        try:
            payload = await request.json()
            token = str(payload.get("token", ""))
            pending = _REVIEW_PENDING.get(token)
            if pending is not None and not pending["future"].done():
                pending["loop"].call_soon_threadsafe(
                    _resolve_review, pending["future"], {"action": "cancel"}
                )
            return web.json_response({"ok": True})
        except (ValueError, KeyError, TypeError) as error:
            return web.json_response({"error": str(error)}, status=400)
