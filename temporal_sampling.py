"""Overlapping temporal refinement of MiniMax H3 AV latents."""
from bisect import bisect_left, bisect_right
import logging
import math

import torch
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.sample
import comfy.utils
from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_minimax_h3 import align_frame_count, video_latent_t

AUTO_OVERLAP_FRAMES = 22


def frame_boundary(token):
    groups, remainder = divmod(token, len(FRAME_PER_TOKEN))
    return groups * sum(FRAME_PER_TOKEN) + sum(FRAME_PER_TOKEN[:remainder])


def window_sizes(chunk_frames, overlap_frames):
    chunk = video_latent_t(align_frame_count(max(22, chunk_frames)))
    overlap = video_latent_t(align_frame_count(max(5, overlap_frames)))
    if overlap >= chunk:
        raise ValueError("Pass 2 overlap must be shorter than the chunk after H3 frame alignment.")
    return chunk, overlap


def automatic_window_sizes(tokens, chunks, overlap_frames=AUTO_OVERLAP_FRAMES):
    """Choose an H3-aligned window that covers the latent with ``chunks`` windows."""
    if not isinstance(chunks, int) or chunks < 1:
        raise ValueError("Automatic temporal chunking requires at least one chunk.")
    if tokens < 2 or (tokens - 2) % 5:
        raise ValueError("Temporal Pass 2 requires an H3 latent with 5k + 2 temporal positions.")
    target_overlap = video_latent_t(align_frame_count(max(5, overlap_frames)))
    minimum_chunk = video_latent_t(align_frame_count(22))
    required = math.ceil((tokens + (chunks - 1) * target_overlap) / chunks)
    target_chunk = max(minimum_chunk, ((required - 2 + 4) // 5) * 5 + 2)

    candidates = []
    for chunk in range(minimum_chunk, tokens + 1, 5):
        for overlap in range(2, chunk, 5):
            if len(temporal_windows(tokens, frame_boundary(chunk), frame_boundary(overlap), exact_tail=True)) != chunks:
                continue
            candidates.append((
                abs(chunk - target_chunk),
                abs(overlap - target_overlap),
                -overlap,
                chunk,
                overlap,
            ))
    if not candidates:
        raise ValueError(f"Cannot fit exactly {chunks} temporal chunks for this video duration.")
    _, _, _, chunk, overlap = min(candidates)
    return frame_boundary(chunk), frame_boundary(overlap)


def temporal_windows(tokens, chunk_frames, overlap_frames, exact_tail=False):
    chunk, overlap = window_sizes(chunk_frames, overlap_frames)
    if tokens < 2 or (tokens - 2) % 5:
        raise ValueError("Temporal Pass 2 requires an H3 latent with 5k + 2 temporal positions.")
    if tokens <= chunk:
        return [(0, tokens)]
    # Both lengths are 5k+2. Starts spaced by a multiple of five preserve
    # the (1,4,4,4,4) frame cadence when each window is sampled locally.
    hop = chunk - overlap
    starts = list(range(0, tokens - chunk + 1, hop))
    if exact_tail and starts[-1] + chunk < tokens:
        # Keep the requested overlap at the tail. The final window may be
        # shorter than ``chunk`` but remains on H3's 5k+2 latent grid.
        next_start = starts[-1] + hop
        while next_start < tokens:
            starts.append(next_start)
            if next_start + chunk >= tokens:
                break
            next_start += hop
        return [(start, min(start + chunk, tokens)) for start in starts]
    if starts[-1] != tokens - chunk:
        starts.append(tokens - chunk)
    return [(start, start + chunk) for start in starts]


def window_conditioning(conditioning, frame_start, frame_end, height, width):
    output = []
    for embedding, metadata in conditioning:
        local = dict(metadata)
        if "minimax_keyframes" in metadata:
            guides = []
            for guide in metadata["minimax_keyframes"]:
                # Continuation context is already expressed in the local
                # chunk's coordinate system.  Do not treat it as a source
                # timeline keyframe (it has no resolved_frame_index).
                if guide.get("kind") in ("context", "context_audio"):
                    guides.append(dict(guide))
                    continue
                origin = guide["resolved_frame_index"]
                video = guide.get("latent")
                audio = guide.get("audio_latent")
                if video is not None:
                    bounds = [frame_boundary(i) for i in range(video.shape[2] + 1)]
                    if origin < frame_end and origin + bounds[-1] > frame_start:
                        first = max(0, bisect_right(bounds, frame_start - origin) - 1)
                        first = first // 5 * 5
                        stop = min(video.shape[2], bisect_left(bounds, frame_end - origin))
                        cropped = video[:, :, first:stop].to(device="cpu", copy=True)
                        if cropped.shape[-2:] != (height, width):
                            batch, channels, time, h, w = cropped.shape
                            frames = cropped.movedim(1, 2).reshape(batch * time, channels, h, w)
                            frames = F.interpolate(frames, size=(height, width), mode="bilinear", align_corners=False)
                            cropped = frames.reshape(batch, time, channels, height, width).movedim(2, 1).contiguous()
                        item = dict(guide)
                        item.pop("audio_latent", None)
                        item["latent"] = cropped
                        item["resolved_frame_index"] = origin + bounds[first] - frame_start
                        guides.append(item)
                if audio is not None:
                    first = max(0, math.ceil((frame_start - origin) * FRAME_RESCALE))
                    stop = min(audio.shape[-1], math.ceil((frame_end - origin) * FRAME_RESCALE))
                    if first < stop:
                        item = dict(guide)
                        item.pop("latent", None)
                        item["audio_latent"] = audio[..., first:stop].to(device="cpu", copy=True)
                        item["resolved_frame_index"] = origin + first / FRAME_RESCALE - frame_start
                        guides.append(item)
                if video is None and audio is None and frame_start <= origin < frame_end:
                    guides.append({**guide, "resolved_frame_index": origin - frame_start})
            if guides:
                local["minimax_keyframes"] = guides
            else:
                local.pop("minimax_keyframes")
        # minimax_refs describe independent references, not target-timeline
        # anchors. Keep them intact, including their text modality tags.
        output.append([embedding, local])
    return output


def refine_audio(model, seed, steps, start_step, cfg, sampler_name, scheduler,
                 positive, negative, latent, add_noise):
    """Refine the complete audio track against frozen, pre-upscale video."""
    samples = latent["samples"]
    if not samples.is_nested or len(samples.unbind()) != 2:
        raise ValueError("Audio refinement requires a MiniMax H3 audiovisual latent.")
    video, audio = samples.unbind()
    if video.ndim != 5 or video.shape[1] != 24 or audio.ndim != 4 or audio.shape[1:3] != (32, 2):
        raise ValueError("Audio refinement requires H3 video [B,24,T,H,W] and audio [B,32,2,T].")
    mm.throw_exception_if_processing_interrupted()
    video, audio = video.cpu(), audio.cpu()
    source = NestedTensor((video, audio))
    noise = (comfy.sample.prepare_noise(source, seed, latent.get("batch_index")) if add_noise
             else comfy.sample.prepare_empty_noise(source))
    audio_mask = torch.ones((1, 1, 2, audio.shape[-1]), dtype=torch.float32)
    original_mask = latent.get("noise_mask")
    if original_mask is not None and original_mask.is_nested:
        masks = original_mask.unbind()
        if len(masks) > 1:
            audio_mask = masks[1].to(device="cpu", dtype=torch.float32, copy=True)
    mask = NestedTensor((torch.zeros((1, 1, video.shape[2], 1, 1), dtype=torch.float32), audio_mask))
    progress = comfy.utils.ProgressBar(steps - start_step)

    def callback(step, x0, x, total_steps):
        mm.throw_exception_if_processing_interrupted()
        progress.update_absolute(step + 1)

    logging.info("[YAFV H3] Refining full audio with frozen %dx%d px video: steps %d-%d.",
                 video.shape[-1] * 16, video.shape[-2] * 16, start_step, steps)
    sampled = comfy.sample.sample(
        model, noise, steps, cfg, sampler_name, scheduler, positive, negative,
        source, disable_noise=not add_noise, start_step=start_step, last_step=steps,
        force_full_denoise=True, noise_mask=mask, callback=callback,
        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed,
    )
    mm.throw_exception_if_processing_interrupted()
    result = dict(latent)
    result["samples"] = NestedTensor((video, sampled.unbind()[1].to(device="cpu", dtype=audio.dtype)))
    return result


def sample_temporal(model, seed, steps, cfg, sampler_name, scheduler, positive,
                    negative, latent, start_step, add_noise, chunk_frames,
                    overlap_frames, chunk_count=None, chunk_callback=None):
    samples = latent["samples"]
    if not samples.is_nested or len(samples.unbind()) != 2:
        raise ValueError("Temporal Pass 2 requires a MiniMax H3 audiovisual latent.")
    video, audio = samples.unbind()
    if video.ndim != 5 or video.shape[1] != 24 or audio.ndim != 4 or audio.shape[1:3] != (32, 2):
        raise ValueError("Temporal Pass 2 requires H3 video [B,24,T,H,W] and audio [B,32,2,T].")
    if chunk_count is not None:
        chunk_frames, overlap_frames = automatic_window_sizes(video.shape[2], chunk_count)
    windows = temporal_windows(video.shape[2], chunk_frames, overlap_frames, exact_tail=True)
    if chunk_count is not None and len(windows) != chunk_count:
        raise ValueError(f"Automatic temporal chunking produced {len(windows)} windows instead of {chunk_count}.")
    logging.info("[YAFV H3] Pass 2: %d temporal window(s), chunk=%d frames, overlap=%d frames, up to %d latent positions each, %dx%d px; audio preserved.",
                 len(windows), chunk_frames, overlap_frames, max(stop - start for start, stop in windows),
                 video.shape[-1] * 16, video.shape[-2] * 16)
    video, audio = video.cpu(), audio.cpu()
    source = NestedTensor((video, audio))
    mm.throw_exception_if_processing_interrupted()
    noise = (comfy.sample.prepare_noise(source, seed, latent.get("batch_index")) if add_noise
             else comfy.sample.prepare_empty_noise(source))
    noise_video, noise_audio = noise.unbind()
    # Every window starts from Pass 1 with the same noise at shared positions.
    # Feeding a finished window back into the next one propagates its artifacts.
    accumulated = torch.zeros_like(video, dtype=torch.float32)
    total_weight = torch.zeros((1, 1, video.shape[2], 1, 1), dtype=torch.float32)
    steps_per_chunk = steps - start_step
    progress = comfy.utils.ProgressBar(steps_per_chunk * len(windows))

    for index, (start, stop) in enumerate(windows):
        mm.throw_exception_if_processing_interrupted()
        f0, f1 = frame_boundary(start), frame_boundary(stop)
        a0 = min(audio.shape[-1], round(f0 * FRAME_RESCALE))
        a1 = min(audio.shape[-1], round(f1 * FRAME_RESCALE))
        if a1 <= a0:
            raise ValueError("The H3 audio latent is too short for this video window.")
        if chunk_callback is not None:
            chunk_callback(index + 1, len(windows), f0, f1)
        window_video = video[:, :, start:stop].clone()
        window_audio = audio[..., a0:a1].clone()
        window = NestedTensor((window_video, window_audio))
        window_noise = NestedTensor((
            noise_video[:, :, start:stop].clone(),
            noise_audio[..., a0:a1].clone(),
        ))
        video_mask = torch.ones((1, 1, window_video.shape[2], 1, 1), dtype=torch.float32)
        window_mask = NestedTensor((
            video_mask,
            torch.zeros((1, 1, 2, window_audio.shape[-1]), dtype=torch.float32),
        ))
        cond = window_conditioning(positive, f0, f1, video.shape[-2], video.shape[-1])
        uncond = window_conditioning(negative, f0, f1, video.shape[-2], video.shape[-1])

        def callback(step, x0, x, total_steps):
            mm.throw_exception_if_processing_interrupted()
            progress.update_absolute(index * steps_per_chunk + step + 1)

        sampled = comfy.sample.sample(
            model, window_noise, steps, cfg, sampler_name, scheduler, cond, uncond,
            window, disable_noise=not add_noise, start_step=start_step, last_step=steps,
            force_full_denoise=True, noise_mask=window_mask, callback=callback,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed,
        )
        refined = sampled.unbind()[0].to(device="cpu", dtype=torch.float32)
        weight = torch.ones(stop - start, dtype=torch.float32)
        if index:
            overlap = windows[index - 1][1] - start
            weight[:overlap] = torch.arange(1, overlap + 1, dtype=torch.float32) / (overlap + 1)
        if index + 1 < len(windows):
            overlap = stop - windows[index + 1][0]
            fade = torch.arange(overlap, 0, -1, dtype=torch.float32) / (overlap + 1)
            weight[-overlap:] = torch.minimum(weight[-overlap:], fade)
        weight = weight.reshape(1, 1, -1, 1, 1)
        accumulated[:, :, start:stop].addcmul_(refined, weight)
        total_weight[:, :, start:stop] += weight
        del sampled, refined, window, window_noise, window_mask, cond, uncond
        mm.soft_empty_cache()
        mm.throw_exception_if_processing_interrupted()

    result = dict(latent)
    assembled_video = accumulated.div_(total_weight).to(dtype=video.dtype)
    result["samples"] = NestedTensor((assembled_video, audio))
    return result
