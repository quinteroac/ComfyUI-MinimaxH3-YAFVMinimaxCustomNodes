"""Overlapping temporal refinement of MiniMax H3 AV latents."""
from bisect import bisect_left, bisect_right
import inspect
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
    if exact_tail:
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


def _append_continuation_context(conditioning, previous_video, previous_audio,
                                 video_tokens, audio_tokens, boundary_video):
    """Attach HR Endless-style native continuation context to conditioning.

    Match HR Endless' two continuation signals: the full Video1/Audio1
    reference plus a five-frame visual boundary keyframe.  The keyframe is
    video-only; audio remains represented by the synchronized reference.
    """
    if video_tokens <= 0 or audio_tokens <= 0:
        return conditioning
    # The native continuation reference must cover the complete overlap.  It
    # is also the state used by the original serial adaptation; shortening it
    # to a five-frame boundary changes the lighting/appearance conditioning.
    video_tail = previous_video[:, :, -video_tokens:].to(device="cpu", copy=True)
    audio_tail = previous_audio[..., -audio_tokens:].to(device="cpu", copy=True)
    ref = {
        "kind": "video_audio",
        "latent": video_tail,
        "latent_t": int(video_tail.shape[2]),
        "latent_h": int(video_tail.shape[-2]),
        "latent_w": int(video_tail.shape[-1]),
        "audio_latent": audio_tail,
        "ref_audio_t": int(audio_tail.shape[-1]),
    }
    boundary = boundary_video[:, :, -video_latent_t(5):].to(device="cpu", copy=True)
    supports_context = "frame_count" in inspect.signature(
        comfy.ldm.minimax.model.PackedLayout.__init__
    ).parameters
    output = []
    for embedding, metadata in conditioning:
        local = dict(metadata)
        refs = list(local.get("minimax_refs", ()))
        refs.append(ref)
        local["minimax_refs"] = refs
        if supports_context:
            keyframes = list(local.get("minimax_keyframes", ()))
            # H3 Extend's context contract reserves these rows together with
            # the layout.  A stock first/last keyframe here would be appended
            # by some compatibility patches after layout construction.
            keyframes.append({"kind": "context", "num_frames": boundary.shape[2], "latent": boundary})
            local["minimax_keyframes"] = keyframes
        output.append([embedding, local])
    return output


def serial_chunk_plan(tokens, chunk_frames, overlap_frames):
    """Plan HR Endless-style synthetic-prefix continuation chunks.

    The first chunk owns a normal H3 window.  Later chunks contain a discarded
    five-frame packing prefix followed by fresh source positions.  The
    previous tail is warm-started into the beginning of those retained source
    positions, so it is denoised and emitted once rather than masked overlap.
    """
    chunk, overlap = window_sizes(chunk_frames, overlap_frames)
    prefix = video_latent_t(5)
    if tokens < 2 or (tokens - 2) % 5:
        raise ValueError("Temporal Pass 2 requires an H3 latent with 5k + 2 temporal positions.")
    if tokens <= chunk:
        return [(0, tokens, 0, 0, 0, 0)]
    capacity = chunk - prefix
    plan = []
    source_start = 0
    first = min(chunk, tokens)
    plan.append((0, first, 0, 0, 0, 0))
    source_start = first
    while source_start < tokens:
        new_tokens = min(capacity, tokens - source_start)
        source_stop = source_start + new_tokens
        prefix_audio = round(frame_boundary(prefix) * FRAME_RESCALE)
        source_audio_start = round(frame_boundary(source_start) * FRAME_RESCALE)
        source_audio_stop = round(frame_boundary(source_stop) * FRAME_RESCALE)
        audio_overlap = min(
            round(frame_boundary(overlap) * FRAME_RESCALE),
            source_audio_stop - source_audio_start,
        )
        warm_tokens = min(overlap, new_tokens)
        plan.append((source_start, source_stop, prefix, prefix_audio, warm_tokens, audio_overlap))
        source_start = source_stop
    return plan


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
    serial_plan = serial_chunk_plan(video.shape[2], chunk_frames, overlap_frames)
    _, overlap_tokens = window_sizes(chunk_frames, overlap_frames)
    if chunk_count is not None and len(serial_plan) != chunk_count:
        raise ValueError(f"Automatic temporal chunking produced {len(serial_plan)} windows instead of {chunk_count}.")
    logging.info("[YAFV H3] Pass 2: %d temporal window(s), chunk=%d frames, overlap=%d frames, up to %d latent positions each, %dx%d px; audio preserved.",
                 len(serial_plan), chunk_frames, overlap_frames, max(stop - start for start, stop, *_ in serial_plan),
                 video.shape[-1] * 16, video.shape[-2] * 16)
    video, audio = video.cpu(), audio.cpu()
    source = NestedTensor((video, audio))
    mm.throw_exception_if_processing_interrupted()
    noise = (comfy.sample.prepare_noise(source, seed, latent.get("batch_index")) if add_noise
             else comfy.sample.prepare_empty_noise(source))
    noise_video, noise_audio = noise.unbind()
    # Continue chunks serially.  The overlap is context for the next sample,
    # not a second prediction to average into the output.
    sampled_video_parts = []
    sampled_audio_parts = []
    previous_video = None
    previous_audio = None
    previous_stop = 0
    previous_audio_stop = 0
    steps_per_chunk = steps - start_step
    progress = comfy.utils.ProgressBar(steps_per_chunk * len(serial_plan))

    for index, (start, stop, prefix_tokens, prefix_audio_tokens, warm_tokens, audio_overlap) in enumerate(serial_plan):
        mm.throw_exception_if_processing_interrupted()
        f0, f1 = frame_boundary(start), frame_boundary(stop)
        a0 = min(audio.shape[-1], round(f0 * FRAME_RESCALE))
        a1 = min(audio.shape[-1], round(f1 * FRAME_RESCALE))
        if a1 <= a0:
            raise ValueError("The H3 audio latent is too short for this video window.")
        if chunk_callback is not None:
            chunk_callback(index + 1, len(serial_plan), f0, f1)
        source_video = video[:, :, start:stop].clone()
        source_audio = audio[..., a0:a1].clone()
        if prefix_tokens:
            prefix_video = torch.zeros(
                (video.shape[0], video.shape[1], prefix_tokens, video.shape[-2], video.shape[-1]),
                dtype=video.dtype,
            )
            prefix_audio = torch.zeros(
                (audio.shape[0], audio.shape[1], audio.shape[2], prefix_audio_tokens),
                dtype=audio.dtype,
            )
            window_video = torch.cat((prefix_video, source_video), dim=2)
            window_audio = torch.cat((prefix_audio, source_audio), dim=-1)
        else:
            window_video, window_audio = source_video, source_audio
        if previous_video is not None:
            if warm_tokens <= 0 or prefix_tokens + warm_tokens > window_video.shape[2]:
                raise ValueError("Temporal chunks have an invalid HR warm-start range.")
            window_video[:, :, prefix_tokens:prefix_tokens + warm_tokens] = previous_video[:, :, -warm_tokens:]
        window = NestedTensor((window_video, window_audio))
        if prefix_tokens:
            prefix_source = NestedTensor((prefix_video, prefix_audio))
            prefix_noise = (comfy.sample.prepare_noise(prefix_source, seed + index, latent.get("batch_index"))
                            if add_noise else comfy.sample.prepare_empty_noise(prefix_source))
            prefix_noise_video, prefix_noise_audio = prefix_noise.unbind()
            window_noise = NestedTensor((
                torch.cat((prefix_noise_video, noise_video[:, :, start:stop].clone()), dim=2),
                torch.cat((prefix_noise_audio, noise_audio[..., a0:a1].clone()), dim=-1),
            ))
        else:
            window_noise = NestedTensor((noise_video[:, :, start:stop].clone(), noise_audio[..., a0:a1].clone()))
        # HR Endless fully denoises the retained warm-start positions.  Keep
        # audio frozen for this node's existing Pass 2 contract.
        video_mask = torch.ones((1, 1, window_video.shape[2], 1, 1), dtype=torch.float32)
        window_mask = NestedTensor((
            video_mask,
            torch.zeros((1, 1, 2, window_audio.shape[-1]), dtype=torch.float32),
        ))
        local_f0, local_f1 = frame_boundary(start), frame_boundary(stop)
        cond = window_conditioning(positive, local_f0, local_f1, video.shape[-2], video.shape[-1])
        uncond = window_conditioning(negative, local_f0, local_f1, video.shape[-2], video.shape[-1])
        if previous_video is not None:
            cond = _append_continuation_context(
                cond, previous_video, previous_audio, overlap_tokens,
                round(frame_boundary(overlap_tokens) * FRAME_RESCALE),
                boundary_video=previous_video,
            )

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
        sampled_video_parts.append(refined[:, :, prefix_tokens:])
        # Pass 2 is a video refinement; its audio track remains the original
        # synchronized track, exactly as in the previous implementation.
        sampled_audio_parts.append(window_audio[..., prefix_audio_tokens:])
        previous_video = refined
        previous_audio = window_audio
        previous_stop = stop
        previous_audio_stop = a1
        del sampled, refined, window, window_noise, window_mask, cond, uncond
        mm.soft_empty_cache()
        mm.throw_exception_if_processing_interrupted()

    result = dict(latent)
    assembled_video = torch.cat(sampled_video_parts, dim=2).to(dtype=video.dtype)
    assembled_audio = torch.cat(sampled_audio_parts, dim=-1)
    if assembled_video.shape[2] != video.shape[2] or assembled_audio.shape[-1] != audio.shape[-1]:
        raise RuntimeError("Serial temporal sampling did not reconstruct the original latent duration.")
    result["samples"] = NestedTensor((assembled_video, assembled_audio))
    return result
