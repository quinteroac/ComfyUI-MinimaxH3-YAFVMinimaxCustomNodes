"""Project continuity using native ComfyUI H3 keyframes and AV noise masks."""
import torch
from comfy.nested_tensor import NestedTensor


def apply_context(conditioning, latent, context_latent, context_length=22,
                  video_transition_steps=4, audio_transition_steps=4,
                  video_anchor_only=True):
    count = int(context_length)
    if count not in (5, 22, 39, 56):
        raise ValueError("context_length must be 5, 22, 39 or 56")
    steps = 2 + 5 * ((count - 5) // 17)
    samples = context_latent["samples"]
    if not getattr(samples, "is_nested", False):
        raise ValueError("Connect both video and audio VAEs to Project Context")
    source_video, source_audio = samples.unbind()
    video, audio = latent["samples"].unbind()
    if source_video.shape[2] < steps or (source_video.shape[2] - steps) % 5:
        raise ValueError("Context does not contain enough phase-aligned H3 tokens")
    if (video.shape[2] <= steps or source_video.shape[:2] != video.shape[:2]
            or source_video.shape[-2:] != video.shape[-2:]):
        raise ValueError("Context and target must share resolution, batch and latent channels, and leave room for new frames")
    tail = source_video[:, :, -steps:].to(video)
    audio_steps = min(round(count * 5 / 3), source_audio.shape[-1])
    if audio_steps < 1 or audio_steps >= audio.shape[-1]:
        raise ValueError("Invalid audio context duration")
    audio_tail = source_audio[..., -audio_steps:].to(audio)
    keyframes = []
    position = 0
    for index in range(steps):
        keyframes.append({"resolved_frame_index": position,
                          "latent": tail[:, :, index:index + 1].clone()})
        position += (1, 4, 4, 4, 4)[index % 5]
    keyframes.append({"resolved_frame_index": round(count * 5 / 3) / (5 / 3) - audio_steps / (5 / 3),
                      "audio_latent": audio_tail.clone()})
    positive = []
    for embedding, metadata in conditioning:
        values = dict(metadata)
        values["minimax_keyframes"] = [dict(k) for k in metadata.get("minimax_keyframes", [])
                                       if k.get("resolved_frame_index", 0) >= count] + keyframes
        positive.append([embedding, values])
    video = video.clone()
    audio = audio.clone()
    vm = torch.ones_like(video[:, :1], dtype=torch.float32)
    am = torch.ones_like(audio[:, :1], dtype=torch.float32)
    if video_anchor_only:
        anchor = context_latent.get("anchor_samples", tail[:, :, -2:]).to(video)
        video[:, :, steps - 2:steps] = anchor
        vm[:, :, steps - 2:steps] = 0
    else:
        video[:, :, :steps] = tail
        audio[..., :audio_steps] = audio_tail
        for mask, length, transition in ((vm.movedim(2, -1), steps, video_transition_steps),
                                         (am, audio_steps, audio_transition_steps)):
            ramp = max(0, min(int(transition), length))
            mask[..., :length - ramp] = 0
            if ramp:
                mask[..., length - ramp:length] = torch.arange(1, ramp + 1, device=mask.device) / (ramp + 1)
    old = latent.get("noise_mask")
    if old is not None:
        masks = old.unbind() if getattr(old, "is_nested", False) else (old, None)
        vm = vm if masks[0] is None else vm * masks[0].to(vm)
        am = am if masks[1] is None else am * masks[1].to(am)
    result = dict(latent, samples=NestedTensor((video, audio)), noise_mask=NestedTensor((vm, am)))
    return positive, count, result


def trim_media(images, audio, trim_frames, output_frames, fps=24.0):
    start, count = int(trim_frames), int(output_frames)
    if start < 0 or count < 1 or fps <= 0 or start + count > len(images):
        raise ValueError("Invalid project trim; connect generation_length to H3 length")
    result_audio = None
    if audio is not None:
        rate = int(audio["sample_rate"])
        offset, length = round(start * rate / fps), round(count * rate / fps)
        waveform = audio["waveform"][..., offset:offset + length]
        waveform = torch.nn.functional.pad(waveform, (0, max(0, length - waveform.shape[-1])))
        result_audio = dict(audio, waveform=waveform)
    return images[start:start + count], result_audio
