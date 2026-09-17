"""MiniMax H3 3D upscaler, ported from LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler.
Preserves checkpoint layout, normalization, sizing and temporal blending.
Model lifetime and cancellation are managed by ComfyUI.
"""
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

import folder_paths
import comfy.model_management as mm
import comfy.utils
from comfy.model_patcher import ModelPatcher

MODEL_FOLDER = "latent_upscale_models"
if MODEL_FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(MODEL_FOLDER, os.path.join(folder_paths.models_dir, MODEL_FOLDER))

def model_names():
    return [name for name in folder_paths.get_filename_list(MODEL_FOLDER)
            if os.path.splitext(name)[1].lower() in (".pth", ".safetensors")] or ["(no upscale models found)"]

def target_size(height, width, mode, scale, target_width, target_height, megapixels, align):
    if mode == "scale by multiplier":
        wp, hp = width * 16 * scale, height * 16 * scale
        effective_scale = scale
    elif mode == "target dimensions":
        wp, hp = target_width, target_height
        effective_scale = (wp / (width * 16) + hp / (height * 16)) / 2
    elif mode == "megapixels":
        hp = (megapixels * 1024 * 1024 / (width / height)) ** 0.5
        wp = hp * width / height
        effective_scale = (wp / (width * 16) + hp / (height * 16)) / 2
    else:
        raise ValueError(f"Unknown upscale mode: {mode}")
    alignment = max(1, align)
    w = max(1, round(round(wp / alignment) * alignment / 16))
    h = max(1, round(round(hp / alignment) * alignment / 16))
    if effective_scale < 1 and (w < width or h < height):
        raise ValueError("The H3 upscaler only supports upscaling.")
    return h, w, effective_scale

def resolve_device(device):
    if device == "cpu":
        return torch.device("cpu")
    if device == "rocm":
        if torch.version.hip is None or not torch.cuda.is_available():
            raise ValueError("ROCm requires a HIP-enabled PyTorch build and an available AMD GPU.")
        return torch.device("cuda")
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unknown upscale device: {device}")

LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264
]
LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523
]

def _make_norm_tensors(device, dtype):
    mean = torch.tensor(LATENTS_MEAN, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    return mean, std

def normalization(channels):
    return nn.GroupNorm(32, channels)

class ResBlockEmb3D(nn.Module):
    def __init__(self, channels, emb_channels, out_channels=None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            normalization(channels), nn.SiLU(),
            nn.Conv3d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels),
        )
        self.out_norm = normalization(self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(), nn.Identity(),
            nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1),
        )
        self.skip = (
            nn.Conv3d(channels, self.out_channels, 1)
            if self.out_channels != channels else nn.Identity()
        )

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return self.skip(x) + h

class TemporalConv(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.norm = normalization(channels)
        self.dwconv = nn.Conv3d(channels, channels,
                                kernel_size=(kernel_size, 1, 1),
                                padding=(padding, 0, 0),
                                groups=channels)
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x):
        identity = x
        h = self.norm(x)
        h = F.silu(h)
        h = self.dwconv(h)
        h = self.pwconv(h)
        return identity + h

class LatentResizer3D(nn.Module):
    def __init__(self, in_channels=24, in_blocks=12, out_blocks=12,
                 channels=512,
                 temporal_every=2, temporal_kernel=5):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        embed_dim = 64
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))

        self.in_blocks = nn.ModuleList()
        for b in range(in_blocks):
            self.in_blocks.append(ResBlockEmb3D(channels, embed_dim))
            if temporal_every > 0 and b % temporal_every == 0:
                self.in_blocks.append(TemporalConv(channels, temporal_kernel))

        self.out_blocks = nn.ModuleList()
        for b in range(out_blocks):
            self.out_blocks.append(ResBlockEmb3D(channels, embed_dim))
            if temporal_every > 0 and b % temporal_every == 0:
                self.out_blocks.append(TemporalConv(channels, temporal_kernel))

        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def forward(self, x, scale=None, target_size=None, enable_chunking=True):
        if target_size is not None:
            size = target_size
        elif scale is not None:
            size = tuple(int(round(s * scale)) for s in x.shape[-3:])
        else:
            return x

        if size == x.shape[-3:]:
            return x

        B, C, T, H, W = x.shape

        tk = 0
        for b in self.in_blocks:
            if isinstance(b, TemporalConv):
                tk = b.dwconv.weight.shape[2]
                break

        overlap = tk
        chunk = 32

        if not enable_chunking or T <= chunk:
            return self._forward_seg(x, scale, size)

        x_padded = F.pad(x, (0, 0, 0, 0, overlap, overlap), mode='replicate')

        out_full = torch.zeros(B, C, T, size[-2], size[-1], device=x.device, dtype=x.dtype)
        weight_full = torch.zeros(1, 1, T, 1, 1, device=x.device, dtype=x.dtype)

        start = 0
        while start < T:
            mm.throw_exception_if_processing_interrupted()
            seg_start = start
            seg_end = min(T, start + chunk)
            out_start = max(0, seg_start - overlap)
            out_end = min(T, seg_end + overlap)
            lo = max(0, out_start - overlap)
            hi = min(T + 2 * overlap, out_end + overlap)

            seg = x_padded[:, :, lo:hi].contiguous()
            seg_size = (hi - lo, size[-2], size[-1])
            seg_out = self._forward_seg(seg, scale, seg_size)

            s0 = (out_start + overlap) - lo
            s1 = s0 + (out_end - out_start)
            valid_out = seg_out[:, :, s0:s1]
            n_valid = out_end - out_start

            weight = torch.ones(n_valid, device=x.device, dtype=x.dtype)
            if seg_start > out_start:
                blend_len = seg_start - out_start
                weight[:blend_len] = torch.arange(1, blend_len + 1, device=x.device, dtype=x.dtype) / (blend_len + 1)
            if out_end > seg_end:
                blend_len = out_end - seg_end
                weight[-blend_len:] = torch.arange(blend_len, 0, -1, device=x.device, dtype=x.dtype) / (blend_len + 1)

            out_full[:, :, out_start:out_end] += valid_out * weight.view(1, 1, n_valid, 1, 1)
            weight_full[:, :, out_start:out_end] += weight.view(1, 1, n_valid, 1, 1)

            start += chunk
            del seg, seg_out, valid_out

        out_full = out_full / weight_full.clamp(min=1e-8)
        return out_full

    def _forward_seg(self, x, scale, size):
        scale_emb = torch.tensor(
            [scale - 1 if scale is not None else 0.0],
            dtype=x.dtype, device=x.device).unsqueeze(0)
        emb = self.embed(scale_emb)

        x = self.conv_in(x)
        for b in self.in_blocks:
            mm.throw_exception_if_processing_interrupted()
            if isinstance(b, ResBlockEmb3D):
                emb_t = emb.expand(x.shape[0], -1)
                x = b(x, emb_t)
            else:
                x = b(x)

        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)

        for b in self.out_blocks:
            mm.throw_exception_if_processing_interrupted()
            if isinstance(b, ResBlockEmb3D):
                emb_t = emb.expand(x.shape[0], -1)
                x = b(x, emb_t)
            else:
                x = b(x)

        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x

def _detect_arch(sd):
    cfg = {
        "in_channels": 24, "in_blocks": 12, "out_blocks": 12, "channels": 512,
        "temporal_every": 2, "temporal_kernel": 5,
    }
    conv_key = 'conv_in.weight'
    if conv_key in sd:
        cfg["in_channels"] = sd[conv_key].shape[1]
        cfg["channels"] = sd[conv_key].shape[0]

    in_ids, out_ids = set(), set()
    temporal_in_indices, temporal_out_indices = set(), set()
    for k in sd.keys():
        m = re.match(r'in_blocks\.(\d+)\.in_layers\.', k)
        if m: in_ids.add(int(m.group(1)))
        m = re.match(r'out_blocks\.(\d+)\.in_layers\.', k)
        if m: out_ids.add(int(m.group(1)))
        m = re.match(r'in_blocks\.(\d+)\.dwconv\.weight', k)
        if m: temporal_in_indices.add(int(m.group(1)))
        m = re.match(r'out_blocks\.(\d+)\.dwconv\.weight', k)
        if m: temporal_out_indices.add(int(m.group(1)))

    if in_ids: cfg["in_blocks"] = len(in_ids)
    if out_ids: cfg["out_blocks"] = len(out_ids)

    if temporal_in_indices or temporal_out_indices:
        cfg["temporal_every"] = 2
        for k in sd.keys():
            if 'dwconv.weight' in k and k.endswith('dwconv.weight'):
                cfg["temporal_kernel"] = sd[k].shape[2]
                break
    else:
        cfg["temporal_every"] = 0

    return cfg


class H3Upscaler:
    """One checkpoint per node instance; ComfyUI owns device residency."""

    def __init__(self):
        self.patcher = None
        self.key = None

    def load(self, name, device, precision):
        path = folder_paths.get_full_path_or_raise(MODEL_FOLDER, name)
        stat = os.stat(path)
        key = (path, stat.st_mtime_ns, stat.st_size, str(device), precision)
        if self.key == key:
            return self.patcher
        if self.patcher is not None:
            mm.unload_model_and_clones(self.patcher, unload_additional_models=False)
            self.patcher = None
            self.key = None
        sd = comfy.utils.load_torch_file(path, safe_load=True)
        if "model" in sd:
            sd = sd["model"]
        if any(k.startswith("upscaler.") for k in sd):
            sd = {k[len("upscaler."):]: v for k, v in sd.items() if k.startswith("upscaler.")}
        sd = {k: v.to(torch.float16) if v.dtype == torch.float8_e4m3fn else v for k, v in sd.items()}
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
        with torch.device("meta"):
            model = LatentResizer3D(**_detect_arch(sd))
        model.load_state_dict(sd, strict=True, assign=True)
        del sd
        model.to(dtype=dtype)
        self.patcher = ModelPatcher(model, load_device=device, offload_device=torch.device("cpu"))
        self.key = key
        return self.patcher

    def upscale(self, latent, model_name, mode, scale, width, height, megapixels,
                align, enable_temporal_chunking, force_unload, device, precision):
        src = latent["samples"]
        h, w, effective_scale = target_size(
            src.shape[-2], src.shape[-1], mode, scale, width, height, megapixels, align)
        if (h, w) == tuple(src.shape[-2:]):
            return latent
        mm.throw_exception_if_processing_interrupted()
        dev = resolve_device(device)
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
        patcher = self.load(model_name, dev, precision)
        completed = False
        try:
            mm.load_models_gpu([patcher], force_full_load=True)
            model = patcher.model
            s = src.to(device=dev, dtype=dtype, copy=True)
            was_4d = s.ndim == 4
            if was_4d:
                s = s.unsqueeze(2)
            mean, std = _make_norm_tensors(dev, dtype)
            s = (s - mean) / std
            out = model(s, scale=effective_scale, target_size=(s.shape[2], h, w),
                        enable_chunking=enable_temporal_chunking)
            del s
            out = out * std + mean
            if was_4d:
                out = out.squeeze(2)
            result = dict(latent)
            result["samples"] = out.to(device="cpu", dtype=src.dtype)
            mask = result.get("noise_mask")
            if mask is not None:
                mask = comfy.utils.reshape_mask(mask, src.shape)
                result["noise_mask"] = F.interpolate(
                    mask.float(), size=result["samples"].shape[2:], mode="nearest").to(mask.dtype)
            mm.throw_exception_if_processing_interrupted()
            completed = True
            return result
        finally:
            if force_unload or not completed:
                mm.unload_model_and_clones(patcher, unload_additional_models=False)
            mm.soft_empty_cache()
