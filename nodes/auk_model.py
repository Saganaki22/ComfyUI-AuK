"""AuK Flux2Edit inference backbone. State keys follow Tencent's release."""
import importlib.metadata
import importlib.util
import math

import torch
from torch import nn
import torch.nn.functional as F

import comfy.ops
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import optimized_attention


_ATTENTION_WARNED = set()


def flash_attn_available():
    # find_spec alone lies: other node packs inject stub flash_attn modules.
    try:
        if importlib.util.find_spec("flash_attn") is None:
            return False
        importlib.metadata.version("flash_attn")
    except (importlib.metadata.PackageNotFoundError, ImportError, ValueError):
        return False
    return True


def sage_attn_available():
    try:
        return importlib.util.find_spec("sageattention") is not None
    except (ImportError, ValueError):
        return False


def set_attention(model, mode):
    for module in model.modules():
        if isinstance(module, Attention):
            module.attention_mode = mode


def attend(q, k, v, heads, mask, mode):
    # All branches return (B, S, H*D), matching optimized_attention's skip_reshape contract.
    if mode in ("flash_attention", "sageattention"):
        is_flash = mode == "flash_attention"
        installed = flash_attn_available() if is_flash else sage_attn_available()
        usable = installed and mask is None and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)
        failure = None
        if usable:
            try:
                if is_flash:
                    from flash_attn import flash_attn_func
                    return flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=False).reshape(q.shape[0], q.shape[2], -1)
                from sageattention import sageattn
                return sageattn(q, k, v, tensor_layout="HND", is_causal=False).transpose(1, 2).flatten(2)
            except torch.OutOfMemoryError:
                raise
            except (ImportError, OSError, NotImplementedError, RuntimeError) as error:
                if any(message in str(error).lower() for message in ("device-side assert", "illegal memory", "misaligned address", "unspecified launch failure")):
                    raise
                failure = str(error)
        reason = "kernel" if failure is not None else "missing" if not installed else "mask" if mask is not None else "device/dtype"
        if (mode, reason) not in _ATTENTION_WARNED:
            _ATTENTION_WARNED.add((mode, reason))
            if failure is not None:
                print(f"AuK: {mode} kernel unavailable ({failure}); using sdpa instead.")
            elif not installed:
                print(f"AuK: {mode} is not installed; using sdpa instead.")
            elif mask is not None:
                print(f"AuK: {mode} cannot apply edit masks; masked steps use sdpa.")
            else:
                print(f"AuK: {mode} needs CUDA with fp16/bf16 activations; using sdpa here.")
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).flatten(2)
    if mode == "eager":
        scores = (q @ k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else scores + mask
        out = torch.softmax(scores, -1, dtype=torch.float32).nan_to_num().to(q.dtype) @ v
        return out.transpose(1, 2).flatten(2)
    if mode == "sdpa":
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).flatten(2)
    return optimized_attention(q, k, v, heads, mask=mask, skip_reshape=True)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("inv_freq", torch.empty(dim // 2))

    def forward(self, length, device):
        angle = torch.arange(length, device=device, dtype=torch.float32)[:, None] * self.inv_freq.to(device=device, dtype=torch.float32)
        return torch.stack((angle.cos(), -angle.sin(), angle.sin(), angle.cos()), -1).reshape(1, 1, length, -1, 2, 2)


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, ops):
        super().__init__()
        self.time_mlp = nn.Sequential(ops.Linear(256, dim), nn.SiLU(), ops.Linear(dim, dim))

    def forward(self, time, dtype):
        freq = torch.exp(torch.arange(128, device=time.device, dtype=torch.float32) * (-math.log(10000) / 127))
        angle = 1000 * time[:, None] * freq[None]
        return self.time_mlp(torch.cat((angle.sin(), angle.cos()), -1).to(dtype))


class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim, ops):
        super().__init__()
        self.conv1d = nn.Sequential(ops.Conv1d(dim, dim, 31, padding=15, groups=16), nn.Mish(), ops.Conv1d(dim, dim, 31, padding=15, groups=16), nn.Mish())

    def forward(self, x):
        return self.conv1d(x.transpose(1, 2)).transpose(1, 2)


class AudioPromptEmbedding(nn.Module):
    def __init__(self, latent_dim, dim, ops):
        super().__init__()
        self.linear = ops.Linear(latent_dim, dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim, ops)

    def forward(self, x):
        x = self.linear(x)
        return x + self.conv_pos_embed(x)


class AdaLayerNorm(nn.Module):
    def __init__(self, dim, ops, final=False):
        super().__init__()
        self.linear = ops.Linear(dim, dim * (2 if final else 6))
        self.norm = ops.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final = final

    def forward(self, x, t):
        values = self.linear(F.silu(t))[:, None]
        if self.final:
            scale, shift = values.chunk(2, -1)
            return self.norm(x) * (1 + scale) + shift
        shift, scale, gate, ff_shift, ff_scale, ff_gate = values.chunk(6, -1)
        return self.norm(x) * (1 + scale) + shift, gate, ff_shift, ff_scale, ff_gate


class FeedForward(nn.Module):
    def __init__(self, dim, mult, ops):
        super().__init__()
        self.linear_in = ops.Linear(dim, int(dim * mult) * 2, bias=False)
        self.linear_out = ops.Linear(int(dim * mult), dim, bias=False)

    def forward(self, x):
        gate, value = self.linear_in(x).chunk(2, -1)
        return self.linear_out(F.silu(gate) * value)


class Attention(nn.Module):
    def __init__(self, dim, heads, ops, joint=False):
        super().__init__()
        self.heads = heads
        self.attention_mode = "auto"
        self.to_qkv = ops.Linear(dim, dim * 3)
        self.q_norm = ops.RMSNorm(dim // heads, eps=None)
        self.k_norm = ops.RMSNorm(dim // heads, eps=None)
        self.to_out = nn.ModuleList([ops.Linear(dim, dim), nn.Identity()])
        if joint:
            self.to_qkv_c = ops.Linear(dim, dim * 3)
            self.c_q_norm = ops.RMSNorm(dim // heads, eps=None)
            self.c_k_norm = ops.RMSNorm(dim // heads, eps=None)
            self.to_out_c = ops.Linear(dim, dim)

    def qkv(self, x, projection, q_norm, k_norm, rope):
        q, k, v = (part.reshape(x.shape[0], x.shape[1], self.heads, -1).transpose(1, 2) for part in projection(x).chunk(3, -1))
        # Separate norm/rope preserves PyTorch RMSNorm's dtype-dependent epsilon
        # and intermediate rounding; the shared paired RoPE handles the rotation.
        q, k = apply_rope(q_norm(q), k_norm(k), rope)
        return q, k, v

    def forward(self, x, rope, c=None, c_rope=None, mask=None):
        q, k, v = self.qkv(x, self.to_qkv, self.q_norm, self.k_norm, rope)
        if c is not None:
            cq, ck, cv = self.qkv(c, self.to_qkv_c, self.c_q_norm, self.c_k_norm, c_rope)
            q, k, v = torch.cat((q, cq), 2), torch.cat((k, ck), 2), torch.cat((v, cv), 2)
        out = attend(q, k, v, self.heads, mask, self.attention_mode)
        if c is None:
            return self.to_out[0](out)
        return self.to_out[0](out[:, :x.shape[1]]), self.to_out_c(out[:, x.shape[1]:])


class DoubleBlock(nn.Module):
    def __init__(self, dim, heads, mult, ops):
        super().__init__()
        self.attn_norm_x = AdaLayerNorm(dim, ops)
        self.attn_norm_c = AdaLayerNorm(dim, ops)
        self.attn = Attention(dim, heads, ops, joint=True)
        self.ff_norm_x = ops.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.ff_norm_c = ops.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.ff_x = FeedForward(dim, mult, ops)
        self.ff_c = FeedForward(dim, mult, ops)

    def forward(self, x, c, t, rope, c_rope, mask, c_mask):
        nx, gx, sx, ax, fx = self.attn_norm_x(x, t)
        nc, gc, sc, ac, fc = self.attn_norm_c(c, t)
        dx, dc = self.attn(nx, rope, nc, c_rope, mask)
        dc = dc.masked_fill(~c_mask[:, :, None], 0)
        x = x + gx * dx
        c = c + gc * dc
        return x + fx * self.ff_x(self.ff_norm_x(x) * (1 + ax) + sx), c + fc * self.ff_c(self.ff_norm_c(c) * (1 + ac) + sc)


class SingleBlock(nn.Module):
    def __init__(self, dim, heads, mult, ops):
        super().__init__()
        self.attn_norm = AdaLayerNorm(dim, ops)
        self.attn = Attention(dim, heads, ops)
        self.ff_norm = ops.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.ff = FeedForward(dim, mult, ops)

    def forward(self, x, t, rope, mask):
        norm, gate, shift, scale, ff_gate = self.attn_norm(x, t)
        out = self.attn(norm, rope, mask=mask)
        if mask is not None:
            out = out.masked_fill(~mask[:, 0, 0, :, None], 0)
        x = x + gate * out
        return x + ff_gate * self.ff(self.ff_norm(x) * (1 + scale) + shift)


class Flux2Edit(nn.Module):
    def __init__(self, ops, dim=1536, heads=24, ff_mult=2, text_hidden_dim=2048, num_layers=10, num_single_layers=20, latent_dim=64):
        super().__init__()
        self.time_embed = TimestepEmbedding(dim, ops)
        self.txt_proj = ops.Linear(text_hidden_dim, dim)
        self.txt_norm = ops.RMSNorm(dim, eps=None)
        self.audio_embed = AudioPromptEmbedding(latent_dim, dim, ops)
        self.rotary_embed = RotaryEmbedding(dim // heads)
        self.transformer_blocks = nn.ModuleList([DoubleBlock(dim, heads, ff_mult, ops) for _ in range(num_layers)])
        self.single_transformer_blocks = nn.ModuleList([SingleBlock(dim, heads, ff_mult, ops) for _ in range(num_single_layers)])
        self.norm_out = AdaLayerNorm(dim, ops, final=True)
        self.proj_out = ops.Linear(dim, latent_dim)

    def prepare(self, text, ref, guided):
        c = self.txt_norm(self.txt_proj(text))
        prompt = self.audio_embed(ref) if ref.shape[1] else None
        if guided:
            c = torch.cat((c, torch.zeros_like(c)))
            if prompt is not None:
                prompt = torch.cat((prompt, self.audio_embed(torch.zeros_like(ref))))
        return c, prompt

    def forward(self, x, time, context, prompt, c_mask, guided=False):
        if guided:
            x = torch.cat((x, x))
            c_mask = torch.cat((c_mask, c_mask))
        t = self.time_embed(time.expand(x.shape[0]), x.dtype)
        x = self.audio_embed(x)
        prompt_len = 0 if prompt is None else prompt.shape[1]
        if prompt is not None:
            x = torch.cat((prompt, x), 1)
        c = context
        rope = self.rotary_embed(x.shape[1], x.device)
        c_rope = self.rotary_embed(c.shape[1], x.device)
        # Upstream enables the joint padding mask only when reference audio exists.
        mask = None
        if prompt is not None:
            audio_mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
            mask = torch.cat((audio_mask, c_mask), 1)[:, None, None]
        for block in self.transformer_blocks:
            x, c = block(x, c, t, rope, c_rope, mask, c_mask)
        text_len = c.shape[1]
        x = torch.cat((c, x), 1)
        rope = self.rotary_embed(x.shape[1], x.device)
        if prompt is not None:
            mask = torch.cat((c_mask, audio_mask), 1)[:, None, None]
        for block in self.single_transformer_blocks:
            x = block(x, t, rope, mask)
        return self.proj_out(self.norm_out(x[:, text_len + prompt_len:], t))


class AuKModel(nn.Module):
    def __init__(self, ops, variant, architecture=None, encoder_layers=36):
        super().__init__()
        self.transformer = Flux2Edit(ops, **(architecture or {}))
        self.layer_weights = nn.Parameter(torch.empty(encoder_layers))
        self.layer_scale = nn.Parameter(torch.empty(1))
        self.is_flash = variant == "flash"
