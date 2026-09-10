import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
import yaml
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import Qwen2_5OmniProcessor

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

import comfy.model_management as mm
import comfy.model_patcher
import comfy.ops
import comfy.utils
from comfy.quant_ops import QUANT_ALGOS

from .auk_model import AuKModel, flash_attn_available, sage_attn_available, set_attention
from .auk_vae import BigVGANFlowVAE, fold_weight_norm
from .encoder import AuKEncoder, make_encoder


FLASH_TIMES = [0.0, 0.07612049579620361, 0.2928932309150696, 0.6173166036605835, 1.0]
FRAMES_PER_SECOND = 50
ENCODER_ASSETS = Path(__file__).resolve().parents[1] / "assets" / "qwen2.5-omni-3b"


def make_patcher(model, load_device, offload_device):
    mm.archive_model_dtypes(model)
    return comfy.model_patcher.CoreModelPatcher(model, load_device=load_device, offload_device=offload_device)


def metadata(path):
    with safe_open(str(path), framework="pt", device="cpu") as file:
        return file.metadata() or {}


def compute_dtype(choice, device):
    if choice == "auto":
        return torch.bfloat16 if mm.supports_dtype(device, torch.bfloat16) and device.type != "cpu" else torch.float32
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[choice]
    if not mm.supports_dtype(device, dtype):
        raise ValueError(f"{device} does not support {choice}; select auto or fp32.")
    return dtype


def operations(state, dtype):
    for key, value in state.items():
        if key.endswith(".comfy_quant"):
            kind = json.loads(value.numpy().tobytes())["format"]
            if kind not in QUANT_ALGOS:
                raise ValueError(f"This ComfyUI does not support {kind}. Update ComfyUI/Comfy Kitchen or use the BF16 checkpoint.")
    return comfy.ops.mixed_precision_ops(compute_dtype=dtype)


def model_variant(path, info):
    """Converted metadata is authoritative; canonical upstream names are unambiguous.

    Original Base and Flash weights have no metadata and often share a model
    directory. Its generic config.yaml can belong to either model.
    """
    variant = info.get("auk_variant")
    if variant is None:
        variant = {"auk_base.safetensors": "base", "auk_flash.safetensors": "flash"}.get(Path(path).name.lower())
    if variant is None:
        raise ValueError("Cannot identify this AuK checkpoint. Keep the original auk_base.safetensors/auk_flash.safetensors filename or use a converted checkpoint with variant metadata. Model configs are bundled in the node pack.")
    if variant not in ("base", "flash"):
        raise ValueError(f"Unsupported AuK variant: {variant}")
    return variant


def model_architecture(variant):
    config_path = Path(__file__).resolve().parents[1] / "assets" / f"auk_{variant}" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["model"]
    expected = {"base": "AuK", "flash": "AuK-Flash"}[variant]
    if config["name"] != expected or config["backbone"] != "Flux2Edit":
        raise ValueError(f"Incorrect bundled AuK config: {config_path}")
    # Training/backend/path options are handled by ComfyUI, not the backbone.
    keys = ("dim", "heads", "ff_mult", "text_hidden_dim", "num_layers", "num_single_layers")
    return {**{key: config["arch"][key] for key in keys}, "latent_dim": config["vae"]["latent_dim"]}


def load_model(path, precision, attention="auto"):
    variant = model_variant(path, metadata(path))
    state = load_file(str(path))
    device = mm.get_torch_device()
    dtype = compute_dtype(precision, device)
    model = AuKModel(operations(state, dtype), variant, architecture=model_architecture(variant))
    model.load_state_dict(state, strict=True)
    model.eval()
    set_attention(model, resolve_attention(attention))
    patcher = make_patcher(model, device, mm.unet_offload_device())
    patcher.auk_dtype = dtype
    print(f"AuK: loaded {Path(path).name} as {variant}; compute={dtype}, attention={attention}. "
          + ("Fixed 4-step sampling, guidance disabled." if variant == "flash" else "Base sampling uses the requested steps and guidance."))
    return patcher


def resolve_attention(choice):
    if choice not in ("auto", "eager", "sdpa", "flash_attention", "sageattention"):
        raise ValueError(f"Unknown AuK attention mode: {choice}")
    return choice


def load_encoder(path, precision):
    info = metadata(path)
    if info.get("auk_component") != "encoder":
        raise ValueError("Select a converted AuK Qwen encoder. Run tools/convert.py on the original Qwen directory first.")
    device = mm.text_encoder_device()
    dtype = compute_dtype(precision, device)
    state = load_file(str(path))
    model = make_encoder(str(ENCODER_ASSETS), operations(state, dtype), dtype)
    missing = model.load_state_dict(state, strict=False).missing_keys
    if set(missing) - {"lm_head.weight"}:
        raise ValueError(f"Checkpoint does not match the AuK encoder architecture (missing {sorted(missing)[:4]}); re-run tools/convert.py with the current version.")
    model.has_lm_head = "lm_head.weight" not in missing
    managed = torch.nn.Module()
    managed.add_module("encoder", model)
    patcher = make_patcher(managed, device, mm.text_encoder_offload_device())
    processor = Qwen2_5OmniProcessor.from_pretrained(str(ENCODER_ASSETS), local_files_only=True, use_fast=False)
    return AuKEncoder(patcher, processor, dtype)


class AuKVAE:
    audio_sample_rate = 24000
    audio_sample_rate_output = 24000
    latent_channels = 64

    def __init__(self, path):
        state = fold_weight_norm(load_file(str(path)))
        model = BigVGANFlowVAE()
        model.load_state_dict(state, strict=True)
        model.eval()
        self.patcher = make_patcher(model, mm.vae_device(), mm.vae_offload_device())

    def encode_reference(self, waveform, seed):
        mm.load_models_gpu([self.patcher])
        device = self.patcher.load_device
        waveform = waveform.to(device=device, dtype=torch.float32)
        # The strided encoder needs at least one codec frame.
        if waveform.shape[-1] < 480:
            waveform = F.pad(waveform, (0, 480 - waveform.shape[-1]))
        generator = torch.Generator(device=device).manual_seed(seed)
        return self.patcher.model.encode(waveform, generator).cpu()

    def encode(self, waveform):
        return self.encode_reference(waveform.movedim(-1, 1).mean(1, keepdim=True), 0).transpose(1, 2)

    def decode(self, samples):
        return self.decode_target(samples.transpose(1, 2)).movedim(1, -1)

    def decode_target(self, latent):
        mm.load_models_gpu([self.patcher])
        device = self.patcher.load_device
        return self.patcher.model.decode(latent.to(device=device, dtype=torch.float32)).cpu()


def encode_instruction(encoder, model, instruction, audio=None):
    weights = model.model.layer_weights.detach().cpu()
    scale = model.model.layer_scale.detach().cpu()
    count = 1 if audio is None else audio["waveform"].shape[0]
    result = []
    for index in range(count):
        mm.throw_exception_if_processing_interrupted()
        source = None if audio is None else {"waveform": audio["waveform"][index:index + 1].clone(), "sample_rate": audio["sample_rate"]}
        hidden, mask = encoder.encode(instruction, source, weights, scale)
        result.append([hidden, {"auk_attention_mask": mask, "auk_audio": source}])
    return result


def time_grid(is_flash, steps, sway):
    if is_flash:
        return FLASH_TIMES
    return [t + sway * (math.cos(math.pi * t / 2) - 1 + t) for t in (i / steps for i in range(steps + 1))]


def generate(model, vae, conditioning, seconds, seed, steps, guidance, sway):
    if not isinstance(vae, AuKVAE):
        raise ValueError("AuK Generate / Edit requires the VAE from AuK VAE Loader.")
    outputs = []
    for index, (hidden, options) in enumerate(conditioning):
        if "auk_attention_mask" not in options:
            raise ValueError("Use conditioning from AuK Instruction Encode.")
        source = options["auk_audio"]
        item_seed = (seed + index) % (2**64)
        if source is None:
            if seconds <= 0:
                raise ValueError("Set a positive generation duration for text-only speech.")
            ref = torch.empty(1, 0, 64)
        else:
            waveform = source["waveform"].mean(1, keepdim=True)
            waveform = torchaudio.functional.resample(waveform, source["sample_rate"], 24000)
            ref = vae.encode_reference(waveform, item_seed)
        target_seconds = seconds if seconds > 0 else source["waveform"].shape[-1] / source["sample_rate"]
        length = max(1, math.ceil(target_seconds * FRAMES_PER_SECOND))
        mm.throw_exception_if_processing_interrupted()
        mm.load_models_gpu([model])
        device, dtype = model.load_device, model.auk_dtype
        ref = ref.to(device=device, dtype=dtype)
        hidden = hidden.to(device=device, dtype=dtype)
        mask = options["auk_attention_mask"].to(device)
        backbone = model.model.transformer
        guided = not model.model.is_flash and guidance >= 1e-5
        context, prompt = backbone.prepare(hidden, ref, guided)
        times = time_grid(model.model.is_flash, steps, sway)
        rng = torch.Generator(device=device).manual_seed(item_seed)
        x = torch.randn((1, length, 64), device=device, dtype=torch.float32, generator=rng)
        progress = comfy.utils.ProgressBar(len(times) - 1)
        pairs = list(zip(times[:-1], times[1:]))
        if tqdm is not None:
            label = f"AuK {'flash' if model.model.is_flash else 'base'}"
            if len(conditioning) > 1:
                label += f" {index + 1}/{len(conditioning)}"
            pairs = tqdm(pairs, desc=label, unit="step", leave=True)
        for start, end in pairs:
            mm.throw_exception_if_processing_interrupted()
            velocity = backbone(x.to(dtype), torch.tensor(start, device=device, dtype=torch.float32), context, prompt, mask, guided)
            if guided:
                conditional, unconditional = velocity.chunk(2)
                velocity = conditional + guidance * (conditional - unconditional)
            x = x + (end - start) * velocity.float()
            progress.update(1)
        del context, prompt, ref, hidden, velocity
        audio = vae.decode_target(x.cpu())
        if not torch.isfinite(audio).all():
            raise RuntimeError("AuK generated non-finite audio; try the BF16 model with fp32 compute.")
        outputs.append(audio[..., :round(target_seconds * 24000)])
    return {"waveform": torch.cat(outputs), "sample_rate": 24000}
