"""Optional Whisper transcription for crafting edit instructions and checking output. Local models only."""
from pathlib import Path
from functools import lru_cache

import torch
from huggingface_hub import snapshot_download

import comfy.model_management as mm
import comfy.ops
import comfy.utils
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from .runtime import make_patcher

LANGUAGES = ["auto", "english", "chinese", "german", "spanish", "russian", "korean", "french", "japanese", "portuguese", "turkish", "polish", "catalan"]
WINDOW = 30
DOWNLOAD_SIZES = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]
DOWNLOAD_REPO = "openai/whisper-{size}"


class ManagedWhisper(WhisperForConditionalGeneration):
    @property
    def device(self):
        # HF generation creates tokens on self.device. Dynamic offloading keeps
        # parameter storage on CPU, so storage location is not execution location.
        return getattr(self, "_auk_execution_device", super().device)


def model_root():
    import folder_paths
    return Path(folder_paths.models_dir)


def model_choices():
    root = model_root()
    choices = set(DOWNLOAD_SIZES)
    for parent in (root / "whisper", root / "audio_encoders"):
        if parent.is_dir():
            choices.update(child.name for child in parent.iterdir() if child.is_dir() and (child / "config.json").is_file())
    return sorted(choices)


def resolve(model_name):
    if not model_name or "/" in model_name or "\\" in model_name or ".." in model_name:
        raise ValueError("Select a local Whisper model folder.")
    root = model_root()
    for parent in (root / "whisper", root / "audio_encoders"):
        resolved = (parent / model_name).resolve()
        if resolved.is_dir() and (resolved / "config.json").is_file() and resolved.is_relative_to(root):
            return resolved
    raise FileNotFoundError(f"Whisper model '{model_name}' not found under {root}/whisper or {root}/audio_encoders. Enable download if missing to fetch it, or place a checkpoint folder there yourself.")


def ensure_model(size):
    if size not in DOWNLOAD_SIZES:
        raise ValueError(f"Unknown Whisper download size '{size}'.")
    target = model_root() / "whisper" / size
    if (target / "config.json").is_file():
        return target
    repo = DOWNLOAD_REPO.format(size=size)
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo, local_dir=target, allow_patterns=["*.json", "*.txt", "model.safetensors"])
    if not any(target.glob("*.safetensors")):
        snapshot_download(repo, local_dir=target, allow_patterns=["pytorch_model.bin"])
    if not (target / "config.json").is_file() or not (any(target.glob("*.safetensors")) or (target / "pytorch_model.bin").is_file()):
        raise RuntimeError(f"Download of {repo} did not produce a usable checkpoint in {target}.")
    return target


def load(model_name):
    resolved = resolve(model_name)
    device = mm.text_encoder_device()
    dtype = torch.bfloat16 if mm.supports_dtype(device, torch.bfloat16) and device.type != "cpu" else torch.float32
    return _load_cached(str(resolved), dtype, str(device))


@lru_cache(maxsize=1)
def _load_cached(resolved, dtype, device_name):
    device = torch.device(device_name)
    processor = WhisperProcessor.from_pretrained(resolved, local_files_only=True)
    model = ManagedWhisper.from_pretrained(resolved, local_files_only=True, dtype=dtype, attn_implementation="sdpa")
    model._auk_execution_device = device
    # Keep positional embeddings native: HF reads their .weight directly.
    # The patcher loads those small tensors eagerly and manages castable layers dynamically.
    for name, module in list(model.named_modules()):
        replacement = None
        if isinstance(module, torch.nn.Linear):
            replacement = comfy.ops.manual_cast.Linear(module.in_features, module.out_features, bias=module.bias is not None, dtype=dtype)
        elif isinstance(module, torch.nn.Conv1d):
            replacement = comfy.ops.manual_cast.Conv1d(module.in_channels, module.out_channels, module.kernel_size, stride=module.stride, padding=module.padding, dtype=dtype)
        elif isinstance(module, torch.nn.LayerNorm):
            replacement = comfy.ops.manual_cast.LayerNorm(module.normalized_shape, eps=module.eps, dtype=dtype)
        if replacement is not None:
            replacement.load_state_dict(module.state_dict(), assign=True)
            model.set_submodule(name, replacement)
    model.eval()
    managed = torch.nn.Module()
    managed.add_module("asr", model)
    return make_patcher(managed, device, mm.text_encoder_offload_device()), processor, dtype


def transcribe(audio, model_name, language, task, download_if_missing=False, model_size="large-v3"):
    try:
        patcher, processor, dtype = load(model_name)
    except FileNotFoundError:
        if not download_if_missing:
            raise
        size = model_name if model_name in DOWNLOAD_SIZES else model_size
        ensure_model(size)
        patcher, processor, dtype = load(size)
    waveforms = audio["waveform"].detach().cpu().float().mean(1)
    rate = audio["sample_rate"]
    target_rate = processor.feature_extractor.sampling_rate
    if rate != target_rate:
        import torchaudio
        waveforms = torchaudio.functional.resample(waveforms, rate, target_rate)
    mm.load_models_gpu([patcher])
    pieces = []
    for waveform in waveforms:
        mm.throw_exception_if_processing_interrupted()
        if waveform.numel() == 0:
            raise ValueError("Whisper requires non-empty audio.")
        is_long = waveform.numel() > WINDOW * target_rate
        inputs = processor(waveform.numpy(), sampling_rate=target_rate, return_tensors="pt",
                           truncation=False, padding="longest" if is_long else "max_length",
                           return_attention_mask=True)
        inputs = {key: value.to(device=patcher.load_device, dtype=dtype if value.is_floating_point() else value.dtype) for key, value in inputs.items()}
        frames = inputs["input_features"].shape[-1]
        progress = comfy.utils.ProgressBar(frames)
        bar = None
        try:
            from tqdm import tqdm
            bar = tqdm(total=frames, desc="Whisper", unit="frame", leave=True)
        except ImportError:
            pass

        def monitor(p_batch):
            mm.throw_exception_if_processing_interrupted()
            current = int(p_batch[:, 0].max().item())
            progress.update_absolute(current, frames)
            if bar is not None:
                bar.update(max(0, current - bar.n))

        options = {"task": task, "return_timestamps": is_long, "monitor_progress": monitor,
                   "condition_on_prev_tokens": False, "return_dict_in_generate": False}
        if language != "auto":
            options["language"] = language
        try:
            with torch.inference_mode():
                predicted = patcher.model.asr.generate(**inputs, **options)
            pieces.append(processor.batch_decode(predicted, skip_special_tokens=True)[0].strip())
            progress.update_absolute(frames, frames)
            if bar is not None:
                bar.update(max(0, frames - bar.n))
        finally:
            if bar is not None:
                bar.close()
    return "\n".join(pieces).strip()
