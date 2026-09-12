"""Local instruction enhancement using the language head of the loaded Qwen encoder.

Port of the classify/render stage of upstream pe.py: map a loose request onto one
AuK task, snap its values, render the canonical instruction template, and
estimate the generation duration. All LLM calls run in-process on the encoder
loaded by AuK Encoder Loader; nothing here touches the network.
"""
import json
import math
import re

import torch

SPEED_CHOICES = (0.5, 0.75, 1.25, 1.5, 2.0)
DECIBEL_CHOICES = (5, 10, 15)
SECONDS_PER_UTF8_BYTE = 0.075
TTS_TASKS = ("Voice description TTS", "Voice cloning")
FRAMES_PER_SECOND = 50
WHISPER_TARGET_RMS = 0.0064
WHISPER_TO_NORMAL_TARGET_RMS = 0.000707945784384138
WHISPER_PEAK_CEILING = 0.95
EMOTION_MULTIPLIERS = {
    "sad": 1.22,
    "fearful": 1.16,
    "surprised": 0.92,
    "excited": 0.90,
}

SYSTEM_PROMPT = """You map a loose audio request to exactly one AuK task and fill its slots.
Available tasks, name -> canonical instruction template:
{capabilities}
Rules:
- Pick the single closest task. Use "Voice description TTS" when the user wants new speech with no reference clip and "Voice cloning" when a reference clip is supplied.
- Copy quoted words into slots exactly as written. Derive slots the request implies but does not state.
- Snap values: pitch to whole semitones 1-12, speed to one of 0.5/0.75/1.25/1.5/2.0, volume to one of 5/10/15 dB.
- "seconds": 0 means match the source audio duration (use it for every edit task). For TTS tasks estimate the spoken length in seconds (~2.5 words per second, minimum 1).
- If no task fits, use task "unsupported" and explain in params.reason.
Answer ONLY minified JSON: {{"task": "...", "params": {{...}}, "seconds": 0}}"""


def capabilities_text(tasks):
    return "\n".join(f"- {name}: {template}" for name, (template, _) in tasks.items())


def _balanced_object(text, start):
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return None


def _loads_repaired(candidate):
    for attempt in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue
    return None


def parse_reply(text):
    text = str(text or "")
    candidates = []
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence is not None:
        candidates.append(fence.group(1))
    brace = text.find("{")
    if brace >= 0:
        balanced = _balanced_object(text, brace)
        if balanced is not None:
            candidates.append(balanced)
    for candidate in candidates:
        obj = _loads_repaired(candidate)
        if obj is not None:
            return obj
    # Salvage fields individually; quantized language heads drop a quote or comma now and then.
    task = re.search(r'"task"\s*:\s*"([^"]*)"', text)
    if task is not None:
        params = {}
        params_at = text.find('"params"')
        if params_at >= 0:
            brace_at = text.find("{", params_at)
            if brace_at >= 0:
                raw_params = _balanced_object(text, brace_at)
                if raw_params is not None:
                    for key, value in re.findall(r'"(\w+)"\s*:\s*"([^"]*)"', raw_params):
                        params[key] = value
                    for key, value in re.findall(r'"(\w+)"\s*:\s*([0-9.]+)', raw_params):
                        params.setdefault(key, float(value))
        seconds = re.search(r'"seconds"\s*:\s*([0-9.]+)', text)
        return {"task": task.group(1), "params": params, "seconds": float(seconds.group(1)) if seconds else 0}
    raise ValueError(f"The language model returned malformed JSON: {text[:200]}")


def snap_params(task, params):
    params = dict(params)
    if task == "Change speed":
        value = float(params.get("factor") or 1.0)
        params["factor"] = min(SPEED_CHOICES, key=lambda choice: abs(choice - value))
    elif task == "Raise pitch" or task == "Lower pitch":
        semitones = min(12, max(1, round(float(params.get("semitones") or 2))))
        params["semitones"] = float(semitones)
    elif task == "Increase volume" or task == "Decrease volume":
        decibels = min(DECIBEL_CHOICES, key=lambda choice: abs(choice - float(params.get("decibels") or 5)))
        params["decibels"] = float(decibels)
    return params


def render(tasks, task, params):
    if task not in tasks:
        reason = params.get("reason") if isinstance(params, dict) else None
        raise ValueError(f"Request is outside AuK's capabilities{f': {reason}' if reason else ''}.")
    template, fields = tasks[task]
    values = {key: params.get(key, default) for key, default in fields.items()}
    missing = [key for key, value in values.items() if value is None or (isinstance(value, str) and not value.strip())]
    if missing:
        raise ValueError(f"The request lacks {missing} for {task}.")
    return template.format(**values)


def estimate_seconds(text, seconds):
    if seconds and seconds > 0:
        return round(min(float(seconds), 3600.0), 2)
    return round(max(1.0, len(str(text).encode("utf-8")) * SECONDS_PER_UTF8_BYTE), 1)


def _spoken_duration(text):
    value = str(text or "")
    chinese = len(re.findall(r"[\u3400-\u9fff]", value))
    english = len(re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)*", value))
    return chinese * 0.21 + english * 0.30


def _source_seconds(audio):
    if audio is None:
        return None
    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if not torch.is_tensor(waveform) or waveform.ndim != 3 or waveform.shape[-1] == 0:
        raise ValueError("Prompt Enhance audio must be a non-empty ComfyUI AUDIO value.")
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("Prompt Enhance audio has an invalid sample rate.")
    return waveform.shape[-1] / sample_rate


def _content_seconds(task, params, source_seconds, transcript):
    slots = {
        "Replace speech": ("replacement", "original"),
        "Insert speech before": ("text", None),
        "Insert speech after": ("text", None),
        "Remove speech": (None, "text"),
        "Edit lyrics": ("replacement", "original"),
    }
    add_slot, remove_slot = slots[task]
    original = _spoken_duration(transcript)
    if original > 0:
        edited = original + (_spoken_duration(params.get(add_slot)) if add_slot else 0)
        edited -= _spoken_duration(params.get(remove_slot)) if remove_slot else 0
        return source_seconds * max(0.05, edited) / original
    if add_slot and remove_slot:
        removed = _spoken_duration(params.get(remove_slot))
        replacement = _spoken_duration(params.get(add_slot))
        return source_seconds * replacement / removed if removed > 0 else source_seconds
    return source_seconds


def task_seconds(task, params, audio, transcript, suggested):
    source_seconds = _source_seconds(audio)
    if task in TTS_TASKS:
        return estimate_seconds(params.get("text"), suggested)
    if source_seconds is None:
        raise ValueError(f"{task} requires audio for Prompt Enhance duration calculation.")
    if task == "Change speed":
        duration = source_seconds / float(params.get("factor") or 1.0)
    elif task in ("Replace speech", "Insert speech before", "Insert speech after", "Remove speech", "Edit lyrics"):
        duration = _content_seconds(task, params, source_seconds, transcript)
    elif task == "Change emotion":
        duration = source_seconds * EMOTION_MULTIPLIERS.get(str(params.get("description", "")).lower(), 1.0)
    elif task == "Add nonverbal sound":
        duration = source_seconds + 0.5
    elif task == "Remove nonverbal sounds":
        duration = max(0.1, source_seconds - 0.3)
    else:
        duration = source_seconds
    return max(1, round(duration * FRAMES_PER_SECOND)) / FRAMES_PER_SECOND


def prepare_audio(audio, task):
    if audio is None:
        return None
    waveform = audio["waveform"].detach().cpu().float().mean(1, keepdim=True)
    if not torch.isfinite(waveform).all():
        raise ValueError("Prompt Enhance audio contains NaN or Inf.")
    target_rms = None
    if task == "Convert to whisper":
        target_rms = WHISPER_TARGET_RMS
    elif task == "Whisper to speech":
        target_rms = WHISPER_TO_NORMAL_TARGET_RMS
    if target_rms is not None:
        rms = waveform.double().square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-9)
        waveform = waveform * (target_rms / rms).float()
        peak = waveform.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-9)
        waveform = waveform * torch.minimum(torch.ones_like(peak), WHISPER_PEAK_CEILING / peak)
    return {"waveform": waveform, "sample_rate": audio["sample_rate"]}


def enhance(encoder, instruction, tasks, context=None, max_new_tokens=256, return_details=False):
    system = SYSTEM_PROMPT.format(capabilities=capabilities_text(tasks))
    user = str(instruction or "").strip()
    if not user:
        raise ValueError("Instruction cannot be empty.")
    if context:
        user = f"{user}\n\nTranscript of the source audio:\n{context}"
    reply = encoder.complete(system, user, max_new_tokens)
    obj = parse_reply(reply)
    task = str(obj.get("task") or "unsupported")
    params = snap_params(task, obj.get("params") or {})
    instruction_out = render(tasks, task, params)
    seconds = float(obj.get("seconds") or 0)
    if task in TTS_TASKS:
        seconds = estimate_seconds(params.get("text") or instruction, seconds)
    else:
        seconds = 0.0
    result = (instruction_out, seconds, task)
    return (*result, params) if return_details else result


def prepare(encoder, instruction, tasks, audio=None, context=None, max_new_tokens=256):
    text, suggested, task, params = enhance(
        encoder, instruction, tasks, context=context,
        max_new_tokens=max_new_tokens, return_details=True,
    )
    seconds = task_seconds(task, params, audio, context, suggested)
    return text, seconds, task, prepare_audio(audio, task)
