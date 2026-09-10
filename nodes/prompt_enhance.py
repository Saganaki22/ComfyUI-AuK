"""Local instruction enhancement using the language head of the loaded Qwen encoder.

Port of the classify/render stage of upstream pe.py: map a loose request onto one
AuK task, snap its values, render the canonical instruction template, and
estimate the generation duration. All LLM calls run in-process on the encoder
loaded by AuK Encoder Loader; nothing here touches the network.
"""
import json
import re

SPEED_CHOICES = (0.5, 0.75, 1.25, 1.5, 2.0)
DECIBEL_CHOICES = (5, 10, 15)
SECONDS_PER_UTF8_BYTE = 0.075
TTS_TASKS = ("Voice description TTS", "Voice cloning")

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


def enhance(encoder, instruction, tasks, context=None, max_new_tokens=256):
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
    return instruction_out, seconds, task
