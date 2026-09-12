# ComfyUI-AuK

English | [中文](README_ZH.md)

Local AuK Base and AuK-Flash speech generation, voice cloning, editing, enhancement
and separation. Uses ComfyUI model management, attention and quantized operations.
Audio input/output uses the core ComfyUI `AUDIO` type.

<img width="2115" height="698" alt="image" src="https://github.com/user-attachments/assets/741b37dc-72eb-46ac-982d-eb90591e64d7" />


> **Read this before installing:** this model's prompt adherence is not strong — voice cloning, TTS and some editing tasks work, but others are unstable. This is a limitation of the model itself, not of this integration. See [Prompt adherence](#prompt-adherence) for the tested task list before you download anything.

## Nodes

| Node | Inputs | Output |
| --- | --- | --- |
| AuK Model Loader | Base/Flash checkpoint, compute precision, attention | AUK_MODEL |
| AuK Encoder Loader | Converted Qwen2.5-Omni encoder, compute precision | AUK_ENCODER |
| AuK VAE Loader | Original, unquantized AuK VAE | VAE |
| AuK Instruction Encode | Encoder, AuK model, instruction, optional audio | CONDITIONING |
| AuK Generate / Edit | Model, VAE, conditioning, duration, seed, steps, guidance, sway | AUDIO |
| AuK Instruction Builder | Task and its relevant fields | STRING |
| AuK Whisper Transcribe | Audio, model, language, task | STRING |
| AuK Prompt Enhance | Encoder, request, optional ASR context/audio | instruction, seconds, task, prepared audio |

Instruction Encode receives the AuK model because Base and Flash contain their own
learned Qwen layer-fusion weights. Encoding happens once before denoising.

Use core **Load Audio / Record Audio**, trimming/mixing nodes, **Preview Audio** and
**Save Audio (Advanced)**. AuK models use a dedicated socket: native KSampler and
CLIP Text Encode do not implement AuK's sampling or multimodal conditioning.

## Installation

Place this directory in `ComfyUI/custom_nodes/ComfyUI-AuK`, then install the small
dependency list using the Python environment that runs ComfyUI:

```shell
python -m pip install -r requirements.txt
```

Requires Transformers 5.3.x or a compatible 5.x version. This package does not pin
or replace your PyTorch build. Quantized checkpoints require ComfyUI and Comfy
Kitchen builds exposing `int8_tensorwise` with ConvRot, `convrot_w4a4` or
`asym_w4a8_int8`. Those formats are present in the development installation;
their availability should not be assumed for every ComfyUI release. BF16 is the
baseline format.

Restart ComfyUI after installing the nodes or adding new model directories.

## Models and locations

Download ready-to-use checkpoints from **[drbaph/AuK-comfyui](https://huggingface.co/drbaph/AuK-comfyui)**. Choose one Base or Flash checkpoint, one Qwen encoder, and the unquantized VAE. Model and encoder formats can be mixed; you do not need every file.

Sizes are decimal GB (1 GB = 1,000,000,000 bytes), measured from the converted files, and are not VRAM requirements. All twelve checkpoint files are available in the model repository.

| Model file | Size | Folder under ComfyUI/models/ | Direct download |
| --- | ---: | --- | --- |
| `auk_base_fp32.safetensors` | 6.122 GB | `diffusion_models/` | [⬇ Download](https://huggingface.co/drbaph/AuK-comfyui/resolve/main/diffusion_models/auk_base_fp32.safetensors?download=true) |
| `auk_flash_fp32.safetensors` | 6.122 GB | `diffusion_models/` | [⬇ Download](https://huggingface.co/drbaph/AuK-comfyui/resolve/main/diffusion_models/auk_flash_fp32.safetensors?download=true) |
| `auk_base_bf16.safetensors` | 3.062 GB | `diffusion_models/` | [⬇ Download](https://huggingface.co/drbaph/AuK-comfyui/resolve/main/diffusion_models/auk_base_bf16.safetensors?download=true) |
| `auk_base_int8.safetensors` | 1.546 GB | `diffusion_models/` | [⬇ Download](https://huggingface.co/drbaph/AuK-comfyui/resolve/main/diffusion_models/auk_base_int8.safetensors?download=true) |
| `auk_base_w4a8.safetensors` | 0.880 GB | `diffusion_models/` | [⬇ Download](https://huggingface.co/drbaph/AuK-comfyui/resolve/main/diffusion_models/auk_base_w4a8.safetensors?download=true) |
| `auk_flash_bf16.safetensors` | 3.062 GB | `diffusion_models/` | [⬇ Download](https://huggingface.co/drbaph/AuK-comfyui/resolve/main/diffusion_models/auk_flash_bf16.safetensors?download=true) |
| `auk_flash_int8.safetensors` | 1.546 GB | `diffusion_models/` | [⬇ Download](https://huggingface.co/drbaph/AuK-comfyui/resolve/main/diffusion_models/auk_flash_int8.safetensors?download=true) |
| `auk_flash_w4a8.safetensors` | 0.880 GB | `diffusion_models/` | [⬇ Download](https://huggingface.co/drbaph/AuK-comfyui/resolve/main/diffusion_models/auk_flash_w4a8.safetensors?download=true) |
| `qwen_omni_bf16.safetensors` | 8.070 GB | `text_encoders/` | [⬇ Download](https://huggingface.co/drbaph/AuK-comfyui/resolve/main/text_encoders/qwen_omni_bf16.safetensors?download=true) |
| `qwen_omni_int8.safetensors` | 4.669 GB | `text_encoders/` | [⬇ Download](https://huggingface.co/drbaph/AuK-comfyui/resolve/main/text_encoders/qwen_omni_int8.safetensors?download=true) |
| `qwen_omni_w4a8.safetensors` | 3.179 GB | `text_encoders/` | [⬇ Download](https://huggingface.co/drbaph/AuK-comfyui/resolve/main/text_encoders/qwen_omni_w4a8.safetensors?download=true) |
| `auk_vae.safetensors` | 0.637 GB | `vae/` | [⬇ Download](https://huggingface.co/drbaph/AuK-comfyui/resolve/main/vae/auk_vae.safetensors?download=true) |

```text
📂 ComfyUI/
└── 📂 models/
    ├── 📂 diffusion_models/
    │   ├── auk_base_fp32.safetensors
    │   ├── auk_flash_fp32.safetensors
    │   ├── auk_base_bf16.safetensors
    │   ├── auk_base_int8.safetensors
    │   ├── auk_base_w4a8.safetensors
    │   ├── auk_flash_bf16.safetensors
    │   ├── auk_flash_int8.safetensors
    │   └── auk_flash_w4a8.safetensors
    ├── 📂 text_encoders/
    │   ├── qwen_omni_bf16.safetensors
    │   ├── qwen_omni_int8.safetensors
    │   └── qwen_omni_w4a8.safetensors
    └── 📂 vae/
        └── auk_vae.safetensors
```

**Only download the model weights.** Base and Flash configs are bundled in `assets/auk_base/config.yaml` and `assets/auk_flash/config.yaml`; no config belongs beside the weights. Keep original checkpoint filenames so their variant can be identified. Qwen config, tokenizer, processor and license files are bundled in this node pack under `assets/qwen2.5-omni-3b/` and loaded locally from there. No sidecars are needed in `models/text_encoders/`. Place the weights directly in the three model folders shown above.

The full original Qwen checkpoint is **11.973 GB**. This integration retains its text/audio encoder plus the language-model head used by the Prompt Enhance: **8.070 GB BF16**, **4.669 GB INT8**, or **3.179 GB W4A8** (the head itself always stays BF16). The **0.637 GB VAE remains unquantized and computes in FP32**. W4A8 means four-bit weights with eight-bit activations; W4A4 (four-bit activations) is also supported by the converter but is not distributed here.

Peak VRAM depends mainly on audio length, the attention backend and ComfyUI's offloading strategy. One reported measurement with a quantized diffusion model and encoder pair was about **8 GB** peak; treat it as an observation, not a minimum requirement. The older 3.88 GB validation figure measures PyTorch allocations only, not total VRAM.

Nodes read local files and never download during execution. Checkpoint metadata
selects quantization; the loader's `precision` controls compute dtype and its
`attention` control selects the attention backend — `auto`/`sdpa` match
upstream inference exactly, while `flash_attention`/`sageattention` are opt-in
and fall back to SDPA for masked edit steps or unsupported dtypes.

## Workflows

[Task guide — instructions and examples for every task](docs/GUIDE.md)

Drag a JSON from `example_workflows/` onto ComfyUI:

1. **01_text_to_speech.json** — description-based speech without reference audio.
2. **02_audio_edit_or_clone.json** — upload source/reference audio and change the instruction; covers every edit task.
3. **03_voice_clone_plus.json** — voice cloning plus Whisper Transcribe (can fetch its
   checkpoint when missing), Instruction Builder and Prompt Enhance wired in.

Select the installed checkpoints in the loaders. The audio examples require you
to select/upload an input in Load Audio. They do not ship an unrelated recording.
Base and Flash perform the same tasks; pick one per run. Flash is the distilled
model — four fixed steps, no guidance — for speed. Base takes 32 steps with
guidance and is the choice for quality. On Windows,
ComfyUI lists subfolder models with backslashes; if a loader shows its value as
missing after loading a workflow, re-select the checkpoint in the dropdown.

Connect the optional Instruction Builder's STRING output to the instruction input
of Instruction Encode, or type any supported instruction directly.

**AuK Prompt Enhance** reproduces the useful PE stages locally with the
Qwen2.5-Omni-3B language head. Connect source audio and, preferably, the source
transcript from AuK Whisper Transcribe. It classifies the request, renders the
canonical instruction, calculates task-aware duration, and applies the upstream
whisper RMS targets. Wire `instruction` to Instruction Encode, `seconds`
to Generate / Edit, and `prepared_audio` to Instruction Encode's audio input.
It needs an encoder checkpoint containing the language head (all released
encoder files above qualify). This avoids the external OpenAI-compatible LLM;
classification quality can therefore differ from upstream's recommended `hy3`.

**AuK Whisper Transcribe** is an optional helper: it transcribes audio with a
Whisper model. Place any Whisper checkpoint folder containing
`config.json` under `ComfyUI/models/whisper/` or `ComfyUI/models/audio_encoders/`,
or enable its `download_if_missing` toggle to fetch a known openai/whisper
checkpoint (tiny to large-v3-turbo) into `models/whisper/<size>/` once and reuse
the local copy afterwards. AuK never needs a transcript of reference audio —
its encoder listens to the clip directly — but a transcript helps you write
target-speaker, replace-speech and lyric-edit instructions, and to check what
AuK actually said. Clips longer than 30 seconds use Whisper's native timestamp-guided long-form processing without truncating the source.
It can also transcribe AuK's own output as a quality check.

| Task | Example instruction / control |
| --- | --- |
| Voice-description TTS | Describe the voice and specify the text to speak. Set target seconds. |
| Voice cloning | `Say the following with the same voice: "Your text".` Supply reference audio. |
| Replace / insert / remove speech | `Replace 'old words' with 'new words'.` |
| Lyric editing | `Change "old lyrics" to "new lyrics" in the vocal recording.` Use isolated vocals. |
| Pitch | `Raise the pitch by 2 semitones.` |
| Speed | `Adjust the speech speed to 1.25x.` Set seconds to source duration / 1.25. |
| Volume | `Increase the volume by 5 dB.` |
| Emotion / timbre | Describe the desired emotion or voice quality. |
| Accent removal | `Remove the regional accent while preserving the speaker's voice and content.` |
| Nonverbal sounds | Describe breaths, laughs or coughs to add/remove. |
| Whisper conversion | Convert normal speech to whisper or whisper to normal speech. |
| Enhancement | Remove noise/reverberation while retaining the speech. |
| Speaker separation | Identify the speaker by order of speaking. |
| Music separation | Extract singing, or retain all human voices. |
| Target-speaker extraction | Identify the speaker by the words they say. |

These are model instructions, not guarantees of perfect edits. See the
[upstream cookbook](https://github.com/Tencent-Hunyuan/AuK/blob/main/docs/COOKBOOK.md)
for English and Chinese examples.

`seconds=0` matches the source duration. Text-only generation requires positive
seconds. Content changes may need a longer or shorter duration. Output is mono
24 kHz. Source audio is downmixed/resampled at the integration boundary.

Base defaults to 32 Euler steps, guidance 2, sway -1. AuK guidance is
`conditional + strength * (conditional - unconditional)`. Flash always uses its
four fixed steps and disables guidance. The seed covers reference VAE sampling
and diffusion noise. There is no combined source/target duration cap in this node pack.
Longer audio takes more memory and processing time.

The generation node decodes directly to preserve amplitude. Native VAE Decode
Audio in the tested ComfyUI normalizes loud outputs, which can alter volume edits.
The VAE adapter implements standard untiled audio encode/decode; tiled VAE nodes
are not implemented.

## Prompt adherence

After rigorous testing on both FP32 and BF16 checkpoints, I have come to the conclusion that this model's prompt adherence is unfortunately not strong in some tasks. This is a limitation of the model itself, not of this integration. Check the prompt adherence list I made, from most to least reliable:

Tests against the Hugging Face demo used the same source audio, instructions and
Prompt Enhance path. For difficult edits, setting the output **0.2–0.5 seconds
shorter than the source** often improved adherence (for example, 5.5 seconds for
a 6-second input), even when the replacement would normally need more time. We
also found that some unstable tasks succeed more often with the Chinese cookbook
instruction than with its English version. This indicates that both duration
conditioning and instruction language can materially affect the result.

**Stable**

1. TTS (description-based)
2. Voice cloning
3. Increase / decrease volume
4. Enhance speech
5. Extract singing
6. Separate speaker
7. Denoise only
8. Repair quality

**Less stable**

9. Keep human voices
10. Change emotion
11. Extract target speaker
12. Change speed
13. Raise / lower pitch
14. Change timbre
15. Remove speech
16. Add / remove nonverbal sounds
17. Remove accent
18. Insert speech before / after, replace speech, edit lyrics, whisper conversion (either direction)

## Attribution

Adapted from [Tencent-Hunyuan/AuK](https://github.com/Tencent-Hunyuan/AuK), commit
`d9f30ffe4231dbc90b48cc83a35d310fece0b060`. AuK code and weights retain their MIT
license. The BigVGAN codec derives from NVIDIA BigVGAN/HiFi-GAN; its alias-free
resampling derives from alias-free-torch. Existing ComfyUI components are reused
by import. Qwen weights retain the upstream Qwen Research license, bundled with
the node under `assets/qwen2.5-omni-3b/`; conversion does not relicense them.

## Citation

Research using AuK can cite the following entry ([arXiv:2609.08936](https://arxiv.org/abs/2609.08936)):

```bibtex
@misc{ma2026auktechnicalreportopensource,
  title={AuK Technical Report: An Open-Source Foundational Model for Speech Generation and Editing},
  author={Ziyang Ma and Zhikang Niu and Wenming Tu and Tianrui Wang and Ruiqi Yan and Junxi Liu and Yanru Huo and Nickk Huang and Yang Liu and Qicong Xie and Zeyu Xie and Hui Wang and Haitao Li and Zixuan Jiang and Yalin Li and Jie Fang and Yifan Duan and Zeyue Tian and Guangzheng Li and Haina Zhu and Shuyi Wang and Jinwen Wang and Mingyu Cui and Tian Tan and Auden and Sen Liang and Steve Yves and Shan Yang and Liefeng Bo and Zilong Zheng and Kai Yu and Eng-Siong Chng and Xie Chen},
  year={2026},
  eprint={2609.08936},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2609.08936},
}
```
