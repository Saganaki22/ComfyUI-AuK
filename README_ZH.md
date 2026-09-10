# ComfyUI-AuK

[English](README.md) | 中文

本地运行的 AuK Base 与 AuK-Flash 语音生成、声音克隆、语音编辑、增强与人声分离。
使用 ComfyUI 的模型管理、注意力与量化算子。音频输入输出使用 ComfyUI 原生的 `AUDIO` 类型。

<img width="2115" height="698" alt="image" src="https://github.com/user-attachments/assets/5ef6e87b-4a24-4ae4-86fc-fcd3e54b3cab" />


> **安装前请先阅读：** 该模型的指令遵循度并不强——声音克隆、TTS 与部分编辑任务可用，但许多任务并不总是有效。这是模型本身的局限，与本集成无关。下载任何模型前，请先查看[指令遵循度](#指令遵循度)中的实测任务列表。

## 节点

| 节点 | 输入 | 输出 |
| --- | --- | --- |
| AuK Model Loader | Base/Flash 检查点、计算精度 | AUK_MODEL |
| AuK Encoder Loader | 转换后的 Qwen2.5-Omni 编码器、计算精度 | AUK_ENCODER |
| AuK VAE Loader | 原始未量化的 AuK VAE | VAE |
| AuK Instruction Encode | 模型、编码器、指令文本、可选音频 | CONDITIONING |
| AuK Generate / Edit | 模型、条件、VAE、时长、种子、步数、引导、sway | AUDIO |
| AuK Instruction Builder | 任务模板及其字段 | STRING |
| AuK Whisper Transcribe | 音频、模型、语言、任务 | STRING |

Instruction Encode 需要接入 AuK 模型，因为 Base 与 Flash 各自带有学习到的
Qwen 层融合权重。条件编码在去噪之前一次性完成。

音频加载、录音、裁剪、混音请使用核心的 **Load Audio / Record Audio** 等节点，
保存与预览使用 **Save Audio / Preview Audio**。AuK 使用专用插座：原生 KSampler
与 CLIP Text Encode 并未实现 AuK 的采样方式与多模态条件。

## 安装

将本目录放入 `ComfyUI/custom_nodes/ComfyUI-AuK`，然后用运行 ComfyUI 的
Python 环境安装少量依赖：

```shell
python -m pip install -r requirements.txt
```

需要 Transformers 5.3.x 或兼容的 5.x 版本。本包不会固定或替换你的 PyTorch。
量化检查点需要提供 `int8_tensorwise`（含 ConvRot）、`convrot_w4a4` 或
`asym_w4a8_int8` 的 ComfyUI 与 Comfy Kitchen 版本；这些格式已存在于开发环境，
但不能假设每个 ComfyUI 发行版都有。BF16 是基线格式。

安装节点或添加模型目录后请重启 ComfyUI。

## 模型与目录

预转换模型：[drbaph/AuK-comfyui](https://huggingface.co/drbaph/AuK-comfyui)。选择一个 Base 或 Flash 模型、一个 Qwen 编码器，再下载未量化 VAE。三种量化版本可以独立搭配，无需全部下载。

下表使用十进制 GB（1 GB = 1,000,000,000 字节），表示文件大小，不是显存需求。全部十二个检查点文件均已上传，下载链接已核对。

| 模型文件 | 大小 | ComfyUI/models/ 下的目录 | 直接下载 |
| --- | ---: | --- | --- |
| `auk_base_fp32.safetensors` | 6.122 GB | `diffusion_models/` | [⬇ 下载](https://huggingface.co/drbaph/AuK-comfyui/resolve/main/diffusion_models/auk_base_fp32.safetensors?download=true) |
| `auk_flash_fp32.safetensors` | 6.122 GB | `diffusion_models/` | [⬇ 下载](https://huggingface.co/drbaph/AuK-comfyui/resolve/main/diffusion_models/auk_flash_fp32.safetensors?download=true) |
| `auk_base_bf16.safetensors` | 3.062 GB | `diffusion_models/` | [⬇ 下载](https://huggingface.co/drbaph/AuK-comfyui/resolve/main/diffusion_models/auk_base_bf16.safetensors?download=true) |
| `auk_base_int8.safetensors` | 1.546 GB | `diffusion_models/` | [⬇ 下载](https://huggingface.co/drbaph/AuK-comfyui/resolve/main/diffusion_models/auk_base_int8.safetensors?download=true) |
| `auk_base_w4a8.safetensors` | 0.880 GB | `diffusion_models/` | [⬇ 下载](https://huggingface.co/drbaph/AuK-comfyui/resolve/main/diffusion_models/auk_base_w4a8.safetensors?download=true) |
| `auk_flash_bf16.safetensors` | 3.062 GB | `diffusion_models/` | [⬇ 下载](https://huggingface.co/drbaph/AuK-comfyui/resolve/main/diffusion_models/auk_flash_bf16.safetensors?download=true) |
| `auk_flash_int8.safetensors` | 1.546 GB | `diffusion_models/` | [⬇ 下载](https://huggingface.co/drbaph/AuK-comfyui/resolve/main/diffusion_models/auk_flash_int8.safetensors?download=true) |
| `auk_flash_w4a8.safetensors` | 0.880 GB | `diffusion_models/` | [⬇ 下载](https://huggingface.co/drbaph/AuK-comfyui/resolve/main/diffusion_models/auk_flash_w4a8.safetensors?download=true) |
| `qwen_omni_bf16.safetensors` | 8.070 GB | `text_encoders/` | [⬇ 下载](https://huggingface.co/drbaph/AuK-comfyui/resolve/main/text_encoders/qwen_omni_bf16.safetensors?download=true) |
| `qwen_omni_int8.safetensors` | 4.669 GB | `text_encoders/` | [⬇ 下载](https://huggingface.co/drbaph/AuK-comfyui/resolve/main/text_encoders/qwen_omni_int8.safetensors?download=true) |
| `qwen_omni_w4a8.safetensors` | 3.179 GB | `text_encoders/` | [⬇ 下载](https://huggingface.co/drbaph/AuK-comfyui/resolve/main/text_encoders/qwen_omni_w4a8.safetensors?download=true) |
| `auk_vae.safetensors` | 0.637 GB | `vae/` | [⬇ 下载](https://huggingface.co/drbaph/AuK-comfyui/resolve/main/vae/auk_vae.safetensors?download=true) |

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

**只需下载模型权重。** Base 与 Flash 配置分别内置于 `assets/auk_base/config.yaml` 和 `assets/auk_flash/config.yaml`，无需放在权重旁边。原始检查点请保留原文件名，以便识别模型版本。 Qwen 配置、分词器、处理器与许可证文件已内置在节点包的 `assets/qwen2.5-omni-3b/`，从该目录本地加载。无需在 `models/text_encoders/` 中放置附属文件。请将权重直接放入上述三个模型目录。

原始完整 Qwen 检查点约 **11.973 GB**；本包保留文本与音频编码部分以及语言模型头（Prompt Enhance 需要）：BF16 为 **8.070 GB**，INT8 为 **4.669 GB**，W4A8 为 **3.179 GB**（语言头始终为 BF16）。VAE 保持原始未量化权重，以 FP32 计算。W4A8 是四比特权重与八比特激活；转换器也支持 W4A4（四比特激活），但不在发布之列。

显存峰值主要取决于音频长度、注意力后端与 ComfyUI 的卸载策略。一次量化模型与编码器组合的实测峰值约 **8 GB**，这只是观测值，不是最低要求。旧验证表中的 3.88 GB 仅是 PyTorch 分配量，不代表总显存需求。

节点只读取本地文件，不会自动下载模型。加载器 `precision` 控制计算精度，文件元数据决定量化格式；`attention` 控件选择注意力后端——`auto`/`sdpa` 与上游推理完全一致，`flash_attention`/`sageattention` 为可选实验，遇掩码或不支持的精度时回退 SDPA。

## 工作流

[任务指南：各项操作步骤与示例](docs/GUIDE.md)

将 `example_workflows/` 中的 JSON 拖入 ComfyUI：

1. **01_text_to_speech.json** — 无参考音频、按描述生成语音。
2. **02_audio_edit_or_clone.json** — 上传源/参考音频并修改指令，覆盖全部编辑任务。
3. **03_voice_clone_plus.json** — 声音克隆，内置 Whisper Transcribe（缺少模型时
   可自动下载）、Instruction Builder 与 Prompt Enhance。

在加载器中选择已安装的检查点。音频示例需要你在 Load Audio 中选择/上传输入，
不附带任何示例录音。Base 与 Flash 执行相同任务，每次运行任选其一：Flash 是
蒸馏模型，固定四步、无引导，速度快；Base 为 32 步加引导，追求质量。

在 Windows 上，ComfyUI 以反斜杠列出子目录模型；载入工作流后如果加载器显示
数值缺失，请在下拉框中重新选择一次检查点。

可将可选的 Instruction Builder 的 STRING 输出连接到 Instruction Encode 的
instruction 输入，或直接输入任意受支持的指令。

**AuK Whisper Transcribe** 是可选辅助节点：使用 Whisper 模型转写音频。
将包含 `config.json` 的 Whisper 检查点目录放入 `ComfyUI/models/whisper/`
或 `ComfyUI/models/audio_encoders/`；也可以开启 `download_if_missing` 开关，
从 openai/whisper 已知检查点（tiny 到 large-v3-turbo）下载一次到
`models/whisper/<size>/`，之后复用本地副本。AuK 从不需要参考
音频的文字稿——它的编码器直接听取音频——但文字稿有助于编写目标说话人、
替换语音与歌词编辑指令，也可用于检查 AuK 实际说了什么。超过 30 秒的音频
将分窗转写。

| 任务 | 示例指令 / 控制 |
| --- | --- |
| 描述生成语音 | 描述音色并给出要说的内容，设定目标时长。 |
| 声音克隆 | `Say the following with the same voice: "Your text".` 并提供参考音频。 |
| 替换 / 插入 / 删除语音 | `Replace 'old words' with 'new words'.` |
| 歌词编辑 | `Change "old lyrics" to "new lyrics" in the vocal recording.` 请使用已分离的人声。 |
| 音调 | `Raise the pitch by 2 semitones.` |
| 语速 | `Adjust the speech speed to 1.25x.` 将 seconds 设为源时长 / 1.25。 |
| 音量 | `Increase the volume by 5 dB.` |
| 情感 / 音色 | 描述目标情感或音色。 |
| 口音去除 | `Remove the regional accent while preserving the speaker's voice and content.` |
| 非人声编辑 | 描述要添加/去除的呼吸声、笑声或咳嗽。 |
| 耳语转换 | 普通语音转耳语，或耳语转普通语音。 |
| 增强 | 去除噪声/混响并保留语音。 |
| 说话人分离 | 按说话顺序指定保留的说话人。 |
| 音乐分离 | 仅提取歌声，或保留所有人声。 |
| 目标说话人提取 | 按目标说话人所说的话进行识别。 |

以上是模型指令，并非完美编辑的保证。更多中英文示例见
[上游 cookbook](https://github.com/Tencent-Hunyuan/AuK/blob/main/docs/COOKBOOK.md)。

`seconds=0` 表示与源时长一致。纯文本生成必须为正数秒。内容变化可能需要更长
或更短的时长。输出为单声道 24 kHz。源音频在集成边界完成降混/重采样。

Base 默认 32 步 Euler、引导 2、sway -1。AuK 引导为
`conditional + strength * (conditional - unconditional)`。Flash 始终使用固定的
四步调度并关闭引导。种子同时覆盖参考 VAE 采样与扩散噪声。本节点包不限制参考/源音频与生成目标的合计时长。
更长的音频需要更多显存与处理时间。

生成节点直接解码以保持幅度。经测试的 ComfyUI 原生 VAE Decode Audio 会对
大音量输出做归一化，可能改变音量编辑的效果。VAE 适配器实现了标准的非分块
音频编解码；未实现分块 VAE 节点。

## 指令遵循度

在 FP32 与 BF16 检查点上经过反复测试，我的结论是：遗憾的是，该模型的指令遵循度在某些任务上并不强。这是模型本身的局限，与本集成无关。下面是我实测的指令遵循度列表，从最可靠到最不可靠排序：

**稳定**

1. TTS（按描述生成）
2. 声音克隆
3. 提高音量 / 降低音量
4. 语音增强
5. 歌声提取
6. 说话人分离
7. 仅降噪
8. 音质修复

**不稳定**

9. 保留所有人声
10. 更换情感
11. 目标说话人提取
12. 调整语速
13. 升调 / 降调
14. 更换音色
15. 删除语音
16. 添加 / 删除非语言声音
17. 去除口音
18. 在前面 / 后面插入语音、替换语音、改写歌词、耳语转换（双向）

## 版权与致谢

改编自 [Tencent-Hunyuan/AuK](https://github.com/Tencent-Hunyuan/AuK)，提交
`d9f30ffe4231dbc90b48cc83a35d310fece0b060`。AuK 代码与权重遵循其 MIT 许可证。
BigVGAN 编解码源自 NVIDIA BigVGAN/HiFi-GAN；其 alias-free 重采样源自
alias-free-torch。现有 ComfyUI 组件通过导入复用。Qwen 权重遵循上游 Qwen
Research 许可证，其副本已随本节点包置于 `assets/qwen2.5-omni-3b/`；转换不会
重新授权。

## 引用

研究使用 AuK 可引用以下条目（[arXiv:2609.08936](https://arxiv.org/abs/2609.08936)）：

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
