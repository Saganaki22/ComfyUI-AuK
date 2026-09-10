"""Local Qwen2.5-Omni text/audio encoder using ComfyUI-managed operations."""
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from transformers import AttentionInterface, Qwen2_5OmniConfig, Qwen2_5OmniProcessor
from transformers.masking_utils import AttentionMaskInterface, eager_mask
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniRotaryEmbedding, Qwen2_5OmniRMSNorm, SinusoidsPositionEmbedding

import comfy.ops
import comfy.model_management as mm
from comfy.ldm.modules.attention import optimized_attention


def comfy_attention(module, query, key, value, attention_mask=None, scaling=None, **kwargs):
    heads = query.shape[1]
    if key.shape[1] != heads:
        repeats = heads // key.shape[1]
        key, value = key.repeat_interleave(repeats, 1), value.repeat_interleave(repeats, 1)
    if scaling is not None:
        query = query * (scaling * query.shape[-1] ** 0.5)
    out = optimized_attention(query, key, value, heads, mask=attention_mask, skip_reshape=True)
    return out.reshape(query.shape[0], query.shape[2], heads, query.shape[-1]), None


AttentionInterface.register("auk_comfy", comfy_attention)
AttentionMaskInterface.register("auk_comfy", eager_mask)


class QwenEncoder(Qwen2_5OmniThinkerForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        del self.visual

    def encode_layers(self, inputs):
        input_ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        embeds = self.model.embed_tokens(input_ids)
        lengths = None
        if "input_features" in inputs:
            features = self.get_audio_features(inputs["input_features"], feature_attention_mask=inputs["feature_attention_mask"], return_dict=True).last_hidden_state
            embeds = embeds.masked_scatter((input_ids == self.config.audio_token_index)[:, :, None], features.to(embeds))
            lengths = inputs["feature_attention_mask"].sum(-1)
        positions, _ = self.get_rope_index(input_ids, attention_mask=mask, use_audio_in_video=False, audio_seqlens=lengths)
        output = self.model(inputs_embeds=embeds, attention_mask=mask, position_ids=positions, use_cache=False, output_hidden_states=True, return_dict=True)
        return output.hidden_states[1:]

    def complete(self, inputs, stop_ids, max_new_tokens=256):
        # Single-sample greedy decode with KV cache. Returns generated token ids.
        token = inputs["input_ids"]
        mask = inputs["attention_mask"]
        features = {key: inputs[key] for key in ("input_features", "feature_attention_mask") if key in inputs}
        lengths = inputs["feature_attention_mask"].sum(-1) if "feature_attention_mask" in inputs else None
        positions, _ = self.get_rope_index(inputs["input_ids"], attention_mask=mask, use_audio_in_video=False, audio_seqlens=lengths)
        past = None
        result = []
        for _ in range(max_new_tokens):
            kwargs = {"input_ids": token, "attention_mask": mask, "position_ids": positions, "past_key_values": past, "use_cache": True, "return_dict": True}
            if past is None:
                kwargs.update(features)
            out = self(**kwargs)
            past = out.past_key_values
            token = out.logits[:, -1].argmax(-1, keepdim=True)
            value = token.item()
            if value in stop_ids:
                break
            result.append(value)
            mask = torch.cat((mask, torch.ones_like(token)), 1)
            positions = positions[:, :, -1:] + 1
        return result


def make_encoder(config_path, ops, dtype):
    config = Qwen2_5OmniConfig.from_pretrained(config_path, local_files_only=True).thinker_config
    config._attn_implementation = "auk_comfy"
    config.text_config._attn_implementation = "auk_comfy"
    config.audio_config._attn_implementation = "auk_comfy"
    config.vision_config._attn_implementation = "eager"
    with torch.device("meta"):
        model = QwenEncoder(config)
    for name, module in list(model.named_modules()):
        replacement = None
        if isinstance(module, nn.Linear):
            replacement = ops.Linear(module.in_features, module.out_features, bias=module.bias is not None, dtype=dtype, device="cpu")
        elif isinstance(module, nn.Embedding):
            replacement = comfy.ops.manual_cast.Embedding(module.num_embeddings, module.embedding_dim, padding_idx=module.padding_idx, dtype=dtype, device="cpu")
        elif isinstance(module, Qwen2_5OmniRMSNorm):
            replacement = comfy.ops.manual_cast.RMSNorm(module.weight.shape[0], eps=module.variance_epsilon, dtype=dtype, device="cpu")
        elif isinstance(module, nn.LayerNorm):
            replacement = comfy.ops.manual_cast.LayerNorm(module.normalized_shape, eps=module.eps, elementwise_affine=module.elementwise_affine, dtype=dtype, device="cpu")
        elif isinstance(module, nn.Conv1d):
            replacement = comfy.ops.manual_cast.Conv1d(module.in_channels, module.out_channels, module.kernel_size, stride=module.stride, padding=module.padding, dilation=module.dilation, groups=module.groups, bias=module.bias is not None, dtype=dtype, device="cpu")
        if replacement is not None:
            model.set_submodule(name, replacement)
    model.model.rotary_emb = Qwen2_5OmniRotaryEmbedding(config.text_config, device="cpu")
    model.audio_tower.positional_embedding = SinusoidsPositionEmbedding(config.audio_config.max_source_positions, config.audio_config.d_model)
    model.eval()
    return model


class AuKEncoder:
    def __init__(self, patcher, processor, dtype):
        self.patcher = patcher
        self.processor = processor
        self.dtype = dtype

    def encode(self, instruction, audio, layer_weights, layer_scale):
        content = [{"type": "text", "text": instruction if audio is not None else instruction + "|<no_prompt_audio>|"}]
        kwargs = {}
        if audio is not None:
            content.append({"type": "audio", "audio": "local-reference"})
            waveform = audio["waveform"].mean(1)
            rate = self.processor.feature_extractor.sampling_rate
            waveform = torchaudio.functional.resample(waveform, audio["sample_rate"], rate)
            kwargs["audio"] = [waveform[0].cpu().numpy()]
        text = self.processor.tokenizer.apply_chat_template([{"role": "user", "content": content}], chat_template=self.processor.chat_template, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], padding=True, return_tensors="pt", **kwargs)
        mm.load_models_gpu([self.patcher])
        device = self.patcher.load_device
        inputs = {key: value.to(device=device, dtype=self.dtype if value.is_floating_point() else value.dtype) for key, value in inputs.items()}
        states = self.patcher.model.encoder.encode_layers(inputs)
        weights = layer_weights.to(device=device, dtype=torch.float32).softmax(0)
        fused = None
        for weight, state in zip(weights, states):
            value = F.layer_norm(state, (state.shape[-1],)).float() * weight
            fused = value if fused is None else fused + value
        fused = fused * layer_scale.to(device=device, dtype=torch.float32)
        return fused.cpu(), inputs["attention_mask"].bool().cpu()

    def complete(self, system, content, max_new_tokens=256):
        model = self.patcher.model.encoder
        if not getattr(model, "has_lm_head", False):
            raise ValueError("This converted encoder lacks the language head; re-run tools/convert.py --component encoder on the original Qwen directory to enable the Prompt Enhance.")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        text = self.processor.tokenizer.apply_chat_template(messages, chat_template=self.processor.chat_template, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], padding=True, return_tensors="pt")
        mm.load_models_gpu([self.patcher])
        device = self.patcher.load_device
        inputs = {key: value.to(device=device, dtype=self.dtype if value.is_floating_point() else value.dtype) for key, value in inputs.items()}
        tokenizer = self.processor.tokenizer
        stop_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
        ids = model.complete(inputs, stop_ids, max_new_tokens)
        return tokenizer.decode(ids, skip_special_tokens=True).strip()
