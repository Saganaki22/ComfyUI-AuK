"""Inference-only AuK BigVGAN codec; weight normalization is folded on load."""
import torch
from torch import nn
import torch.nn.functional as F

import comfy.ops
from comfy.ldm.mmaudio.vae.activations import SnakeBeta
from comfy.ldm.mmaudio.vae.alias_free_torch import UpSample1d as CoreUpSample1d


class CausalConv1d(comfy.ops.manual_cast.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, dilation=dilation, bias=bias)
        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, x):
        return super().forward(F.pad(x, (self.left_padding, 0)))


class CausalConvTranspose1d(comfy.ops.manual_cast.ConvTranspose1d):
    def forward(self, x, output_size=None):
        return super().forward(x, output_size)[..., :-self.stride[0]]


class UpSample1d(CoreUpSample1d):
    def __init__(self):
        nn.Module.__init__(self)
        self.ratio = self.stride = 2
        self.pad = 5
        self.pad_left = self.pad_right = 15
        self.register_buffer("filter", torch.empty(1, 1, 12))


class LowPassFilter1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("filter", torch.empty(1, 1, 12))

    def forward(self, x):
        filt = comfy.ops.cast_to_input(self.filter, x).expand(x.shape[1], -1, -1)
        return F.conv1d(F.pad(x, (11, 0), mode="replicate"), filt, stride=2, groups=x.shape[1])


class DownSample1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.lowpass = LowPassFilter1d()

    def forward(self, x):
        return self.lowpass(x)


class Activation1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.act = SnakeBeta(channels, alpha_logscale=True)
        self.upsample = UpSample1d()
        self.downsample = DownSample1d()

    def forward(self, x):
        return self.downsample(self.act(self.upsample(x)))


class Conv1dS(nn.Module):
    def __init__(self, inputs, outputs, kernel, stride=1):
        super().__init__()
        self.layer = comfy.ops.manual_cast.Conv1d(inputs, outputs, kernel, stride=stride, padding=(kernel - 1) // 2)

    def forward(self, x):
        return self.layer(x)


class ResStack(nn.Module):
    def __init__(self, channels):
        super().__init__()
        ops = comfy.ops.manual_cast
        self.layers = nn.ModuleList([nn.Sequential(nn.LeakyReLU(), ops.Conv1d(channels, channels, 3, dilation=2**i, padding=2**i), nn.LeakyReLU(), ops.Conv1d(channels, channels, 3, padding=1)) for i in range(6)])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [12, 24, 48, 96, 192, 384, 768]
        layers = [Conv1dS(1, 12, 3), nn.LeakyReLU(0.2)]
        for inputs, outputs, factor in zip(channels[:-1], channels[1:], [2, 2, 2, 3, 4, 5]):
            layers.extend([Conv1dS(inputs, outputs, factor * 2, factor), ResStack(outputs), nn.LeakyReLU(0.2)])
        layers.append(Conv1dS(768, 128, 3))
        self.generator = nn.Sequential(*layers)

    def forward(self, x):
        return self.generator(x)


class AMPBlock(nn.Module):
    def __init__(self, channels, kernel):
        super().__init__()
        self.convs1 = nn.ModuleList([CausalConv1d(channels, channels, kernel, dilation=d) for d in [1, 3, 5]])
        self.convs2 = nn.ModuleList([CausalConv1d(channels, channels, kernel) for _ in range(3)])
        self.activations = nn.ModuleList([Activation1d(channels) for _ in range(6)])

    def forward(self, x):
        for i, (c1, c2) in enumerate(zip(self.convs1, self.convs2)):
            x = x + c2(self.activations[2 * i + 1](c1(self.activations[2 * i](x))))
        return x


class BigVGANFlowVAE(nn.Module):
    def __init__(self):
        super().__init__()
        ops = comfy.ops.manual_cast
        self.register_buffer("global_mean", torch.empty(64))
        self.register_buffer("global_log_std", torch.empty(64))
        self.audio_encoder = Encoder()
        self.conv_pre = ops.Conv1d(64, 1536, 7, padding=3)
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        for i, (rate, kernel) in enumerate(zip([5, 4, 3, 2, 2, 2], [10, 8, 6, 4, 4, 4])):
            inputs, outputs = 1536 // 2**i, 1536 // 2**(i + 1)
            self.ups.append(nn.ModuleList([CausalConvTranspose1d(inputs, outputs, kernel, stride=rate)]))
            self.resblocks.extend([AMPBlock(outputs, k) for k in [3, 7, 11]])
        self.activation_post = Activation1d(24)
        self.conv_post = CausalConv1d(24, 1, 7, bias=False)

    def encode(self, waveform, generator=None):
        mean, log_std = self.audio_encoder(waveform).chunk(2, 1)
        noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        latent = (mean + noise * log_std.exp()).transpose(1, 2)
        center = comfy.ops.cast_to_input(self.global_mean, latent)
        variance = comfy.ops.cast_to_input(self.global_log_std, latent)
        return ((latent - center) / variance.sqrt())[:, :waveform.shape[-1] // 480].clone()

    def decode(self, latent):
        center = comfy.ops.cast_to_input(self.global_mean, latent)
        variance = comfy.ops.cast_to_input(self.global_log_std, latent)
        x = self.conv_pre((latent * variance.sqrt() + center).transpose(1, 2))
        for i, up in enumerate(self.ups):
            x = up[0](x)
            x = sum(self.resblocks[i * 3 + j](x) for j in range(3)) / 3
        return self.conv_post(self.activation_post(x)).clamp(-1, 1)


def fold_weight_norm(state):
    state = {key: value for key, value in state.items() if not key.startswith("flow.")}
    for key in list(state):
        if key.endswith("weight_g"):
            prefix = key[:-8]
            gain, value = state.pop(key), state.pop(prefix + "weight_v")
            state[prefix + "weight"] = torch._weight_norm(value, gain, 0)
    return state
