import torch
import torch.nn as nn
from torch.nn import functional as F

class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        padding: int = 2,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=0.1),
        )

    def forward(self, waveform):
        return self.decoder(waveform)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 15,
        padding: int = 7,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=0.1),
        )

    def forward(self, waveform):
        return self.encoder(waveform)


class Wavenet(nn.Module):
    def __init__(
        self,
        num_channels: int = 1,
        depth: int = 12,
        initial_output_channels: int = 24,
    ):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        out_channels = initial_output_channels
        for layer in range(depth):
            encoder = Encoder(num_channels, out_channels)
            self.encoders.append(encoder)

            num_channels = out_channels
            out_channels += initial_output_channels
            if layer == depth - 1:
                decoder = Decoder(
                    depth * initial_output_channels + num_channels, num_channels
                )
            else:
                decoder = Decoder(num_channels + out_channels, num_channels)

            self.decoders.insert(0, decoder)

        bottleneck_dim = depth * initial_output_channels
        self.bottleneck = nn.Sequential(
            nn.Conv1d(bottleneck_dim, bottleneck_dim, 15, stride=1, padding=7),
            nn.BatchNorm1d(bottleneck_dim),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.final = nn.Sequential(
            nn.Conv1d(1 + initial_output_channels, 1, kernel_size=1, stride=1),
            nn.Tanh(),
        )

    def forward(self, waveform):
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        if waveform.size(1) != 1:
            raise TypeError(
                f"mono channel audio expected, input has {waveform.size(1)} channels"
            )

        encoder_outputs = []
        out = waveform
        for encoder in self.encoders:
            out = encoder(out)
            encoder_outputs.insert(0, out)
            out = out[:, :, ::2]

        out = self.bottleneck(out)

        for layer, decoder in enumerate(self.decoders):
            out = F.interpolate(out, scale_factor=2, mode="linear")
            out = self.fix_last_dim(out, encoder_outputs[layer])
            out = torch.cat([out, encoder_outputs[layer]], dim=1)
            out = decoder(out)

        out = torch.cat([out, waveform], dim=1)
        out = self.final(out)
        return out

    def fix_last_dim(self, x, target):
        assert (
            x.shape[-1] >= target.shape[-1]
        ), "input dimension cannot be larger than target dimension"
        if x.shape[-1] == target.shape[-1]:
            return x

        diff = x.shape[-1] - target.shape[-1]
        if diff % 2 != 0:
            x = F.pad(x, (0, 1))
            diff += 1

        crop = diff // 2
        return x[:, :, crop:-crop]