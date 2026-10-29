"""CycleGAN ResNet generator matching the official PyTorch model zoo."""

from __future__ import annotations

import torch
from torch import nn


class ResnetBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(channels, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(channels, affine=False, track_running_stats=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.conv_block(value)


class ResnetGenerator(nn.Module):
    """9-block, 256px ResNet generator used by the official CycleGAN weights."""

    def __init__(self, input_channels: int = 3, output_channels: int = 3,
                 base_channels: int = 64, residual_blocks: int = 9) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, base_channels, kernel_size=7, padding=0, bias=True),
            nn.InstanceNorm2d(base_channels, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
        ]
        current = base_channels
        for _ in range(2):
            layers.extend([
                nn.Conv2d(current, current * 2, kernel_size=3, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(current * 2, affine=False, track_running_stats=True),
                nn.ReLU(inplace=True),
            ])
            current *= 2
        layers.extend(ResnetBlock(current) for _ in range(residual_blocks))
        for _ in range(2):
            layers.extend([
                nn.ConvTranspose2d(current, current // 2, kernel_size=3, stride=2,
                                   padding=1, output_padding=1, bias=True),
                nn.InstanceNorm2d(current // 2, affine=False, track_running_stats=True),
                nn.ReLU(inplace=True),
            ])
            current //= 2
        layers.extend([
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_channels, output_channels, kernel_size=7, padding=0),
            nn.Tanh(),
        ])
        self.model = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.model(value)
