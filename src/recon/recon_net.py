import torch
import torch.nn as nn
from typing import Dict


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ReconNet(nn.Module):
    """
    Minimal reconstruction baseline.

    Input:
        frames: [B, T, 1, H, W]

    Output dict:
        {
            "objects": [B, L, H, W]
        }
    """

    def __init__(
        self,
        in_frames: int,
        out_lambda: int,
        hidden: int = 64,
        nonnegative_output: bool = False,
    ):
        super().__init__()
        self.in_frames = in_frames
        self.out_lambda = out_lambda
        self.nonnegative_output = nonnegative_output

        self.net = nn.Sequential(
            ConvBlock(in_frames, hidden),
            ConvBlock(hidden, hidden),
            nn.Conv2d(hidden, hidden // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden // 2, out_lambda, kernel_size=3, padding=1),
        )

    def forward(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        frames: [B, T, 1, H, W]
        """
        if frames.ndim != 5:
            raise ValueError(f"`frames` must be [B, T, 1, H, W], got {frames.shape}")

        bsz, num_frames, channels, height, width = frames.shape
        if channels != 1:
            raise ValueError(f"Expected single-channel frames, got {frames.shape}")
        if num_frames != self.in_frames:
            raise ValueError(
                f"Configured in_frames={self.in_frames}, but got frames with T={num_frames}"
            )

        x = frames.reshape(bsz, num_frames, height, width)  # [B, T, H, W]
        objects = self.net(x)  # [B, L, H, W]

        if self.nonnegative_output:
            objects = torch.clamp_min(objects, 0.0)

        return {
            "objects": objects,
        }