from __future__ import annotations

from pathlib import Path

import torch


def normalize_image(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().float()
    min_value = tensor.amin()
    max_value = tensor.amax()
    scale = (max_value - min_value).clamp_min(1e-8)
    return (tensor - min_value) / scale


def summarize_tensor(tensor: torch.Tensor) -> dict[str, object]:
    return {"shape": tuple(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device)}


def save_tensor(path: str | Path, tensor: torch.Tensor) -> None:
    Path(path).write_text(str(summarize_tensor(tensor)), encoding="utf-8")

