import torch
import torch.nn.functional as F
from typing import Optional


def conv2d_same_per_sample(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Per-sample spatial convolution.

    Args:
        x: [B, 1, H, W]
        k: [B, 1, Kh, Kw]

    Returns:
        y: [B, 1, H, W]
    """
    if x.ndim != 4 or k.ndim != 4:
        raise ValueError(f"Expected 4D tensors, got x={x.shape}, k={k.shape}")
    if x.shape[0] != k.shape[0]:
        raise ValueError(f"Batch mismatch: x={x.shape}, k={k.shape}")
    if x.shape[1] != 1 or k.shape[1] != 1:
        raise ValueError(f"Expected single-channel inputs, got x={x.shape}, k={k.shape}")

    kh, kw = k.shape[-2:]
    pad_h = kh // 2
    pad_w = kw // 2

    outs = []
    for i in range(x.shape[0]):
        yi = F.conv2d(
            x[i:i + 1],
            k[i:i + 1],
            padding=(pad_h, pad_w),
        )
        outs.append(yi)

    return torch.cat(outs, dim=0)


def render_frames(
    objects: torch.Tensor,
    psfs: torch.Tensor,
    spectral_response: Optional[torch.Tensor] = None,
    noise_std: float = 0.0,
    clip: bool = False,
) -> torch.Tensor:
    """
    Non-coherent image formation:
        I_t = sum_lambda Q(lambda) * (O_lambda * PSF_{t,lambda}) + n_t

    Args:
        objects: [B, L, H, W]
        psfs: [B, T, L, Hp, Wp]
        spectral_response: [L] or None
        noise_std: additive Gaussian noise std
        clip: whether to clip output to nonnegative

    Returns:
        frames: [B, T, 1, H, W]
    """
    if objects.ndim != 4:
        raise ValueError(f"`objects` must be [B, L, H, W], got {objects.shape}")
    if psfs.ndim != 5:
        raise ValueError(f"`psfs` must be [B, T, L, Hp, Wp], got {psfs.shape}")

    bsz, num_lambda, height, width = objects.shape
    bsz2, num_frames, num_lambda2, _, _ = psfs.shape

    if bsz != bsz2:
        raise ValueError(f"Batch mismatch: objects={objects.shape}, psfs={psfs.shape}")
    if num_lambda != num_lambda2:
        raise ValueError(f"Lambda mismatch: objects={objects.shape}, psfs={psfs.shape}")

    if spectral_response is None:
        spectral_response = torch.ones(
            num_lambda,
            dtype=objects.dtype,
            device=objects.device,
        )
    else:
        if spectral_response.ndim != 1 or spectral_response.shape[0] != num_lambda:
            raise ValueError(
                f"`spectral_response` must be [L], got {spectral_response.shape}"
            )
        spectral_response = spectral_response.to(dtype=objects.dtype, device=objects.device)

    frames = []
    for t in range(num_frames):
        yt = torch.zeros(
            (bsz, 1, height, width),
            dtype=objects.dtype,
            device=objects.device,
        )

        for l in range(num_lambda):
            obj_l = objects[:, l:l + 1]         # [B, 1, H, W]
            psf_l = psfs[:, t, l:l + 1]         # [B, 1, Hp, Wp]
            yt = yt + spectral_response[l] * conv2d_same_per_sample(obj_l, psf_l)

        frames.append(yt)

    frames = torch.stack(frames, dim=1)        # [B, T, 1, H, W]

    if noise_std > 0.0:
        frames = frames + noise_std * torch.randn_like(frames)

    if clip:
        frames = frames.clamp_min(0.0)

    return frames