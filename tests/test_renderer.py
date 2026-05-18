from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.forward.renderer import render_frames


def test_renderer_output_shape():
    objects = torch.rand(2, 3, 32, 32)        # [B,L,H,W]
    psfs = torch.rand(2, 4, 3, 9, 9)          # [B,T,L,Hp,Wp]
    psfs = psfs / (psfs.sum(dim=(-2, -1), keepdim=True) + 1e-8)

    frames = render_frames(objects, psfs)

    assert frames.shape == (2, 4, 1, 32, 32)
    assert torch.isfinite(frames).all()


def test_renderer_with_spectral_response():
    objects = torch.rand(1, 3, 16, 16)
    psfs = torch.rand(1, 2, 3, 7, 7)
    psfs = psfs / (psfs.sum(dim=(-2, -1), keepdim=True) + 1e-8)

    q = torch.tensor([1.0, 0.5, 0.25], dtype=objects.dtype)
    frames = render_frames(objects, psfs, spectral_response=q)

    assert frames.shape == (1, 2, 1, 16, 16)
    assert torch.isfinite(frames).all()


def test_renderer_delta_object_returns_embedded_psf():
    bsz, num_lambda, h, w = 1, 1, 17, 17
    hp, wp = 5, 5

    objects = torch.zeros(bsz, num_lambda, h, w)
    objects[0, 0, h // 2, w // 2] = 1.0

    psf = torch.rand(bsz, 1, num_lambda, hp, wp)
    psf = psf / (psf.sum(dim=(-2, -1), keepdim=True) + 1e-8)

    frames = render_frames(objects, psf)
    frame = frames[0, 0, 0]

    center_patch = frame[h // 2 - hp // 2 : h // 2 + hp // 2 + 1,
                         w // 2 - wp // 2 : w // 2 + wp // 2 + 1]

    assert center_patch.shape == (hp, wp)
    assert torch.isfinite(center_patch).all()
    assert torch.allclose(center_patch.sum(), torch.tensor(1.0), atol=1e-4, rtol=1e-4)