from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.forward.psf_basis import PSFBasisForward
from src.forward.complex_field_basis import ComplexFieldBasisForward


def test_psf_basis_forward_shape_and_normalization():
    model = PSFBasisForward(
        n_lambda=3,
        psf_size=32,
        rank=8,
        embed_dim=64,
        normalize_psf=True,
    )

    masks = torch.rand(2, 4, 1, 64, 64)  # [B,T,1,Hm,Wm]
    out = model(masks)

    assert isinstance(out, dict)
    assert "psfs" in out
    assert "coeff" in out

    psfs = out["psfs"]
    coeff = out["coeff"]

    assert psfs.shape == (2, 4, 3, 32, 32)
    assert coeff.shape == (2, 4, 3, 8)
    assert torch.isfinite(psfs).all()
    assert torch.isfinite(coeff).all()

    totals = psfs.sum(dim=(-2, -1))
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-4, rtol=1e-4)


def test_complex_field_basis_forward_shape_and_normalization():
    model = ComplexFieldBasisForward(
        n_lambda=2,
        psf_size=32,
        rank=8,
        embed_dim=64,
        normalize_psf=True,
        coeff_scale=1.0,
    )

    masks = torch.rand(2, 3, 1, 64, 64)
    out = model(masks)

    assert isinstance(out, dict)
    for key in ["psfs", "coeff_real", "coeff_imag", "field_real", "field_imag"]:
        assert key in out

    psfs = out["psfs"]
    coeff_real = out["coeff_real"]
    coeff_imag = out["coeff_imag"]

    assert psfs.shape == (2, 3, 2, 32, 32)
    assert coeff_real.shape == (2, 3, 2, 8)
    assert coeff_imag.shape == (2, 3, 2, 8)

    assert torch.isfinite(psfs).all()
    assert torch.isfinite(coeff_real).all()
    assert torch.isfinite(coeff_imag).all()

    totals = psfs.sum(dim=(-2, -1))
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-4, rtol=1e-4)


def test_forward_model_backward():
    model = ComplexFieldBasisForward(
        n_lambda=1,
        psf_size=16,
        rank=4,
        embed_dim=32,
        normalize_psf=True,
    )

    masks = torch.rand(2, 2, 1, 64, 64)
    out = model(masks)
    loss = out["psfs"].mean()
    loss.backward()

    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad