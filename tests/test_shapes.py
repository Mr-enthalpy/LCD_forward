from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.h5_dataset import ForwardH5Dataset, ReconH5Dataset
from src.forward.complex_field_basis import ComplexFieldBasisForward
from src.forward.psf_basis import PSFBasisForward
from src.forward.renderer import render_frames
from src.recon.recon_net import ReconNet


def _make_tiny_h5(
    path: Path,
    n: int = 6,
    t: int = 2,
    l: int = 1,
    hm: int = 64,
    wm: int = 64,
    hp: int = 17,
    wp: int = 17,
    h: int = 64,
    w: int = 64,
) -> None:
    rng = np.random.default_rng(0)

    masks = rng.random((n, t, 1, hm, wm), dtype=np.float32)
    psfs = rng.random((n, t, l, hp, wp), dtype=np.float32)
    psfs = psfs / (psfs.sum(axis=(-2, -1), keepdims=True) + 1e-8)
    objects = rng.random((n, l, h, w), dtype=np.float32)
    frames = rng.random((n, t, 1, h, w), dtype=np.float32)
    wavelengths = np.linspace(550.0, 550.0 + 10.0 * (l - 1), l, dtype=np.float32)
    spectral_response = np.ones((l,), dtype=np.float32)

    with h5py.File(path, "w") as f:
        f.create_dataset("masks", data=masks)
        f.create_dataset("psfs", data=psfs)
        f.create_dataset("objects", data=objects)
        f.create_dataset("frames", data=frames)
        f.create_dataset("wavelengths", data=wavelengths)
        f.create_dataset("spectral_response", data=spectral_response)


def test_forward_h5_dataset_shapes():
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = Path(tmpdir) / "tiny.h5"
        _make_tiny_h5(h5_path, n=5, t=3, l=2, hp=11, wp=11)

        ds = ForwardH5Dataset(h5_path)
        item = ds[0]

        assert "masks" in item
        assert "psfs" in item
        assert item["masks"].shape == (3, 1, 64, 64)
        assert item["psfs"].shape == (3, 2, 11, 11)

        loader = DataLoader(ds, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        assert batch["masks"].shape == (2, 3, 1, 64, 64)
        assert batch["psfs"].shape == (2, 3, 2, 11, 11)


def test_recon_h5_dataset_shapes():
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = Path(tmpdir) / "tiny.h5"
        _make_tiny_h5(h5_path, n=5, t=2, l=3)

        ds = ReconH5Dataset(h5_path)
        item = ds[0]

        assert "objects" in item
        assert "masks" in item
        assert "frames" in item
        assert item["objects"].shape == (3, 64, 64)
        assert item["masks"].shape == (2, 1, 64, 64)
        assert item["frames"].shape == (2, 1, 64, 64)

        loader = DataLoader(ds, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        assert batch["objects"].shape == (2, 3, 64, 64)
        assert batch["masks"].shape == (2, 2, 1, 64, 64)
        assert batch["frames"].shape == (2, 2, 1, 64, 64)


def test_forward_to_renderer_to_recon_shape_chain():
    batch_size = 2
    num_frames = 2
    num_lambda = 1
    mask_size = 64
    psf_size = 17
    image_size = 64

    masks = torch.rand(batch_size, num_frames, 1, mask_size, mask_size)
    objects = torch.rand(batch_size, num_lambda, image_size, image_size)

    forward_model = PSFBasisForward(
        n_lambda=num_lambda,
        psf_size=psf_size,
        rank=8,
        embed_dim=64,
        normalize_psf=True,
    )
    forward_out = forward_model(masks)
    psfs = forward_out["psfs"]

    assert psfs.shape == (batch_size, num_frames, num_lambda, psf_size, psf_size)

    frames = render_frames(objects, psfs)
    assert frames.shape == (batch_size, num_frames, 1, image_size, image_size)

    recon_model = ReconNet(
        in_frames=num_frames,
        out_lambda=num_lambda,
        hidden=32,
        nonnegative_output=False,
    )
    recon_out = recon_model(frames)
    pred_objects = recon_out["objects"]

    assert pred_objects.shape == (batch_size, num_lambda, image_size, image_size)


def test_complex_field_forward_chain_shape():
    batch_size = 2
    num_frames = 3
    num_lambda = 2
    mask_size = 64
    psf_size = 21

    masks = torch.rand(batch_size, num_frames, 1, mask_size, mask_size)

    model = ComplexFieldBasisForward(
        n_lambda=num_lambda,
        psf_size=psf_size,
        rank=8,
        embed_dim=64,
        normalize_psf=True,
        coeff_scale=1.0,
    )
    out = model(masks)

    assert out["psfs"].shape == (batch_size, num_frames, num_lambda, psf_size, psf_size)
    assert out["coeff_real"].shape == (batch_size, num_frames, num_lambda, 8)
    assert out["coeff_imag"].shape == (batch_size, num_frames, num_lambda, 8)
    assert out["field_real"].shape == (batch_size, num_frames, num_lambda, psf_size, psf_size)
    assert out["field_imag"].shape == (batch_size, num_frames, num_lambda, psf_size, psf_size)