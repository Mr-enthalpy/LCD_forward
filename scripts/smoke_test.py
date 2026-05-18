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
from src.losses.losses import forward_loss, recon_loss
from src.recon.recon_net import ReconNet


def make_synthetic_h5(
    h5_path: Path,
    n: int = 8,
    t: int = 2,
    l: int = 1,
    hm: int = 64,
    wm: int = 64,
    hp: int = 17,
    wp: int = 17,
    h: int = 64,
    w: int = 64,
) -> None:
    rng = np.random.default_rng(123)

    masks = rng.random((n, t, 1, hm, wm), dtype=np.float32)

    # synthetic normalized PSFs
    psfs = rng.random((n, t, l, hp, wp), dtype=np.float32)
    psfs = psfs / (psfs.sum(axis=(-2, -1), keepdims=True) + 1e-8)

    # synthetic objects
    objects = rng.random((n, l, h, w), dtype=np.float32)

    wavelengths = np.linspace(550.0, 550.0 + 10.0 * (l - 1), l, dtype=np.float32)
    spectral_response = np.ones((l,), dtype=np.float32)

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("masks", data=masks)
        f.create_dataset("psfs", data=psfs)
        f.create_dataset("objects", data=objects)
        f.create_dataset("wavelengths", data=wavelengths)
        f.create_dataset("spectral_response", data=spectral_response)


def run_forward_smoke(device: torch.device, h5_path: Path) -> None:
    ds = ForwardH5Dataset(h5_path)
    loader = DataLoader(ds, batch_size=2, shuffle=False)

    batch = next(iter(loader))
    masks = batch["masks"].to(device)   # [B,T,1,Hm,Wm]
    gt_psfs = batch["psfs"].to(device)  # [B,T,L,Hp,Wp]

    model = ComplexFieldBasisForward(
        n_lambda=gt_psfs.shape[2],
        psf_size=gt_psfs.shape[-1],
        rank=8,
        embed_dim=64,
        normalize_psf=True,
        coeff_scale=1.0,
    ).to(device)

    out = model(masks)
    pred_psfs = out["psfs"]

    assert pred_psfs.shape == gt_psfs.shape, (
        f"Forward output shape mismatch: pred={pred_psfs.shape}, gt={gt_psfs.shape}"
    )
    assert torch.isfinite(pred_psfs).all(), "Non-finite values in predicted PSFs"

    loss_dict = forward_loss(pred_psfs, gt_psfs)
    loss = loss_dict["loss"]
    loss.backward()

    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad, "No gradients found in forward model"

    print(
        "[forward smoke] ok | "
        f"shape={tuple(pred_psfs.shape)} "
        f"loss={float(loss.item()):.6f}"
    )


def run_recon_smoke(device: torch.device, h5_path: Path) -> None:
    ds = ReconH5Dataset(h5_path)
    loader = DataLoader(ds, batch_size=2, shuffle=False)

    batch = next(iter(loader))
    objects = batch["objects"].to(device)  # [B,L,H,W]
    masks = batch["masks"].to(device)      # [B,T,1,Hm,Wm]

    num_lambda = objects.shape[1]
    num_frames = masks.shape[1]

    forward_model = PSFBasisForward(
        n_lambda=num_lambda,
        psf_size=17,
        rank=8,
        embed_dim=64,
        normalize_psf=True,
    ).to(device)

    forward_out = forward_model(masks)
    psfs = forward_out["psfs"]

    frames = render_frames(objects=objects, psfs=psfs)
    assert frames.shape == (objects.shape[0], num_frames, 1, objects.shape[-2], objects.shape[-1]), (
        f"Rendered frame shape mismatch: got={frames.shape}"
    )
    assert torch.isfinite(frames).all(), "Non-finite values in rendered frames"

    recon_model = ReconNet(
        in_frames=num_frames,
        out_lambda=num_lambda,
        hidden=32,
        nonnegative_output=False,
    ).to(device)

    recon_out = recon_model(frames)
    pred_objects = recon_out["objects"]

    assert pred_objects.shape == objects.shape, (
        f"Recon output shape mismatch: pred={pred_objects.shape}, gt={objects.shape}"
    )
    assert torch.isfinite(pred_objects).all(), "Non-finite values in reconstructed objects"

    loss_dict = recon_loss(pred_objects, objects)
    loss = loss_dict["loss"]
    loss.backward()

    has_grad = any(p.grad is not None for p in recon_model.parameters() if p.requires_grad)
    assert has_grad, "No gradients found in recon model"

    print(
        "[recon smoke] ok | "
        f"frames={tuple(frames.shape)} "
        f"objects={tuple(pred_objects.shape)} "
        f"loss={float(loss.item()):.6f}"
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = Path(tmpdir) / "synthetic_smoke.h5"
        make_synthetic_h5(h5_path)

        run_forward_smoke(device, h5_path)
        run_recon_smoke(device, h5_path)

    print("[smoke test] all checks passed.")


if __name__ == "__main__":
    main()