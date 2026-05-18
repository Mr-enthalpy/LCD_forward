from __future__ import annotations

from pathlib import Path
import sys

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.h5_dataset import ReconH5Dataset
from src.forward.complex_field_basis import ComplexFieldBasisForward
from src.forward.psf_basis import PSFBasisForward
from src.recon.recon_net import ReconNet
from src.train.train_recon_impl import fit_recon
from src.utils.seed import set_seed


def load_yaml_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_forward_model(cfg: dict):
    model_name = cfg.get("forward_model_name", "complex_field_basis")

    common_kwargs = dict(
        n_lambda=cfg["num_wavelengths"],
        psf_size=cfg["psf_size"],
        rank=cfg.get("forward_rank", 16),
        embed_dim=cfg.get("forward_embed_dim", 128),
        normalize_psf=cfg.get("normalize_psf", True),
    )

    if model_name == "complex_field_basis":
        return ComplexFieldBasisForward(
            **common_kwargs,
            coeff_scale=cfg.get("coeff_scale", None),
        )
    elif model_name == "psf_basis":
        return PSFBasisForward(**common_kwargs)
    else:
        raise ValueError(f"Unknown forward model: {model_name}")


def main() -> None:
    cfg = load_yaml_config(ROOT / "configs" / "recon_single_lambda.yaml")
    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = ReconH5Dataset(ROOT / "data" / "sample" / "train.h5")
    val_set = ReconH5Dataset(ROOT / "data" / "sample" / "val.h5")

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 0),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 0),
    )

    forward_model = build_forward_model(cfg).to(device)

    forward_ckpt = cfg.get("forward_checkpoint", None)
    if forward_ckpt:
        ckpt = torch.load(forward_ckpt, map_location=device)
        forward_model.load_state_dict(ckpt["model_state"])

    recon_model = ReconNet(
        in_frames=cfg["num_frames"],
        out_lambda=cfg["num_wavelengths"],
        hidden=cfg.get("recon_hidden", 64),
        nonnegative_output=cfg.get("nonnegative_output", False),
    ).to(device)

    optimizer = torch.optim.Adam(
        recon_model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 0.0),
    )

    spectral_response = cfg.get("spectral_response", None)
    if spectral_response is not None:
        spectral_response = torch.tensor(spectral_response, dtype=torch.float32, device=device)

    fit_recon(
        forward_model=forward_model,
        recon_model=recon_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=cfg["epochs"],
        output_dir=str(ROOT / "outputs" / "recon"),
        recon_loss_cfg=cfg.get("loss", {}),
        spectral_response=spectral_response,
        noise_std=cfg.get("noise_std", 0.0),
        scheduler=None,
    )


if __name__ == "__main__":
    main()