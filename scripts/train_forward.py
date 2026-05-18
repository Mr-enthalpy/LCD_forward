from __future__ import annotations

from pathlib import Path
import sys

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.h5_dataset import ForwardH5Dataset
from src.forward.complex_field_basis import ComplexFieldBasisForward
from src.forward.psf_basis import PSFBasisForward
from src.train.train_forward_impl import fit_forward
from src.utils.seed import set_seed


def load_yaml_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_forward_model(cfg: dict):
    model_name = cfg.get("model_name", "complex_field_basis")

    common_kwargs = dict(
        n_lambda=cfg["num_wavelengths"],
        psf_size=cfg["psf_size"],
        rank=cfg.get("rank", 16),
        embed_dim=cfg.get("embed_dim", 128),
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
    cfg = load_yaml_config(ROOT / "configs" / "forward_single_lambda.yaml")
    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = ForwardH5Dataset(ROOT / "data" / "sample" / "train.h5")
    val_set = ForwardH5Dataset(ROOT / "data" / "sample" / "val.h5")

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

    model = build_forward_model(cfg).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 0.0),
    )

    fit_forward(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=cfg["epochs"],
        output_dir=str(ROOT / "outputs" / "forward"),
        loss_cfg=cfg.get("loss", {}),
        scheduler=None,
    )


if __name__ == "__main__":
    main()