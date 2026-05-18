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
from src.train.train_forward_impl import validate_forward
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

    val_set = ForwardH5Dataset(ROOT / "data" / "sample" / "val.h5")
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 0),
    )

    model = build_forward_model(cfg).to(device)

    ckpt_path = cfg.get("forward_checkpoint", None)
    default_ckpt = ROOT / "outputs" / "forward" / "best_forward.pt"
    if ckpt_path is None and default_ckpt.exists():
        ckpt_path = default_ckpt

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
    else:
        print("[eval_forward] warning: no checkpoint found, evaluating randomly initialized model.")

    metrics = validate_forward(
        model=model,
        loader=val_loader,
        device=device,
        loss_cfg=cfg.get("loss", {}),
    )

    print(
        {
            "forward_eval_loss": metrics["loss"],
            "forward_eval_l1": metrics["loss_l1"],
            "forward_eval_mse": metrics["loss_mse"],
            "forward_eval_fft": metrics["loss_fft"],
            "forward_eval_energy": metrics["loss_energy"],
            "forward_eval_centroid": metrics["loss_centroid"],
        }
    )


if __name__ == "__main__":
    main()