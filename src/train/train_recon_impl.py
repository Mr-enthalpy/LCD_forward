import os
from typing import Dict, Optional

import torch

from src.forward.renderer import render_frames
from src.losses.losses import recon_loss, recon_metrics


def train_one_epoch_recon(
    forward_model,
    recon_model,
    loader,
    optimizer,
    device,
    recon_loss_cfg: Optional[Dict] = None,
    spectral_response: Optional[torch.Tensor] = None,
    noise_std: float = 0.0,
) -> Dict[str, float]:
    forward_model.eval()
    recon_model.train()
    recon_loss_cfg = recon_loss_cfg or {}

    total = {
        "loss": 0.0,
        "loss_l1": 0.0,
        "loss_mse": 0.0,
        "loss_tv": 0.0,
    }
    n_batches = 0

    for batch in loader:
        objects = batch["objects"].to(device)  # [B,L,H,W]
        masks = batch["masks"].to(device)      # [B,T,1,Hm,Wm]

        with torch.no_grad():
            forward_out = forward_model(masks)
            psfs = forward_out["psfs"]
            frames = render_frames(
                objects=objects,
                psfs=psfs,
                spectral_response=spectral_response,
                noise_std=noise_std,
                clip=False,
            )

        recon_out = recon_model(frames)
        pred_objects = recon_out["objects"]

        loss_dict = recon_loss(pred_objects, objects, **recon_loss_cfg)
        loss = loss_dict["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for k in total:
            total[k] += float(loss_dict[k].item())
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in total.items()}


@torch.no_grad()
def validate_recon(
    forward_model,
    recon_model,
    loader,
    device,
    recon_loss_cfg: Optional[Dict] = None,
    spectral_response: Optional[torch.Tensor] = None,
    noise_std: float = 0.0,
) -> Dict[str, float]:
    forward_model.eval()
    recon_model.eval()
    recon_loss_cfg = recon_loss_cfg or {}

    total = {
        "loss": 0.0,
        "loss_l1": 0.0,
        "loss_mse": 0.0,
        "loss_tv": 0.0,
        "mae": 0.0,
        "mse": 0.0,
        "psnr": 0.0,
    }
    n_batches = 0

    for batch in loader:
        objects = batch["objects"].to(device)
        masks = batch["masks"].to(device)

        forward_out = forward_model(masks)
        psfs = forward_out["psfs"]
        frames = render_frames(
            objects=objects,
            psfs=psfs,
            spectral_response=spectral_response,
            noise_std=noise_std,
            clip=False,
        )

        recon_out = recon_model(frames)
        pred_objects = recon_out["objects"]

        loss_dict = recon_loss(pred_objects, objects, **recon_loss_cfg)
        metric_dict = recon_metrics(pred_objects, objects)

        total["loss"] += float(loss_dict["loss"].item())
        total["loss_l1"] += float(loss_dict["loss_l1"].item())
        total["loss_mse"] += float(loss_dict["loss_mse"].item())
        total["loss_tv"] += float(loss_dict["loss_tv"].item())

        total["mae"] += float(metric_dict["mae"].item())
        total["mse"] += float(metric_dict["mse"].item())
        total["psnr"] += float(metric_dict["psnr"].item())
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in total.items()}


def fit_recon(
    forward_model,
    recon_model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs: int,
    output_dir: str,
    recon_loss_cfg: Optional[Dict] = None,
    spectral_response: Optional[torch.Tensor] = None,
    noise_std: float = 0.0,
    scheduler=None,
):
    os.makedirs(output_dir, exist_ok=True)
    best_val = float("inf")

    for epoch in range(epochs):
        train_metrics = train_one_epoch_recon(
            forward_model=forward_model,
            recon_model=recon_model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            recon_loss_cfg=recon_loss_cfg,
            spectral_response=spectral_response,
            noise_std=noise_std,
        )

        val_metrics = validate_recon(
            forward_model=forward_model,
            recon_model=recon_model,
            loader=val_loader,
            device=device,
            recon_loss_cfg=recon_loss_cfg,
            spectral_response=spectral_response,
            noise_std=noise_std,
        )

        if scheduler is not None:
            if hasattr(scheduler, "step"):
                scheduler.step()

        print(
            f"[Epoch {epoch + 1}] "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_psnr={val_metrics['psnr']:.4f}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            ckpt = {
                "epoch": epoch + 1,
                "model_state": recon_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": best_val,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }
            torch.save(ckpt, os.path.join(output_dir, "best_recon.pt"))