import os
from typing import Dict, Optional

import torch

from src.losses.losses import forward_loss


def train_one_epoch_forward(
    model,
    loader,
    optimizer,
    device,
    loss_cfg: Optional[Dict] = None,
) -> Dict[str, float]:
    model.train()
    loss_cfg = loss_cfg or {}

    total = {
        "loss": 0.0,
        "loss_l1": 0.0,
        "loss_mse": 0.0,
        "loss_fft": 0.0,
        "loss_energy": 0.0,
        "loss_centroid": 0.0,
    }
    n_batches = 0

    for batch in loader:
        masks = batch["masks"].to(device)   # [B,T,1,Hm,Wm]
        gt_psfs = batch["psfs"].to(device)  # [B,T,L,Hp,Wp]

        out = model(masks)
        pred_psfs = out["psfs"]

        loss_dict = forward_loss(pred_psfs, gt_psfs, **loss_cfg)
        loss = loss_dict["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for k in total:
            total[k] += float(loss_dict[k].item())
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in total.items()}


@torch.no_grad()
def validate_forward(
    model,
    loader,
    device,
    loss_cfg: Optional[Dict] = None,
) -> Dict[str, float]:
    model.eval()
    loss_cfg = loss_cfg or {}

    total = {
        "loss": 0.0,
        "loss_l1": 0.0,
        "loss_mse": 0.0,
        "loss_fft": 0.0,
        "loss_energy": 0.0,
        "loss_centroid": 0.0,
    }
    n_batches = 0

    for batch in loader:
        masks = batch["masks"].to(device)
        gt_psfs = batch["psfs"].to(device)

        out = model(masks)
        pred_psfs = out["psfs"]

        loss_dict = forward_loss(pred_psfs, gt_psfs, **loss_cfg)

        for k in total:
            total[k] += float(loss_dict[k].item())
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in total.items()}


def fit_forward(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs: int,
    output_dir: str,
    loss_cfg: Optional[Dict] = None,
    scheduler=None,
):
    os.makedirs(output_dir, exist_ok=True)
    best_val = float("inf")

    for epoch in range(epochs):
        train_metrics = train_one_epoch_forward(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_cfg=loss_cfg,
        )
        val_metrics = validate_forward(
            model=model,
            loader=val_loader,
            device=device,
            loss_cfg=loss_cfg,
        )

        if scheduler is not None:
            if hasattr(scheduler, "step"):
                scheduler.step()

        print(
            f"[Epoch {epoch + 1}] "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_l1={val_metrics['loss_l1']:.6f} "
            f"val_fft={val_metrics['loss_fft']:.6f}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            ckpt = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": best_val,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }
            torch.save(ckpt, os.path.join(output_dir, "best_forward.pt"))