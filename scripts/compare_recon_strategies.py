from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.optic_handoff import load_psf_dictionary
from src.recon.linear_spectral_recon import (
    generate_synthetic_object,
    render_frames_fft,
    frequency_domain_ridge_reconstruct,
    single_frame_reconstruct,
    compute_recon_metrics,
)


def select_diverse_masks(data, count=9):
    masks_np = data["masks"]
    ids = data["mask_id"]
    families = data["mask_family"]
    n = data["n_samples"]

    family_order = ["deterministic", "task_related", "random_midfreq", "random_lowfreq"]
    seen = set()
    indices = []

    for fam in family_order:
        for i in range(n):
            if families[i] == fam and i not in seen:
                seen.add(i)
                indices.append(i)
                if len(indices) >= count:
                    break
        if len(indices) >= count:
            break

    while len(indices) < count:
        for i in range(n):
            if i not in seen:
                seen.add(i)
                indices.append(i)
                if len(indices) >= count:
                    break

    indices = indices[:count]
    return (
        torch.from_numpy(masks_np[indices]).float(),
        torch.from_numpy(data["psfs"][indices][:, 0]).float(),
        [ids[i] for i in indices],
        [families[i] for i in indices],
    )


def compute_psf_diversity_score(psfs_t):
    psfs_np = psfs_t[:, :, :, :].numpy()
    n_frames, n_wl, h, w = psfs_np.shape
    psf_fft = np.fft.fft2(psfs_np)

    scores = []
    for i in range(h):
        for j in range(w):
            H = psf_fft[:, :, i, j]
            HTH = H.conj().T @ H
            s = np.linalg.svdvals(HTH)
            scores.append(s[-1] / (s[0] + 1e-12))
    return float(np.mean(scores))


def main():
    train_h5 = r"D:\datasets\optic_system\phase3_release_20260520\lcd_forward\psf_dictionary\train.h5"
    psf_size = (128, 128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = load_psf_dictionary(train_h5, psf_working_size=psf_size)
    obj_np = generate_synthetic_object(n_channels=3, spatial_size=(128, 128), seed=42)
    obj = torch.from_numpy(obj_np).float().to(device)

    print("\n=== Mask Selection Strategies ===")
    strategies = {}

    s1_m, s1_p, s1_ids, s1_fam = select_diverse_masks(data, count=9)
    strategies["diverse_9"] = (s1_p.to(device), s1_ids)
    print(f"  diverse_9: {s1_ids}")

    s1_m, s1_p, s1_ids, s1_fam = select_diverse_masks(data, count=18)
    strategies["diverse_18"] = (s1_p.to(device), s1_ids)
    print(f"  diverse_18: {s1_ids[:5]}...")

    all_det = [i for i in range(data["n_samples"]) if data["mask_family"][i] == "deterministic"]
    det_ids = [data["mask_id"][i] for i in all_det]
    det_psfs = torch.from_numpy(data["psfs"][all_det][:, 0]).float().to(device)
    strategies["deterministic_8"] = (det_psfs, det_ids)
    print(f"  deterministic_8: {det_ids}")

    all_det_plus = all_det + [i for i in range(data["n_samples"]) if data["mask_family"][i] == "task_related"][:4]
    strategies["det_plus_task_12"] = (
        torch.from_numpy(data["psfs"][all_det_plus][:, 0]).float().to(device),
        [data["mask_id"][i] for i in all_det_plus],
    )
    print(f"  det_plus_task_12: {len(all_det_plus)} masks")

    print("\n=== Regularization Strategies ===")
    alpha_strategies = [
        ("none", 0.0, "none"),
        ("alpha_1e-6", 1e-6, "adaptive"),
        ("alpha_1e-4", 1e-4, "adaptive"),
        ("thresh_1e-6", 1e-6, "threshold"),
    ]

    results = []

    for strat_name, (psfs_t, mask_ids) in strategies.items():
        t_count = psfs_t.shape[0]
        if t_count < 3:
            continue

        frames = render_frames_fft(obj, psfs_t)

        for alpha_name, alpha_val, policy in alpha_strategies:
            try:
                recon_multi = frequency_domain_ridge_reconstruct(
                    frames, psfs_t, alpha=alpha_val, policy=policy
                )
                recon_single = single_frame_reconstruct(
                    frames, psfs_t, alpha=alpha_val, policy=policy
                )

                m_s = compute_recon_metrics(obj, recon_single)
                m_m = compute_recon_metrics(obj, recon_multi)

                gain = m_m["mean"]["psnr"] - m_s["mean"]["psnr"]

                results.append({
                    "strategy": strat_name,
                    "alpha_policy": alpha_name,
                    "T": t_count,
                    "single_psnr": m_s["mean"]["psnr"],
                    "multi_psnr": m_m["mean"]["psnr"],
                    "multi_gain": gain,
                    "single_corr": m_s["mean"]["correlation"],
                    "multi_corr": m_m["mean"]["correlation"],
                })
                print(f"  {strat_name:20s} | {alpha_name:22s} | single={m_s['mean']['psnr']:.2f} multi={m_m['mean']['psnr']:.2f} gain={gain:+.2f}dB")
            except Exception as e:
                print(f"  {strat_name:20s} | {alpha_name:22s} | FAILED: {e}")

    print("\n=== Best Combinations ===")
    results.sort(key=lambda r: r["multi_psnr"], reverse=True)
    print(f"{'Strategy':<22s} {'Policy':<24s} {'T':>4s} {'S-PSNR':>8s} {'M-PSNR':>8s} {'Gain':>8s} {'S-Corr':>8s} {'M-Corr':>8s}")
    print("-" * 90)
    for r in results[:15]:
        print(f"{r['strategy']:<22s} {r['alpha_policy']:<24s} {r['T']:>4d} {r['single_psnr']:>8.2f} {r['multi_psnr']:>8.2f} {r['multi_gain']:>+7.2f} {r['single_corr']:>8.4f} {r['multi_corr']:>8.4f}")

    print("\n=== Best Gain (multi - single) ===")
    results.sort(key=lambda r: r["multi_gain"], reverse=True)
    for r in results[:8]:
        print(f"  {r['strategy']:<22s} {r['alpha_policy']:<24s} gain={r['multi_gain']:+.4f}dB | S={r['single_psnr']:.2f} M={r['multi_psnr']:.2f}")


if __name__ == "__main__":
    main()
