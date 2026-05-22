from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.optic_handoff import load_psf_dictionary
from scripts.compare_recon_strategies import select_diverse_masks
from src.recon.linear_spectral_recon import (
    generate_synthetic_object,
    render_frames_fft,
    frequency_domain_ridge_reconstruct,
    single_frame_reconstruct,
    compute_recon_metrics,
)
from src.utils.figures import (
    plot_synthetic_objects, plot_mask_grid, plot_rendered_frames,
    plot_reconstruction, plot_recon_comparison, plot_recon_rgb_pseudocolor_comparison,
)

ALPHA = 5e-15
POLICY = "adaptive"
H5_PATH = r"D:\datasets\optic_system\phase3_release_20260520\lcd_forward\psf_dictionary\train.h5"
HANDOFF_SYN_DIR = Path(
    r"D:\datasets\LCD_forward\lcd_forward_phase3_5_3_6_release_20260520"
    r"\thesis\phase3_6_linear_recon_synthetic"
)

psf_working_size = (256, 256)
obj_size = (256, 256)
n_masks = 12
wavelengths = np.array([450.0, 550.0, 650.0], dtype=np.float32)

print("Loading PSF dictionary...")
psf_data = load_psf_dictionary(H5_PATH, psf_working_size=psf_working_size)

print(f"Selecting {n_masks} masks (diverse_family_first)...")
sel_masks_t, sel_psfs_t, sel_ids, sel_families = select_diverse_masks(psf_data, count=n_masks)
sel_masks_display = np.array([psf_data["masks"][psf_data["mask_id"].index(mid)] for mid in sel_ids])[:, 0, 0, :, :]

print("Generating synthetic object...")
obj_np = generate_synthetic_object(n_channels=3, spatial_size=obj_size, seed=42)
obj = torch.from_numpy(obj_np).float()

psfs_t = sel_psfs_t.float()

print("Rendering frames...")
frames = render_frames_fft(obj, psfs_t)
frames_np = frames.numpy()

print("Reconstructing single-frame...")
recon_single = single_frame_reconstruct(frames, psfs_t, alpha=ALPHA, policy=POLICY)
recon_single_np = recon_single.numpy()

print("Reconstructing multi-frame...")
recon_multi = frequency_domain_ridge_reconstruct(frames, psfs_t, alpha=ALPHA, policy=POLICY)
recon_multi_np = recon_multi.numpy()

print("Computing metrics...")
metrics_single = compute_recon_metrics(obj, recon_single)
metrics_multi = compute_recon_metrics(obj, recon_multi)

out_dir = Path(HANDOFF_SYN_DIR)

print("Saving figures...")
plot_synthetic_objects(obj_np, out_dir / "figures" / "synthetic_objects.png")
plot_mask_grid(sel_masks_display, sel_ids, out_dir / "figures" / "selected_masks.png",
               title="Selected Encoding Masks")
plot_rendered_frames(frames_np, out_dir / "figures" / "rendered_frames.png")
plot_reconstruction(obj_np, recon_single_np, out_dir / "figures" / "recon_single_frame.png",
                    wavelengths_nm=wavelengths, title="Single-Frame Reconstruction")
plot_reconstruction(obj_np, recon_multi_np, out_dir / "figures" / "recon_multiframe.png",
                    wavelengths_nm=wavelengths, title="Multi-Frame Reconstruction")
plot_recon_comparison(obj_np, recon_single_np, recon_multi_np,
                      out_dir / "figures" / "recon_comparison.png", wavelengths_nm=wavelengths)
plot_recon_comparison(obj_np, recon_single_np, recon_multi_np,
                      out_dir / "figures" / "recon_per_band_comparison.png", wavelengths_nm=wavelengths)
plot_recon_rgb_pseudocolor_comparison(
    obj_np, recon_single_np, recon_multi_np,
    out_dir / "figures" / "recon_rgb_pseudocolor_comparison.png",
    wavelengths_nm=wavelengths,
)

print("Saving NPZ...")
np.savez_compressed(
    out_dir / "data" / "recon_appendix_arrays.npz",
    gt_object=obj_np.astype(np.float32),
    recon_single=recon_single_np.astype(np.float32),
    recon_multi=recon_multi_np.astype(np.float32),
    rendered_frames=frames_np.astype(np.float32),
    wavelengths_nm=wavelengths.astype(np.float32),
    selected_mask_ids=np.asarray(sel_ids, dtype="U"),
    mask_families=np.asarray(sel_families, dtype="U"),
    source_psf_h5=np.asarray([H5_PATH], dtype="U"),
)

results = {
    "object_size": list(obj_size),
    "n_masks": n_masks,
    "alpha": ALPHA,
    "single_frame_metrics": metrics_single,
    "multi_frame_metrics": metrics_multi,
    "selected_masks": sel_ids,
    "mask_families": sel_families,
}

with open(out_dir / "metrics" / "reconstruction_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Single: PSNR={metrics_single['mean']['psnr']:.2f}, "
      f"SSIM_display={metrics_single['mean']['ssim_display']:.4f}")
print(f"Multi:  PSNR={metrics_multi['mean']['psnr']:.2f}, "
      f"SSIM_display={metrics_multi['mean']['ssim_display']:.4f}")
print("Done.")
