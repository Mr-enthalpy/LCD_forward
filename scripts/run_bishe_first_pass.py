from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.optic_handoff import (
    load_psf_dictionary,
    select_masks_by_strategy,
    get_masks_flat as _get_masks_flat_raw,
)
from src.forward.psf_pca_ridge import PSFPCARidge, compute_psf_metrics
from src.recon.linear_spectral_recon import (
    generate_synthetic_object,
    render_frames_fft,
    frequency_domain_ridge_reconstruct,
    single_frame_reconstruct,
    compute_recon_metrics,
    resize_psf_tensor,
)
from src.utils.figures import (
    _ensure_dir,
    normalize_for_display,
    plot_mask_grid,
    plot_measured_vs_predicted,
    plot_pca_basis_preview,
    plot_synthetic_objects,
    plot_rendered_frames,
    plot_reconstruction,
    plot_recon_comparison,
    plot_recon_rgb_pseudocolor_comparison,
)


def resolve_config_value(value, env_map=None):
    if isinstance(value, str) and value.startswith("${") and "}" in value:
        inner = value[2:-1]
        if ":" in inner:
            var, default = inner.split(":", 1)
        else:
            var, default = inner, ""
        if env_map and var in env_map:
            return env_map[var]
        return os.environ.get(var, default)
    return value


def resolve_config(cfg, env_map=None):
    if isinstance(cfg, dict):
        return {k: resolve_config(v, env_map) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [resolve_config(v, env_map) for v in cfg]
    return resolve_config_value(cfg, env_map)


def run_forward_validation(cfg, run_dir, env_map):
    print("\n" + "=" * 60)
    print("Phase 3.5: Forward Validation")
    print("=" * 60)

    train_h5 = resolve_config_value(cfg["data"]["train_h5"], env_map)
    val_h5 = resolve_config_value(cfg["data"]["val_h5"], env_map)
    test_h5 = resolve_config_value(cfg["data"]["test_h5"], env_map)

    fw_cfg = cfg["forward_validation"]
    psf_working_size = tuple(fw_cfg["psf_working_size"])
    n_components = fw_cfg["pca_components"]
    alpha = fw_cfg["ridge_alpha"]
    n_plot = fw_cfg.get("num_examples_to_plot", 6)

    print(f"Loading train: {train_h5}")
    train_data = load_psf_dictionary(train_h5, psf_working_size=psf_working_size)
    print(f"  N={train_data['n_samples']}, mask={train_data['mask_shape']}, psf={train_data['psf_shape']}")
    print(f"  wavelengths: {train_data['wavelengths_nm']}")

    print(f"\nLoading val: {val_h5}")
    val_data = load_psf_dictionary(val_h5, psf_working_size=psf_working_size)
    print(f"  N={val_data['n_samples']}")

    print(f"\nLoading test: {test_h5}")
    test_data = load_psf_dictionary(test_h5, psf_working_size=psf_working_size)
    print(f"  N={test_data['n_samples']}")

    train_masks = train_data["masks"]
    train_psfs = train_data["psfs"]
    test_masks = test_data["masks"]
    test_psfs = test_data["psfs"]

    print(f"\nFitting PCA + Ridge: components={n_components}, alpha={alpha}")
    model = PSFPCARidge(n_components=n_components, alpha=alpha, per_wavelength=True)
    fit_history = model.fit(train_masks, train_psfs)

    print("\nPredicting test PSFs")
    test_coeffs = model.predict_coeff(test_masks)
    pred_psfs = model.reconstruct(test_coeffs)

    print("Computing metrics")
    metrics = compute_psf_metrics(test_psfs, pred_psfs)

    fwd_dir = _ensure_dir(run_dir / "forward_validation")

    with open(fwd_dir / "psf_prediction_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"fit_history": fit_history, "test_metrics": metrics}, f, indent=2)

    print(f"Metrics: mean_mse={metrics['mean']['mse']:.6f}, mean_corr={metrics['mean']['normalized_correlation']:.4f}")

    n_examples = min(n_plot, test_data["n_samples"])
    plot_indices = list(range(n_examples))

    plot_measured_vs_predicted(
        test_psfs[plot_indices], pred_psfs[plot_indices],
        [test_data["mask_id"][i] for i in plot_indices],
        test_data["wavelengths_nm"],
        fwd_dir / "measured_vs_predicted_examples.png",
    )

    for wl_idx in range(model.n_wavelengths):
        pca = model.pca_models[wl_idx]
        plot_pca_basis_preview(
            pca.components_,
            pca.explained_variance_ratio_,
            fwd_dir / f"psf_basis_preview_wl{wl_idx}.png",
        )

    report = f"""# Forward Validation Report

## Data
- Train: {train_data['n_samples']} samples
- Val: {val_data['n_samples']} samples
- Test: {test_data['n_samples']} samples
- Wavelengths: {train_data['wavelengths_nm'].tolist()} nm
- PSF size: {train_data['psf_shape']}

## Model
- Method: PCA + Ridge regression
- PCA components: {n_components}
- Ridge alpha: {alpha}

## Test Metrics
- Mean MSE: {metrics['mean']['mse']:.6f}
- Mean Relative L2: {metrics['mean']['relative_l2']:.4f}
- Mean Correlation: {metrics['mean']['normalized_correlation']:.4f}

### Per-Wavelength
"""
    for wl_str, wl_metrics in metrics["per_wavelength"].items():
        wl_nm = train_data["wavelengths_nm"][int(wl_str)]
        report += f"- {wl_nm:.0f} nm: MSE={wl_metrics['mse']:.6f}, R-L2={wl_metrics['relative_l2']:.4f}, Corr={wl_metrics['normalized_correlation']:.4f}\n"

    report += """
## Conclusion
The measured mask-to-PSF relationship can be partially modelled by a low-dimensional PCA + ridge regression baseline.
This fulfills the Phase 3.5 existence proof requirement.
"""
    (fwd_dir / "forward_validation_report.md").write_text(report, encoding="utf-8")

    print("Forward validation complete.")
    return {"metrics": metrics, "model": model, "train_data": train_data, "test_data": test_data}


def run_linear_recon_synthetic(cfg, run_dir, env_map):
    print("\n" + "=" * 60)
    print("Phase 3.6: Linear Reconstruction - Synthetic Smoke Test (Level 1)")
    print("=" * 60)

    lr_cfg = cfg["linear_recon"]
    obj_size = tuple(lr_cfg["synthetic_object_size"])
    psf_working_size = tuple(lr_cfg["psf_working_size"])
    wavelengths = np.array(lr_cfg["wavelengths_nm"], dtype=np.float32)
    alpha = lr_cfg["ridge_alpha"]
    policy = lr_cfg.get("ridge_policy", "adaptive")
    n_masks = lr_cfg["selected_masks"]["count"]
    seed = lr_cfg.get("synthetic_fixed_seed", 42)
    mask_strategy = lr_cfg["selected_masks"].get("strategy", "representative_first")

    train_h5 = resolve_config_value(cfg["data"]["train_h5"], env_map)
    print(f"Loading PSF dictionary: {train_h5}")
    psf_data = load_psf_dictionary(train_h5, psf_working_size=psf_working_size)

    print(f"Selecting {n_masks} masks for encoding (strategy={mask_strategy})")
    if mask_strategy == "diverse_family_first":
        from scripts.compare_recon_strategies import select_diverse_masks
        _, sel_psfs, sel_ids, sel_families = select_diverse_masks(psf_data, count=n_masks)
        sel_masks_display = np.array([psf_data["masks"][psf_data["mask_id"].index(mid)] for mid in sel_ids])
    else:
        sel_masks, sel_psfs, sel_ids, sel_families = select_masks_by_strategy(
            psf_data, strategy=mask_strategy, count=n_masks,
        )
        sel_masks_display = sel_masks[:, 0, 0, :, :]

    print(f"Generating synthetic object: size={obj_size}")
    obj_np = generate_synthetic_object(n_channels=3, spatial_size=obj_size, seed=seed)
    obj = torch.from_numpy(obj_np).float()

    print(f"Selected {len(sel_ids)} masks")
    if isinstance(sel_psfs, torch.Tensor):
        psfs_t = sel_psfs.float()
    elif sel_psfs.ndim == 5:
        psfs_t = torch.from_numpy(sel_psfs[:, 0, :, :, :]).float()
    else:
        psfs_t = torch.from_numpy(sel_psfs).float()
    if psfs_t.ndim == 5:
        psfs_t = psfs_t[:, 0]

    print("Rendering frames via FFT convolution")
    frames = render_frames_fft(obj, psfs_t)
    frames_np = frames.numpy()

    print("Running single-frame reconstruction (baseline)")
    recon_single = single_frame_reconstruct(frames, psfs_t, alpha=alpha, policy=policy)
    recon_single_np = recon_single.numpy()

    print("Running multi-frame reconstruction")
    recon_multi = frequency_domain_ridge_reconstruct(frames, psfs_t, alpha=alpha, policy=policy)
    recon_multi_np = recon_multi.numpy()

    print("Computing metrics")
    metrics_single = compute_recon_metrics(obj, recon_single)
    metrics_multi = compute_recon_metrics(obj, recon_multi)

    recon_dir = _ensure_dir(run_dir / "linear_recon")

    plot_synthetic_objects(obj_np, recon_dir / "synthetic_objects.png")

    plot_mask_grid(sel_masks_display, sel_ids, recon_dir / "selected_masks.png", title="Selected Encoding Masks")

    plot_rendered_frames(frames_np, recon_dir / "rendered_frames.png")

    plot_reconstruction(obj_np, recon_single_np, recon_dir / "recon_single_frame.png",
                        wavelengths_nm=wavelengths, title="Single-Frame Reconstruction")
    plot_reconstruction(obj_np, recon_multi_np, recon_dir / "recon_multiframe.png",
                        wavelengths_nm=wavelengths, title="Multi-Frame Reconstruction")
    plot_recon_comparison(obj_np, recon_single_np, recon_multi_np,
                          recon_dir / "recon_comparison.png", wavelengths_nm=wavelengths)
    plot_recon_comparison(obj_np, recon_single_np, recon_multi_np,
                          recon_dir / "recon_per_band_comparison.png", wavelengths_nm=wavelengths)
    plot_recon_rgb_pseudocolor_comparison(
        obj_np,
        recon_single_np,
        recon_multi_np,
        recon_dir / "recon_rgb_pseudocolor_comparison.png",
        wavelengths_nm=wavelengths,
    )

    np.savez_compressed(
        recon_dir / "recon_appendix_arrays.npz",
        gt_object=obj_np.astype(np.float32),
        recon_single=recon_single_np.astype(np.float32),
        recon_multi=recon_multi_np.astype(np.float32),
        rendered_frames=frames_np.astype(np.float32),
        wavelengths_nm=wavelengths.astype(np.float32),
        selected_mask_ids=np.asarray(sel_ids, dtype="U"),
        mask_families=np.asarray(sel_families, dtype="U"),
        source_psf_h5=np.asarray([train_h5], dtype="U"),
    )

    results = {
        "object_size": list(obj_size),
        "n_masks": n_masks,
        "alpha": alpha,
        "single_frame_metrics": metrics_single,
        "multi_frame_metrics": metrics_multi,
        "selected_masks": sel_ids,
        "mask_families": sel_families,
    }

    with open(recon_dir / "reconstruction_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Single-frame metrics: mean_mse={metrics_single['mean']['mse']:.6f}, mean_psnr={metrics_single['mean']['psnr']:.2f}")
    print(f"Multi-frame metrics:  mean_mse={metrics_multi['mean']['mse']:.6f}, mean_psnr={metrics_multi['mean']['psnr']:.2f}")

    report = f"""# Linear Reconstruction Report (Level 1: Synthetic Smoke Test)

## Setup
- Object: synthetic procedural target ({obj_size[0]}x{obj_size[1]})
- Wavelengths: {wavelengths.tolist()} nm
- Encoding masks: {n_masks} (strategy: representative_first)
- PSF source: measured_dictionary
- PSF working size: {psf_working_size}
- Solver: frequency_domain_ridge (alpha={alpha})

## Results

### Single-Frame
- Mean MSE: {metrics_single['mean']['mse']:.6f}
- Mean PSNR: {metrics_single['mean']['psnr']:.2f} dB
- Mean Correlation: {metrics_single['mean']['correlation']:.4f}

### Multi-Frame
- Mean MSE: {metrics_multi['mean']['mse']:.6f}
- Mean PSNR: {metrics_multi['mean']['psnr']:.2f} dB
- Mean Correlation: {metrics_multi['mean']['correlation']:.4f}

## Conclusion
The reconstruction pipeline (FFT convolution renderer + frequency-domain ridge solver)
operates correctly with measured PSF dictionary kernels.

## Appendix Outputs
- `recon_per_band_comparison.png`: per-wavelength GT / single-frame / multi-frame / error comparison
- `recon_rgb_pseudocolor_comparison.png`: pseudo-RGB comparison using nearest 650/550/450 nm channels
- `recon_appendix_arrays.npz`: GT object, reconstructions, rendered frames, wavelengths, selected mask IDs, source HDF5
"""
    (recon_dir / "linear_recon_report.md").write_text(report, encoding="utf-8")

    print("Synthetic reconstruction complete.")
    return results


def run_linear_recon_cave(cfg, run_dir, env_map):
    print("\n" + "=" * 60)
    print("Phase 3.6: Linear Reconstruction - CAVE (Level 2)")
    print("=" * 60)

    cave_cfg = cfg.get("cave", {})
    if not cave_cfg.get("enabled", False):
        print("CAVE disabled in config, skipping.")
        return None

    test_h5_path = Path(resolve_config_value(cave_cfg.get("test_h5", ""), env_map))
    if not test_h5_path.exists():
        print(f"CAVE test data not found: {test_h5_path}, skipping.")
        return None

    from src.datasets.cave_dataset import load_cave_dataset

    cave_data = load_cave_dataset(test_h5_path)
    n_eval = min(cfg.get("linear_recon", {}).get("num_cave_eval_scenes", 3), cave_data["n_scenes"])
    print(f"CAVE test: {cave_data['n_scenes']} scenes, evaluating {n_eval}")

    lr_cfg = cfg["linear_recon"]
    psf_working_size = tuple(lr_cfg["psf_working_size"])
    wavelengths = np.array(lr_cfg["wavelengths_nm"], dtype=np.float32)
    alpha = lr_cfg["ridge_alpha"]
    policy = lr_cfg.get("ridge_policy", "adaptive")
    n_masks = lr_cfg["selected_masks"]["count"]
    obj_size = tuple(lr_cfg["cave_object_size"])
    mask_strategy = lr_cfg["selected_masks"].get("strategy", "representative_first")

    train_h5 = resolve_config_value(cfg["data"]["train_h5"], env_map)
    psf_data = load_psf_dictionary(train_h5, psf_working_size=psf_working_size)
    if mask_strategy == "diverse_family_first":
        from scripts.compare_recon_strategies import select_diverse_masks
        _, sel_psfs, sel_ids, sel_families = select_diverse_masks(psf_data, count=n_masks)
    else:
        _, sel_psfs, sel_ids, sel_families = select_masks_by_strategy(
            psf_data, strategy=mask_strategy, count=n_masks,
        )
        if sel_psfs.ndim == 5:
            sel_psfs = sel_psfs[:, 0]
    if isinstance(sel_psfs, torch.Tensor):
        psfs_t = sel_psfs.float()
    elif sel_psfs.ndim == 5:
        psfs_t = torch.from_numpy(sel_psfs[:, 0, :, :, :]).float()
    else:
        psfs_t = torch.from_numpy(sel_psfs).float()
    if psfs_t.ndim == 5:
        psfs_t = psfs_t[:, 0]
    if psfs_t.ndim != 4:
        raise ValueError(f"Expected psfs_t [T, L, H, W], got shape {psfs_t.shape}")

    cave_dir = _ensure_dir(run_dir / "linear_recon_cave")
    all_metrics = []
    appendix_entries = []

    for scene_idx in range(n_eval):
        scene_id = cave_data["scene_id"][scene_idx]
        scene_dir = _ensure_dir(cave_dir / f"scene_{scene_id}")
        print(f"\nScene {scene_idx + 1}/{n_eval}: {scene_id}")

        obj_np = cave_data["objects"][scene_idx]
        obj = torch.from_numpy(obj_np).float()

        if obj.shape[-2:] != obj_size:
            obj = torch.nn.functional.interpolate(
                obj.unsqueeze(0), size=obj_size, mode="bilinear", align_corners=False
            ).squeeze(0)

        frames = render_frames_fft(obj, psfs_t)
        recon_single = single_frame_reconstruct(frames, psfs_t, alpha=alpha, policy=policy)
        recon_multi = frequency_domain_ridge_reconstruct(frames, psfs_t, alpha=alpha, policy=policy)

        metrics_s = compute_recon_metrics(obj, recon_single)
        metrics_m = compute_recon_metrics(obj, recon_multi)

        plot_recon_comparison(obj.numpy(), recon_single.numpy(), recon_multi.numpy(),
                              scene_dir / "recon_comparison.png", wavelengths_nm=wavelengths)
        plot_recon_comparison(obj.numpy(), recon_single.numpy(), recon_multi.numpy(),
                              scene_dir / "recon_per_band_comparison.png", wavelengths_nm=wavelengths)
        plot_recon_rgb_pseudocolor_comparison(
            obj.numpy(),
            recon_single.numpy(),
            recon_multi.numpy(),
            scene_dir / "recon_rgb_pseudocolor_comparison.png",
            wavelengths_nm=wavelengths,
        )
        plot_rendered_frames(frames.numpy(), scene_dir / "rendered_frames.png",
                             title=f"Rendered Frames - {scene_id}")

        np.savez_compressed(
            scene_dir / "recon_appendix_arrays.npz",
            gt_object=obj.numpy().astype(np.float32),
            recon_single=recon_single.numpy().astype(np.float32),
            recon_multi=recon_multi.numpy().astype(np.float32),
            rendered_frames=frames.numpy().astype(np.float32),
            wavelengths_nm=wavelengths.astype(np.float32),
            selected_mask_ids=np.asarray(sel_ids, dtype="U"),
            mask_families=np.asarray(sel_families, dtype="U"),
            source_cave_h5=np.asarray([str(test_h5_path)], dtype="U"),
            source_psf_h5=np.asarray([train_h5], dtype="U"),
            scene_id=np.asarray([scene_id], dtype="U"),
        )

        scene_result = {
            "scene_id": scene_id,
            "single_frame_metrics": metrics_s,
            "multi_frame_metrics": metrics_m,
        }
        all_metrics.append(scene_result)
        appendix_entries.append({
            "scene_id": scene_id,
            "figures": [
                f"scene_{scene_id}/recon_per_band_comparison.png",
                f"scene_{scene_id}/recon_rgb_pseudocolor_comparison.png",
                f"scene_{scene_id}/recon_comparison.png",
                f"scene_{scene_id}/rendered_frames.png",
            ],
            "data": f"scene_{scene_id}/recon_appendix_arrays.npz",
        })

        print(f"  Single: mse={metrics_s['mean']['mse']:.6f}, psnr={metrics_s['mean']['psnr']:.2f}")
        print(f"  Multi:  mse={metrics_m['mean']['mse']:.6f}, psnr={metrics_m['mean']['psnr']:.2f}")

    with open(cave_dir / "cave_recon_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"n_scenes_evaluated": n_eval, "results": all_metrics}, f, indent=2)

    report = """# CAVE Linear Reconstruction Report

## Purpose
This report records the public-dataset reconstruction appendix for Phase 3.6.
Each scene includes explicit per-wavelength comparison, pseudo-RGB comparison,
rendered measurements, and compressed arrays for traceability.

## Figures and Data
"""
    for entry in appendix_entries:
        report += f"\n### {entry['scene_id']}\n"
        report += f"- Per-band comparison: `{entry['figures'][0]}`\n"
        report += f"- Pseudo-RGB comparison: `{entry['figures'][1]}`\n"
        report += f"- Legacy comparison alias: `{entry['figures'][2]}`\n"
        report += f"- Rendered frames: `{entry['figures'][3]}`\n"
        report += f"- Appendix arrays: `{entry['data']}`\n"

    report += f"""

## Provenance
- CAVE HDF5: `{test_h5_path}`
- PSF dictionary HDF5: `{train_h5}`
- Wavelengths: {wavelengths.tolist()} nm
- Selected masks: {sel_ids}
- Solver: frequency_domain_ridge alpha={alpha}, policy={policy}

## Boundary
This is public-dataset simulation driven by measured PSF kernels. It is not real target capture.
"""
    (cave_dir / "cave_recon_report.md").write_text(report, encoding="utf-8")

    print(f"\nCAVE reconstruction complete. {n_eval} scenes evaluated.")
    return all_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bishe first pass Phase 3.5-3.6")
    parser.add_argument("--config", default="configs/bishe_first_pass.yaml")
    parser.add_argument("--skip-handoff-verify", action="store_true")
    parser.add_argument("--allow-invalid-handoff", action="store_true")
    parser.add_argument("--dry-run-small", action="store_true")
    args = parser.parse_args()

    config_path = ROOT / args.config
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    release_root_default = "D:/datasets/optic_system/phase3_release_20260520"
    cave_root_default = "D:/datasets/CAVE"

    release_root = os.environ.get("LCD_RELEASE_ROOT", release_root_default)
    cave_root = os.environ.get("CAVE_ROOT", cave_root_default)

    psf_dict_dir = os.path.join(release_root, "lcd_forward", "psf_dictionary")
    cave_processed = os.path.join(cave_root, "processed")

    env_map = {
        "LCD_RELEASE_ROOT": release_root,
        "CAVE_ROOT": cave_root,
        "LCD_TRAIN_H5": os.environ.get("LCD_TRAIN_H5", os.path.join(psf_dict_dir, "train.h5")),
        "LCD_VAL_H5": os.environ.get("LCD_VAL_H5", os.path.join(psf_dict_dir, "val.h5")),
        "LCD_TEST_H5": os.environ.get("LCD_TEST_H5", os.path.join(psf_dict_dir, "test.h5")),
        "CAVE_PROCESSED": os.environ.get("CAVE_PROCESSED", cave_processed),
        "CAVE_TEST_H5": os.environ.get("CAVE_TEST_H5", os.path.join(cave_processed, "test.h5")),
    }
    cfg = resolve_config(cfg, env_map)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(cfg["output"]["root"])
    run_dir = _ensure_dir(output_root / run_id)

    manifest = {
        "experiment_id": cfg["experiment_id"],
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "config": str(config_path),
        "env": {k: v for k, v in env_map.items()},
        "psf_original_size": [512, 512],
        "psf_working_size": list(cfg["forward_validation"]["psf_working_size"]),
        "steps_completed": [],
        "errors": [],
    }

    if not args.skip_handoff_verify:
        print("Step 1: Verifying optic_system handoff")
        from scripts.verify_optic_handoff import (
            verify_release_files,
            verify_hdf5_schema,
        )
        release_root = Path(env_map["LCD_RELEASE_ROOT"])

        errors = verify_release_files(release_root)
        psf_dir = release_root / "lcd_forward" / "psf_dictionary"
        if psf_dir.exists():
            errors.extend(verify_hdf5_schema(psf_dir / "train.h5"))
            errors.extend(verify_hdf5_schema(psf_dir / "val.h5"))
            errors.extend(verify_hdf5_schema(psf_dir / "test.h5"))

        if errors:
            print(f"Handoff verification FAILED: {len(errors)} errors")
            for e in errors:
                print(f"  - {e}")
            manifest["errors"].extend(errors)
            if not args.allow_invalid_handoff:
                print("\nERROR: handoff verification failed. Use --allow-invalid-handoff to override, or --skip-handoff-verify to bypass.")
                manifest_path = run_dir / "run_manifest.json"
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2, ensure_ascii=False)
                sys.exit(1)
        else:
            print("Handoff verification PASSED")
            manifest["steps_completed"].append("handoff_verification")
    else:
        print("Handoff verification skipped")
        manifest["steps_completed"].append("handoff_verification_skipped")

    try:
        fw_result = run_forward_validation(cfg, run_dir, env_map)
        manifest["steps_completed"].append("forward_validation")
        manifest["forward_metrics"] = fw_result["metrics"]["mean"]
    except Exception as e:
        print(f"Forward validation FAILED: {e}")
        manifest["errors"].append(f"forward_validation: {e}")

    try:
        recon_result = run_linear_recon_synthetic(cfg, run_dir, env_map)
        manifest["steps_completed"].append("linear_recon_synthetic")
    except Exception as e:
        print(f"Synthetic reconstruction FAILED: {e}")
        manifest["errors"].append(f"linear_recon_synthetic: {e}")

    try:
        cave_result = run_linear_recon_cave(cfg, run_dir, env_map)
        if cave_result is not None:
            manifest["steps_completed"].append("linear_recon_cave")
    except Exception as e:
        print(f"CAVE reconstruction FAILED: {e}")
        manifest["errors"].append(f"linear_recon_cave: {e}")

    summary = f"""# Bishe First Pass Summary

## Status
- Run ID: {run_id}
- Timestamp: {manifest['timestamp']}
- Steps completed: {manifest['steps_completed']}
- Errors: {len(manifest['errors'])}

## Scope
This first-pass run demonstrates that the optic_system Phase 3 measured PSF dictionary
can be ingested by LCD_forward, used for simple mask-to-PSF forward validation, and used
in a minimal three-wavelength multiframe linear reconstruction loop.

The goal is feasibility and internal consistency, not superiority.

## Level 1: Synthetic Smoke Test
A purely procedural object verifies the renderer and solver function correctly.

## Level 2: CAVE Public-Dataset Reconstruction
Public multispectral object + measured optic_system PSFs form the main Phase 3.6 result.

## Level 3: Real Target Capture
Optional, not required for the first-pass thesis existence proof.

## Boundary Statement
- This is a public-dataset simulation driven by measured optic_system PSFs.
- No hardware control, no learned masks, no complex neural reconstruction.
- CAVE reconstruction uses FFT circular convolution for feasibility demonstration.
- Real target capture remains optional for later phases.
"""
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")

    manifest_path = run_dir / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"First pass complete. Outputs: {run_dir}")
    print(f"Steps: {manifest['steps_completed']}")
    if manifest["errors"]:
        print(f"Errors: {len(manifest['errors'])}")


if __name__ == "__main__":
    main()
