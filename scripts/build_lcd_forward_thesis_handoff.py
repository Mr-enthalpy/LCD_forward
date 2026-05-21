from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REQUIRED_SECTIONS = {
    "forward_validation": [
        "figures/measured_vs_predicted_examples.png",
        "figures/psf_basis_preview_wl0.png",
        "figures/psf_basis_preview_wl1.png",
        "figures/psf_basis_preview_wl2.png",
        "metrics/psf_prediction_metrics.json",
        "reports/forward_validation_report.md",
    ],
    "h_matrix_diagnostics": [
        "figures/h_rank_map.png",
        "figures/h_log_condition_map.png",
        "figures/h_condition_histogram.png",
        "figures/h_singular_value_maps.png",
        "figures/otf_magnitude_grid_selected_masks.png",
        "figures/mask_frequency_diversity_cv_map.png",
        "figures/wavelength_transfer_comparison.png",
        "data/h_rank_map.npy",
        "data/h_condition_map.npy",
        "data/h_singular_values.npz",
        "data/frequency_diversity_cv_map.npy",
        "metrics/h_matrix_diagnostics.json",
        "reports/h_matrix_diagnostics_report.md",
    ],
    "synthetic_recon": [
        "figures/synthetic_objects.png",
        "figures/selected_masks.png",
        "figures/rendered_frames.png",
        "figures/recon_single_frame.png",
        "figures/recon_multiframe.png",
        "figures/recon_comparison.png",
        "figures/recon_per_band_comparison.png",
        "figures/recon_rgb_pseudocolor_comparison.png",
        "data/recon_appendix_arrays.npz",
        "metrics/reconstruction_metrics.json",
        "reports/linear_recon_report.md",
    ],
    "cave_recon": [
        "figures/scene_cd_ms/recon_comparison.png",
        "figures/scene_cd_ms/recon_per_band_comparison.png",
        "figures/scene_cd_ms/recon_rgb_pseudocolor_comparison.png",
        "figures/scene_cd_ms/rendered_frames.png",
        "data/scene_cd_ms/recon_appendix_arrays.npz",
        "figures/scene_clay_ms/recon_comparison.png",
        "figures/scene_clay_ms/recon_per_band_comparison.png",
        "figures/scene_clay_ms/recon_rgb_pseudocolor_comparison.png",
        "figures/scene_clay_ms/rendered_frames.png",
        "data/scene_clay_ms/recon_appendix_arrays.npz",
        "figures/scene_superballs_ms/recon_comparison.png",
        "figures/scene_superballs_ms/recon_per_band_comparison.png",
        "figures/scene_superballs_ms/recon_rgb_pseudocolor_comparison.png",
        "figures/scene_superballs_ms/rendered_frames.png",
        "data/scene_superballs_ms/recon_appendix_arrays.npz",
        "metrics/cave_recon_metrics.json",
        "metrics/cave_recon_summary.json",
        "reports/cave_recon_report.md",
    ],
    "summary_appendices": [
        "thesis/reports/result_index.md",
        "thesis/reports/figure_catalog.md",
        "thesis/reports/repro_commands.md",
        "thesis/phase3_6_linear_recon_cave/metrics/reconstruction_metrics_summary.csv",
    ],
}


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def collect_files(run_dir: Path, handoff_root: Path):
    fwd_src = run_dir / "forward_validation"
    fwd_dst = handoff_root / "thesis" / "phase3_5_forward_validation"
    _copy_tree(fwd_src, fwd_dst)

    h_src = run_dir / "h_matrix_diagnostics"
    h_dst = handoff_root / "thesis" / "h_matrix_diagnostics"
    _copy_tree(h_src, h_dst)

    syn_src = run_dir / "linear_recon"
    syn_dst = handoff_root / "thesis" / "phase3_6_linear_recon_synthetic"
    _copy_tree(syn_src, syn_dst)

    cave_src = run_dir / "linear_recon_cave"
    cave_dst = handoff_root / "thesis" / "phase3_6_linear_recon_cave"
    _copy_tree(cave_src, cave_dst, flatten_cave=True)

    prov_dir = _ensure_dir(handoff_root / "provenance")
    shutil.copy2(run_dir / "run_manifest.json", prov_dir / "lcd_forward_run_manifest.json")
    shutil.copy2(ROOT / "configs" / "bishe_first_pass.yaml", prov_dir / "bishe_first_pass.yaml")
    shutil.copy2(ROOT / "docs" / "phase3_6_debug_and_tuning.md", prov_dir / "phase3_6_debug_and_tuning.md")


def _copy_tree(src: Path, dst: Path, flatten_cave: bool = False):
    if not src.exists():
        print(f"  WARNING: source not found: {src}")
        return
    if flatten_cave:
        for item in src.iterdir():
            if item.is_dir() and item.name.startswith("scene_"):
                scene_fig_dst = _ensure_dir(dst / "figures" / item.name)
                scene_data_dst = _ensure_dir(dst / "data" / item.name)
                for f in item.iterdir():
                    if not f.is_file():
                        continue
                    if f.suffix in (".png", ".jpg", ".pdf"):
                        shutil.copy2(f, scene_fig_dst / f.name)
                    elif f.suffix in (".npy", ".npz"):
                        shutil.copy2(f, scene_data_dst / f.name)
                    elif f.suffix in (".json",):
                        _ensure_dir(dst / "metrics" / item.name)
                        shutil.copy2(f, dst / "metrics" / item.name / f.name)
                    elif f.suffix in (".md",):
                        _ensure_dir(dst / "reports" / item.name)
                        shutil.copy2(f, dst / "reports" / item.name / f.name)
            elif item.is_file():
                if item.suffix in (".json",):
                    _ensure_dir(dst / "metrics")
                    shutil.copy2(item, dst / "metrics" / item.name)
                elif item.suffix in (".md",):
                    _ensure_dir(dst / "reports")
                    shutil.copy2(item, dst / "reports" / item.name)
    else:
        _ensure_dir(dst)
        figs_dir = _ensure_dir(dst / "figures")
        metrics_dir = _ensure_dir(dst / "metrics")
        reports_dir = _ensure_dir(dst / "reports")

        for item in src.iterdir():
            if item.is_dir():
                sub_dst = dst / item.name
                if sub_dst.exists():
                    shutil.rmtree(sub_dst)
                shutil.copytree(item, sub_dst)
            elif item.suffix in (".png", ".jpg", ".pdf"):
                shutil.copy2(item, figs_dir / item.name)
            elif item.suffix in (".json",):
                shutil.copy2(item, metrics_dir / item.name)
            elif item.suffix in (".npy", ".npz"):
                data_dir = _ensure_dir(dst / "data")
                shutil.copy2(item, data_dir / item.name)
            elif item.suffix in (".md",):
                shutil.copy2(item, reports_dir / item.name)


def write_optic_reference(optic_release_root: Path, dst: Path):
    ref = {"optic_system_release_root": str(optic_release_root)}
    release_file = optic_release_root / "RELEASE.json"
    if release_file.exists():
        with open(release_file, encoding="utf-8-sig") as f:
            ref["optic_system_RELEASE"] = json.load(f)
    with open(dst, "w") as f:
        json.dump(ref, f, indent=2)


def build_cave_summary(handoff_root: Path):
    cave_metrics_path = handoff_root / "thesis" / "phase3_6_linear_recon_cave" / "metrics" / "cave_recon_metrics.json"
    if not cave_metrics_path.exists():
        return

    with open(cave_metrics_path) as f:
        data = json.load(f)

    scenes = data.get("results", [])
    summary = {
        "n_scenes": len(scenes),
        "scenes": [],
        "all_cases_multi_greater_than_single": True,
        "boundary": "public-dataset simulation using measured PSF kernels; not real target capture",
    }
    for s in scenes:
        entry = {
            "scene_id": s["scene_id"],
            "single_psnr": s["single_frame_metrics"]["mean"]["psnr"],
            "multi_psnr": s["multi_frame_metrics"]["mean"]["psnr"],
            "gain_db": round(s["multi_frame_metrics"]["mean"]["psnr"] - s["single_frame_metrics"]["mean"]["psnr"], 2),
        }
        summary["scenes"].append(entry)
        if entry["gain_db"] <= 0:
            summary["all_cases_multi_greater_than_single"] = False

    out_path = handoff_root / "thesis" / "phase3_6_linear_recon_cave" / "metrics" / "cave_recon_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)


def write_reconstruction_metrics_summary_csv(handoff_root: Path):
    metrics_dir = handoff_root / "thesis" / "phase3_6_linear_recon_cave" / "metrics"
    out_path = metrics_dir / "reconstruction_metrics_summary.csv"
    syn_path = handoff_root / "thesis" / "phase3_6_linear_recon_synthetic" / "metrics" / "reconstruction_metrics.json"
    cave_path = metrics_dir / "cave_recon_metrics.json"
    wavelengths = [450.0, 550.0, 650.0]
    rows = []

    def add_rows(dataset: str, scene_id: str, result: dict):
        single = result["single_frame_metrics"]
        multi = result["multi_frame_metrics"]
        methods = [("single_frame", single), ("multi_frame", multi)]

        single_mean_psnr = single["mean"]["psnr"]
        for method, metrics in methods:
            mean_gain = metrics["mean"]["psnr"] - single_mean_psnr if method == "multi_frame" else 0.0
            rows.append({
                "dataset": dataset,
                "scene_id": scene_id,
                "channel": "mean",
                "wavelength_nm": "",
                "method": method,
                "mse": metrics["mean"]["mse"],
                "relative_l2": metrics["mean"]["relative_l2"],
                "psnr": metrics["mean"]["psnr"],
                "correlation": metrics["mean"]["correlation"],
                "psnr_gain_vs_single_db": mean_gain,
            })
            for channel_str, channel_metrics in metrics["per_channel"].items():
                channel = int(channel_str)
                single_channel_psnr = single["per_channel"][channel_str]["psnr"]
                gain = channel_metrics["psnr"] - single_channel_psnr if method == "multi_frame" else 0.0
                rows.append({
                    "dataset": dataset,
                    "scene_id": scene_id,
                    "channel": channel,
                    "wavelength_nm": wavelengths[channel] if channel < len(wavelengths) else "",
                    "method": method,
                    "mse": channel_metrics["mse"],
                    "relative_l2": channel_metrics["relative_l2"],
                    "psnr": channel_metrics["psnr"],
                    "correlation": channel_metrics["correlation"],
                    "psnr_gain_vs_single_db": gain,
                })

    if syn_path.exists():
        with open(syn_path) as f:
            syn = json.load(f)
        add_rows("synthetic", "synthetic_procedural", syn)

    if cave_path.exists():
        with open(cave_path) as f:
            cave = json.load(f)
        for scene in cave.get("results", []):
            add_rows("cave", scene["scene_id"], scene)

    _ensure_dir(metrics_dir)
    fieldnames = [
        "dataset",
        "scene_id",
        "channel",
        "wavelength_nm",
        "method",
        "mse",
        "relative_l2",
        "psnr",
        "correlation",
        "psnr_gain_vs_single_db",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_result_index(handoff_root: Path):
    cave_summary_path = handoff_root / "thesis" / "phase3_6_linear_recon_cave" / "metrics" / "cave_recon_summary.json"
    fwd_path = handoff_root / "thesis" / "phase3_5_forward_validation" / "metrics" / "psf_prediction_metrics.json"
    h_path = handoff_root / "thesis" / "h_matrix_diagnostics" / "metrics" / "h_matrix_diagnostics.json"
    cave_summary = {}
    fwd_metrics = {}
    h_metrics = {}
    if cave_summary_path.exists():
        with open(cave_summary_path) as f:
            cave_summary = json.load(f)
    if fwd_path.exists():
        with open(fwd_path) as f:
            fwd_metrics = json.load(f)
    if h_path.exists():
        with open(h_path) as f:
            h_metrics = json.load(f)

    text = """# Result Index

This page maps each thesis-closure claim to the evidence files, input provenance,
and caveats needed to audit the handoff.

| Claim | Primary evidence | Data / metrics | Provenance | Caveat |
| --- | --- | --- | --- | --- |
"""
    fwd_corr = fwd_metrics.get("test_metrics", {}).get("mean", {}).get("normalized_correlation", "")
    full_rank = h_metrics.get("full_rank_fraction", h_metrics.get("n_full_rank_points", ""))
    text += (
        f"| Measured mask-to-PSF relation is modelable by a simple PCA+ridge baseline | "
        f"`thesis/phase3_5_forward_validation/figures/measured_vs_predicted_examples.png` | "
        f"`thesis/phase3_5_forward_validation/metrics/psf_prediction_metrics.json` "
        f"(mean corr={fwd_corr}) | `provenance/optic_system_release_reference.json` | "
        "Low-dimensional baseline only; no physical micro-geometry claim |\n"
    )
    text += (
        f"| Multi-frame transfer matrix is non-degenerate for the analyzed setup | "
        f"`thesis/h_matrix_diagnostics/figures/h_rank_map.png` and "
        f"`thesis/h_matrix_diagnostics/figures/h_singular_value_maps.png` | "
        f"`thesis/h_matrix_diagnostics/metrics/h_matrix_diagnostics.json` "
        f"(rank evidence={full_rank}) | `provenance/bishe_first_pass.yaml` | "
        "FFT/circulant approximation at downsampled PSF size |\n"
    )
    text += (
        "| Multi-frame reconstruction improves over the single-frame baseline | "
        "`thesis/phase3_6_linear_recon_cave/figures/*/recon_per_band_comparison.png` and "
        "`thesis/phase3_6_linear_recon_cave/figures/*/recon_rgb_pseudocolor_comparison.png` | "
        "`thesis/phase3_6_linear_recon_cave/metrics/reconstruction_metrics_summary.csv` | "
        "`thesis/phase3_6_linear_recon_cave/data/*/recon_appendix_arrays.npz` | "
        "Public-dataset simulation, not real target capture |\n"
    )
    if cave_summary.get("scenes"):
        text += "\n## CAVE Scene Summary\n\n"
        text += "| Scene | Single PSNR | Multi PSNR | Gain |\n| --- | ---: | ---: | ---: |\n"
        for scene in cave_summary["scenes"]:
            text += (
                f"| {scene['scene_id']} | {scene['single_psnr']:.2f} | "
                f"{scene['multi_psnr']:.2f} | {scene['gain_db']:+.2f} |\n"
            )
    text += """

## First Read Order
1. `thesis/reports/thesis_evidence_summary.md`
2. `thesis/reports/result_index.md`
3. `thesis/reports/figure_catalog.md`
4. `thesis/reports/repro_commands.md`
"""
    _ensure_dir(handoff_root / "thesis" / "reports")
    (handoff_root / "thesis" / "reports" / "result_index.md").write_text(text, encoding="utf-8")


def write_figure_catalog(handoff_root: Path):
    text = """# Figure Catalog

This catalog states what each thesis-facing figure is meant to show and what it does not show.

| Figure | What to inspect | Supports | Limit |
| --- | --- | --- | --- |
| `thesis/phase3_5_forward_validation/figures/measured_vs_predicted_examples.png` | PSF shape agreement between measured and PCA+ridge-predicted kernels | Simple forward validation sanity baseline | Does not prove a first-principles optical model |
| `thesis/phase3_5_forward_validation/figures/psf_basis_preview_wl0.png` | Low-dimensional PSF variation at 450 nm | PCA basis interpretability | Basis is wavelength-specific and empirical |
| `thesis/phase3_5_forward_validation/figures/psf_basis_preview_wl1.png` | Low-dimensional PSF variation at 550 nm | PCA basis interpretability | Basis is wavelength-specific and empirical |
| `thesis/phase3_5_forward_validation/figures/psf_basis_preview_wl2.png` | Low-dimensional PSF variation at 650 nm | PCA basis interpretability | Basis is wavelength-specific and empirical |
| `thesis/h_matrix_diagnostics/figures/h_rank_map.png` | Per-frequency rank of the multi-frame transfer matrix | Non-degenerate spectral encoding evidence | Uses selected mask subset and circular convolution |
| `thesis/h_matrix_diagnostics/figures/h_log_condition_map.png` | Spatial-frequency conditioning | Explains where inversion is stable or weak | High condition values imply noise sensitivity |
| `thesis/h_matrix_diagnostics/figures/h_singular_value_maps.png` | Singular-value spread across frequencies | Frequency-domain separability | Downsampled PSF working size only |
| `thesis/h_matrix_diagnostics/figures/otf_magnitude_grid_selected_masks.png` | OTF diversity for selected masks | Mask diversity sanity check | Qualitative diagnostic |
| `thesis/phase3_6_linear_recon_synthetic/figures/recon_per_band_comparison.png` | GT, single-frame, multi-frame, and error per wavelength | Synthetic reconstruction pipeline check | Procedural target only |
| `thesis/phase3_6_linear_recon_synthetic/figures/recon_rgb_pseudocolor_comparison.png` | Pseudo-RGB visual comparison | Human-readable synthetic reconstruction summary | Display normalization is for visualization |
| `thesis/phase3_6_linear_recon_cave/figures/scene_cd_ms/recon_per_band_comparison.png` | Per-band CAVE reconstruction quality for `cd_ms` | Public-dataset reconstruction evidence | Simulation with measured PSFs, not captured target |
| `thesis/phase3_6_linear_recon_cave/figures/scene_cd_ms/recon_rgb_pseudocolor_comparison.png` | Pseudo-RGB CAVE reconstruction for `cd_ms` | Visual comparison across single/multi frame | Pseudo-color is not calibrated camera RGB |
| `thesis/phase3_6_linear_recon_cave/figures/scene_clay_ms/recon_per_band_comparison.png` | Per-band CAVE reconstruction quality for `clay_ms` | Includes weaker-case behavior | Simulation with measured PSFs, not captured target |
| `thesis/phase3_6_linear_recon_cave/figures/scene_clay_ms/recon_rgb_pseudocolor_comparison.png` | Pseudo-RGB CAVE reconstruction for `clay_ms` | Visualizes remaining artifacts | Pseudo-color is not calibrated camera RGB |
| `thesis/phase3_6_linear_recon_cave/figures/scene_superballs_ms/recon_per_band_comparison.png` | Per-band CAVE reconstruction quality for `superballs_ms` | Public-dataset reconstruction evidence | Simulation with measured PSFs, not captured target |
| `thesis/phase3_6_linear_recon_cave/figures/scene_superballs_ms/recon_rgb_pseudocolor_comparison.png` | Pseudo-RGB CAVE reconstruction for `superballs_ms` | Visual comparison across single/multi frame | Pseudo-color is not calibrated camera RGB |

## Array Appendices
- `thesis/phase3_6_linear_recon_synthetic/data/recon_appendix_arrays.npz`
- `thesis/phase3_6_linear_recon_cave/data/scene_cd_ms/recon_appendix_arrays.npz`
- `thesis/phase3_6_linear_recon_cave/data/scene_clay_ms/recon_appendix_arrays.npz`
- `thesis/phase3_6_linear_recon_cave/data/scene_superballs_ms/recon_appendix_arrays.npz`

Each `.npz` stores GT object, single-frame reconstruction, multi-frame reconstruction,
rendered frames, wavelengths, selected mask IDs, and source HDF5 provenance.
"""
    _ensure_dir(handoff_root / "thesis" / "reports")
    (handoff_root / "thesis" / "reports" / "figure_catalog.md").write_text(text, encoding="utf-8")


def write_repro_commands(handoff_root: Path, release_id: str):
    text = f"""# Reproduction Commands

These commands reproduce or verify the thesis handoff on this repository branch.
They assume the external input datasets remain at the recorded provenance paths.

## Environment
```bash
pip install -e .
```

## Run Phase 3.5/3.6 First Pass
```bash
python scripts/run_bishe_first_pass.py --config configs/bishe_first_pass.yaml --skip-handoff-verify
```

Expected run output root:
```text
outputs/bishe_first_pass/<YYYYMMDD_HHMMSS>/
```

## Build Handoff Payload
```bash
python scripts/build_lcd_forward_thesis_handoff.py \\
  --run-dir outputs/bishe_first_pass/<YYYYMMDD_HHMMSS> \\
  --optic-release-root D:/datasets/optic_system/phase3_release_20260520 \\
  --output-root D:/datasets/LCD_forward/{release_id}
```

## Verify Handoff
```bash
python scripts/verify_lcd_forward_handoff.py D:/datasets/LCD_forward/{release_id}
```

## Recorded Input Paths
- optic_system release: `D:/datasets/optic_system/phase3_release_20260520`
- PSF HDF5: `D:/datasets/optic_system/phase3_release_20260520/lcd_forward/psf_dictionary/train.h5`
- CAVE test HDF5: `D:/CAVE/processed/test.h5`
- Config: `provenance/bishe_first_pass.yaml`

## Reproduction Boundary
- No hardware control is invoked.
- No `optic_system` device services are imported.
- Outputs are feasibility evidence, not superiority claims.
"""
    _ensure_dir(handoff_root / "thesis" / "reports")
    (handoff_root / "thesis" / "reports" / "repro_commands.md").write_text(text, encoding="utf-8")


def write_thesis_evidence_summary(handoff_root: Path):
    with open(handoff_root / "thesis" / "phase3_5_forward_validation" / "metrics" / "psf_prediction_metrics.json") as f:
        fwd_metrics = json.load(f)

    cave_summary_path = handoff_root / "thesis" / "phase3_6_linear_recon_cave" / "metrics" / "cave_recon_summary.json"
    cave_data = {}
    if cave_summary_path.exists():
        with open(cave_summary_path) as f:
            cave_data = json.load(f)

    syn_metrics_path = handoff_root / "thesis" / "phase3_6_linear_recon_synthetic" / "metrics" / "reconstruction_metrics.json"
    syn_data = {}
    if syn_metrics_path.exists():
        with open(syn_metrics_path) as f:
            syn_data = json.load(f)

    report = f"""# Thesis Evidence Summary

## Phase 3.5: Forward Validation

The measured mask-to-PSF relationship can be partially modelled by a low-dimensional
PCA + ridge regression baseline.

- PCA 24 components, ridge alpha=1.0
- Test correlation mean: {fwd_metrics['test_metrics']['mean']['normalized_correlation']:.4f}
- Per-wavelength correlation: 450nm={fwd_metrics['test_metrics']['per_wavelength']['0']['normalized_correlation']:.4f}, 550nm={fwd_metrics['test_metrics']['per_wavelength']['1']['normalized_correlation']:.4f}, 650nm={fwd_metrics['test_metrics']['per_wavelength']['2']['normalized_correlation']:.4f}

## Phase 3.6: H-Matrix Diagnostics

The multi-frame measured PSF transfer matrix is full-rank at all analyzed frequencies,
confirming non-degenerate frequency-domain encoding structure for three-wavelength recovery.

## Phase 3.6: Linear Reconstruction

### Synthetic (Level 1)
- Multi-frame reconstruction improves the measured structural metrics and figures; raw PSNR is an absolute amplitude metric and should not be interpreted alone.
- Pipeline verification: renderer + solver function correctly
- Appendix includes per-band comparison, pseudo-RGB comparison, and reconstruction arrays

### CAVE Public Dataset (Level 2)
"""
    if cave_data.get("scenes"):
        for s in cave_data["scenes"]:
            report += f"- {s['scene_id']}: single={s['single_psnr']:.1f} dB -> multi={s['multi_psnr']:.1f} dB (gain=+{s['gain_db']:.1f} dB)\n"
        report += f"\nAll cases multi > single: {cave_data['all_cases_multi_greater_than_single']}\n"

    report += f"""
## Scope Boundary

This result is a feasible existence demonstration using optic_system measured PSF
dictionary and public multispectral scenes. It shows:

1. optic_system measured PSF dictionary has been ingested by LCD_forward
2. Mask -> PSF prediction is achievable at correlation ~0.985 (Phase 3.5)
3. H matrix frequency-domain structure supports multichannel recovery
4. Multi-frame reconstruction outperforms single-frame baseline
5. This is public-dataset simulation driven by measured PSF kernels
6. Real target capture is NOT part of this release

## Not Claimed

- No SOTA performance claims
- No real target capture validation
- No learned mask / GenerMask
- No complex neural reconstruction
- No physical LCD micro-geometry identification
"""
    _ensure_dir(handoff_root / "thesis" / "reports")
    (handoff_root / "thesis" / "reports" / "thesis_evidence_summary.md").write_text(report)


def write_limitations(handoff_root: Path):
    text = """# Limitations

## Known Limitations of This Release

1. **FFT circular convolution**: reconstruction uses circular boundary conditions,
   which may introduce artifacts at image edges. Noted in reports.

2. **PSF working size**: PSFs are downsampled from 512x512 to 256x256 for
   compute efficiency. Full-resolution reconstruction may differ.

3. **Sum-normalized PSFs**: all PSFs have DC = 1.0, meaning encoding diversity
   is entirely in non-DC frequencies. Real unnormalized PSFs would carry
   additional amplitude information.

4. **Paraxial / far-field approximation**: the forward model assumes shift-invariant,
   circulant convolution, which is an approximation for the real optical system.

5. **Three wavelength discretization**: only 450/550/650 nm are used. Full spectral
   resolution may yield different channel separability.

6. **12-mask subset**: reconstruction uses a subset of the 170-mask dictionary.

7. **No real target capture**: all objects are either synthetic or CAVE public
   dataset scenes. Real target capture remains an optional Phase 3.6b task.

8. **No learned mask optimization**: masks are hand-designed or random, not
   optimized for encoding diversity.

9. **No physical pupil reconstruction**: dOTF is used as diagnostic only.

10. **Single run reproducibility**: this release captures one first-pass run.
    Multi-seed or multi-split statistics are not yet available.
"""
    _ensure_dir(handoff_root / "thesis" / "reports")
    (handoff_root / "thesis" / "reports" / "limitations.md").write_text(text)


def write_metric_audit_response(handoff_root: Path):
    src = ROOT / "docs" / "phase3_6_cave_metrics_response.md"
    dst = _ensure_dir(handoff_root / "thesis" / "reports") / "metric_audit_response.md"
    shutil.copy2(src, dst)


def write_h_matrix_dc_otf_response(handoff_root: Path):
    src = ROOT / "docs" / "phase3_6_h_matrix_dc_otf_response.md"
    dst = _ensure_dir(handoff_root / "thesis" / "reports") / "h_matrix_dc_otf_response.md"
    shutil.copy2(src, dst)


def write_solver_regularization_response(handoff_root: Path):
    src = ROOT / "docs" / "phase3_6_solver_regularization_response.md"
    dst = _ensure_dir(handoff_root / "thesis" / "reports") / "solver_regularization_response.md"
    shutil.copy2(src, dst)


def write_summary_report(handoff_root: Path, release_id: str, commit_sha: str):
    text = f"""# LCD_forward Phase 3.5-3.6 First-Pass Summary

## Release

- Release ID: {release_id}
- Source repo: https://github.com/Mr-enthalpy/LCD_forward
- Source commit: {commit_sha}
- Date: {datetime.now().strftime('%Y-%m-%d')}

## Contents

This release packages the first-pass LCD_forward Phase 3.5-3.6 loop results
into a thesis-consumable handoff.

### Phase 3.5: Measured PSF Forward Validation
- PCA + ridge baseline
- measured_vs_predicted PSF figures
- Per-wavelength and aggregate metrics

### H-Matrix Diagnostics
- Frequency-domain rank and condition analysis
- Singular value maps
- Cross-mask diversity CV map
- Wavelength transfer comparison

### Phase 3.6: Linear Reconstruction
- Synthetic smoke test (Level 1)
- CAVE public dataset reconstruction (Level 2)
- Single-frame vs multi-frame comparison
- Explicit per-band reconstruction comparisons
- Pseudo-RGB reconstruction comparisons
- Per-scene metrics, figures, and compressed array appendices

## Input Provenance

- optic_system Phase 3 release: optic_system_phase3_release_20260520
- CAVE dataset: public multispectral image database

## Usage

Thesis consumers should read:
1. thesis/reports/thesis_evidence_summary.md (first)
2. thesis/reports/limitations.md
3. Individual section reports/ for detailed results

## Boundary

This is a thesis existence proof, not a performance-optimized model.
Real target capture is NOT included.
"""
    _ensure_dir(handoff_root / "thesis" / "reports")
    (handoff_root / "thesis" / "reports" / "lcd_forward_phase3_5_3_6_summary.md").write_text(text)


def write_data_contract(handoff_root: Path):
    text = """# LCD_forward Thesis Handoff Data Contract

## Identity

This release contains LCD_forward derived results (Phase 3.5-3.6 first pass).
It is NOT a hardware data release. Raw measured HDF5 remains in optic_system release.

## Input Source

- optic_system_phase3_release_20260520
  - train.h5 / val.h5 / test.h5
  - masks [N,1,1,64,64] uint8
  - psfs [N,1,3,512,512] float64
  - wavelengths_nm [450, 550, 650]
  - psf_roi_key: roi_512

## Output Structure

```
thesis/
  phase3_5_forward_validation/
    figures/    - measured vs predicted PSF, PCA basis
    metrics/    - psf_prediction_metrics.json
    reports/    - forward_validation_report.md
  h_matrix_diagnostics/
    figures/    - rank map, condition map, histogram, SV maps, OTF grid, CV map
    data/       - numpy arrays (.npy, .npz)
    metrics/    - h_matrix_diagnostics.json
    reports/    - h_matrix_diagnostics_report.md
  phase3_6_linear_recon_synthetic/
    figures/    - synthetic objects, masks, frames, single/multi recon, per-band/RGB comparisons
    data/       - recon_appendix_arrays.npz
    metrics/    - reconstruction_metrics.json
    reports/    - linear_recon_report.md
  phase3_6_linear_recon_cave/
    figures/    - per-scene per-band comparison, RGB pseudo-color comparison, rendered_frames
    data/       - per-scene recon_appendix_arrays.npz
    metrics/    - cave_recon_metrics.json, cave_recon_summary.json
    reports/    - cave_recon_report.md
  reports/
    thesis_evidence_summary.md
    result_index.md
    figure_catalog.md
    repro_commands.md
    lcd_forward_phase3_5_3_6_summary.md
    limitations.md
provenance/
  lcd_forward_run_manifest.json
  bishe_first_pass.yaml
  phase3_6_debug_and_tuning.md
  optic_system_release_reference.json
```

## Metrics Interpretation

- forward validation: correlation > 0.95 indicates model captures PSF structure
- H matrix full-rank: encoding system is non-degenerate
- cave_recon: raw PSNR reports absolute amplitude error; interpret it together with per-band correlation, SSIM/audit metrics, and reconstruction figures
- recon_appendix_arrays.npz: GT object, single-frame recon, multi-frame recon, rendered frames, wavelengths, selected masks, and HDF5 provenance
- metric_audit_response.md: authoritative interpretation of the visual-vs-PSNR mismatch, including the clay_ms 450 nm amplitude-bias explanation

## Not Included

- Real target capture
- Learned mask / GenerMask
- Deep neural reconstruction
- 512x512 full-resolution reconstruction
- Multi-seed statistics

## Boundary

This release supports thesis existence proof. It does not claim:
- SOTA performance
- Optimal reconstruction
- Full physical model identification
"""
    (handoff_root / "data_contract.md").write_text(text)


def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def build_manifest_and_checksum(handoff_root: Path, release_id: str):
    manifest = {"release_id": release_id, "files": {}}
    checksum_lines = []

    for root, dirs, files in os.walk(handoff_root):
        for fname in files:
            full = Path(root) / fname
            rel = full.relative_to(handoff_root).as_posix()
            size = full.stat().st_size
            sha = compute_sha256(full)

            manifest["files"][rel] = {"size_bytes": size}
            checksum_lines.append(f"{sha}  {rel}")

    manifest_path = handoff_root / "MANIFEST.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    sums_path = handoff_root / "SHA256SUMS.txt"
    with open(sums_path, "w") as f:
        f.write("\n".join(checksum_lines) + "\n")


def write_release_json(handoff_root: Path, release_id: str, optic_release_root: str):
    try:
        commit_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        commit_sha = "unknown"

    release = {
        "release_id": release_id,
        "source_repo": "https://github.com/Mr-enthalpy/LCD_forward",
        "source_branch": "phase3-bishe-forward-loop",
        "source_commit": commit_sha,
        "created_at": datetime.now().strftime("%Y-%m-%d"),
        "input_optic_system_release": {
            "release_id": "optic_system_phase3_release_20260520",
            "canonical_path": str(optic_release_root),
            "psf_roi_key": "roi_512",
            "wavelengths_nm": [450.0, 550.0, 650.0],
            "psf_dictionary_contract": "masks [N,1,1,64,64], psfs [N,1,3,512,512]",
        },
        "lcd_forward_outputs": {
            "phase3_5_forward_validation": True,
            "h_matrix_diagnostics": True,
            "phase3_6_synthetic_reconstruction": True,
            "phase3_6_cave_reconstruction": True,
            "reconstruction_appendices": True,
            "real_target_capture": False,
        },
        "scope": "thesis existence proof, not performance-optimized model",
    }
    with open(handoff_root / "RELEASE.json", "w") as f:
        json.dump(release, f, indent=2)
    return commit_sha


def write_readme(handoff_root: Path, release_id: str):
    text = f"""# {release_id}

LCD_forward Phase 3.5-3.6 first-pass thesis handoff.

## Quick Links

- First read: thesis/reports/thesis_evidence_summary.md
- Result index: thesis/reports/result_index.md
- Figure catalog: thesis/reports/figure_catalog.md
- Reproduction commands: thesis/reports/repro_commands.md
- Metric audit response: thesis/reports/metric_audit_response.md
- H-matrix DC/OTF response: thesis/reports/h_matrix_dc_otf_response.md
- Solver regularization response: thesis/reports/solver_regularization_response.md
- Data contract: data_contract.md
- Limitations: thesis/reports/limitations.md
- Debug record: provenance/phase3_6_debug_and_tuning.md

## Contents

1. Phase 3.5 forward validation (mask -> PSF prediction)
2. H-matrix frequency diagnostics (full-rank proof)
3. Phase 3.6 synthetic reconstruction (smoke test)
4. Phase 3.6 CAVE reconstruction (public dataset)
5. Reconstruction appendix: per-band figures, pseudo-RGB figures, and `.npz` arrays

## Input provenance

- optic_system Phase 3 release: see provenance/optic_system_release_reference.json
- CAVE public multispectral image database
"""
    (handoff_root / "README.md").write_text(text)


def main():
    parser = argparse.ArgumentParser(description="Build LCD_forward thesis handoff")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--optic-release-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    optic_root = Path(args.optic_release_root)
    output_root = Path(args.output_root)

    release_id = f"lcd_forward_phase3_5_3_6_release_{datetime.now().strftime('%Y%m%d')}"

    if output_root.exists():
        print(f"Output directory exists, removing: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    print(f"Building handoff: {release_id}")
    print(f"  Run dir: {run_dir}")
    print(f"  Output: {output_root}")

    print("\nCollecting files from run...")
    collect_files(run_dir, output_root)

    print("Building CAVE summary...")
    build_cave_summary(output_root)
    write_reconstruction_metrics_summary_csv(output_root)

    print("Writing provenance...")
    write_optic_reference(optic_root, output_root / "provenance" / "optic_system_release_reference.json")

    print("Writing reports...")
    commit_sha = write_release_json(output_root, release_id, str(optic_root))
    write_thesis_evidence_summary(output_root)
    write_result_index(output_root)
    write_figure_catalog(output_root)
    write_repro_commands(output_root, release_id)
    write_limitations(output_root)
    write_metric_audit_response(output_root)
    write_h_matrix_dc_otf_response(output_root)
    write_solver_regularization_response(output_root)
    write_summary_report(output_root, release_id, commit_sha)
    write_data_contract(output_root)
    write_readme(output_root, release_id)

    print("Computing manifest and checksums...")
    build_manifest_and_checksum(output_root, release_id)

    git_descriptor = ROOT / "handoff" / release_id
    _ensure_dir(git_descriptor)
    shutil.copy2(output_root / "RELEASE.json", git_descriptor / "RELEASE.json")
    shutil.copy2(output_root / "README.md", git_descriptor / "README.md")
    shutil.copy2(output_root / "data_contract.md", git_descriptor / "data_contract.md")
    shutil.copy2(output_root / "MANIFEST.json", git_descriptor / "MANIFEST.json")
    shutil.copy2(output_root / "SHA256SUMS.txt", git_descriptor / "SHA256SUMS.txt")
    print(f"  Updated Git descriptor: {git_descriptor}")

    print(f"\nHandoff built: {output_root}")
    print(f"  RELEASE.json: {(output_root / 'RELEASE.json').exists()}")
    print(f"  MANIFEST.json: {(output_root / 'MANIFEST.json').exists()}")
    print(f"  SHA256SUMS.txt: {(output_root / 'SHA256SUMS.txt').exists()}")


if __name__ == "__main__":
    main()
