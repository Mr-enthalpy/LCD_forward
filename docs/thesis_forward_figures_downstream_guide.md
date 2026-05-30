# Downstream Thesis Figure Guide: Chapter 4 Forward Figures

## Purpose

This handoff publishes thesis-ready replacements for Chapter 4 forward-model figures from `LCD_forward` Phase 3.5 artifacts. It is a figure/data handoff only: do not change thesis claims, do not reinterpret optical assumptions, and do not retrain or regenerate figures downstream unless the source handoff is replaced.

Update note, 2026-05-30: sample metadata and NC values were removed from the forward prediction and residual figures. The figures now show only Chinese labels and compact sample indices; sample IDs, mask families, and NC values are published in the companion Markdown/CSV tables for thesis table formatting.

## Published Root

Use this local release root:

```text
D:/datasets/LCD_forward/lcd_forward_phase3_5_3_6_release_20260520
```

Chapter 4 figure package:

```text
thesis/thesis_figures/
- fig4_pca_basis_subset.pdf
- fig4_pca_basis_subset.png
- fig4_forward_prediction_subset.pdf
- fig4_forward_prediction_subset.png
- fig4_forward_prediction_residuals.pdf
- fig4_forward_prediction_residuals.png
- fig4_forward_prediction_subset_metrics.csv
- fig4_forward_prediction_subset_metrics.md
- thesis_figures_manifest.json
```

## Files To Copy Into Thesis Repo

Copy these PDFs into the thesis figure asset directory:

```text
source: D:/datasets/LCD_forward/lcd_forward_phase3_5_3_6_release_20260520/thesis/thesis_figures/fig4_pca_basis_subset.pdf
target: ../thesis_mono_lcd_nju/assets/figures/fig4_pca_basis_subset.pdf

source: D:/datasets/LCD_forward/lcd_forward_phase3_5_3_6_release_20260520/thesis/thesis_figures/fig4_forward_prediction_subset.pdf
target: ../thesis_mono_lcd_nju/assets/figures/fig4_forward_prediction_subset.pdf

source: D:/datasets/LCD_forward/lcd_forward_phase3_5_3_6_release_20260520/thesis/thesis_figures/fig4_forward_prediction_residuals.pdf
target: ../thesis_mono_lcd_nju/assets/figures/fig4_forward_prediction_residuals.pdf
```

Do not overwrite unrelated thesis figures. If the thesis repo is somewhere else, keep the same filenames and copy them to that repo's `assets/figures/` directory.

## LaTeX Usage

Use the PDFs directly:

```latex
\includegraphics[width=0.92\textwidth]{fig4_pca_basis_subset.pdf}
\includegraphics[width=0.92\textwidth]{fig4_forward_prediction_subset.pdf}
\includegraphics[width=0.92\textwidth]{fig4_forward_prediction_residuals.pdf}
```

The PNG files are preview/raster fallbacks only; the thesis should prefer PDF for vector text.

## Figure 4-1 Replacement

Use:

```text
thesis/thesis_figures/fig4_pca_basis_subset.pdf
```

This replaces the dense all-component PCA montage. It shows only PC1-PC3 for 450/550/650 nm measured PSFs. The intended statement is limited:

- measured PSF variations have visible low-dimensional structure;
- only a readable subset is shown;
- the full PCA model still used 24 components during Phase 3.5 validation.

Read supporting provenance from:

```text
thesis/thesis_figures/thesis_figures_manifest.json
```

Important manifest fields:

```text
pca_recomputed: true
pca_basis.num_components_total: 24
pca_basis.num_components_shown: 3
pca_basis.normalization: per-component symmetric p99
```

## Figure 4-3 Replacement

Use:

```text
thesis/thesis_figures/fig4_forward_prediction_subset.pdf
```

This replaces the dense measured-vs-predicted prediction montage. It shows four representative held-out test masks with:

```text
样本 | 掩膜 | 实测 PSF | 预测 PSF
```

Use this companion figure when discussing residual structure:

```text
thesis/thesis_figures/fig4_forward_prediction_residuals.pdf
```

It shows:

```text
样本 | 掩膜 | 残差
```

Pseudo-RGB mapping:

```text
R = 650 nm
G = 550 nm
B = 450 nm
```

Selected held-out sample IDs:

```text
- random_lowfreq_043
- random_midfreq_013
- task_related_012
- random_lowfreq_005
```

Selection policy:

```text
family_balanced_with_quantile_fill
```

Mean normalized correlation over the selected examples:

```text
0.9887
```

## Numeric Values For Thesis Text

Use this Markdown table for direct thesis table formatting:

```text
thesis/thesis_figures/fig4_forward_prediction_subset_metrics.md
```

Use this CSV for machine-readable representative per-wavelength and per-sample values:

```text
thesis/thesis_figures/fig4_forward_prediction_subset_metrics.csv
```

Columns:

```text
sample_id, mask_family, wavelength_nm, norm_corr, mse, mae, residual_l1, display_note, mean_norm_corr
```

Recommended downstream rule:

1. Use rows where `wavelength_nm` is `450.0`, `550.0`, or `650.0` for per-wavelength NC.
2. Use rows where `wavelength_nm` is `mean` for one mean NC per selected sample.
3. Use `fig4_forward_prediction_residuals.pdf` for residual visibility; do not rely on the main prediction figure for residual inspection.
4. Do not estimate metrics from figure pixels.
5. Do not mix these representative-sample metrics with the full-test aggregate unless the text explicitly says which is which.

Full-test aggregate remains the Phase 3.5 metric in:

```text
thesis/phase3_5_forward_validation/metrics/psf_prediction_metrics.json
```

## Provenance And Reproducibility

The figure package was generated from existing measured PSF HDF5 files:

```text
train_h5: D:/datasets/optic_system/phase3_release_20260520/lcd_forward/psf_dictionary/train.h5
test_h5:  D:/datasets/optic_system/phase3_release_20260520/lcd_forward/psf_dictionary/test.h5
```

Generation command, if upstream regeneration is needed:

```bash
python scripts/export_thesis_forward_figures.py ^
  --dataset D:/datasets/optic_system/phase3_release_20260520/lcd_forward/psf_dictionary/train.h5 ^
  --test-dataset D:/datasets/optic_system/phase3_release_20260520/lcd_forward/psf_dictionary/test.h5 ^
  --forward-artifacts outputs/bishe_first_pass/20260521_224543/forward_validation ^
  --out-dir outputs/thesis_figures ^
  --num-pcs 3 ^
  --num-examples 4 ^
  --allow-recompute ^
  --dpi 300 ^
  --format both
```

Reason for `--allow-recompute`: the Phase 3.5 handoff preserved report figures and metrics, but not the full PCA basis arrays and predicted-test PSF arrays. The script therefore recomputed the same simple PCA+ridge baseline deterministically from the measured PSF dictionary and recorded this in `thesis_figures_manifest.json`.

## Boundaries

Do not use this handoff to claim:

- first-principles optical model identification;
- hardware target capture;
- SOTA forward prediction;
- learned reconstruction performance;
- mask optimization.

The correct scope is: a thesis-readable visual export supporting the existing Phase 3.5 feasibility claim that a simple PCA+ridge model captures the dominant measured mask-to-PSF relationship.
