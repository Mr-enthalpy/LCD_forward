# Phase 3.6 CAVE Metrics Audit Response

## Authoritative Explanation

The current handoff uses the precision-explicit solver setting:

```yaml
ridge_alpha: 3.0e-15
ridge_policy: adaptive
internal_solve_precision: complex128
```

The earlier visual-vs-PSNR concern is resolved differently after the alpha/precision sweep: multi-frame reconstruction remains structurally superior, and `clay_ms` 450 nm now also has positive raw-PSNR gain. Raw PSNR is still an absolute amplitude metric and must be interpreted together with correlation, SSIM, and figures, but the previous `clay_ms` 450 nm negative-gain case no longer applies to the current handoff.

The defensible thesis claim is:

> Under the measured-PSF forward model, multi-frame observations contain substantially more recoverable spectral-spatial information than a single frame. In the current complex128 solver regime, this is supported by raw PSNR gains, per-band correlation, SSIM / display-normalized diagnostics, and reconstruction figures.

## Updated CAVE Metrics

| Scene | Method | Raw PSNR | Raw correlation | Raw SSIM | Display-normalized SSIM |
|---|---:|---:|---:|---:|---:|
| cd_ms | Single | 21.07 | 0.7888 | 0.7310 | 0.6814 |
| cd_ms | Multi | 33.36 | 0.9996 | 0.8030 | 0.8448 |
| clay_ms | Single | 14.31 | 0.3298 | 0.3225 | 0.2355 |
| clay_ms | Multi | 20.54 | 0.9968 | 0.5250 | 0.9205 |
| superballs_ms | Single | 18.82 | 0.6882 | 0.5117 | 0.4874 |
| superballs_ms | Multi | 42.96 | 0.9970 | 0.7125 | 0.8518 |

## clay_ms 450 nm Detail

| Method | Raw PSNR | Correlation | Display min-max PSNR | Affine-calibrated PSNR | GT mean | Recon mean |
|---|---:|---:|---:|---:|---:|---:|
| Single-frame | 16.10 dB | 0.1063 | 8.09 dB | 30.33 dB | 0.0268 | 0.1607 |
| Multi-frame | 26.31 dB | 1.0000 | 82.45 dB | 93.53 dB | 0.0268 | 0.0752 |

The current multi-frame result has positive raw-PSNR gain and near-perfect structural correlation in the 450 nm band. The old negative-gain explanation is retained only as historical context for the earlier solver setting, not as the current handoff conclusion.

## Normalization and Display

CAVE preprocessing applies per-scene min-max normalization once when creating the HDF5 dataset. Raw PSNR compares reconstruction output directly against that normalized GT. Reconstruction outputs are not independently min-max normalized before raw PSNR.

The visual figures are different:

- Per-band comparison panels use Matplotlib image autoscaling unless fixed display limits are supplied.
- Pseudo-RGB comparison uses per-panel percentile normalization.

Thus figures emphasize structural similarity, while raw PSNR measures absolute amplitude agreement. Both should be reported.

## Required Handoff Wording

Use this wording:

> The current CAVE multi-frame reconstructions are structurally superior to the single-frame baseline and also improve mean raw PSNR in all evaluated scenes. Raw PSNR is retained as an absolute amplitude metric and is interpreted together with per-band correlation, SSIM, display-normalized diagnostics, and figures. The earlier `clay_ms` 450 nm negative raw-PSNR case was tied to the previous solver precision/alpha setting and is not present in the current complex128/alpha=3e-15 handoff.

Do not use this wording:

> Alpha tuning proves optimal reconstruction performance.

The alpha sweep is a thesis-closure stability sweep over a simple ridge baseline, not a claim of global optimality.

## Thesis Position

- The measured-PSF-driven multi-frame system demonstrates recoverable spectral-spatial information beyond the single-frame baseline.
- The evidence is feasibility and internal consistency, not calibrated radiometric superiority.
- Alpha is a numerical solver parameter and must be reported together with internal precision.
