# Phase 3.6 CAVE Metrics Audit Response

## Authoritative Explanation

The apparent conflict is not that the multi-frame reconstruction is bad, and it is not a PSNR implementation bug. The correct interpretation is:

> Multi-frame reconstruction recovers scene structure and channel separation much better than the single-frame baseline, while raw PSNR also penalizes residual absolute amplitude / DC calibration error. The CAVE figures are display-normalized, so they mainly expose structural recovery; raw PSNR is computed on absolute `[0, 1]` tensors and therefore remains sensitive to low-frequency brightness bias.

This means the thesis claim should not be phrased as "PSNR alone proves the visual result." The defensible claim is:

> Under the measured-PSF forward model, multi-frame observations contain substantially more recoverable spectral-spatial information than a single frame. This is supported by reconstruction figures, per-band correlation, SSIM / display-normalized diagnostics, and raw PSNR when interpreted as an absolute amplitude metric.

## Why clay_ms Looks Much Better Than Its Raw PSNR Gain

The `clay_ms` 450 nm band is the decisive case. Recomputed metrics from the saved arrays show:

| Method | Raw PSNR | Correlation | Display min-max PSNR | Affine-calibrated PSNR | GT mean | Recon mean |
|---|---:|---:|---:|---:|---:|---:|
| Single-frame | 16.15 dB | 0.0929 | 7.96 dB | 30.32 dB | 0.0268 | 0.1627 |
| Multi-frame | 15.80 dB | 0.9994 | 48.06 dB | 59.65 dB | 0.0268 | 0.1890 |

The multi-frame result is nearly perfectly correlated with GT, so its spatial structure is correct. Raw PSNR is low because the reconstructed 450 nm band has a large positive low-frequency / DC offset relative to a very dark GT band. In MSE terms, a broad brightness bias dominates even when the spatial pattern is correct.

Therefore, the negative 450 nm raw-PSNR gain is an amplitude-calibration artifact of the current linear solver / normalized CAVE setup, not evidence that single-frame reconstruction contains more useful information.

## Answers to the Audit Questions

### (a) Spatial error distribution

The initial "flat region dilution" hypothesis is not the strongest explanation. For `clay_ms` 450 nm, error is broad and low-frequency:

| Method | Flat-50 MSE | Structure-25 MSE | Structure-25 SSE share |
|---|---:|---:|---:|
| Single-frame | 0.02629 | 0.01985 | 0.2043 |
| Multi-frame | 0.02631 | 0.02634 | 0.2501 |

The mismatch is not concentrated in fine structure. It is dominated by global amplitude / DC bias.

### (b) clay_ms 450 nm metric calculation

The saved arrays and metrics pass the blocking checks:

- `gt_object`, `recon_single`, and `recon_multi` all have shape `[3, 256, 256]`.
- Recomputed PSNR matches stored PSNR with maximum discrepancy `6.15e-7 dB`.
- Single-frame and multi-frame use the same rendered frames, PSF tensor, GT crop, ridge alpha, and ridge policy.
- The single-frame branch intentionally truncates to the first frame / first PSF before using the same frequency-domain ridge solver.
- No silent broadcasting issue is present in the saved metric calculation.

### (c) SSIM and correlation

Correlation and display-normalized SSIM align with visual quality:

| Scene | Method | Raw PSNR | Raw correlation | Raw SSIM | Display-normalized SSIM |
|---|---:|---:|---:|---:|---:|
| cd_ms | Single | 20.78 | 0.7887 | 0.6882 | 0.6587 |
| cd_ms | Multi | 39.17 | 1.0000 | 0.9622 | 0.9923 |
| clay_ms | Single | 14.36 | 0.3528 | 0.3273 | 0.2343 |
| clay_ms | Multi | 18.04 | 0.9929 | 0.4444 | 0.8851 |
| superballs_ms | Single | 18.99 | 0.6875 | 0.5101 | 0.4861 |
| superballs_ms | Multi | 25.53 | 0.9983 | 0.5824 | 0.8637 |

Raw SSIM still penalizes luminance / contrast error, so it improves less dramatically for `clay_ms`. Display-normalized SSIM and correlation capture the structural improvement seen in the figures.

### (d) Frequency-domain error

For `clay_ms` 450 nm, squared error energy is almost entirely low-frequency:

| Method | Low frequency `<=0.1` | Mid frequency `0.1-0.3` | High frequency `>0.3` |
|---|---:|---:|---:|
| Single-frame | 0.9860 | 0.0100 | 0.0040 |
| Multi-frame | 1.0000 | 0.0000 | 0.0000 |

The anomaly is not a hidden high-frequency failure. It is a low-frequency amplitude bias.

### (e) Normalization

CAVE preprocessing applies per-scene min-max normalization once when creating the HDF5 dataset. Raw PSNR compares reconstruction output directly against that normalized GT. Reconstruction outputs are not independently min-max normalized before raw PSNR.

The visual figures are different:

- Per-band comparison panels use Matplotlib image autoscaling unless fixed display limits are supplied.
- Pseudo-RGB comparison uses per-panel percentile normalization.

The visual panels therefore show structural similarity after display scaling, while raw PSNR measures absolute amplitude agreement.

## Required Handoff Wording

Use this wording in the handoff:

> The CAVE multi-frame reconstructions are visually and structurally superior to the single-frame baseline. Raw PSNR is retained as an absolute amplitude metric, but it understates structural recovery when the reconstruction has residual low-frequency / DC brightness bias. For this release, the reconstruction evidence should be read from the figures together with raw PSNR, per-band correlation, SSIM, and the metric audit. The `clay_ms` 450 nm negative raw-PSNR gain is explained by amplitude bias in a very dark GT band, not by loss of recovered structure.

Do not use this wording:

> Multi-frame is better because PSNR is always much higher.

That is too strong and fails on the `clay_ms` 450 nm per-band audit.

## Thesis Position

The thesis position remains valid if stated precisely:

- The measured-PSF-driven multi-frame system demonstrates recoverable spectral-spatial information beyond the single-frame baseline.
- The evidence is feasibility and internal consistency, not calibrated radiometric superiority.
- Remaining amplitude bias is a limitation of the current simple ridge baseline and normalization setup, not a contradiction of the multi-frame encoding result.
