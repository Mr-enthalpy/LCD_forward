# Phase 3.6 CAVE Metrics Audit

This note audits the apparent mismatch between CAVE visual reconstruction quality and PSNR in the Phase 3.6 handoff.

## Inputs

- Release root: `D:/datasets/LCD_forward/lcd_forward_phase3_5_3_6_release_20260520`
- Saved arrays: `thesis/phase3_6_linear_recon_cave/data/*/recon_appendix_arrays.npz`
- Stored metrics: `thesis/phase3_6_linear_recon_cave/metrics/cave_recon_metrics.json`
- Audit command:

```bash
python scripts/audit_cave_recon_metrics.py \
  --release-root D:/datasets/LCD_forward/lcd_forward_phase3_5_3_6_release_20260520 \
  --output-dir outputs/metric_audit
```

The audit is offline-only. It uses saved CAVE appendices and does not invoke hardware control.

## Blocking Checks

### Metric recomputation

The saved PSNR values reproduce from `recon_appendix_arrays.npz`.

- Maximum stored-vs-recomputed PSNR difference: `6.15e-7 dB`
- Saved shapes for all CAVE scenes: `gt_object == recon_single == recon_multi == [3, 256, 256]`
- No silent broadcasting was found in the saved metric calculation.

### Reconstruction path

For CAVE reconstruction, `scripts/run_bishe_first_pass.py` renders one frame stack and then calls:

- `single_frame_reconstruct(frames, psfs_t, alpha=alpha, policy=policy)`
- `frequency_domain_ridge_reconstruct(frames, psfs_t, alpha=alpha, policy=policy)`

Both paths use the same `frames`, same `psfs_t`, same GT tensor `obj`, and same ridge `alpha` / policy. The single-frame path intentionally truncates to the first frame and first PSF before calling the same frequency-domain ridge solver.

## Scene-Level Cross-Check

| Scene | Method | Raw PSNR | Raw corr. | Raw SSIM | Min-max SSIM | Min-max PSNR |
|---|---:|---:|---:|---:|---:|---:|
| cd_ms | single | 20.78 | 0.7887 | 0.6882 | 0.6587 | 20.14 |
| cd_ms | multi | 39.17 | 1.0000 | 0.9622 | 0.9923 | 55.72 |
| clay_ms | single | 14.36 | 0.3528 | 0.3273 | 0.2343 | 10.88 |
| clay_ms | multi | 18.04 | 0.9929 | 0.4444 | 0.8851 | 36.63 |
| superballs_ms | single | 18.99 | 0.6875 | 0.5101 | 0.4861 | 18.81 |
| superballs_ms | multi | 25.53 | 0.9983 | 0.5824 | 0.8637 | 49.44 |

Correlation and min-max SSIM agree with the visual interpretation: multi-frame reconstruction recovers structure much better than single-frame reconstruction in all three scenes. Raw SSIM remains lower for `clay_ms` because SSIM also penalizes luminance and contrast mismatch.

## clay_ms 450 nm Finding

| Method | Raw PSNR | Corr. | Min-max PSNR | Affine PSNR | GT mean | Pred mean | Low-frequency error share |
|---|---:|---:|---:|---:|---:|---:|---:|
| single | 16.15 | 0.0929 | 7.96 | 30.32 | 0.0268 | 0.1627 | 0.9860 |
| multi | 15.80 | 0.9994 | 48.06 | 59.65 | 0.0268 | 0.1890 | 1.0000 |

The negative raw-PSNR gain for `clay_ms` at 450 nm is not a structural reconstruction failure. The multi-frame reconstruction is almost perfectly correlated with GT, but it carries a large absolute amplitude / DC offset relative to a very dark GT band. Raw PSNR penalizes that offset heavily. After independent min-max normalization or affine calibration, the multi-frame result is far better than the single-frame result.

## Spatial Error Distribution

For `clay_ms` 450 nm:

| Method | Flat-50 MSE | Structure-25 MSE | Structure-25 SSE share |
|---|---:|---:|---:|
| single | 0.02629 | 0.01985 | 0.2043 |
| multi | 0.02631 | 0.02634 | 0.2501 |

The error is not primarily a high-structure localized failure. It is broad low-frequency / amplitude error. This rejects the initial hypothesis that PSNR is mainly diluted by spatially flat regions; the stronger explanation is that visual figures normalize display ranges while raw PSNR uses absolute `[0, 1]` values.

## Frequency Error Distribution

For `clay_ms` 450 nm:

| Method | Low `<=0.1` | Mid `0.1-0.3` | High `>0.3` |
|---|---:|---:|---:|
| single | 0.9860 | 0.0100 | 0.0040 |
| multi | 1.0000 | 0.0000 | 0.0000 |

The `clay_ms` 450 nm raw-PSNR anomaly is almost entirely low-frequency / DC error. High-frequency error is not the dominant issue.

## Normalization Audit

CAVE preprocessing performs per-scene min-max normalization once when writing the HDF5 dataset. The GT stored in `test.h5` and saved in the NPZ appendices is already normalized. Reconstruction outputs are not independently min-max normalized before raw PSNR calculation.

The figures are different:

- `plot_recon_comparison` uses Matplotlib autoscaling per image panel unless fixed limits are supplied.
- `plot_recon_rgb_pseudocolor_comparison` explicitly applies per-panel percentile normalization.

Therefore, visual figures emphasize structural similarity, while raw PSNR measures absolute radiometric agreement. Both observations can be simultaneously true.

## Conclusion

This is primarily a metric-interpretation and visualization-scaling issue, not evidence that multi-frame reconstruction is worse. The existing raw PSNR values are numerically reproducible, but raw PSNR alone is insufficient for the thesis claim.

Recommended reporting:

- Keep raw PSNR as an absolute amplitude metric.
- Add per-band correlation and SSIM as standard companion metrics.
- Label display-normalized or affine-calibrated PSNR as diagnostic structural metrics only.
- Explicitly state that `clay_ms` multi-frame reconstruction recovers structure but retains amplitude calibration error, especially at 450 nm and 650 nm.
