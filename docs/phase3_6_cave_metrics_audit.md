# Phase 3.6 CAVE Metrics Audit

Generated: 2026-05-21T22:53:40

## Scope

- Release root: `D:/datasets/LCD_forward/lcd_forward_phase3_5_3_6_release_20260520`
- Source arrays: `thesis/phase3_6_linear_recon_cave/data/*/recon_appendix_arrays.npz`
- Stored metrics: `thesis/phase3_6_linear_recon_cave/metrics/cave_recon_metrics.json`
- This audit is offline-only and uses saved CAVE appendices; it does not invoke hardware control.

## Scene-Level Metric Cross-Check

| Scene | Method | Raw PSNR | Raw corr. | Raw SSIM | Min-max SSIM | Min-max PSNR |
|---|---:|---:|---:|---:|---:|---:|
| cd_ms | single | 21.07 | 0.7888 | 0.7310 | 0.6814 | 20.49 |
| cd_ms | multi | 33.36 | 0.9996 | 0.8030 | 0.8448 | 72.81 |
| clay_ms | single | 14.31 | 0.3298 | 0.3225 | 0.2355 | 10.51 |
| clay_ms | multi | 20.54 | 0.9968 | 0.5250 | 0.9205 | 62.60 |
| superballs_ms | single | 18.82 | 0.6882 | 0.5117 | 0.4874 | 18.71 |
| superballs_ms | multi | 42.96 | 0.9970 | 0.7125 | 0.8518 | 65.38 |

## clay_ms 450 nm Detail

| Method | Raw PSNR | Corr. | Min-max PSNR | Affine PSNR | GT mean | Pred mean | FFT low share |
|---|---:|---:|---:|---:|---:|---:|---:|
| single | 16.10 | 0.1063 | 8.09 | 30.33 | 0.0268 | 0.1607 | 0.9861 |
| multi | 26.31 | 1.0000 | 82.45 | 93.53 | 0.0268 | 0.0752 | 1.0000 |

## Findings

- Stored PSNR values reproduce from the NPZ arrays; no metric recomputation mismatch was found.
- Single-frame and multi-frame reconstructions have identical `[3, 256, 256]` shapes against GT, so there is no silent broadcasting in the saved metric calculation.
- The updated clay_ms 450 nm multi-frame result has positive raw-PSNR gain (+10.20 dB) and near-perfect correlation; the earlier negative-gain case is not present for this solver precision / alpha setting.
- The per-band and pseudo-RGB figures use independent display autoscaling / per-panel normalization, so they can show strong structural recovery while raw PSNR remains low.
- Raw PSNR is still valid as an absolute radiometric error metric, but it is insufficient as the only thesis-facing quality metric for this normalized CAVE simulation.

## Recommendation

- Report raw PSNR together with correlation and SSIM.
- Add a display-normalized or affine-calibrated metric only as a diagnostic for structural recovery, not as a replacement for raw radiometric error.
- For clay_ms, explicitly state that multi-frame reconstruction recovers structure; remaining raw-PSNR limits should be interpreted as absolute amplitude / low-frequency calibration error rather than channel-separation failure.
