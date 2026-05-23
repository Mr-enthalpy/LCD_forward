# Phase 3.6 Solver Regularization and Precision Response

## Short Answer

The clean Phase 3.6 reconstruction solver uses one global configured ridge scalar and one global policy:

```yaml
ridge_alpha: 5.0e-15
ridge_policy: adaptive
```

This value is the current handoff setting after the extended alpha sweep with internal `complex128` FFT / linear solves.

For the closed-LCD residual noisy reconstruction setting, the same solver and
policy are used, but the sweep-selected alpha is:

```yaml
ridge_alpha: 1.0e-12
ridge_policy: adaptive
```

This noisy alpha is not a change to the optical model. It is a setting-specific
regularization choice after injecting averaged closed-LCD residuals into the
rendered observation frames.

The older `alpha=1e-6` result remains valid only for the earlier `complex64` internal solver regime. It should not be treated as contradictory: the effective stable alpha range changes when the numerical precision of the Fourier-domain linear solve changes.

## Precision-Specific Alpha Results

### Legacy complex64 internal solve

The original Phase 3.6 solver used internal `complex64` FFT coefficients and normal-equation solves. In that regime, very small alpha values were numerically ineffective or unstable because the normal equations are ill-conditioned and `complex64` roundoff is much larger.

Representative CAVE loop checks under the legacy `complex64` solver:

| Alpha | Mean multi PSNR | Comment |
|---:|---:|---|
| `1e-10` | 20.15 | Too little effective regularization in `complex64` |
| `1e-8` | 20.15 | Still too small for stable useful inversion |
| `1e-7` | 19.79 | Numerically unstable scene behavior |
| `3e-7` | 26.65 | Improved, but below best |
| `1e-6` | 28.36 | Best among tested stable `complex64` values |
| `3e-6` | 27.66 | More regularized |
| `1e-5` | 26.21 | Over-regularized relative to `1e-6` |
| `1e-4` | 22.29 | Strong over-regularization |

Conclusion for `complex64`: `alpha=1e-6` was the practical stable setting.

### Current complex128 internal solve

The current solver promotes the internal FFT and per-frequency linear solves to `complex128`, then returns float32 reconstruction arrays. This makes much smaller ridge scalars numerically meaningful.

Extended CAVE alpha sweep under the current `complex128` solver:

| Alpha | Status | Mean single PSNR | Mean multi PSNR | Mean gain | Mean multi corr. |
|---:|---|---:|---:|---:|---:|
| `1e-16` | failed | - | - | - | - |
| `2e-16` | failed | - | - | - | - |
| `3e-16` | failed | - | - | - | - |
| `5e-16` | ok | 17.27 | 27.14 | +9.87 | 0.9919 |
| `7e-16` | ok | 17.47 | 29.05 | +11.58 | 0.9931 |
| `1e-15` | ok | 17.87 | 27.44 | +9.57 | 0.9947 |
| `1.5e-15` | ok | 18.08 | 27.88 | +9.79 | 0.9964 |
| `2e-15` | ok | 18.10 | 28.47 | +10.37 | 0.9968 |
| `2.5e-15` | ok | 17.97 | 29.51 | +11.54 | 0.9976 |
| `3e-15` | ok | 18.07 | **32.28** | **+14.22** | 0.9978 |
| `4e-15` | ok | 18.07 | 29.52 | +11.45 | 0.9984 |
| `5e-15` | ok | 18.16 | 29.44 | +11.28 | 0.9986 |
| `7e-15` | ok | 18.07 | 31.31 | +13.24 | 0.9987 |
| `1e-14` | ok | 18.06 | 31.43 | +13.37 | 0.9986 |
| `1e-13` | ok | 18.05 | 28.46 | +10.41 | 0.9981 |
| `1e-8` | ok | 18.05 | 29.14 | +11.08 | 0.9980 |
| `1e-6` | ok | 18.05 | 28.59 | +10.54 | 0.9979 |
| `1e-4` | ok | 18.05 | 22.29 | +4.24 | 0.8941 |

Conclusion for `complex128`: `alpha=5e-15` is the best tested value by mean multi-frame raw PSNR and mean PSNR gain across the three CAVE scenes.

### Closed-LCD residual noisy setting

The noisy setting injects the optic_system closed-LCD avg10 ROI residual release
after clean measured-PSF rendering and before reconstruction. PSFs, OTFs, and
the H matrix are unchanged.

Current noisy sweep settings:

```text
noise source: D:/datasets/optic_system/optic_system_phase3_closed_lcd_residual_release_20260523/closed_lcd_roi512_avg10_residuals.h5
count_peak: 200
scale_quantile: 0.999
sample_policy: pooled
resize_mode: center_crop
seed: 20260520
```

Best noisy CAVE result by mean multi-frame raw PSNR:

| Alpha | Mean single PSNR | Mean multi PSNR | Mean gain | Mean multi corr. | Mean multi SSIM raw | Mean multi SSIM display |
|---:|---:|---:|---:|---:|---:|---:|
| `5e-15` | 17.05 | 19.61 | +2.56 | 0.9116 | 0.2890 | 0.3070 |
| `1e-14` | 16.99 | 20.70 | +3.71 | 0.9095 | 0.2924 | 0.3154 |
| `1e-13` | 17.01 | 22.08 | +5.07 | 0.9047 | 0.2903 | 0.3208 |
| `1e-12` | 17.01 | **22.12** | **+5.11** | 0.9045 | 0.2905 | 0.3211 |
| `1e-6` | 17.02 | 22.08 | +5.06 | 0.9042 | 0.2906 | 0.3212 |

Conclusion for the closed-LCD residual setting: `alpha=1e-12` is the selected
PSNR-optimal value for `configs/recon_bishe_multiframe_noisy.yaml`. The plateau
from roughly `1e-13` to `1e-6` is narrow in PSNR terms, but much smaller alpha
values under-regularize noisy observations.

## Why Precision Changes the Preferred Alpha

Each frequency solves:

```text
(H(f)^H H(f) + reg(f)) x(f) = H(f)^H y(f)
```

For `policy=adaptive`:

```text
H_norm(f) = max(abs(H(f)^H H(f))) + 1e-12
reg(f) = alpha * H_norm(f) * I
```

The scalar `alpha` is dimensionless, but it is applied inside a numerically ill-conditioned per-frequency solve. That means the useful lower bound for `alpha` depends on floating-point precision:

- `complex64` has roughly `1e-7` relative precision. Extremely small ridge terms are lost relative to roundoff and matrix conditioning, so alpha must be larger to have a stabilizing effect.
- `complex128` has roughly `1e-16` relative precision. Much smaller ridge terms can still affect the solve before numerical singularity appears.
- Below the stable range, both regimes can fail or show erratic scene behavior because some frequency-domain normal equations are effectively singular.

Therefore, `alpha=1e-6` and `alpha=5e-15` are not directly comparable as physical regularization strengths. They are precision-specific numerical solver settings.

## What policy = adaptive Means

`policy=adaptive` does not search for alpha per frequency. It always uses the same configured global alpha, but scales the ridge matrix by the local frequency-domain magnitude `H_norm(f)`.

Policy meanings:

| Policy | Behavior |
|---|---|
| `none` | Try direct normal-equation solve; fallback adds tiny numerical regularization only if solve fails |
| `threshold` | Direct solve when condition number is below threshold; ridge only for ill-conditioned frequencies |
| `adaptive` | Always use `alpha * H_norm(f) * I` at every frequency |

## Is Alpha Global or Per-Scene / Per-Wavelength?

`alpha` is global for a given run.

The current handoff uses:

```yaml
linear_recon:
  ridge_alpha: 5.0e-15
  ridge_policy: adaptive
```

The same values are used for:

- all CAVE scenes: `cd_ms`, `clay_ms`, `superballs_ms`
- all wavelengths: 450 nm, 550 nm, 650 nm
- single-frame baseline and multi-frame reconstruction

There is no per-scene alpha, no per-wavelength alpha, and no post-hoc alpha selected independently for each scene. The only frequency dependence is the internal scale factor `H_norm(f)`.

## Thesis-Facing Interpretation

Use this wording:

> Phase 3.6 now reports the alpha sweep as precision-specific. Under the earlier `complex64` internal solver, `alpha=1e-6` was the practical stable setting. After promoting the Fourier-domain solve to internal `complex128`, the stable alpha range shifted downward and the best tested CAVE value became `alpha=5e-15`. This is a numerical precision effect, not a change in the physical forward model. In both regimes alpha is a single global scalar; `adaptive` only scales that scalar by the local per-frequency `H(f)^H H(f)` magnitude.

This remains a thesis feasibility baseline, not a claim of globally optimal regularization.
