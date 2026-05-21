# Phase 3.6 Solver Regularization Response

## Short Answer

The Phase 3.6 reconstruction solver uses one global configured ridge scalar:

```yaml
ridge_alpha: 1.0e-6
ridge_policy: adaptive
```

This scalar is not selected per scene and not selected per wavelength. It is passed unchanged to both the single-frame and multi-frame reconstruction calls.

With `policy=adaptive`, the solver applies that global scalar at every Fourier frequency after scaling it by the local normal-equation magnitude:

```text
reg(f) = alpha * max(abs(H(f)^H H(f))) * I
```

So the scalar `alpha` is global, but the effective ridge matrix magnitude is frequency-dependent because `H(f)` changes with frequency.

## Why alpha = 1e-6

The value `1e-6` was chosen from the Phase 3.6 thesis-closure sweep documented in `docs/phase3_6_debug_and_tuning.md`.

The sweep was not an exhaustive hyperparameter optimization. It was a minimal stability check on the synthetic 3-channel target at 128x128, consistent with the thesis-closure constraint to use simple baselines.

Relevant sweep entries:

| Strategy | Policy | T | Single PSNR | Multi PSNR | Gain |
|---|---:|---:|---:|---:|---:|
| det_plus_task_12 | alpha_1e-6 | 12 | 14.80 | 29.20 | +14.40 |
| det_plus_task_12 | thresh_1e-6 | 12 | 14.80 | 29.20 | +14.40 |
| deterministic_8 | alpha_1e-6 | 8 | 14.80 | 25.85 | +11.04 |
| diverse_9 | alpha_1e-6 | 9 | 14.80 | 25.61 | +10.80 |
| det_plus_task_12 | alpha_1e-4 | 12 | 14.72 | 21.10 | +6.38 |
| diverse_9 | alpha_1e-4 | 9 | 14.72 | 19.94 | +5.21 |

Additional debugging showed:

- `alpha=1.0` over-regularized the inverse and collapsed multi-frame benefit.
- `alpha=1e-4` was stable but materially worse than `1e-6` in the sweep.
- `alpha=0` / direct solve was not robust because some frequencies are ill-conditioned.
- `threshold` with `1e-6` gave similar results to `adaptive` with `1e-6`, so `adaptive` was kept as the simpler always-regularized path.

Therefore, `1e-6` is not claimed to be globally optimal. It is the smallest tested stable ridge value that preserved the multi-frame gain while avoiding direct-solve instability.

## What policy = adaptive Means

In `src/recon/linear_spectral_recon.py`, every frequency point solves:

```text
(H(f)^H H(f) + reg(f)) x(f) = H(f)^H y(f)
```

For `policy=adaptive`:

```text
H_norm(f) = max(abs(H(f)^H H(f))) + 1e-12
reg(f) = alpha * H_norm(f) * I
```

This is "adaptive" only in the sense that the regularization magnitude follows the local frequency-domain scale of `H(f)`. It does not run a per-frequency search, cross-validation, or scene-specific alpha tuning.

Policy meanings:

| Policy | Behavior |
|---|---|
| `none` | Try direct normal-equation solve; fallback adds tiny numerical regularization only if solve fails |
| `threshold` | Direct solve when condition number is below threshold; ridge only for ill-conditioned frequencies |
| `adaptive` | Always use `alpha * H_norm(f) * I` at every frequency |

## Is alpha Global or Per-Scene / Per-Wavelength?

`alpha` is global for the Phase 3.6 run.

It is configured once in `configs/bishe_first_pass.yaml`:

```yaml
linear_recon:
  ridge_alpha: 1.0e-6
  ridge_policy: adaptive
```

The same values are used for:

- all CAVE scenes: `cd_ms`, `clay_ms`, `superballs_ms`
- all wavelengths: 450 nm, 550 nm, 650 nm
- single-frame baseline and multi-frame reconstruction

There is no per-scene alpha, no per-wavelength alpha, and no post-hoc alpha selected from CAVE test metrics.

The only frequency dependence is the internal scale factor `H_norm(f)` used by `policy=adaptive`.

## Thesis-Facing Interpretation

The correct handoff wording is:

> Phase 3.6 uses a global ridge scalar `alpha=1e-6` with an adaptive per-frequency scale factor derived from `H(f)^H H(f)`. This value was selected from a minimal synthetic-target stability sweep, not from CAVE test-set optimization. The sweep showed `1e-6` preserved multi-frame gain, `1e-4` was more over-regularized, `1.0` was too large, and direct solve was unstable. The same global alpha and policy are used for every CAVE scene and wavelength.

This is sufficient for a thesis feasibility baseline. It is not a claim that `1e-6` is the globally optimal regularization value.
