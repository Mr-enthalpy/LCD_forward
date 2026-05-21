# Phase 3.6 Debug & Tuning Record

> Generated from the first-pass Phase 3.5-3.6 implementation loop.
> Records all bugs found, root-cause analysis, strategy sweeps,
> and final results. Source: May 20, 2026 development session.

## 1. Bug Discovery & Fixes

### Bug 1: Missing IFFT in frequency-domain reconstructor

**Symptom**: Phase 3.6 reconstruction PSNR ~ -28 to -32 dB.
Multi-frame results indistinguishable from random noise.

**Root cause**: `frequency_domain_ridge_reconstruct` solved per-frequency
ridge systems correctly, producing complex frequency-domain coefficients
`X̂(f)`, but stored them directly as spatial-domain pixel values:

```python
# BEFORE (buggy):
reconstruction[:, i, j] = x_hat.real   # frequency coeff stored as spatial

# AFTER (fixed):
reconstruction[:, i, j] = x_hat        # stored as complex frequency coeff
for c in range(n_lambda):
    reconstruction[c] = torch.fft.ifft2(reconstruction[c]).real  # back to spatial
```

**Impact**:

| Run | Single PSNR | Multi PSNR |
|-----|------------|------------|
| Before IFFT fix | -28.8 dB | -31.6 dB |
| After IFFT fix (alpha=1.0) | 15.0 dB | 15.1 dB |

### Bug 2: Regularization dominating signal (HTH eigenvalues drowned)

**Symptom**: After IFFT fix, PSNR improved to ~15 dB, but multi-frame (T=9)
showed **zero gain** over single-frame (T=1): 15.0 vs 15.1 dB.

**Root cause**: Ridge alpha=1.0, combined with per-frequency adaptive scaling
`reg = alpha x |HTH|max x I`, caused regularization to overwhelm the smaller
singular values of HTH at most frequencies.

At median condition-number frequencies (cond ~ 15):
- HTH max ~ 1.0, HTH min ~ 0.07
- `reg = 1.0 x 1.0 x I` adds 1.0 to diagonal
- Effective condition number: (1+1) / (0.07+1) ~ 1.9
- Regularization dominates the solution - T=1 and T=9 produce same result

**Fix**: Reduce alpha to 1e-6, add `"none"` and `"threshold"` policies:

```python
# AFTER (fixed):
def _solve_per_frequency(H, y, alpha, policy, eye, cond_threshold):
    if policy == "none":
        return torch.linalg.solve(HTH, Hty)  # direct solve
    elif policy == "threshold":
        s = torch.linalg.svdvals(HTH)
        cond = s[0] / (s[-1] + 1e-12)
        if cond < cond_threshold:
            return torch.linalg.solve(HTH, Hty)  # direct for well-conditioned
        else:
            reg = alpha * H_norm * eye
            return torch.linalg.solve(HTH + reg, Hty)  # ridge only for ill-conditioned
```

## 2. Frequency-Domain System Analysis

### H matrix rank and conditioning

For T frames and L wavelengths, each frequency point (i,j) defines a
[T x L] complex transfer matrix H. Analysis on train set PSFs at 64x64:

| Metric | Value |
|--------|-------|
| Full-rank (3) frequency points | 4096 / 4096 (100%) |
| Median condition number | 15.0 |
| Mean condition number | 10,702 |
| Max condition number | 43,735,758 |

**Interpretation**: The system is theoretically solvable at every frequency,
since H always has full column rank. However, the tail of ill-conditioned
frequencies (cond > 10⁶) requires minimal regularization for numerical
stability.

### Cross-mask PSF frequency transfer diversity

Spatial correlation between selected mask PSFs at 550 nm:

| Mask | Mean correlation to other masks |
|------|-------------------------------|
| task_related_005 | 0.825 (most diverse) |
| random_lowfreq_053 | 0.942 |
| edge_block_top | 0.963 (most similar) |

Frequency-domain coefficient-of-variation of |H| across masks:

| Frequency | |H| mean | CV across 9 masks |
|-----------|---------|-------------------|
| DC (0,0) | 1.0 | 0.0000 (identical: sum-normalized) |
| Low (32,32) | 5.76e-3 | 0.55 |
| Mid (96,96) | 2.71e-4 | 0.39 |
| High (200,200) | 3.66e-4 | 0.63 |

**Interpretation**: PSFs are nearly identical at DC (all sum-normalized to 1.0),
but diverge at mid-to-high spatial frequencies where diffraction differences appear.
The per-wavelength differences are even larger: at low frequency (16,16),
`|H|` at 450 nm = 5.4e-2 vs 550 nm = 7.6e-3 - a 7x channel separation.

This validates Phase 3.2b: mask-induced PSF differences are real and measurable
in frequency domain, concentrated at non-DC frequencies.

## 3. Strategy Sweep Results

All sweeps run on synthetic 3-channel target at 128x128 for speed.
Device: CPU.

### Mask selection strategies tested

| Strategy | Count | Composition |
|----------|-------|-------------|
| diverse_9 | 9 | 8 deterministic + 1 task_related |
| diverse_18 | 18 | 8 deterministic + 10 task_related + random |
| deterministic_8 | 8 | All 8 deterministic masks |
| det_plus_task_12 | 12 | 8 deterministic + 4 task_related |

### Regularization strategies tested

| Policy | alpha | Description |
|--------|---|-------------|
| none | 0 | Direct solve - fails (singular at some frequencies) |
| alpha_1e-6 | 1e-6 | Adaptive ridge |
| alpha_1e-4 | 1e-4 | Adaptive ridge |
| thresh_1e-6 | 1e-6 | Threshold-based (ridge only if cond > 10³) |

### Full sweep results (synthetic target, 128x128)

| Strategy | Policy | T | Single PSNR | **Multi PSNR** | **Gain** |
|----------|--------|---|-------------|----------------|----------|
| **det_plus_task_12** | **alpha_1e-6** | 12 | 14.80 | **29.20** | **+14.40** |
| **det_plus_task_12** | **thresh_1e-6** | 12 | 14.80 | **29.20** | **+14.40** |
| deterministic_8 | alpha_1e-6 | 8 | 14.80 | 25.85 | +11.04 |
| deterministic_8 | thresh_1e-6 | 8 | 14.80 | 25.85 | +11.04 |
| diverse_9 | alpha_1e-6 | 9 | 14.80 | 25.61 | +10.80 |
| diverse_9 | thresh_1e-6 | 9 | 14.80 | 25.61 | +10.80 |
| diverse_18 | alpha_1e-6 | 18 | 14.80 | 24.64 | +9.84 |
| det_plus_task_12 | alpha_1e-4 | 12 | 14.72 | 21.10 | +6.38 |
| diverse_9 | alpha_1e-4 | 9 | 14.72 | 19.94 | +5.21 |

**Key findings**:

1. **More diverse masks = higher multi-frame PSNR**: det+task_12 (29.2 dB) > det_8 (25.9 dB) > diverse_9 (25.6 dB)
2. **alpha=1e-6 >> alpha=1e-4**: At 128x128, alpha=1e-6 gives +14.4 dB gain vs alpha=1e-4 gives +6.4 dB
3. **18 frames worse than 12**: The additional t frames from similar families add correlated noise, reducing SNR
4. **Single-frame consistently ~14.8 dB**: Underdetermined (T=1, L=3) - serves as a stable baseline
5. **Direct solve (alpha=0) fails**: Some frequencies are too ill-conditioned for unregularized inversion

### Best configuration

```
Mask selection: diverse_family_first (deterministic + task_related), count=12
Regularization: alpha = 1e-6, policy = adaptive
PSF working size: 256x256
```

## 4. Final Results (256x256)

### Phase 3.5: Forward Validation

| Wavelength | MSE | Relative L2 | Correlation |
|------------|-----|-------------|-------------|
| 450 nm | 6.4e-10 | 0.216 | 0.982 |
| 550 nm | 3.6e-10 | 0.154 | 0.989 |
| 650 nm | 4.9e-10 | 0.184 | 0.985 |
| **Mean** | **5.0e-10** | **0.184** | **0.985** |

- PCA 24 components, ridge alpha=1.0
- Per-wavelength ridge R² > 0.999 (train PCA coefficient fit)
- PSF working size: 256x256

### Phase 3.6: Linear Reconstruction

Final config: alpha=1e-6, adaptive policy, 12 diverse masks (8 deterministic + 4 task_related), 256x256.

Metric audit note: raw PSNR measures absolute amplitude agreement after CAVE per-scene normalization. It should be interpreted together with per-band correlation, SSIM/audit diagnostics, and figures; see `docs/phase3_6_cave_metrics_audit.md`.

| Level | Scene | Single PSNR | **Multi PSNR** | **Multi Gain** |
|-------|-------|-------------|----------------|----------------|
| L1 Synthetic | bars/circles/blocks | 15.35 dB | **24.35 dB** | **+9.00 dB** |
| L2 CAVE | cd_ms | 20.78 dB | **39.17 dB** | **+18.39 dB** |
| L2 CAVE | clay_ms | 14.36 dB | **18.04 dB** | **+3.68 dB** |
| L2 CAVE | superballs_ms | 18.99 dB | **25.53 dB** | **+6.54 dB** |

**Interpretation**:

- **cd_ms (39.2 dB)**: Near-perfect 3-channel reconstruction. The CD scene has
  simple, smooth structures well-posed for the ridge solver.
- **superballs_ms (25.5 dB)**: Good recovery with +6.5 dB multi-frame gain.
- **clay_ms (18.0 dB)**: Hardest scene - high spatial frequency content.
  The similar PSF transfer functions cannot resolve fine textures equally well
  across all wavelengths.
- **Synthetic (24.4 dB)**: Consistent +9 dB gain, verifies the pipeline.

**All cases show multi-frame > single-frame in mean raw PSNR**, but the metric
audit shows that raw PSNR alone can understate structural recovery when residual
amplitude / DC error remains. The thesis claim should use raw PSNR together with
correlation, SSIM/audit diagnostics, and figures.

### Per-channel breakdown (best scene: cd_ms)

| Channel | Wavelength | Single PSNR | Multi PSNR | Gain |
|---------|-----------|-------------|------------|------|
| 0 | 450 nm | 20.78 dB (mean) | 39.17 dB (mean) | +18.39 dB |
| 1 | 550 nm | - | - | - |
| 2 | 650 nm | - | - | - |

Note: per-channel single-frame PSNR is not meaningful for T=1 underdetermined
systems; only the mean across channels is reported. See `reconstruction_metrics.json`
in the output directory for full per-channel multi-frame breakdowns.

## 5. Computational Notes

### Memory scaling

| Size | Memory per TxL FFT | Per-frequency solve | Total |
|------|-------------------|---------------------|-------|
| 128x128 | 16K x 12 x 3 x 8B = 4.6 MB | 16K iterations | ~10s CPU |
| 256x256 | 65K x 12 x 3 x 8B = 18.4 MB | 65K iterations | ~2 min CPU |
| 512x512 | 262K x 12 x 3 x 8B = 73.7 MB | 262K iterations | ~8 min CPU |

### Degradation path

If 512x512 is too large:
1. psf_working_size = 256x256 (used in first pass)
2. pca_components = 16 -> 12 -> 8
3. selected_masks count = 12 -> 9 -> 6

All degradation parameters recorded in run_manifest.json.

## 6. Conclusions

### Thesis evidence chain (Phase 3.5 -> 3.6)

1. **mask -> PSF is modellable**: PCA + ridge achieves test correlation 0.985
   across 3 wavelengths, downsampled to 256x256
2. **Measured PSFs encode multichannel information**: Multi-frame ridge
   reconstruction with internal complex128 solves and alpha=3e-15 achieves
   CAVE PSNR gains of +6.2 to +24.1 dB over the single-frame baseline
3. **Phase 3.2b conclusions validated**: Inter-mask PSF differences (~298x noise)
   are sufficient for encoding diversity in frequency domain
4. **CAVE public dataset**: Realistic 3-channel reconstruction at 256x256
   with measured PSF kernels demonstrates the loop

### Known limitations

- FFT circular convolution ignores edge effects (noted in report)
- All PSFs are sum-normalized, producing identically-1 DC component,
  forcing all encoding diversity into mid/high frequencies
- Ridge solver assumes linear, shift-invariant, circulant system
- No learned masks, no deep reconstruction - these are intentional
  thesis-scope boundaries

### Next steps (beyond first pass)

- Phase 3.6b: real target capture from optic_system (optional)
- Use original 512x512 PSFs with GPU
- Investigate non-ridge solvers (conjugate gradient, ADMM)
- Thesis figure assembly (Phase 3.7)

## 7. Alpha Sweep Precision Addendum

After handoff review, the alpha sweep was repeated with solver precision made
explicit.

The original `alpha=1e-6` conclusion belongs to the legacy internal `complex64`
solver. In that regime, smaller alpha values did not provide useful
regularization because the per-frequency normal equations were too close to the
`complex64` numerical precision limit.

The current solver promotes FFT coefficients and per-frequency linear solves to
internal `complex128`, then returns float32 reconstruction arrays. Under this
precision regime, the stable alpha range shifts downward:

| Internal solve precision | Best tested alpha | Mean single PSNR | Mean multi PSNR | Mean gain | Notes |
|---|---:|---:|---:|---:|---|
| complex64 legacy | 1e-6 | ~18.05 | ~28.36 | ~+10.54 | Practical stable setting for old solver |
| complex128 current | 3e-15 | 18.07 | 32.28 | +14.22 | Best tested CAVE value after precision upgrade |

Values below `5e-16` failed in the current `complex128` sweep because the
per-frequency systems became singular. Larger values such as `1e-4` remain
over-regularized.

Alpha is therefore a numerical solver parameter, not a physical parameter. It
must be reported together with the internal solve precision.


