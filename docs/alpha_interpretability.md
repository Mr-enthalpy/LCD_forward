# Alpha Interpretability: Why α ≈ 10⁻¹⁵?

## Summary

The Phase 3.6 ridge regularization parameter α ∈ [3×10⁻¹⁵, 5×10⁻¹⁵] is approximately 12 orders of magnitude smaller than typical Tikhonov regularizers. This extreme value is a direct consequence of the imaging physics: non-coherent mono-LCD encoding concentrates channel-separation information at mid spatial frequencies where the OTF magnitude is orders of magnitude weaker than at DC. An α of zero causes numerical collapse because the per-frequency transfer matrix is nearly singular at multiple frequency points (especially DC).

This document explains both constraints and connects them to the measurable H-matrix diagnostics.

---

## 1. Forward Model in the Fourier Domain

The multi-frame rendering equation in the spatial domain is:

```
I_t(x,y) = Σ_λ H_t,λ(x,y) ∗ O_λ(x,y)
```

After 2D FFT, the convolution becomes point-wise multiplication at each frequency (u,v):

```
y(u,v) = H(u,v) · x(u,v)     where H ∈ C^(T×L)
```

- **T** = number of frames (diverse LCD masks)
- **L** = number of wavelength channels (3: 450, 550, 650 nm)
- **y(u,v) ∈ C^T** = observed frame DFT coefficients
- **x(u,v) ∈ C^L** = unknown spectral object DFT coefficients
- **H(u,v) ∈ C^(T×L)** = per-frequency transfer matrix (OTF samples from T masks × L wavelengths)

Reconstruction solves the linear system at each (u,v) independently:

```
x̂(u,v) = (HᴴH + α·||HᴴH||_max·I)⁻¹ Hᴴ y(u,v)
```

The regularization term `α·||HᴴH(f)||_max·I` is **frequency-scaled**: the scalar α multiplies the local spectral norm of HᴴH.

---

## 2. Why α Must Be Extremely Small (~10⁻¹⁵)

### 2.1 The Frequency-Layer Structure of H(f)

The transfer matrix H(u,v) has dramatically different behavior across the frequency spectrum:

| Frequency band | H(f) properties | Condition number | Encoding utility |
|---|---|---|---|
| **DC (0,0)** | All rows ≈ [1,1,1] (sum-normalized PSFs) | ~3.3×10⁷ | Near rank-1: no channel separation |
| **Low (0–0.05 Nyquist)** | Slow divergence from DC | ~10³–10⁵ | Weak separation emerging |
| **Mid (0.05–0.3 Nyquist)** | Cross-mask diversity + wavelength-specific OTF differences | ~10¹–10³ | **Maximum encoding information** |
| **High (>0.3 Nyquist)** | OTF magnitude roll-off | ~10²–10⁴ | Signal-to-noise degrades |

### 2.2 Dynamic Range Imbalance

At DC, `||HᴴH||_max` is proportional to (sum of normalized PSF power)² ≈ (L·T)² ≈ 36² = ~1300. At mid frequencies, the same quantity is 10⁴–10⁶× smaller because PSF OTF magnitudes drop sharply away from DC.

The regularization term at frequency (u,v) is:

```
reg(f) = α · ||HᴴH(f)||_max · I
```

Since the scalar α is global, the **effective regularization ratio** at mid frequencies is:

```
reg(mid) / signal(mid) ≈ α · (large DC norm) / (tiny mid signal)
```

This means:

| α | Effective mid-freq regularization | Effect on reconstruction |
|---|---|---|
| 1e-2 | Massive — completely overwhelms mid-freq encoding | No channel separation; PSNR ≈ 18.8 dB; SSIM ≈ 0.67 |
| 1e-6 | Significant — damps mid-freq channel separation | Degraded reconstruction; PSNR ≈ 28.6 dB; SSIM ≈ 0.91 |
| 3e-15 | Just enough to stabilize DC, negligible at mid frequencies | Full encoding preserved; PSNR ≈ 29.5–32.3 dB; SSIM ≈ 0.96–0.99 |

### 2.3 Physical Interpretation

The fundamental issue is **architectural, not numerical**: non-coherent imaging with sum-normalized mono LCD PSFs creates a huge dynamic range between DC and mid-frequency transfer. The encoding information lives at frequencies where the OTF magnitude is weakest, so regularization must be proportionally tiny.

> The optimal α is the smallest value that prevents numerical singularity at DC while introducing negligible attenuation at the mid frequencies where channel-separation encoding resides.

---

## 3. Why α Cannot Be Zero

### 3.1 Singular Value Collapse at DC

At α = 0, the per-frequency solve reduces to the (regularized) pseudo-inverse:

```
x̂(u,v) = (HᴴH)⁻¹ Hᴴ y(u,v)                     (if H is full rank)
```

The H-matrix diagnostics (from `thesis/h_matrix_diagnostics/h_matrix_diagnostics.json`) show:

```
Singular values at DC: [5.9999997, 2.16×10⁻⁷, 1.82×10⁻⁷]
Condition number at DC: 32,992,378
```

The DC condition number of ~3.3×10⁷ means that the pseudo-inverse amplifies any measurement noise or numerical rounding error at DC by a factor of ~10⁷. This contamination spreads to all spatial positions via the inverse FFT — the enormous DC inversion error dominates the entire reconstruction.

### 3.2 Other Near-Singular Frequencies

While the median condition number across the 256×256 frequency grid is a few tens, the distribution has a heavy tail:

- Many frequency points have condition > 10³
- Some exceed 10⁶

Without regularization, these near-singular points produce large-magnitude inversion artifacts that appear as structured noise in the spatial reconstruction.

### 3.3 Numerical Failure Boundary

In practice, the Fourier-domain linear solve via `torch.linalg.solve(HᴴH + reg·I, Hᴴy)` uses Cholesky decomposition internally. When `α·||HᴴH||_max` falls below the diagonal of HᴴH by more than the floating-point precision (complex128: ~2×10⁻¹⁶), the regularization has no numerical effect and the matrix is treated as singular.

This produces an **abrupt transition**:

| α | Behavior |
|---|---|
| ≥ 5×10⁻¹⁶ | Solves succeed; reconstruction stable |
| 3×10⁻¹⁶ | Solves begin failing on some frequency points |
| ≤ 1×10⁻¹⁸ | Catastrophic failure: `torch.linalg.solve` raises `RuntimeError` |

The gap between the stable lower bound (5×10⁻¹⁶) and the optimal value (3×10⁻¹⁵) is only ~1 order of magnitude — an extremely narrow window.

---

## 4. Connection to H-Matrix Diagnostics

The H-matrix frequency diagnostics (`scripts/analyze_h_matrix_diagnostics.py`) confirm the structure that drives α selection:

| Diagnostic | Value | Relevance to α |
|---|---|---|
| DC condition number | 3.3×10⁷ | Explains why α ≠ 0: DC would dominate reconstruction error |
| Non-DC full-rank % | 100% | Confirms encoding is non-degenerate; reconstruction is possible |
| Median condition number | ~10–100 | Mid frequencies are well-conditioned |
| Max condition (non-DC) | ~10⁶ | Heavy tail requires regularization even away from DC |
| Cross-mask CV peak location | Mid-high frequencies | Diversity is concentrated where OTF magnitude is weakest |

The H-rank map (both unshifted and FFT-shifted) shows full rank (3) across all non-DC frequency points. The singular value maps show that the second and third singular values are orders of magnitude smaller at DC than elsewhere, confirming that DC is the singular-value-collapse locus.

---

## 5. Practical Implications

### 5.1 Scene Dependence

The "optimal" α varies across scenes because each scene has a different spectral distribution of spatial frequencies. The cd_ms scene (a CD cover with high-frequency text/detail) benefits from slightly more regularization at α = 5×10⁻¹⁵ (PSNR = 61.7 dB), while clay_ms and superballs_ms peak earlier at α ∈ [4×10⁻¹⁵, 7×10⁻¹⁵]. The mean multi-scene PSNR peaks at α = 5×10⁻¹⁵, but any value in [3×10⁻¹⁵, 7×10⁻¹⁵] produces within 1–3 dB of the best result. This reinforces that α is a numerical stability parameter, not a physically tunable regularizer — the system's effective regularization is dominated by the frequency-scaled H_norm structure, not the global scalar α.

### 5.1 Precision Dependence

The usable α range depends on the internal numerical precision:

| Precision | Machine epsilon | Stable α range | Optimal α |
|---|---|---|---|
| complex64 | ~6×10⁻⁸ | 3×10⁻⁷ to 10⁻² | ~10⁻⁶ |
| complex128 | ~1×10⁻¹⁶ | 5×10⁻¹⁶ to 10⁻² | ~3×10⁻¹⁵ |

The optimal α is precision-specific, not physically absolute. The complex128 solver was chosen precisely because it enables the extremely small regularization that the physics demands.

### 5.2 Policy: Why "adaptive" Rather Than "threshold" or "none"

| Policy | Behavior | Why not chosen |
|---|---|---|
| `none` | Direct solve at all frequencies | Fails at DC and near-DC frequencies |
| `threshold` | Ridge only where cond > 10³ | Creates discontinuity in regularization; DC gets different treatment than adjacent frequencies |
| `adaptive` | Always α·||HᴴH||·I | Smooth frequency-scaled regularization; stable and well-behaved |

The `adaptive` policy produces the smoothest frequency-domain behavior because every frequency receives regularization proportional to its local norm — α is truly a single dimensionless scalar controlling the global regularization strength.

### 5.3 Single Frame Baseline is Unaffected

Single-frame reconstruction (T=1) is inherently ill-posed regardless of α: H ∈ C^(1×L) is a row vector, so the system is severely underdetermined. Single-frame PSNR (~18 dB) is invariant across α and reflects only the panchromatic averaging inherent in a single capture. The multi-frame gain (up to +14 dB) is the key reconstruction metric.

---

## 6. Thesis-Facing Statement

> The optimal ridge regularization parameter for the LCD Phase 3.6 Fourier-domain reconstruction lies in the range α ∈ [3×10⁻¹⁵, 5×10⁻¹⁵]. This extreme value follows from the imaging physics: the LCD PSF encoding diversity is concentrated at mid spatial frequencies where OTF magnitudes are 10⁴–10⁶× weaker than at DC, but the per-frequency regularization penalty is scaled by the local (DC-dominated) spectral norm of HᴴH. Alpha cannot be zero because the DC frequency point has a condition number exceeding 3×10⁷, causing numerical collapse of the Fourier-domain linear solve. The optimal α is the smallest value that stabilizes the DC inversion while introducing negligible attenuation at mid frequencies — a gap of approximately one order of magnitude above the complex128 numerical precision floor. This is a precision-specific solver setting, not a claim of globally optimal regularization.

## 7. Reproduction

```bash
# Run the alpha sweep (produces JSON+CSV+MD in outputs/alpha_sweep/)
python scripts/sweep_cave_recon_alpha.py \
  --release-root D:/datasets/LCD_forward/lcd_forward_phase3_5_3_6_release_20260520 \
  --train-h5 D:/datasets/optic_system/phase3_release_20260520/lcd_forward/psf_dictionary/train.h5 \
  --policy adaptive

# Generate the PSNR+SSIM visualization
python scripts/plot_alpha_metrics.py \
  --sweep-json outputs/alpha_sweep/cave_alpha_sweep.json \
  --output-dir outputs/alpha_sweep
```

Resulting figures in `outputs/alpha_sweep/`:
- `alpha_psnr.png` / `.pdf` - PSNR-vs-alpha figure
- `alpha_ssim.png` / `.pdf` - raw-SSIM-vs-alpha figure
- `alpha_display_ssim.png` / `.pdf` - display-normalized-SSIM-vs-alpha figure
