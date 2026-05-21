# Phase 3.6 H-Matrix DC and OTF Display Response

## OTF Figure Update

The OTF magnitude grid is a qualitative sanity figure, not a quantitative H-rank proof. The previous grid displayed too many masks in one panel, making the layout visually dense.

The regenerated handoff now uses a representative subset:

- Figure: `thesis/h_matrix_diagnostics/figures/otf_magnitude_grid_selected_masks.png`
- FFT-shifted figure: `thesis/h_matrix_diagnostics/figures/otf_magnitude_grid_selected_masks_fftshifted.png`
- Layout: 3 wavelengths by 4 representative masks
- Purpose: show representative OTF magnitude diversity without overcrowding the panel

The H-rank and condition-number conclusions are still based on all selected masks and all 256x256 frequency points, not on the displayed OTF subset.

## DC Coordinate

The H diagnostics use NumPy `fft2` indexing before display.

- In the unshifted FFT arrays, DC is at pixel `(0, 0)`.
- In FFT-shifted figures, DC is displayed at pixel `(128, 128)` for the 256x256 grid.

Therefore, the answer to "(a)" is:

> In FFT-shifted H/OTF figures, the DC point is at pixel `(128, 128)`.

## DC Rank Behavior

At the unshifted DC point `(0, 0)`, the singular values are:

```text
[5.99999972024777, 2.1571531786409853e-07, 1.8186017724191896e-07]
```

The configured relative rank threshold is `1e-8`, so the absolute cutoff is approximately:

```text
1e-8 * 5.99999972024777 = 5.99999972024777e-08
```

Both small singular values are above this cutoff, so the stored rank map reports:

```text
rank(DC, threshold=1e-8) = 3
```

However, the same point is extremely ill-conditioned:

```text
condition(DC) = 32992378.034290873
```

Rank sensitivity at DC:

| Relative threshold | DC rank |
|---:|---:|
| `1e-6` | 1 |
| `1e-8` | 3 |
| `1e-10` | 3 |
| `1e-12` | 3 |

Therefore, the answer to "(b)" is:

> The saved rank map reports DC as rank 3 under the `1e-8` numerical threshold. This should not be interpreted as strong DC channel separation. Physically and effectively, DC is near rank-1 because sum-normalized PSFs make the DC response nearly identical across masks/wavelengths; the rank-3 result comes from tiny numerical residual singular values above the strict `1e-8` cutoff.

## Rank Excluding DC

After excluding the DC point:

```text
full-rank points excluding DC = 65535 / 65535
percentage = 100.0%
```

Therefore, the answer to "(c)" is:

> Excluding DC, the remaining frequency points are still 100.0% full-rank under the same `1e-8` threshold.

## Thesis Interpretation

The correct thesis-facing statement is:

> The multi-frame H matrix is numerically full-rank across the analyzed grid under the configured threshold. The DC point is a special low-frequency case: it is reported as rank 3 at `1e-8`, but is effectively near rank-1 and very ill-conditioned. The non-DC frequencies remain full-rank after removing DC, so the non-degenerate encoding evidence does not depend on the DC point.

This interpretation is more precise than saying simply "all frequencies are full-rank" without qualification.
