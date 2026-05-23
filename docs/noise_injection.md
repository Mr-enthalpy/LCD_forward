# Closed-LCD Residual Noise Injection Plan

This document plans how LCD_forward should consume the optic_system closed-LCD
averaged residual release. It is a design and integration contract, not an optical
claim change.

## Positioning

LCD_forward should use the release only after measured-PSF rendering and before
reconstruction:

```text
object + measured PSF
  -> clean frames
  -> count-domain scaling
  -> add closed-LCD averaged residual
  -> background-subtracted noisy frames
  -> same H-matrix / ridge reconstruction
  -> metrics
```

The residual is injected into observation frames `y_t` only. It must not be added
to PSFs, OTFs, PCA bases, forward-surrogate targets, or the H matrix. The test
question is: how stable is the measured-PSF inverse under averaged closed-LCD
observation residuals?

This module injects exposure-matched 10-frame averaged closed-LCD ROI residuals.
It does not model real camera read noise, shot noise, PRNU, or sensor-only dark.

## Mathematical Contract

The clean measured-operator frame is:

```text
Y_t^clean = sum_lambda X_lambda * h_{t,lambda}
```

The count-domain conceptual model is:

```text
C_t = B + g Y_t^clean + R_t
```

where:

- `B` is the closed-LCD average dark/background frame.
- `R_t` is a 10-frame averaged residual crop sampled from the optic_system release.
- `g` maps LCD_forward normalized clean frames to camera counts.

The reconstruction input remains background-subtracted:

```text
Y_t^noisy = (C_t - B) / g = Y_t^clean + R_t / g
```

Implementation should therefore directly use:

```text
Y_t^noisy = Y_t^clean + R_t / g
```

This matches the current normalized-frame Phase 3.6 pipeline and avoids adding a
background that is immediately subtracted.

## Count-Domain Scale

Default:

```text
g = S_peak / Q_0.999(Y^clean)
S_peak = 200 counts
```

Interpretation: the clean frame 99.9% quantile corresponds to about 200 counts
above the closed-LCD background. Optional stress tests may use 100 and 50 counts.
The main thesis table should use one fixed default unless explicitly labeled as a
sensitivity appendix.

## Residual Release Contract

Expected HDF5 fields:

```text
/closed_lcd/residuals_avg10   [L, R, 512, 512]
/closed_lcd/mean_avg10        [L, 512, 512]
/metadata/wavelengths_nm      [3]
/metadata/exposure_us         [3]
/metadata/n_avg_frames        10
/metadata/source_mask_id      "all_closed_window"
```

Required semantic checks:

- `n_avg_frames == 10`.
- `source_mask_id == "all_closed_window"`.
- The release must be treated as closed-LCD residual data, not as a normal mask
  sequence.
- `is_sensor_dark` must be recorded as `false` in LCD_forward metadata unless the
  upstream release explicitly says otherwise.
- `is_single_frame_burst` must be recorded as `false`.
- Residual mean should be close to zero after the upstream subtraction.

## Resize / Crop Policy

Current Phase 3.6 reconstruction uses 256x256 objects/frames while measured PSF
inputs originate from roi_512 and are downsampled/cropped for the working size.
Residual preprocessing must match the frame preprocessing. The first supported
mode should be:

```yaml
residual_transform: center_crop_512_to_256
resize_mode: center_crop
```

Do not introduce arbitrary interpolation for the residual unless the clean frame
path uses the same transform. If a later pipeline switches to frequency-domain
resize/downsample, the residual transform must be changed and recorded together
with that preprocessing.

## Sampling Policy

Default:

```text
pooled
```

For each observation frame, sample one residual crop from the pooled residual bank
over wavelength and repeat. This is the correct main setting because a rendered
detector frame is a sum over wavelengths and does not have a unique wavelength
label.

Supported policies:

- `pooled`: recommended main result; sample from all `[wavelength, repeat]`
  residuals.
- `cycle_by_frame`: deterministic periodic sampling by frame index for strict
  reproducibility and debugging.
- `per_lambda_weighted`: optional only; not recommended for main results because
  it creates a wavelength label that the detector frame does not physically have.

## Proposed Module

```text
src/noise/
├── __init__.py
└── closed_lcd_residual.py
```

Core interface:

```python
class ClosedLCDResidualNoise:
    def __init__(
        self,
        h5_path: str,
        target_shape: tuple[int, int],
        sample_policy: str = "pooled",
        count_peak: float = 200.0,
        quantile: float = 0.999,
        resize_mode: str = "center_crop",
        seed: int = 0,
    ):
        ...

    def apply(self, frames_clean: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        frames_clean: [N, T, 1, H, W] or [T, H, W] normalized clean frames.
        returns:
            frames_noisy: same shape as frames_clean
            metadata: scale/noise sampling information
        """
```

The implementation must preserve dtype/device where practical. Sampling should be
driven by a local generator seeded from the config, not global random state.

## Configuration

Add under `linear_recon.add_noise` in `configs/bishe_first_pass.yaml` or a new
`configs/recon_bishe_multiframe_noisy.yaml`:

```yaml
add_noise:
  enabled: true
  type: closed_lcd_avg10_residual
  source_h5: D:/datasets/.../closed_lcd_roi512_avg10_residuals.h5

  source_mask_id: all_closed_window
  n_avg_frames: 10
  is_sensor_dark: false
  is_single_frame_burst: false

  count_peak: 200.0
  scale_quantile: 0.999
  sample_policy: pooled
  resize_mode: center_crop
  seed: 20260520

  claim_level: averaged_frame_closed_lcd_residual
```

The existing `linear_recon.add_noise.enabled: false` placeholder is the right
integration point.

## Integration Point

In `scripts/run_bishe_first_pass.py`, the clean Phase 3.6 path currently does:

```python
frames = render_frames_fft(obj, psfs_t)
recon_single = single_frame_reconstruct(frames, psfs_t, alpha=alpha, policy=policy)
recon_multi = frequency_domain_ridge_reconstruct(frames, psfs_t, alpha=alpha, policy=policy)
```

Noisy integration should become:

```python
frames_clean = render_frames_fft(obj, psfs_t)

if noise_cfg.enabled:
    noise_model = ClosedLCDResidualNoise.from_config(noise_cfg, target_shape=frames_clean.shape[-2:])
    frames_input, noise_meta = noise_model.apply(frames_clean)
else:
    frames_input = frames_clean
    noise_meta = {"enabled": False}

recon_single = single_frame_reconstruct(frames_input, psfs_t, alpha=alpha, policy=policy)
recon_multi = frequency_domain_ridge_reconstruct(frames_input, psfs_t, alpha=alpha, policy=policy)
```

`psfs_t`, `psfs_fft`, `H(f)`, ridge alpha, and ridge policy stay unchanged.

For appendix arrays, store both:

```text
rendered_frames_clean
rendered_frames_input
noise_metadata
```

If backward compatibility is needed, keep `rendered_frames` as the actual
reconstruction input and add `rendered_frames_clean` separately.

## Reporting

Do not overwrite clean results. Report settings as separate rows:

```text
scene | setting | method | PSNR | corr | display-SSIM
synthetic | clean | single | ...
synthetic | clean | multi | ...
synthetic | closed_lcd_residual | single | ...
synthetic | closed_lcd_residual | multi | ...
cd_ms | closed_lcd_residual | single | ...
cd_ms | closed_lcd_residual | multi | ...
```

Recommended hierarchy:

- `clean measured-operator reconstruction`: upper-bound / sanity check.
- `closed-LCD residual noisy reconstruction`: main empirical residual setting.
- `count_peak stress tests`: appendix or sensitivity analysis.

Main interpretation should emphasize single-frame vs multi-frame relative gain,
not absolute PSNR. Clean metrics can be moved to appendix if they are too ideal.

## Thesis Wording

Suggested Chapter 5 wording:

```text
为降低 clean measured-operator reconstruction 的理想化程度，本文进一步使用 optic_system 发布的
closed-LCD averaged-frame residual release 构造 noisy reconstruction setting。该 release 从
all_closed_window 样本中导出与 PSF dictionary 曝光匹配、10 帧平均后的 roi_512 残差。
LCD_forward 在 measured PSF renderer 生成 clean frames 后，将 residual 按 count-domain scale
注入观测帧，再使用相同的 measured PSF H 矩阵进行频域 ridge reconstruction。该设置用于评估当前
平均采集流程下经验背景残差对重建的影响，不被解释为完整传感器噪声模型。
```

## Tests

Add `tests/test_closed_lcd_residual_noise.py` with at least:

1. Reads residual HDF5.
2. Validates `residuals_avg10` shape.
3. Validates `n_avg_frames == 10`.
4. Validates `is_sensor_dark == false` in LCD_forward metadata.
5. Checks residual mean is close to zero.
6. `apply()` preserves frame shape.
7. Same seed gives reproducible output.
8. `enabled=false` path returns old clean behavior.
9. `all_closed_window` cannot be used as a normal mask sequence.

Use a synthetic tiny HDF5 fixture for CI; do not require the D: release in tests.

## Implementation Order

1. Add `src/noise/closed_lcd_residual.py` and unit tests.
2. Add noisy config with `enabled: false` default preserved in existing clean config.
3. Integrate into synthetic and CAVE branches of `run_bishe_first_pass.py`.
4. Store `noise_metadata` in JSON and `.npz` outputs.
5. Add clean/noisy split to summary CSV and reports.
6. Run default `count_peak=200` experiment.
7. Optionally run `count_peak=100` and `50` as sensitivity appendix.

## Non-Goals

Do not:

- Add residuals to PSF, OTF, or H matrices.
- Train forward surrogates with closed-LCD residual data.
- Treat `all_closed_window` as a normal mask.
- Claim this is real sensor noise, read noise, shot noise, PRNU, or a full camera
  noise model.
- Use per-frame min-max normalization that cancels the injected residual.
- Change optical claims based on this residual injection experiment alone.
