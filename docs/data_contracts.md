# Data contracts

This document defines the HDF5 data format that `LCD_forward` expects from `optic_system` Phase 3.4 exports.

## Source location

```
outputs/psf_dictionary/export_lcd_forward/
    train.h5
    val.h5
    test.h5
```

## Local placement in LCD_forward

```
data/optic_system/psf_dictionary/
    train.h5
    val.h5
    test.h5
```

Or configure external paths via `configs/forward_validation.yaml` and `configs/linear_recon.yaml`.

Current local thesis-machine placement:

```
D:/datasets/optic_system/phase3_release_20260520/
    lcd_forward/psf_dictionary/train.h5
    lcd_forward/psf_dictionary/val.h5
    lcd_forward/psf_dictionary/test.h5

D:/datasets/LCD_forward/
    lcd_forward_phase3_5_3_6_release_20260520/

D:/datasets/CAVE/
    processed/train.h5
    processed/val.h5
    processed/test.h5
```

The `D:/datasets/LCD_forward/` handoff payload contains large derived artifacts that are intentionally not synchronized through Git. The Git `handoff/` directory only stores the small descriptor files needed to audit identity, checksums, and provenance.

## PSF dictionary HDF5 format

### Core tensors

| Key    | Shape                     | Dtype   | Description                                |
|--------|---------------------------|---------|--------------------------------------------|
| masks  | `[N, T, 1, Hm, Wm]`      | uint8                | Low-resolution LCD control variables       |
| psfs   | `[N, T, L, Hp, Wp]`      | float32 or float64   | Measured PSF crops from `psf_roi.json`     |

Phase 3.4 single-wavelength single-frame dictionary:

```
T = 1
L = 1
masks: [N, 1, 1, 64, 64]
psfs:  [N, 1, 1, Hp, Wp]
```

### Auxiliary fields

| Key              | Dtype   | Description                          |
|------------------|---------|--------------------------------------|
| mask_id          | str     | Unique mask identifier per sample, e.g. `all_open_window`, `random_lowfreq_001` |
| mask_family      | str     | Mask family or category label        |
| metadata_json    | str     | JSON string with provenance data     |

### metadata_json content

```json
{
  "source_raw_h5": "<path to source raw HDF5>",
  "pupil_window_source": "<path to pupil_window.json>",
  "psf_roi_source": "<path to psf_roi.json>",
  "camera_params_source": "<path to camera_params.json>",
  "wavelength_nm": 550.0,
  "T": 1,
  "L": 1,
  "mask_shape": [1, 64, 64],
  "psf_shape": [Hp, Wp],
  "normalization": "background_subtract_then_sum_normalize"
}
```

## Data interpretation rules

### masks

- Low-resolution LCD control variables, **not** physical LCD pixel arrays.
- Values may be uint8 code values (0/255). Dataset loader should normalize to float [0, 1] internally if required, and must record this transform.

### psfs

- Measured PSF crops exported from `optic_system`, already associated with `psf_roi.json`.
- Shape `[Hp, Wp]` is determined by the PSF ROI calibration (Phase 3.2a).
- Dtype may be `float32` or `float64` depending on `optic_system` export version. Loaders should cast to `float32` internally when needed.

### psf normalization

- `optic_system` export may provide sum-normalized PSFs (field: `normalization` in `metadata_json`).
- `LCD_forward` must read `metadata_json` and avoid silently assuming raw intensity units.

### Provenance

Every output in `outputs/` must be traceable back to:
1. The source HDF5 file
2. The config that produced it
3. The relevant git commit

## Reconstruction data format (for Phase 3.6)

When `optic_system` provides real target captures for reconstruction:

| Key         | Shape                     | Description                       |
|-------------|---------------------------|-----------------------------------|
| objects     | `[N, L, H, W]`            | Ground-truth spectral object      |
| frames      | `[N, T, 1, H, W]`         | Multi-frame observations (optional) |
| masks       | `[N, T, 1, Hm, Wm]`       | Masks used for each frame         |

Optional metadata:
- `wavelengths`: `[L]` - wavelength values in nm
- `spectral_response`: `[L]` - sensor spectral response per band

