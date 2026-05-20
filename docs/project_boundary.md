# Project boundary

This document defines the hard boundary between `optic_system` and `LCD_forward`.

## optic_system owns

- Hardware control (camera, LCD, TLS light source)
- Raw capture HDF5 generation
- LCD / TLS / camera synchronization
- Effective pupil window calibration (Phase 3.1)
- Camera-frame PSF ROI calibration (Phase 3.2a)
- PSF repeatability and mask-induced diversity measurement (Phase 3.2b)
- dOTF diagnostic capture (Phase 3.3)
- Measured PSF dictionary capture and `LCD_forward`-compatible HDF5 export (Phase 3.4)
- Metadata and provenance recording
- Experimental diagnostic figure output

## LCD_forward owns

- HDF5 dataset reading (`optic_system` exports)
- Forward model baseline (Phase 3.5)
- Forward validation and predicted-vs-measured PSF comparison
- Frame renderer using measured PSFs
- Minimal linear reconstruction (Phase 3.6)
- Reconstruction metrics
- Thesis-ready figures and reports (Phase 3.7)

## Explicitly prohibited in LCD_forward

Do **not**:

- Move hardware capture logic into `LCD_forward`
- Import camera / LCD / TLS services from `optic_system`
- Duplicate `optic_system` capture scripts
- Edit raw capture assumptions without updating data contracts
- Add hardware control or device communication code
- Create raw capture HDF5 files

## Data flow diagram

```
optic_system
    │
    ├── Phase 3.1: pupil_window.json
    ├── Phase 3.2: psf_roi.json, repeatability reports
    ├── Phase 3.3: dOTF diagnostic figures
    ├── Phase 3.4: export_lcd_forward/
    │       ├── train.h5
    │       ├── val.h5
    │       └── test.h5
    │
    └──[handoff]──────────────────────────────────┐
                                                    │
LCD_forward                                         │
    │                                               │
    ├── reads export HDF5 ◄─────────────────────────┘
    ├── Phase 3.5: forward validation
    │       └── outputs/forward_validation/
    ├── Phase 3.6: linear reconstruction
    │       └── outputs/linear_recon/
    └── Phase 3.7: thesis figures
            └── outputs/bishe_figures/
```

## Communication protocol

When `optic_system` updates its export format, `docs/data_contracts.md` in `LCD_forward` must be updated to match before `LCD_forward` code changes are made.

Do not silently change expectations about:
- HDF5 key names
- tensor shapes
- normalization conventions
- metadata JSON structure

