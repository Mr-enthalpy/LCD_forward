# LCD_forward — thesis modelling and reconstruction backend

LCD_forward is the modelling and reconstruction backend for the mono-LCD programmable diffraction imaging thesis loop. It consumes measured HDF5 exports from `optic_system` and performs forward validation, rendering, and minimal reconstruction. Hardware control remains in `optic_system`.

## Project identity

This is a **thesis-closure project**, not a long-term research platform.

Goal: validate that a mono LCD can act as a stable, programmable diffraction encoder whose mask-dependent PSFs can be measured, modelled at a simple level, and used in a minimal multiframe / multichannel reconstruction loop.

Non-goals:

- SOTA performance or optimal reconstruction quality
- learned mask / end-to-end optimal design
- full first-principles optical modelling
- complete complex pupil reconstruction
- long-term generality or reusable experimental platform

## Relationship with optic_system

```
optic_system                          LCD_forward
-----------                          -----------
hardware control          <--X-->     backend only
raw capture HDF5                      reads HDF5 exports
measured PSF dictionary  ──export──>  forward validation
dOTF diagnostics                      rendering
PSF ROI / pupil calib                 linear reconstruction
                                      thesis figures
```

LCD_forward does not control cameras, LCDs, or TLS light sources, and does not create raw capture HDF5 files.

## Input data

Expected Phase 3.4 export from `optic_system`:

```
data/optic_system/psf_dictionary/
    train.h5
    val.h5
    test.h5
```

See `docs/data_contracts.md` for the full HDF5 format specification.

## Thesis phases

| Phase  | Title                                     | Owner        |
|--------|-------------------------------------------|--------------|
| 3.0–3.4| Hardware calibration & PSF dictionary     | optic_system |
| 3.5    | Measured PSF forward validation           | LCD_forward  |
| 3.6    | Minimal multiframe / multichannel reconstruction | LCD_forward  |
| 3.7    | Thesis figures and report freeze          | LCD_forward  |

Current state: **thesis branch initialized**. Phase 3.5/3.6 implementation starts after `optic_system` Phase 3.4 export is available.

## Quick links

- First read: `docs/thesis_background.md`
- Phase plan: `docs/bishe_plan.md`
- Data contracts: `docs/data_contracts.md`
- Project boundary: `docs/project_boundary.md`
- Server usage: `docs/server_usage.md`
- Agent constraints: `AGENTS.md`

## Installation

```bash
pip install -e .
```

## Repository structure

```
configs/       YAML configs for forward validation and linear reconstruction
data/          HDF5 input data (not in version control)
docs/          Thesis documentation and data contracts
outputs/       Derived figures, metrics, and reports (not in version control)
scripts/       Entry-point scripts
src/           Forward models, renderer, reconstruction, datasets, utilities
tests/         Shape and module tests
```
