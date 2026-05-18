# Outputs directory

`outputs/` contains derived modelling, validation, reconstruction, and figure outputs.

Every output must trace back to a source HDF5 file and config. Do not treat outputs as raw experimental evidence; raw evidence belongs to `optic_system`.

## Subdirectories

```
outputs/
    forward_validation/     Phase 3.5: forward validation figures and metrics
    linear_recon/           Phase 3.6: reconstruction results and metrics
    bishe_figures/          Phase 3.7: aggregated thesis figures
```

## Not in version control

All output files (`*.h5`, `*.npy`, `*.npz`, `*.png`, `*.pdf`, `*.json`, `*.md`) under `outputs/` are git-ignored. Only this `README.md` is tracked.
