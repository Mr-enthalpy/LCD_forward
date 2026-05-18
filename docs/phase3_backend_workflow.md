# Phase 3 backend workflow

This document describes the end-to-end workflow that `LCD_forward` scripts should follow once Phase 3.5 / 3.6 implementation begins.

## Forward validation workflow (Phase 3.5)

```
1. Verify input data
   $ python scripts/check_data_format.py --config configs/forward_validation.yaml

2. Basic data visualization
   $ python scripts/visualize_psf_dictionary.py --config configs/forward_validation.yaml

3. Forward baseline training
   $ python scripts/run_forward_baseline.py --config configs/forward_validation.yaml

4. Forward validation evaluation
   $ python scripts/eval_forward_baseline.py --config configs/forward_validation.yaml

5. Generate forward validation figures
   $ python scripts/plot_forward_validation.py --config configs/forward_validation.yaml
```

Expected outputs per step:

| Step | Output |
|------|--------|
| 1    | Console confirmation of tensor shapes and metadata |
| 2    | `outputs/forward_validation/mask_examples.png`, `outputs/forward_validation/psf_examples.png` |
| 3    | Model checkpoint (optional), log file |
| 4    | `outputs/forward_validation/psf_error_metrics.json` |
| 5    | `outputs/forward_validation/measured_vs_predicted_psf.png`, `outputs/forward_validation/representative_cases/` |

## Linear reconstruction workflow (Phase 3.6)

```
1. Generate or load multi-frame observations
   $ python scripts/prepare_recon_observations.py --config configs/linear_recon.yaml

2. Run linear reconstruction
   $ python scripts/run_linear_recon.py --config configs/linear_recon.yaml

3. Evaluate reconstruction
   $ python scripts/eval_linear_recon.py --config configs/linear_recon.yaml

4. Generate reconstruction figures
   $ python scripts/plot_linear_recon.py --config configs/linear_recon.yaml
```

Expected outputs per step:

| Step | Output |
|------|--------|
| 1    | `outputs/linear_recon/rendered_frames.npy` |
| 2    | `outputs/linear_recon/recon_single_frame.npy`, `outputs/linear_recon/recon_multiframe.npy` |
| 3    | `outputs/linear_recon/reconstruction_metrics.json` |
| 4    | `outputs/linear_recon/recon_comparison.png` |

## Thesis figure aggregation (Phase 3.7)

```
$ python scripts/aggregate_bishe_figures.py --config configs/bishe_figures.yaml
```

Collects figures from all source phases into `outputs/bishe_figures/` and generates `figure_manifest.json`.

## Script naming convention

All Phase 3.5/3.6 scripts should be `scripts/phase35_*.py` or `scripts/phase36_*.py` to clearly distinguish them from the original prototype scripts.

## Configuration

All scripts should accept a `--config` argument pointing to a YAML file in `configs/`. Shared parameters (data paths, model parameters, output directories) live in the config; script-specific parameters may be overridden via command-line flags.

## Error handling

- Missing input HDF5: print clear error message pointing to `docs/data_contracts.md`
- Shape mismatch: print expected vs actual shapes
- Config missing required key: print which key and which section

## Reproducibility

Every run should record:
- Git commit hash
- Config file path (copy to output directory)
- Source HDF5 paths
- Random seed (if applicable)
- Timestamp

This information should be written to `<output_dir>/run_metadata.json`.
