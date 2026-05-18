# Data directory

`data/` is not a repository for large experimental HDF5 files.

Place `optic_system` exports here manually or point configs to their external paths.

## Expected Phase 3.4 export

```
data/optic_system/psf_dictionary/
    train.h5
    val.h5
    test.h5
```

See `docs/data_contracts.md` for the full HDF5 format specification.

## Sample data (from original prototype)

```
data/sample/
    train.h5
    val.h5
    test.h5
```

These are synthetic placeholders from the original prototype. They should not be interpreted as physically faithful mono-LCD forward simulations.

## Data not in version control

All `*.h5` and `*.npy` files under `data/` are git-ignored. Only `README.md` files are tracked.
