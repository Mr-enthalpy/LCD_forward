# Data Layout

This directory holds the sample H5 placeholders used by the prototype.

Expected conventions:
- `data/sample/train.h5`
- `data/sample/val.h5`
- `data/sample/test.h5`

The current project skeleton keeps these files lightweight so the directory
layout exists immediately. The synthetic dataset code in `src/datasets/h5_dataset.py`
can also work without real HDF5 payloads.

