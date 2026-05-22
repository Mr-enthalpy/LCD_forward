# Reconstruction Figure Caption Guide

This note is for downstream thesis writing only. It does not change reconstruction data,
metrics, optical claims, or the Phase 3.6 conclusions.

## Scope

Use this guide for Chapter 5 reconstruction figures derived from:

- `thesis/phase3_6_linear_recon_synthetic/figures/`
- `thesis/phase3_6_linear_recon_cave/figures/scene_*/`

The figures are generated from the corresponding `recon_appendix_arrays.npz` files, which
store `gt_object`, `recon_single`, `recon_multi`, `wavelengths_nm`, selected masks, rendered
frames, and HDF5 provenance.

Long explanatory notes are intentionally kept in this document and in thesis captions, not
embedded below the images.

## Per-band Reconstruction Figures

Applies to every `recon_per_band_comparison.png`.

Actual layout:

- Columns are wavelength channels, usually `450 nm`, `550 nm`, and `650 nm`.
- Rows are, from top to bottom: ground truth, single-frame reconstruction, multi-frame reconstruction, and absolute error.
- In the bottom error row, each panel is split into two halves: left is single-frame absolute error, right is multi-frame absolute error.
- The white vertical line in the error row separates the single-frame and multi-frame error halves.

Recommended Chinese caption sentence:

> 图中列对应不同波长通道，行自上而下依次为真实目标、单帧重建、多帧重建和绝对误差图；底行每个误差图的左半部分为单帧误差，右半部分为多帧误差，白色竖线用于区分两者。

## Pseudo-RGB Reconstruction Figures

Applies to every `recon_rgb_pseudocolor_comparison.png`.

Actual layout:

- First row shows ground truth, single-frame reconstruction, and multi-frame reconstruction.
- Second row shows single-frame absolute pseudo-RGB error and multi-frame absolute pseudo-RGB error.
- Pseudo-RGB uses the nearest available channels as `R/G/B = 650/550/450 nm`.
- Pseudo-RGB display uses per-panel 1st-99th percentile clipping for readability; residual panels share one color scale between single-frame and multi-frame error.
- The pseudo-RGB image is not calibrated camera RGB.

Recommended Chinese caption sentence:

> 第一行从左到右依次为真实目标、单帧重建和多帧重建；第二行给出单帧和多帧重建的绝对伪彩色误差。伪彩色映射采用 R/G/B = 650/550/450 nm，仅用于可视化，不表示相机标定 RGB。

## Figure 5 Usage

Recommended mapping if the thesis uses five Chapter 5 figures:

- Figure 5-1: synthetic per-band reconstruction comparison; use the per-band caption structure.
- Figure 5-2: synthetic pseudo-RGB reconstruction comparison; use the pseudo-RGB caption structure.
- Figures 5-3 to 5-5: CAVE scene reconstruction comparisons; use whichever caption structure matches the inserted image.

Do not rely on surrounding正文 to explain row/column semantics. Each caption should state the actual layout explicitly.
