# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CrossScaleRC is a Julia package for cross-scale reservoir computing (echo state networks) with spatial blocking, used to forecast high-dimensional spatiotemporal systems. Supports single-layer, multi-layer (coarse→fine), Deep ESN, and Next-Generation RC architectures. Applications: Kuramoto-Sivashinsky equation and sea surface temperature (SST) forecasting.

Paper: https://doi.org/10.48550/arXiv.2510.11209

## Common Commands

```bash
# Activate project and install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run a main script
julia --project=. main_single_layer.jl

# Run tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Start Julia REPL with project
julia --project=.
# then: using CrossScaleRC
```

## Architecture

### Core Types (src/types.jl)
- `ResParams{T}`: Hyperparameters (reservoir size N, spectral radius g, connectivity degree, input scalings g_in_rec/g_in_neigh/g_in_layer, leak rate dt/τ)
- `Reservoir{T}`: ESN with sparse recurrent weights W and input weight matrices (W_in_rec, W_in_neigh, W_in_layer)
- `BlockModel{T}`: Trained readout per spatial block with output matrix W_out, state x, and row indices for input types

### Dynamics Engines (src/dynamics/)
- `run_single_layer()`: Single-layer ESN with spatial blocks, local + neighbor inputs
- `run_multi_layer()`: Two-layer pipeline — coarse layer runs first, predictions upscaled and fed as cross-layer input to fine layer
- `DeepESN_train!` / `DeepESN_test_closed_loop`: Deep ESN with quadratic feature augmentation
- `nextgen_closedloop()`: Next-Generation RC using polynomial features + ridge regression (no reservoir)

### Data Pipeline (src/data/)
- `load_data(Q, L, μ)`: Loads KS data from CSV under `data/kuramoto/`, normalizes to zero mean and unit variance
- `load_data(resolutions_vec)`: Loads SST data from JLD2 files under `data/sst/` for multiple resolutions
- `regrid_average(data, divisor)`: Spatial coarse-graining via block averaging
- `make_blocks()`: Creates 1D or 2D spatial block structures defining local/neighbor/cross-layer input indices

### Data Flow
```
Raw data (CSV or JLD2) → load_data() → normalize
  → regrid_average() [optional coarsening]
  → make_blocks() → block structures with row indices
  → run_single_layer() or run_multi_layer()
    → generate_reservoir() per block → ridge regression training → closed-loop test
  → plot_train_test_heatmaps() → visualization
```

### Key Design Patterns
- **Spatial blocking**: Domain decomposition where each block has its own reservoir and readout, with neighbor mixing
- **Multi-scale hierarchy**: Coarse layer predictions flow into fine layer as additional input
- **Input weight modes**: `:structured` (contiguous neuron blocks) or `:random`
- **Regression modes**: `:linear` or `:quadratic` (feature augmentation with squared states)
- **Overlap modes**: `:exclude` or `:include` for cross-scale input handling in multi-layer
- **Per-layer parameters**: Hyperparameters passed as tuples, one element per layer

### Entry Scripts
| Script | Architecture |
|--------|-------------|
| `main_single_layer.jl` | Single-layer block-wise ESN on 1D KS |
| `main_multi_layer.jl` | Two-layer coarse→fine ESN on 1D KS |
| `main_deep_rc.jl` | Deep ESN (no spatial blocking) on KS |
| `main_ngrc.jl` | Next-Generation RC on KS |
| `main_sst.jl` | Two-layer ESN on 2D SST data |
| `run_tuning_*.jl` | Hyperparameter grid search scripts |

### SST Data Preparation
Requires running scripts in order: `scripts/download_sst_data.py` (CDS download) → `scripts/main_regrid.jl` (regrid) → `scripts/main_produce_sst_data.jl` (concatenate into final JLD2).
