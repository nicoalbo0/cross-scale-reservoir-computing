"""
    CrossScaleRC

Cross-scale reservoir computing: single- and multi-layer echo state networks with
spatial blocking, Kuramoto–Sivashinsky and SST data loading, regridding, and plotting.

# Main functions
- **Data**: `load_data`, `regrid_average`, `make_blocks`
- **Dynamics**: `run_single_layer`, `run_multi_layer`
- **Metrics**: `rmse_upto`, `input_dimensions`
- **Plots**: `plot_train_test_heatmaps`, `plot_units_activity`
- **Readout**: `build_W_in` (input weight construction)

See individual function docstrings for usage.
"""
module CrossScaleRC

using Statistics
using Random
using LinearAlgebra
using SparseArrays
using Interpolations
using CSV
using DelimitedFiles
using DataFrames
using Plots
using LaTeXStrings
using Measures
using ProgressMeter
using Base.Threads
using ReservoirComputing
using JLD2
using NCDatasets
using Dates
using DSP

export load_data
export make_blocks
export regrid_average
export read_and_regrid

export run_single_layer
export run_multi_layer
export DeepESN, DeepESN_train!, DeepESN_test_closed_loop
export nextgen_closedloop

export plot_train_test_heatmaps
export plot_units_activity

export rmse_upto
export build_W_in
export input_dimensions
export grid_search
export nino34_index
export skill_score
export sst_grid_coords
export bandpass_decompose
export reconstruct_bands

include("types.jl")
include("data/loading.jl")
include("data/regrid.jl")
include("data/grids.jl")
include("data/temporal_decomposition.jl")
include("utils/generic.jl")
include("utils/deeprc_utils.jl")
include("utils/ngrc_utils.jl")
include("utils/gridsearch_utils.jl")
include("dynamics/single_layer.jl")
include("dynamics/multi_layer.jl")
include("plots.jl")

end # module CrossScaleRC
