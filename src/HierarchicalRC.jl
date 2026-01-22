module HierarchicalRC

using Statistics
using Random
using LinearAlgebra
using SparseArrays
using Interpolations
using CSV
using DataFrames
using Plots
using LaTeXStrings
using Measures
using ProgressMeter
using Base.Threads
using ReservoirComputing

export load_data
export make_blocks
export plot_train_test_heatmaps
export plot_units_activity
export run_single_layer
export run_multi_layer
export regrid_average

export rmse_upto
export build_W_in

include("types.jl")
include("utils.jl")
include("dynamics/single_layer.jl")
include("dynamics/multi_layer.jl")
include("plots.jl")

end # module HierarchicalRC
