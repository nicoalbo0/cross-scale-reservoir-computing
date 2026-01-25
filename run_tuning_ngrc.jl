# Activate environment
using Pkg, Revise
project_root = @__DIR__ # 1. Get the directory of the current script
Pkg.activate(project_root)
Pkg.instantiate()

using HierarchicalRC
using LinearAlgebra, Measures
using Plots
using Random

include(joinpath(project_root, "src", "ngrc_utils.jl"))
include(joinpath(project_root, "src", "gridsearch_utils.jl"))

BLAS.set_num_threads(Threads.nthreads())

##
using DelimitedFiles

# ---------------------------
# Experiment setup
# ---------------------------

Q0 = 128
L = 44
μ = 0.01

resolution_divisor = 4
Q = div(Q0, resolution_divisor)
data, τ = load_data(Q0, L, μ; show_data=false, interpolate_data=false)
data = regrid_average(data, resolution_divisor)

# Experiment configuration
washout     = 1_000
train_len   = 50_000
predict_len = 1_000

M, Ttot = size(data)
washout + train_len + predict_len + 2 ≤ Ttot || error("Not enough data")

function run_once(params)
    past, degree, ridge = params

    preds_test, preds_train, Wout, Φ = nextgen_closedloop(
        data,
        train_len,
        predict_len;
        washout = washout,
        past = past,
        degree = degree,
        ridge = ridge,
    )

    t_start = washout + past
    test_data = data[:, (t_start + train_len + 1):(t_start + train_len + predict_len)]

    error_curve = [rmse_upto(test_data, preds_test; T=t) for t in 1:size(test_data, 2)]

    er_skip = max(1, round(Int, length(error_curve) / 4))
    error_grid = error_curve[1:er_skip:end][1:4]
    return error_grid
end

# ---------------------------
# Choose grids + wrapper
# ---------------------------
grids = Dict(
    :past   => [1, 2, 3, 4],
    :degree => [1, 2],
    :ridge  => [1e-4, 1e-2, 1e-1],
)

run_once_grid(p) = run_once((p.past, p.degree, p.ridge))

grid_search(
    run_once_grid,
    grids;
    nrep = 1,
    outfile = "gridsearch_ngrc_Q$(Q).csv",
    param_order = [:past, :degree, :ridge],
    error_names = [:rmse1, :rmse2, :rmse3, :rmse4],
    progress = true,
)