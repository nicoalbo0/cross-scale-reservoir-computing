# Activate environment
using Pkg, Revise
project_root = @__DIR__ # 1. Get the directory of the current script
Pkg.activate(project_root)
Pkg.instantiate()

using CrossScaleRC
using LinearAlgebra, Measures
using Plots
using Random

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

num_networks = 4
mixing       = 2
blocks       = make_blocks(size(data, 1), num_networks, mixing)
washout      = 1_000
train_len    = 50_000
predict_len  = 1_000
warmup       = 1_000

M, Ttot = size(data)
train_len + predict_len ≤ Ttot || error("Not enough data")

function run_once(params)
    res_size, res_radius, degree, g_in_rec, g_in_neigh, g_in_layer, ridge_param = params
    res_params = (res_size, res_radius, degree, g_in_rec, g_in_neigh, g_in_layer)

    preds_test, _, _, _, _ = run_single_layer(
        res_params,
        data,
        zeros(size(data)),   # no previous multilayer input
        train_len,
        predict_len,
        blocks;
        washout = washout,
        warmup = warmup,
        ridge_parameter = ridge_param,
        show_progress = false,
        input_mode = :structured,
    )

    test_data = data[:, train_len - warmup + 1 : train_len - warmup + size(preds_test, 2)]

    error_curve = [rmse_upto(test_data[:, warmup:end], preds_test[:, warmup:end]; T=t)
                   for t in 1:size(test_data[:, warmup:end], 2)]

    er_skip = max(1, round(Int, length(error_curve) / 4))
    error_grid = error_curve[1:er_skip:end][1:4]   # 4 elements
    return error_grid
end

# Wrapper: take a NamedTuple of scalars from the grid and pack it into your tuple `params`
function run_once_grid(p)
    params = (
        p.res_size,
        p.res_radius,
        p.degree,
        p.g_in_rec,
        p.g_in_neigh,
        p.g_in_layer,
        p.ridge_param,
    )
    return run_once(params)
end

# ---------------------------
# Define the scalar grids you want to scan
# ---------------------------

# Baseline example values you gave:
# res_size     = 500
# res_radius   = 0.1
# degree       = 10
# g_in_rec     = 2.5/sqrt(Q/num_networks)
# g_in_neigh   = 2.5/sqrt(mixing)
# g_in_layer   = 0.0
# ridge_param  = 1e-5

grids = Dict(
    :res_size    => [300],
    :res_radius  => [0.1, 0.5],
    :degree      => [10],
    :g_in_rec    => [2.0 / sqrt(Q / num_networks),
                     3.0 / sqrt(Q / num_networks)],
    :g_in_neigh  => [2.0 / sqrt(mixing),
                    3.0 / sqrt(mixing)],
    :g_in_layer  => [0.0],
    :ridge_param => [1e-6, 1e-5, 1e-4],
)

# Run
grid_search(
    run_once_grid,
    grids;
    nrep = 3,
    outfile = "gridsearch_singlelayer_Q$(Q).csv",
    param_order = [:res_size, :res_radius, :degree, :g_in_rec, :g_in_neigh, :g_in_layer, :ridge_param],
    error_names = [:rmse1, :rmse2, :rmse3, :rmse4],
    progress = true,
)