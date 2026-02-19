# Activate environment
using Pkg, Revise
Pkg.activate(".")

using CrossScaleRC
using LinearAlgebra
using Plots, Measures
using Random
using DelimitedFiles

BLAS.set_num_threads(1)

# Experiment setup
Q0 = 128
L = 44
μ = 0.01

resolution_divisor = 1
Q = div(Q0, resolution_divisor)
data, τ = load_data(Q0, L, μ; show_data=false, refinement=1)
fine = regrid_average(data, resolution_divisor)

resolution_divisor_upper_layer = 4
coarse = regrid_average(data, resolution_divisor_upper_layer)

washout     = 1_000
train_len   = 50_000
predict_len = 1_000
warmup      = 1_000

M, Ttot = size(data)
train_len + predict_len ≤ Ttot || error("Not enough data")

num_networks = [4, 8]
mixing = [2, 2]

blocks_coarse = make_blocks(size(coarse, 1), num_networks[1], mixing[1])
blocks_fine   = make_blocks(size(fine, 1), resolution_divisor_upper_layer, num_networks[2], mixing[2], num_networks[1]; overlap_mode = :exclude)

blocks = [blocks_coarse, blocks_fine]

# Per-layer τ, dt for reservoir
τ_default = [0.25, 0.25]
dt_default = [0.25, 0.25]

function run_once(params)
    res_size, res_radius, degree, g_in_rec, g_in_neigh, g_in_layer, ridge_param = params
    res_params = (res_size, res_radius, degree, g_in_rec, g_in_neigh, g_in_layer, τ_default, dt_default)

    preds_fine, preds_coarse, train_pred_fine, train_pred_coarse,
    train_data_coarse, train_data_fine, data_coarse, X_coarse, X_fine = run_multi_layer(
        res_params,
        fine,
        coarse,
        train_len,
        predict_len,
        blocks;
        washout = washout,
        warmup = warmup,
        ridge_parameter = ridge_param,
        show_progress = false,
        input_mode = :structured,
        regression_mode = [:quadratic, :quadratic],
    )

    test_fine = data[:, train_len - warmup + 1 : train_len - warmup + size(preds_fine, 2)]
    error_curve = [rmse_upto(test_fine[:, warmup:end], preds_fine[:, warmup:end]; T=t)
                  for t in 1:size(test_fine[:, warmup:end], 2)]


    er_skip = max(1, round(Int, length(error_curve) / 4))
    error_grid = error_curve[1:er_skip:end][1:4]   # ensure 4 elements
    return error_grid
end


# Baseline vector params (length-2) + grids for only the [2] entries
base = (
    res_size   = [1000, 1000],
    res_radius = [0.1,  0.6],
    degree     = [10,   10],
    g_in_rec   = [2.5 / sqrt(div(Q, resolution_divisor_upper_layer) / num_networks[1]),
                 1.0 / sqrt(Q / num_networks[2])],
    g_in_neigh = [2.5 / sqrt(mixing[1]),
                 2.0 / sqrt(mixing[2])],
    g_in_layer = [0.0,
                 2.0 / sqrt(div(Q, resolution_divisor_upper_layer) / num_networks[2])],
    ridge_param = [1e-5, 1e0],
)

# Build fresh vectors each time (avoid mutating shared arrays across reps/grid points).
function build_params(base, p)
    res_size = deepcopy(base.res_size);       res_size[2] = p.res_size2
    res_radius = deepcopy(base.res_radius);   res_radius[2] = p.res_radius2
    degree = deepcopy(base.degree)            # fixed here, but kept for completeness

    g_in_rec = deepcopy(base.g_in_rec);       g_in_rec[2] = p.g_in_rec2
    g_in_neigh = deepcopy(base.g_in_neigh);   g_in_neigh[2] = p.g_in_neigh2
    g_in_layer = deepcopy(base.g_in_layer);   g_in_layer[2] = p.g_in_layer2

    ridge_param = deepcopy(base.ridge_param); ridge_param[2] = p.ridge2

    return (res_size, res_radius, degree, g_in_rec, g_in_neigh, g_in_layer, ridge_param)
end

run_once_grid(p) = run_once(build_params(base, p))

grids = Dict(
    :res_size2   => [1000],
    :res_radius2 => [0.1],
    :g_in_rec2   => [1.0 / sqrt(Q / num_networks[2]), 0.0],
    :g_in_neigh2 => [2.0 / sqrt(mixing[2]), 0.0],
    :g_in_layer2 => [2.0 / sqrt(div(Q, resolution_divisor_upper_layer) / num_networks[2])],
    :ridge2      => [1e-4, 1e1],
)

grid_search(
    run_once_grid,
    grids;
    nrep = 3,
    outfile = "gridsearch_multilayer_Q$(Q).csv",
    param_order = [:res_size2, :res_radius2, :g_in_rec2, :g_in_neigh2, :g_in_layer2, :ridge2],
    error_names = [:rmse1, :rmse2, :rmse3, :rmse4],
    progress = true,
)
