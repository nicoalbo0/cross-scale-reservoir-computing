if length(ARGS) != 2
    error("Use the following command:\n- julia ...jl g name")
end
# Activate environment
using Pkg, Revise
Pkg.activate(".")

using CrossScaleRC
using LinearAlgebra, Measures
using Plots
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
data = regrid_average(data, resolution_divisor)

num_networks = 16
mixing       = 8
blocks       = make_blocks(size(data, 1), num_networks, mixing)
washout      = 1_000
train_len    = 50_000
predict_len  = 1_000
warmup       = 1_000

M, Ttot = size(data)
train_len + predict_len ≤ Ttot || error("Not enough data")

# τ, dt for reservoir (fixed or from data)
τ_default = 0.25
dt_default = 0.25

function run_once(params)
    res_size, res_radius, degree, g_in_rec, g_in_neigh, g_in_layer, ridge_param = params
    res_params = (res_size, res_radius, degree, g_in_rec, g_in_neigh, g_in_layer, τ_default, dt_default)

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

# Scalar grid definition
grids = Dict(
    :res_size    => [400],
    :res_radius  => [parse(Float64, ARGS[1])],
    :degree      => [10],
    :g_in_rec    => [10.0^k / sqrt(Q / num_networks) for k=-3:0.5:1],
    :g_in_neigh  => [10.0^k / sqrt(mixing) for k=-3:0.5:0.5],
    :g_in_layer  => [0.0],
    :ridge_param => [10.0^k for k=-6:-2],
)

# Run
time0 = time()
grid_search(
    run_once_grid,
    grids;
    nrep = 20,
    outfile = "gridsearch_singlelayer_Q$(Q)_$(ARGS[2]).csv",
    param_order = [:res_size, :res_radius, :degree, :g_in_rec, :g_in_neigh, :g_in_layer, :ridge_param],
    error_names = [:rmse1, :rmse2, :rmse3, :rmse4],
    progress = true,
)
println("time spent $(time() - time0)")