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

resolution_divisor = 4
Q = div(Q0, resolution_divisor)
data, τ = load_data(Q0, L, μ; show_data=false, refinement=1)
data = regrid_average(data, resolution_divisor)

washout      = 1_000
train_len    = 50_000
predict_len  = 1_000
warmup       = 1_000

M, Ttot = size(data)
train_len + predict_len ≤ Ttot || error("Not enough data")

# Deep RC
nu = Q
ny = Q

function run_once(params)
    nr, nl, reservoir_params = params

    input_data = data[:, 1:(washout + train_len - 1)]
    target_data = data[:, 2:(washout + train_len)]

    deep_rc = DeepESN(nu, ny;
        nl = nl,
        nr = nr,
        rho = reservoir_params[:radius],
        leak = reservoir_params[:leaky_coeff],
        input_scale = reservoir_params[:input_scale],
        inter_scale = 1.0,
        sparsity = reservoir_params[:sparsity],
    )

    DeepESN_train!(deep_rc, input_data, target_data;
        washout = washout,
        ridge = reservoir_params[:ridge_param],
    )

    test_start_idx = train_len - warmup + 1
    warmup_u = data[:, test_start_idx:(test_start_idx + warmup - 1)]

    preds_test = DeepESN_test_closed_loop(deep_rc;
        steps = predict_len,
        warmup = warmup_u,
        reset_state = true,
    )

    test_data = data[:, (test_start_idx + warmup):(test_start_idx + warmup + predict_len - 1)]

    error_curve = [
        rmse_upto(test_data[:, 1:end], preds_test[:, warmup+1:end]; T=t)
        for t in 1:size(test_data, 2)
    ]

    er_skip = max(1, round(Int, length(error_curve) / 4))
    error_grid = error_curve[1:er_skip:end][1:4]   # 4 elements
    return error_grid
end

# Baseline + wrapper that builds (nr, nl, reservoir_params)

base_reservoir_params(nr) = Dict(
    :radius => 0.1,
    :sparsity => 10 / nr,
    :input_scale => 2.5 / sqrt(Q),
    :leaky_coeff => 1.0,
    :ridge_param => 1e-4,
)

function run_once_grid(p)
    nr = p.nr
    nl = p.nl

    # Build params fresh per run (so reps don’t share/mutate dictionaries).
    rp = base_reservoir_params(nr)
    rp[:radius] = p.radius
    rp[:leaky_coeff] = p.leaky_coeff
    rp[:input_scale] = p.input_scale
    rp[:ridge_param] = p.ridge_param

    return run_once((nr, nl, rp))
end

# ---------------------------
# Define the grids
# ---------------------------
grids = Dict(
    :nl => [2],
    :nr => [300],
    :radius => [0.05, 0.1, 0.2],
    :leaky_coeff => [1.0],
    :input_scale => [2.5 / sqrt(Q)],
    :ridge_param => [1e-6, 1e-4, 1e-2],
)

grid_search(
    run_once_grid,
    grids;
    nrep = 3,
    outfile = "gridsearch_deepesn_Q$(Q).csv",
    param_order = [:nl, :nr, :radius, :leaky_coeff, :input_scale, :ridge_param],
    error_names = [:rmse1, :rmse2, :rmse3, :rmse4],
    progress = true,
) 