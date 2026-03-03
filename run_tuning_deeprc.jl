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

BLAS.set_num_threads(Threads.nthreads())
#BLAS.set_num_threads(round(Int, Threads.nthreads()/2))

##
Q0 = 128
L = 44
μ = 0.01

resolution_divisor = 1
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
        inter_scale = reservoir_params[:inter_scale],
        sparsity = reservoir_params[:sparsity],
    )

    train!(deep_rc, input_data, target_data;
        washout = washout,
        ridge = reservoir_params[:ridge_param],
    )

    test_start_idx = train_len - warmup + 1
    warmup_u = data[:, test_start_idx:(test_start_idx + warmup - 1)]

    preds_test = test_closed_loop(deep_rc;
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

# ---------------------------
# Baseline + wrapper that builds (nr, nl, reservoir_params)
# ---------------------------

base_reservoir_params(nr) = Dict(
    :radius => 0.1,
    :sparsity => 10 / nr,
    :input_scale => 2.5 / sqrt(Q),
    :inter_scale => 2.5 / sqrt(nr),
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
    rp[:inter_scale] = p.inter_scale
    rp[:ridge_param] = p.ridge_param

    return run_once((nr, nl, rp))
end

# ---------------------------
# Define the grids
# ---------------------------
n_nodes = 1000
grids = Dict(
    :nl => [2],
    :nr => [n_nodes],
    :radius => [parse(Float64, ARGS[1])],
    :leaky_coeff => [1.0],
    :input_scale => [10.0^k / sqrt(Q) for k=-3:0.5:1],
    :inter_scale => [10.0^k / sqrt(n_nodes) for k=-3:0.5:1],
    :ridge_param => [10.0^k for k=-6:-2],
)

time0 = time()
grid_search(
    run_once_grid,
    grids;
    nrep = 20,
    outfile = "gridsearch_deepesn_Q$(Q)_$(ARGS[2]).csv",
    param_order = [:nl, :nr, :radius, :leaky_coeff, :input_scale, :inter_scale, :ridge_param],
    error_names = [:rmse1, :rmse2, :rmse3, :rmse4],
    progress = true,
) 
println("time spent $(time() - time0)")