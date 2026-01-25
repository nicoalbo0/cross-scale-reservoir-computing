# Activate environment
using Pkg, Revise
project_root = @__DIR__ # 1. Get the directory of the current script
Pkg.activate(project_root)
Pkg.instantiate()

using HierarchicalRC
using ReservoirComputing
using LinearAlgebra
using Plots, Measures, LaTeXStrings
using Random, Statistics
using SparseArrays

include(joinpath(project_root, "src", "deeprc_utils.jl"))

# Set threading
BLAS.set_num_threads(6)

# ============================================================================
# Main Script
# ============================================================================

##
# Data parameters
println("Loading KS data...")
Q0 = 64
L = 22
μ = 0.01
resolution_divisor = 4
Q = div(Q0, resolution_divisor)

data, τ = load_data(Q0, L, μ; show_data=false, interpolate_data=false)
# regrid to lower the resolution
data = regrid_average(data, resolution_divisor)

# Experiment configuration
washout = 1_000
train_len = 50_000
warmup = 1_000
predict_len = 1_000

M_data, T_tot = size(data)
train_len + predict_len ≤ T_tot || error("Not enough data")

# Deep RC hyperparameters
nu = Q
ny = Q
nl = 2
nr = 500

reservoir_params = Dict(
    :radius => 0.1,
    :sparsity => 10/nr,
    :input_scale => 2.5/√(Q),
    :leaky_coeff => 1.0,
    :ridge_param => 1e-4
)

# ============================================================================
# TRAINING PHASE
# ============================================================================
input_data = data[:, 1:(washout + train_len - 1)]
target_data = data[:, 2:(washout + train_len)]

println("\\n" * "="^70)
println("DEEP RESERVOIR COMPUTING ON KURAMOTO–SIVASHINSKY EQUATION")
println("="^70)
println("Data shape: $(size(data))")
println("Training samples: $train_len")
println("Prediction horizon: $predict_len")
println("Number of layers: $nl")
println("Layer sizes: $nr")
println("="^70 * "\\n")

# Build and train DeepESN
rng = MersenneTwister(42)


deep_rc = DeepESN(nu, ny;
    nl = nl,
    nr = nr,
    rho = reservoir_params[:radius],
    leak = reservoir_params[:leaky_coeff],
    input_scale = reservoir_params[:input_scale],
    inter_scale = 1.0,
    sparsity = reservoir_params[:sparsity],
    rng = rng
)

println("\\nTraining readout...")
preds_train = train!(deep_rc, input_data, target_data;
    washout = washout,
    ridge = reservoir_params[:ridge_param],
    reset_state = true,
    return_output = true
)

println("\\nDeep RC architecture built successfully!")

# ============================================================================
# TESTING PHASE
# ============================================================================
##
println("Generating test predictions...")
test_start_idx = train_len - warmup + 1    # = 49_001
warmup_u = data[:, test_start_idx:(test_start_idx + warmup - 1)]

# Now predict
preds_test = test_closed_loop(deep_rc;
    steps = predict_len,
    warmup = warmup_u,
    reset_state = true
)

# Extract the corresponding true test data
test_data = data[:, (test_start_idx + warmup):(test_start_idx + warmup + predict_len - 1)]


println("\\nTest data range: indices $(test_start_idx + warmup) to $(test_start_idx + warmup + predict_len - 1)")
println("Training data range: indices 1 to $(washout + train_len)")
println("No overlap: $(test_start_idx + warmup) > $(washout + train_len) is $(test_start_idx + warmup > washout + train_len)")

## Plotting
h1 = heatmap(test_data, clim=(-3, 3), cmap=:RdBu)
h2 = heatmap(preds_test[:, warmup+1:end], clim=(-3, 3), cmap=:RdBu)
h3 = heatmap(test_data - preds_test[:, warmup+1:end], clim=(-3, 3), cmap=:RdBu)
vline!(h3, [1], lw=2, color=:red, legend=false)

error_curve = collect(
    rmse_upto(test_data[:, 1:end], preds_test[:, warmup+1:end]; T=t) 
    for t in 1:size(test_data, 2)
)

p1 = plot(error_curve, grid=false, lw=2, color=:black, 
    ylabel="rmse_upto", xlabel="timestep", legend=false, title="Total err.")

display(plot(h1, h2, h3, p1, size=(800, 500)))