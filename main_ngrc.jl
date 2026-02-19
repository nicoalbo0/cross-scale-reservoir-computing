# Activate environment
using Pkg, Revise
project_root = @__DIR__ # 1. Get the directory of the current script
Pkg.activate(project_root)
Pkg.instantiate()

using CrossScaleRC
using LinearAlgebra
using Plots

BLAS.set_num_threads(1)

# Data
Q = 128
L = 44
μ = 0.01
resolution_divisor = 4
Qeffective = div(Q, resolution_divisor)

data, τ = load_data(Q, L, μ; show_data=false, refinement=1)
data = regrid_average(data, resolution_divisor)

# Experiment configuration
washout     = 1_000
train_len   = 50_000
predict_len = 1_000

M, Ttot = size(data)
washout + train_len + predict_len + 2 ≤ Ttot || error("Not enough data")


# Run NextGen
past   = 3      # number of past values (k)
degree = 2      # polynomial degree (p)
ridge  = 1e-2

preds_test, preds_train, Wout, Φ = nextgen_closedloop(
    data,
    train_len,
    predict_len;
    washout = washout,
    past = past,
    degree = degree,
    ridge = ridge
)

# Optional quick plot for one component (e.g. component 1)
t_start = washout + past
train_data = data[:, (t_start + 1):(t_start + train_len)];
test_data  = data[:, (t_start + train_len + 1):(t_start + train_len + predict_len)];

## Plotting
h1 = heatmap(test_data, clim=(-3, 3), cmap=:RdBu);
h2 = heatmap(preds_test, clim=(-3, 3), cmap=:RdBu);
h3 = heatmap(test_data - preds_test, clim=(-3, 3), cmap=:RdBu);
vline!(h3, [1], lw=2, color=:red, legend=false);

error_curve = collect(
    rmse_upto(test_data, preds_test; T=t) 
    for t in 1:size(test_data, 2)
);

p1 = plot(error_curve, grid=false, lw=2, color=:black, 
    ylabel="rmse_upto", xlabel="timestep", legend=false, title="Total err.")

display(plot(h1, h2, h3, p1, size=(800, 500)))