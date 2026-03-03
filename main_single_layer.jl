# Activate environment
using Pkg, Revise
Pkg.activate(".")
Pkg.instantiate()

using CrossScaleRC
using LinearAlgebra
using Plots, Measures, LaTeXStrings

BLAS.set_num_threads(1)

Q0 = 128
L = 44
μ = 0.01

resolution_divisor = 1;
Q = div(Q0,resolution_divisor);
data, τ = load_data(Q0, L, μ; show_data=false, refinement=1);

# regrid to lower the resolution
data = regrid_average(data, resolution_divisor);

# Experiment configuration

num_networks = 16
mixing       = 8
blocks       = make_blocks(size(data, 1), num_networks, mixing)
washout      = 1_000
train_len    = 50_000
predict_len  = 1_000
warmup       = 1_000

M, Ttot = size(data)
train_len + predict_len ≤ Ttot || error("Not enough data")

res_size     = 400
res_radius   = 0.3
degree       = 10
g_in_rec     = 1.0/√(Q/num_networks)
g_in_neigh   = 1.0/√(mixing)
g_in_layer   = 0.0
ridge_param  = 1e-3
dt           = 0.25
τ            = 0.25

res_params = (res_size, res_radius, degree, g_in_rec, g_in_neigh, g_in_layer, τ, dt)

preds_test, preds_train, train_data , X, _ = run_single_layer(
    res_params,
    data,
    zeros(size(data)), # no previous multilayer input
    train_len,
    predict_len,
    blocks;
    washout = washout,
    warmup = warmup,
    ridge_parameter = ridge_param,
    show_progress = false,
    input_mode = :structured,
    regression_mode = :quadratic, #or linear
)

## Plot
test_data   = data[:,train_len - warmup + 1 : train_len - warmup + size(preds_test,2)];
h1 = heatmap(test_data[:, warmup:end], clim=(-3,3), cmap=:RdBu, xlabel="timestep", title="Test Data");
h2 = heatmap(preds_test[:, warmup:end], clim=(-3,3), cmap=:RdBu, xlabel="timestep", title="Test Forecast");
h3 = heatmap(test_data[:, warmup:end] - preds_test[:, warmup:end], clim=(-3,3), cmap=:RdBu, xlabel="timestep", title="Test Error");
vline!(h3, [1], lw=2, color=:red, legend=false);

error_curve = collect(rmse_upto(test_data[:, warmup:end], preds_test[:, warmup:end]; T=t) for t in 1:size(test_data[:, warmup:end], 2));

p1 = plot(error_curve, grid=false, lw=2, color=:black, ylabel="rmse_upto", xlabel="timestep", legend=false, title="Total err.");
#vline!(p1,[warmup, warmup], lw=2, color=:red);

display(plot(h1,h2,h3,p1, size=(800,500)))
