# Activate environment
using Pkg, Revise
Pkg.activate(".")
Pkg.instantiate()

using HierarchicalRC
using LinearAlgebra
using Plots, Measures, LaTeXStrings

#BLAS.set_num_threads(1)
BLAS.set_num_threads(6)

# single network config        | parallel network config
# dt           = 0.25          | dt           = 0.25
# Q            = 64            | Q            = 512
# L            = 22            | L            = 200
# μ            = 0.0/0.01      | μ            = 0.0/0.01
# num_networks = 1             | num_networks = 64
# mixing       = 0             | mixing       = 6/12
# Λ_max        = 0.05          | Λ_max       = 0.09

Q = 64
L = 22
μ = 0.01
resolution_divisor = 4; #4
Qeffective = div(Q,resolution_divisor);

data, τ = load_data(Q, L, μ; show_data=false, interpolate_data=false);
# regrid to lower the resolution
data = regrid_average(data, resolution_divisor);

# Experiment configuration

num_networks = 1
mixing       = 0
blocks       = make_blocks(size(data, 1), num_networks, mixing)
washout      = 1_000
train_len    = 50_000
predict_len  = 1_000
warmup       = 1_000

M, Ttot = size(data)
train_len + predict_len ≤ Ttot || error("Not enough data")

res_size     = 1000
res_radius   = 0.1
degree       = 10
g_in_rec     = 2.5/√(Qeffective/num_networks)
g_in_neigh   = 0.0
g_in_layer   = 0.0
ridge_param  = 1e-4

res_params = (res_size, res_radius, degree, g_in_rec, g_in_neigh, g_in_layer)

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
    input_mode = :structured
)

## Plotting 
#=
p1 = plot_train_test_heatmaps(
    train_data,
    preds_train,
    data,
    preds_test;
    τ = τ,
    λ_max = 0.05,
    warmup = warmup,
    train_len = train_len,
    Q = Qeffective,
    L = L
);

test_data   = data[:,train_len - warmup + 1 : train_len - warmup + size(preds_test,2)];

error_curve = collect(rmse_upto(test_data[:, warmup:end], preds_test[:, warmup:end]; T=t) for t in 1:size(test_data[:, warmup:end], 2));

p2_1 = plot(error_curve, grid=false, lw=2, color=:black, ylabel="rmse_upto", xlabel="timestep", legend=false, title="Total err.");
vline!(p2_1,[warmup, warmup], lw=2, color=:red);

p2_2 = plot_units_activity(X);
plot!(p2_2, title="Network activity");

p2 = plot(p2_1, p2_2, layout=(2,1));
display(plot(p1,p2, layout=grid(1, 2, widths = (2/3, 1/3)), size=(750,500)))
=#

## Plotting Check 2
test_data   = data[:,train_len - warmup + 1 : train_len - warmup + size(preds_test,2)];
h1 = heatmap(test_data[:, warmup:end], clim=(-3,3), cmap=:RdBu);
h2 = heatmap(preds_test[:, warmup:end], clim=(-3,3), cmap=:RdBu);
h3 = heatmap(test_data[:, warmup:end] - preds_test[:, warmup:end], clim=(-3,3), cmap=:RdBu);
vline!(h3, [1], lw=2, color=:red, legend=false);

error_curve = collect(rmse_upto(test_data[:, warmup:end], preds_test[:, warmup:end]; T=t) for t in 1:size(test_data[:, warmup:end], 2));

p1 = plot(error_curve, grid=false, lw=2, color=:black, ylabel="rmse_upto", xlabel="timestep", legend=false, title="Total err.");
#vline!(p1,[warmup, warmup], lw=2, color=:red);

display(plot(h1,h2,h3,p1, size=(800,500)))
