# Activate environment
using Pkg, Revise
Pkg.activate(".")
Pkg.instantiate()

using HierarchicalRC
using LinearAlgebra
using Plots, Measures, LaTeXStrings

BLAS.set_num_threads(1)

# single network config        | parallel network config
# dt           = 0.25          | dt           = 0.25
# Q            = 64            | Q            = 512
# L            = 22            | L            = 200
# μ            = 0.0/0.01      | μ            = 0.0/0.01
# num_networks = 1             | num_networks = 64
# mixing       = 0             | mixing       = 6/12
# Λ_max        = 0.05          | Λ_max       = 0.09

Q::Int = 64
L::Int = 22
μ = 0.01
res_divisor::Int = 8;
Qeffective = div(Q,res_divisor);

data, τ = load_data(Q, L, μ; show_data=false, interpolate_data=false);
# regrid to lower the resolution
data = regrid_average(data, res_divisor);

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

res_size     = 1500
res_radius   = 0.6
degree       = 10
g_in_rec     = 1.0/√(Qeffective/num_networks)
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
    show_progress = true,
    input_mode = :structured
)

## Plotting 

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

error_curve = collect(rmse_upto(test_data, preds_test; T=t) for t in 1:size(test_data, 2))

p2_1 = plot(error_curve, grid=false, lw=2, color=:black, ylabel="rmse_upto", xlabel="timestep", legend=false, title="total err.");
vline!(p2_1,[warmup, warmup], lw=2, color=:red);
p2_2 = plot(X[1][1:50,:]', xlabel="timestep", ylabel="node state", label="", title="\nnetwork activity", ylim=(-1, 1), color=:black, alpha=0.25);
p2 = plot(p2_1, p2_2, layout=(2,1));
display(plot(p1,p2, layout=grid(1, 2, widths = (2/3, 1/3)), size=(750,500)))