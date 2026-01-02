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

Q = 32
L = 22
μ = 0.0

data, τ = load_data(Q, L, μ; show_data=false, interpolate_data=false);

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

res_size     = 5000
res_radius   = 0.6
degree       = 5
g_in_rec     = 1.0
g_in_neigh   = 1.0
g_in_layer   = 1.0
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

p = plot_train_test_heatmaps(
    train_data,
    preds_train,
    data,
    preds_test;
    τ = τ,
    λ_max = 0.05,
    warmup = warmup,
    train_len = train_len,
    Q = Q,
    L = L
)
display(p)

p = plot(X[1][1:50,:]', label="")
display(p)