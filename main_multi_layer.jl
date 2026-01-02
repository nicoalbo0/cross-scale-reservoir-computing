# Activate environment
using Pkg, Revise
Pkg.activate(".")
Pkg.instantiate()

using HierarchicalRC
using LinearAlgebra, Measures
using Plots

BLAS.set_num_threads(1)

# single network config        | parallel network config
# Q            = 64            | Q            = 512
# L            = 22            | L            = 200
# μ            = 0.0/0.01      | μ            = 0.0/0.01
# num_networks = 1             | num_networks = 64
# mixing       = 0             | mixing       = 6/12
# Λ_max        = 0.05          | Λ_max       = 0.09

Q = 64
L = 22
μ = 0.0

data, τ      = load_data(Q, L, μ; show_data=false, interpolate_data=false);
#data_c, _    = load_data(Int(Q/4), Int(L/2) , μ; show_data=false, interpolate_data=false);
div          = 8
data_c       = regrid_average(data, div)

# Experiment configuration

washout      = 1_000
train_len    = 30_000
predict_len  = 1_000
warmup       = 1_000

M, Ttot = size(data)
train_len + predict_len ≤ Ttot || error("Not enough data")

res_size     = [5000, 5000]
res_radius   = [0.6,   0.6]
degree       = [10,     10]
g_in_rec     = [1.0,   0.5]
g_in_neigh   = [0.0,   0.1]
g_in_layer   = [0.0,   0.1]
ridge_param  = [1e-4, 1e-4]
num_networks = [1,       2]
mixing       = [0,       6]

res_params = (res_size, res_radius, degree, g_in_rec, g_in_neigh, g_in_layer)

preds_fine, preds_coarse, train_pred_fine, train_pred_coarse, train_data_coarse, train_data_fine, data_coarse, X_coarse, X_fine = run_multi_layer(
    res_params,
    data,
    data_c,
    train_len,
    predict_len;
    washout = washout,
    warmup = warmup,
    num_networks = num_networks,
    mixing = mixing,
    ridge_parameter = ridge_param,
    show_progress = true,
    div = div
)

## Plotting

p_coarse =
    plot_train_test_heatmaps(
        train_data_coarse,
        train_pred_coarse,
        data_coarse,
        preds_coarse;
        τ = τ,
        λ_max = 0.05,
        warmup = warmup,
        train_len = train_len,
        Q = Int(round(Q / div)),
        L = Int(round(L / div)),
        title_prefix = "Layer 1"
    )

p_fine =
    plot_train_test_heatmaps(
        train_data_fine,
        train_pred_fine,
        data,
        preds_fine;
        τ = τ,
        λ_max = 0.05,
        warmup = warmup,
        train_len = train_len,
        Q = Q,
        L = L,
        title_prefix = "Layer 2"
    )

p = plot(p_coarse, p_fine, layout = (2, 1), size = (1200, 1200), left_margin=5mm)
display(p)

p1, p2 = plot_units_activity(X_coarse), plot_units_activity(X_fine)
p = plot(p1, p2, layout=(2,1), size=(600,800), left_margin=5mm)
display(p)