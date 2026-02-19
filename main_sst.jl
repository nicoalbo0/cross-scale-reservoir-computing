# Activate environment
using Pkg, Revise
Pkg.activate(".")
Pkg.instantiate()

using CrossScaleRC
using LinearAlgebra
using Plots, Measures, LaTeXStrings

BLAS.set_num_threads(1)

data, tau = load_data([18.0, 6.0]; show_data=true, refinement=3);
coarse = data[1];
fine   = data[2];

grids = [
    (2,1),   # coarse layer → 2 blocks in longitude
    (6,3)    # fine layer → 16 blocks
]

mixing = 2
blocks_layers = make_blocks(data, grids, mixing)

blocks_coarse = blocks_layers[1]
blocks_fine   = blocks_layers[2]

# flattening
nlon_c, nlat_c, Ttot = size(coarse)
nlon_f, nlat_f, _    = size(fine)

coarse_mat = reshape(coarse, nlon_c*nlat_c, Ttot)
fine_mat   = reshape(fine,   nlon_f*nlat_f, Ttot)

# Experiment configuration

washout      = 1_000
train_len    = 30_000
predict_len  = 1_000
warmup       = 1_000

_, _, Ttot = size(data[1])
train_len + predict_len ≤ Ttot || error("Not enough data")

# hyperparameters
rec1, neigh1, layer1 = input_dimensions(blocks_coarse)
rec2, neigh2, layer2 = input_dimensions(blocks_fine)

num_networks = [2,      18]
mixing       = [2,       2]
res_size     = [1250, 1250]
res_radius   = [0.4,   0.8]
degree       = [10,     10]
g_in_rec     = [1.0/√rec1,    1.0/√rec2]
g_in_neigh   = [1.0/√neigh1,  1.0/√neigh2]
g_in_layer   = [0.0,          1.0/√layer2]
ridge_param  = [1e-1, 1e0]
dt           = [1.0, 1.0]
τ            = [1.0, 1.0]

res_params = (res_size, res_radius, degree, g_in_rec, g_in_neigh, g_in_layer, τ, dt)

blocks = [blocks_coarse, blocks_fine]

preds_fine, preds_coarse, train_pred_fine, train_pred_coarse, train_data_coarse, train_data_fine, data_coarse, X_coarse, X_fine = run_multi_layer(
    res_params,
    fine_mat,
    coarse_mat,
    train_len,
    predict_len,
    blocks;
    washout = washout,
    warmup = warmup,
    ridge_parameter = ridge_param,
    show_progress = false,
    input_mode = :structured, # :random, :structured,
    regression_mode = [:linear, :quadratic]
)

## Plotting

p_coarse =
    plot_train_test_heatmaps(
        train_data_coarse,
        train_pred_coarse,
        coarse_mat,
        preds_coarse;
        τ = tau,
        λ_max = 0.05,
        warmup = warmup,
        train_len = train_len,
        Q = size(coarse_mat, 1),
        L = size(coarse_mat, 1),
        title_prefix = "Layer 1",
        clims=(-3, 3)
    );

test_coarse   = coarse_mat[:,train_len - warmup + 1 : train_len - warmup + size(preds_coarse,2)];
error_curve_coarse = collect(rmse_upto(test_coarse[:, warmup:end], preds_coarse[:, warmup:end]; T=t) for t in 1:size(test_coarse[:, warmup:end], 2));


p_fine =
    plot_train_test_heatmaps(
        train_data_fine,
        train_pred_fine,
        fine_mat,
        preds_fine;
        τ = tau,
        λ_max = 0.05,
        warmup = warmup,
        train_len = train_len,
        Q = size(fine_mat, 1),
        L = size(fine_mat, 1),
        title_prefix = "Layer 2",
        clims=(-3, 3)
    );
test_fine   = fine_mat[:,train_len - warmup + 1 : train_len - warmup + size(preds_fine,2)];
error_curve_fine = collect(rmse_upto(test_fine[:, warmup:end], preds_fine[:, warmup:end]; T=t) for t in 1:size(test_fine[:, warmup:end], 2));

#--
p1 = plot(p_coarse, p_fine, layout = (2, 1), size = (1200, 1200), left_margin=5mm);


p2_1_2, p2_2_2 = plot_units_activity(X_coarse; n_units = min(res_size[1], 100)), plot_units_activity(X_fine; n_units = min(res_size[2], 100));

p2_1_1 = plot(error_curve_coarse, grid=false, lw=2, color=:black, ylabel="rmse_upto", xlabel="timestep", legend=false, title="Layer 1");
vline!(p2_1_1,[1], lw=2, color=:red);

p2_2_1 = plot(error_curve_fine, grid=false, lw=2, color=:black, ylabel="rmse_upto", xlabel="timestep", legend=false, title="Layer 2");
vline!(p2_2_1,[1], lw=2, color=:red);

p2 = plot(p2_1_1, p2_1_2, p2_2_1, p2_2_2, layout=(4,1));

p = plot(p1, p2, layout=grid(1, 2, widths = (2/3, 1/3)), size=(1000,750), left_margin=2mm);
display(p)