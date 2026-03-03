# Activate environment
using Pkg, Revise
Pkg.activate(".")
Pkg.instantiate()

using CrossScaleRC
using LinearAlgebra
using Plots, Measures, LaTeXStrings

BLAS.set_num_threads(1)

##
sampling_rate = 4; # per day
data, tau = load_data([18.0, 6.0]; show_data=true, refinement=sampling_rate);
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
res_radius   = [0.85,   0.55]
degree       = [10,     10]
g_in_rec     = [10^(-1.5)/√rec1,    10^(-0.5)/√rec2]
g_in_neigh   = [10^(-1.5)/√neigh1,  10^(-0.5)/√neigh2]
g_in_layer   = [0.0,          10^(-1.0)/√layer2]
ridge_param  = [1e-2, 10^(1.0)]
dt           = [1.0/sampling_rate, 1.0/sampling_rate]
τ            = [2.5, 2.5]

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
    input_mode = :random, # :random, :structured,
    regression_mode = [:linear, :linear]
)

# Plotting
test_coarse   = coarse_mat[:,train_len - warmup + 1 : train_len - warmup + size(preds_coarse,2)];
error_curve_coarse = collect(rmse_upto(test_coarse[:, warmup:end], preds_coarse[:, warmup:end]; T=t) for t in 1:size(test_coarse[:, warmup:end], 2));

coarse_scale = 0.5*maximum(abs.(data[1]))
h1 = heatmap(test_coarse[:, warmup:end], clim=(-coarse_scale,coarse_scale), cmap=:RdBu, xlabel="timestep", title="Test Data");
h2 = heatmap(preds_coarse[:, warmup:end], clim=(-coarse_scale,coarse_scale), cmap=:RdBu, xlabel="timestep", title="Test Forecast");
h3 = heatmap(test_coarse[:, warmup:end] - preds_coarse[:, warmup:end], clim=(-coarse_scale,coarse_scale), cmap=:RdBu, xlabel="timestep", title="Test Error");
vline!(h3, [1], lw=2, color=:red, legend=false);
p_coarse = plot(h1,h2,h3, size=(800,500), plot_title="Layer 1");

#--
test_fine   = fine_mat[:,train_len - warmup + 1 : train_len - warmup + size(preds_fine,2)];
error_curve_fine = collect(rmse_upto(test_fine[:, warmup:end], preds_fine[:, warmup:end]; T=t) for t in 1:size(test_fine[:, warmup:end], 2));

fine_scale = 0.5*maximum(abs.(data[2]))
h1 = heatmap(test_fine[:, warmup:end], clim=(-fine_scale,fine_scale), cmap=:RdBu, xlabel="timestep", title="Test Data");
h2 = heatmap(preds_fine[:, warmup:end], clim=(-fine_scale,fine_scale), cmap=:RdBu, xlabel="timestep", title="Test Forecast");
h3 = heatmap(test_fine[:, warmup:end] - preds_fine[:, warmup:end], clim=(-fine_scale,fine_scale), cmap=:RdBu, xlabel="timestep", title="Test Error");
vline!(h3, [1], lw=2, color=:red, legend=false);
p_fine = plot(h1,h2,h3, size=(800,500), plot_title="Layer 2");

#--
p1 = plot(p_coarse, p_fine, layout = (2, 1), size = (1200, 1200), left_margin=5mm);

p2_1_2, p2_2_2 = plot_units_activity(X_coarse; n_units = min(res_size[1], 250)), plot_units_activity(X_fine; n_units = min(res_size[2], 250));

p2_1_1 = plot(error_curve_coarse, grid=false, lw=2, color=:black, ylabel="rmse_upto", xlabel="timestep", legend=false, title="Layer 1");
vline!(p2_1_1,[1], lw=2, color=:red);

p2_2_1 = plot(error_curve_fine, grid=false, lw=2, color=:black, ylabel="rmse_upto", xlabel="timestep", legend=false, title="Layer 2");
vline!(p2_2_1,[1], lw=2, color=:red);

p2 = plot(p2_1_1, p2_1_2, p2_2_1, p2_2_2, layout=(4,1));

p = plot(p1, p2, layout=grid(1, 2, widths = (2/3, 1/3)), size=(1000,750), left_margin=2mm);
display(p)