# Activate environment
using Pkg, Revise
Pkg.activate(".")
Pkg.instantiate()

using CrossScaleRC
using LinearAlgebra, Measures
using Plots

BLAS.set_num_threads(1)

Q = 128
L = 44
μ = 0.01

data, tau      = load_data(Q, L, μ; show_data=false, refinement=1);

#data_c, _    = load_data(Int(Q/4), Int(L/2) , μ; show_data=false, interpolate_data=false);
resolution_divisor_upper_layer          = 4
coarse         = regrid_average(data, resolution_divisor_upper_layer)
fine           = data
# Experiment configuration

washout      = 1_000
train_len    = 50_000
predict_len  = 1_000
warmup       = 1_000

M, Ttot = size(data)
train_len + predict_len ≤ Ttot || error("Not enough data")

num_networks = [4,       16]
mixing       = [2,       8]
res_size     = [1000, 400]
res_radius   = [0.1,   0.2]
degree       = [10,     10]
g_in_rec     = [1.0/√(div(Q,resolution_divisor_upper_layer)/num_networks[1]),   10.0^(-3.0) /√(Q/num_networks[2])]
g_in_neigh   = [10^(-0.5)/√(mixing[1]),   10.0^(-2.5)/√(mixing[2])]
g_in_layer   = [0.0,   1.0/√(div(Q,resolution_divisor_upper_layer)/num_networks[2])]
ridge_param  = [1e-6, 1e-3]
dt           = [0.25, 0.25]
τ            = [0.25, 0.25]


res_params = (res_size, res_radius, degree, g_in_rec, g_in_neigh, g_in_layer, τ, dt)

blocks_coarse = make_blocks(size(coarse, 1), num_networks[1], mixing[1])
blocks_fine   = make_blocks(size(data, 1), resolution_divisor_upper_layer, num_networks[2], mixing[2], num_networks[1]; overlap_mode = :exclude)

blocks = [blocks_coarse, blocks_fine]

preds_fine, preds_coarse, train_pred_fine, train_pred_coarse, train_data_coarse, train_data_fine, data_coarse, X_coarse, X_fine = run_multi_layer(
    res_params,
    fine,
    coarse,
    train_len,
    predict_len,
    blocks;
    washout = washout,
    warmup = warmup,
    ridge_parameter = ridge_param,
    show_progress = false,
    input_mode = :structured, # :random, :structured,
    regression_mode = [:quadratic, :quadratic]
)

## Plotting

p_coarse =
    plot_train_test_heatmaps(
        train_data_coarse,
        train_pred_coarse,
        data_coarse,
        preds_coarse;
        τ = tau,
        λ_max = 0.08,
        warmup = warmup,
        train_len = train_len,
        Q = Int(round(Q / resolution_divisor_upper_layer)),
        L = Int(round(L / resolution_divisor_upper_layer)),
        title_prefix = "Layer 1"
    );

test_coarse   = data_coarse[:,train_len - warmup + 1 : train_len - warmup + size(preds_coarse,2)];
error_curve_coarse = collect(rmse_upto(test_coarse[:, warmup:end], preds_coarse[:, warmup:end]; T=t) for t in 1:size(test_coarse[:, warmup:end], 2));


p_fine =
    plot_train_test_heatmaps(
        train_data_fine,
        train_pred_fine,
        data,
        preds_fine;
        τ = tau,
        λ_max = 0.08,
        warmup = warmup,
        train_len = train_len,
        Q = Q,
        L = L,
        title_prefix = "Layer 2"
    );
test_fine   = data[:,train_len - warmup + 1 : train_len - warmup + size(preds_fine,2)];
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