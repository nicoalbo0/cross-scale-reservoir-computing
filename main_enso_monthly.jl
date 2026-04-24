# ENSO forecasting at MONTHLY resolution.
#
# Rationale: ENSO lives at 2–7 yr timescales. Training a reservoir on daily
# data makes it spend capacity on sub-seasonal noise that's fundamentally
# unpredictable beyond ~2 weeks. By binning the anomaly field into monthly
# means *before* it hits the reservoir, we decouple the time step of the
# model from the sampling rate of the data: one reservoir step = one month,
# so τ=20 hosts a 2-yr memory and the reservoir operates at the signal's
# natural timescale.
#
# Usage:
#   ENSO_SEED=42 julia --threads 4 --project=. main_enso_monthly.jl

using Pkg, Revise
Pkg.activate(".")

using CrossScaleRC
using JLD2
using Dates
using LinearAlgebra
using Random
using Statistics
using Plots, Measures, LaTeXStrings

BLAS.set_num_threads(1)

seed = parse(Int, get(ENV, "ENSO_SEED", "42"))
Random.seed!(seed)
seed_tag = "seed$(seed)"

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

mode = :three_layer
@assert mode in (:three_layer, :two_layer, :single_layer)
mode_tag = String(mode)

# Monthly pipeline: 1 reservoir step = 1 calendar month.
# 1982-01..2015-12 = 34 yr × 12 = 408 months total.
washout     = 12            # 1 yr washout before ridge regression
train_len   = 288           # 24 yr training window (covers ~5 ENSO cycles)
predict_len = 96            # 8 yr autonomous forecast
warmup      = 12            # 1 yr state warm-up from true data
dt          = 1.0           # 1 month / step

lon_range = (126.0, 288.0)
lat_range = (-36.0, 36.0)
start_date = Date(1982, 1, 1)

# ---------------------------------------------------------------------------
# 2. Load daily SST anomalies, then bin to monthly
# ---------------------------------------------------------------------------

println("Loading SST (daily, 3 resolutions) ...")
# load_sst_data with refinement=1 returns daily anomalies (no time upsampling).
# The climatology training window is specified in DAYS.
train_days_for_clim = train_len * 30  # ≈ 24 yr in days — covers the monthly training period
data_vec_daily, _dt = load_data(
    [18.0, 6.0, 2.0];
    show_data     = false,
    refinement    = 1,
    lon_range     = lon_range,
    lat_range     = lat_range,
    anomalies     = true,
    train_indices = 1:train_days_for_clim
)

"""Bin (nlon × nlat × nt_daily) into (nlon × nlat × nt_monthly) using calendar months."""
function daily_to_monthly(daily::AbstractArray{<:Real, 3}, start_date::Date)
    nlon, nlat, nt_daily = size(daily)
    dates  = [start_date + Day(t - 1) for t in 1:nt_daily]
    ym     = [(year(d), month(d)) for d in dates]
    keys   = unique(ym)
    monthly = Array{Float64, 3}(undef, nlon, nlat, length(keys))
    for (k, target) in enumerate(keys)
        mask = [x == target for x in ym]
        monthly[:, :, k] .= dropdims(mean(daily[:, :, mask]; dims=3); dims=3)
    end
    return monthly
end

println("Binning to monthly means ...")
data_vec = [daily_to_monthly(d, start_date) for d in data_vec_daily]

# Re-normalise per pixel at the monthly scale (daily normalisation attenuates
# after averaging, so the reservoir would be under-driven).
for d in data_vec
    nlon, nlat, _ = size(d)
    for i in 1:nlon, j in 1:nlat
        ts = @view d[i, j, :]
        any(isnan, ts) && (ts .= 0.0)
        μ, σ = mean(ts), std(ts)
        σ > 0 && (ts .= (ts .- μ) ./ σ)
    end
end

very_coarse = data_vec[1]
medium      = data_vec[2]
fine        = data_vec[3]

nlon_vc, nlat_vc, Ttot = size(very_coarse)
nlon_m,  nlat_m,  _    = size(medium)
nlon_f,  nlat_f,  _    = size(fine)

println("Domain (lon × lat × months):")
println("  18° : $(nlon_vc)×$(nlat_vc) × $(Ttot)")
println("   6° : $(nlon_m)×$(nlat_m) × $(Ttot)")
println("    2°: $(nlon_f)×$(nlat_f) × $(Ttot)")
println("Need: train+predict = $(train_len)+$(predict_len) = $(train_len + predict_len) ≤ $(Ttot)")
train_len + predict_len ≤ Ttot || error("Not enough data.")

vc_mat = reshape(very_coarse, nlon_vc * nlat_vc, Ttot)
m_mat  = reshape(medium,      nlon_m  * nlat_m,  Ttot)
f_mat  = reshape(fine,        nlon_f  * nlat_f,  Ttot)

# ---------------------------------------------------------------------------
# 3. Block construction (unchanged)
# ---------------------------------------------------------------------------

grids_AB = [(3, 1), (9, 3)]
grids_BC = [(9, 3), (9, 4)]
mixing = 2

blocks_AB = make_blocks(data_vec[1:2], grids_AB, mixing)
blocks_BC = make_blocks(data_vec[2:3], grids_BC, mixing)

blocks_vc   = blocks_AB[1]
blocks_m_AB = blocks_AB[2]
blocks_m_BC = blocks_BC[1]
blocks_f    = blocks_BC[2]

rec_vc, neigh_vc, _       = input_dimensions(blocks_vc)
rec_m,  neigh_m,  layer_m = input_dimensions(blocks_m_AB)
rec_f,  neigh_f,  layer_f = input_dimensions(blocks_f)

# ---------------------------------------------------------------------------
# 4. Hyperparameters (MONTHLY timescales)
# ---------------------------------------------------------------------------

# Memory target: 18° needs 2-4 yr memory (24-48 months). With α=dt/τ, decay
# (1-α)^24 ≈ 0.5 requires α ≈ 0.029 → τ ≈ 35. 6° faster (~1 yr), 2° even faster.
res_size_A = [500,   500]
res_rad_A  = [0.85,  0.75]
degree_A   = [10,    10]
g_rec_A    = [10^(-1.5)/√rec_vc,   10^(-0.5)/√rec_m]
g_neigh_A  = [10^(-1.5)/√neigh_vc, 10^(-0.5)/√neigh_m]
g_layer_A  = [0.0,                  10^(-1.0)/√layer_m]
ridge_A    = [1e-3,  1e-2]
τ_A        = [30.0,  10.0]   # months
dt_A       = [dt,    dt]
params_A   = (res_size_A, res_rad_A, degree_A, g_rec_A, g_neigh_A, g_layer_A, τ_A, dt_A)

res_size_B = [500,   500]
res_rad_B  = [0.75,  0.55]
degree_B   = [10,    10]
g_rec_B    = [10^(-0.5)/√rec_m,   10^(-0.5)/√rec_f]
g_neigh_B  = [10^(-0.5)/√neigh_m, 10^(-0.5)/√neigh_f]
g_layer_B  = [0.0,                 10^(-1.0)/√layer_f]
ridge_B    = [1e-2,  1.0]
τ_B        = [10.0,  3.0]
dt_B       = [dt,    dt]
params_B   = (res_size_B, res_rad_B, degree_B, g_rec_B, g_neigh_B, g_layer_B, τ_B, dt_B)

# ---------------------------------------------------------------------------
# 5. Run pipeline (3-layer by default, same chaining as main_enso.jl)
# ---------------------------------------------------------------------------

if mode == :three_layer
    println("\n--- Call A: 18° → 6° ---")
    preds_m, preds_vc, train_pred_m, train_pred_vc,
        train_data_vc, train_data_m, _, X_vc, X_m = run_multi_layer(
            params_A, m_mat, vc_mat, train_len, predict_len,
            [blocks_vc, blocks_m_AB];
            washout = washout, warmup = warmup,
            ridge_parameter = ridge_A,
            show_progress = true, input_mode = :random,
            regression_mode = [:quadratic, :quadratic])
    coarse_for_B = hcat(train_pred_m, preds_m[:, warmup+1:end])
end

if mode in (:three_layer, :two_layer)
    coarse_for_B = (mode == :two_layer) ? m_mat : coarse_for_B
    println("\n--- Call B: 6° → 2° ---")
    preds_f, preds_m_B, train_pred_f, train_pred_m_B,
        _, train_data_f, _, _, X_f = run_multi_layer(
            params_B, f_mat, coarse_for_B, train_len, predict_len,
            [blocks_m_BC, blocks_f];
            washout = washout, warmup = warmup,
            ridge_parameter = ridge_B,
            show_progress = true, input_mode = :random,
            regression_mode = [:quadratic, :quadratic])
elseif mode == :single_layer
    println("\n--- Single layer 2° ---")
    blocks_f_solo = make_blocks([fine], [(9, 4)], mixing)[1]
    params_f_solo = (res_size_B[2], res_rad_B[2], degree_B[2],
                     g_rec_B[2], g_neigh_B[2], 0.0, τ_B[2], dt_B[2])
    preds_f, train_pred_f, train_data_f, X_f, _ = run_single_layer(
        params_f_solo, f_mat, zeros(eltype(f_mat), size(f_mat)),
        train_len, predict_len, blocks_f_solo;
        washout = washout, warmup = warmup,
        ridge_parameter = ridge_B[2],
        show_progress = true, input_mode = :random,
        regression_mode = :quadratic)
end

# ---------------------------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------------------------

t0 = train_len - warmup
test_f = f_mat[:, t0 + 1 : t0 + size(preds_f, 2)]

function cropped_lons_lats(nlon, nlat, res, lon_range, lat_range)
    lon_start = -180.0 + res / 2
    lat_start = -90.0  + res / 2
    lons_all  = lon_start .+ res .* (0 : round(Int, 360 / res) - 1)
    lats_all  = lat_start .+ res .* (0 : round(Int, 180 / res) - 1)
    lons360   = mod.(lons_all, 360.0)
    lon_mask  = lon_range[1] .<= lons360 .<= lon_range[2]
    lat_mask  = lat_range[1] .<= lats_all .<= lat_range[2]
    return lons_all[lon_mask], lats_all[lat_mask]
end
lons_f_crop, lats_f_crop = cropped_lons_lats(nlon_f, nlat_f, 2.0, lon_range, lat_range)

test_f_3d  = reshape(test_f,  nlon_f, nlat_f, size(test_f,  2))
preds_f_3d = reshape(preds_f, nlon_f, nlat_f, size(preds_f, 2))
n34_true = nino34_index(test_f_3d[:,  :, warmup + 1 : end], lons_f_crop, lats_f_crop)
n34_pred = nino34_index(preds_f_3d[:, :, warmup + 1 : end], lons_f_crop, lats_f_crop)

sc = skill_score(n34_true, n34_pred)
println("\n=== ENSO Monthly Forecast Skill ===")
println("  ACC               : $(round(sc.acc; digits=3))")
println("  RMSE              : $(round(sc.rmse; digits=4))")
println("  RMSE skill vs pers: $(round(sc.rmse_skill; digits=3))")

# Lead-time ACC in months
println("\n=== Lead-time skill (months) ===")
lead_months = [3, 6, 9, 12, 18, 24, 36, 48]
lead_accs = Float64[]
for L in lead_months
    L > length(n34_true) && continue
    s = skill_score(n34_true[1:L], n34_pred[1:L])
    push!(lead_accs, s.acc)
    println("  $(L) mo : ACC=$(round(s.acc; digits=3))   RMSE=$(round(s.rmse; digits=4))")
end

mkpath("results")
# Niño 3.4 time series plot
p1 = plot(n34_true; lw=2.5, color=:black, label="Observed",
          xlabel="month since forecast start", ylabel="Niño 3.4 (norm.)",
          title="Monthly ENSO forecast  seed=$(seed)  ACC=$(round(sc.acc; digits=2))",
          legend=:topright)
plot!(p1, n34_pred; lw=2.0, color=:red, linestyle=:dash, label="Forecast")

# Lead-time curve
p2 = plot(lead_months[1:length(lead_accs)], lead_accs;
          marker=:circle, lw=2, color=:blue, label="ACC",
          xlabel="Lead (months)", ylabel="Niño 3.4 ACC",
          title="Lead-time skill", ylim=(-1, 1), legend=:topright)
hline!(p2, [0.5]; ls=:dash, color=:gray, label="ACC=0.5 (useful)")
hline!(p2, [0.0]; ls=:dot,  color=:black, label="no skill")

p = plot(p1, p2; layout=(2, 1), size=(1200, 700), left_margin=4mm)
savefig(p, "results/enso_monthly_$(mode_tag)_$(seed_tag).png")
println("\nSaved: results/enso_monthly_$(mode_tag)_$(seed_tag).png")

# ── Geographic snapshot maps — 2° resolution, observed vs forecast ─────────
# Show the SST-anomaly field at selected lead times so spatial coverage of
# ENSO-related patterns (warm pool, cold tongue, central-Pacific warming
# during El Niño) can be compared between observed and forecast directly.
snapshot_leads = [6, 12, 18, 24]
snapshot_leads = filter(L -> L ≤ size(preds_f_3d, 3) - warmup, snapshot_leads)
# Fixed color scale (±1.5 σ, since data is already normalized to unit std).
# Using a tight scale reveals patterns rather than washing them out.
sc_map = 1.5

function map_panel(field2d, lons, lats, title_str, scale; colormap=:RdBu)
    heatmap(lons, lats, permutedims(field2d);
            clim=(-scale, scale), c=colormap,
            xlabel="longitude (°E)", ylabel="latitude (°N)",
            title=title_str, aspect_ratio=:equal,
            xlims=(minimum(lons), maximum(lons)),
            ylims=(minimum(lats), maximum(lats)),
            colorbar=true)
end

lons_plot = mod.(lons_f_crop, 360.0)   # 126…288 °E
# Ensure monotonic for heatmap
perm = sortperm(lons_plot)
lons_plot = lons_plot[perm]

map_panels = Plots.Plot[]
for L in snapshot_leads
    col = warmup + L
    obs_2d  = test_f_3d[:,  :, col][perm, :]
    fore_2d = preds_f_3d[:, :, col][perm, :]
    err_2d  = obs_2d .- fore_2d

    push!(map_panels, map_panel(obs_2d,  lons_plot, lats_f_crop,
                                "Observed — lead $L mo", sc_map))
    push!(map_panels, map_panel(fore_2d, lons_plot, lats_f_crop,
                                "Forecast — lead $L mo", sc_map))
    push!(map_panels, map_panel(err_2d,  lons_plot, lats_f_crop,
                                "Obs − Forecast — lead $L mo", sc_map))
end

n_leads = length(snapshot_leads)
p_maps = plot(map_panels...;
              layout = (n_leads, 3),
              size = (1600, 320 * n_leads),
              plot_title = "2° SST anomaly maps  —  seed=$(seed)  mode=$(mode_tag)",
              left_margin = 4mm, bottom_margin = 2mm)
savefig(p_maps, "results/enso_monthly_maps_$(mode_tag)_$(seed_tag).png")
println("Saved: results/enso_monthly_maps_$(mode_tag)_$(seed_tag).png")

# Persist full 2° spatial forecast field so the aggregator can build
# ensemble-mean maps and pattern-correlation diagnostics across seeds.
# test_f_3d/preds_f_3d are (nlon × nlat × (warmup + predict_len)); drop the
# warmup columns so saved arrays line up with scoring axis.
test_f_post  = test_f_3d[:,  :, warmup + 1 : end]
preds_f_post = preds_f_3d[:, :, warmup + 1 : end]

jldsave("results/enso_monthly_preds_$(mode_tag)_$(seed_tag).jld2";
        n34_true=n34_true, n34_pred=n34_pred,
        lead_months=collect(lead_months[1:length(lead_accs)]),
        lead_accs=lead_accs,
        full_acc=sc.acc, full_rmse=sc.rmse, full_rmse_skill=sc.rmse_skill,
        test_f=test_f_post, preds_f=preds_f_post,
        lons_f=collect(lons_f_crop), lats_f=collect(lats_f_crop),
        seed=seed, mode_tag=mode_tag)

println("\n" * "="^50)
println("SUMMARY  [monthly, mode=$(mode_tag), seed=$(seed)]")
println("="^50)
println("Train : $(train_len) mo ($(round(train_len/12; digits=1)) yr)")
println("Predict: $(predict_len) mo ($(round(predict_len/12; digits=1)) yr)")
println("ACC    : $(round(sc.acc; digits=3))")
println("RMSE   : $(round(sc.rmse; digits=4))")
println("="^50)
