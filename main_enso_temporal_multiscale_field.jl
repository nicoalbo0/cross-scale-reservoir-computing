# ENSO temporal cross-scale RC — STAGE B: per-pixel SST field.
#
# Architecture: bandpass-decompose each pixel's monthly SST anomaly time
# series into slow/mid/fast bands. Each band is a 3-D cube (lon × lat × t).
# Each band has its own single-layer 2° spatial pipeline (36 spatial blocks
# × N=500 reservoirs). The final SST forecast is the sum of band-wise
# predicted fields.
#
# Modes (bandpass family — DEPRECATED, kept for reproduction of negative results):
#   :no_xscale_field    — three independent spatial pipelines on bandpass-decomposed
#                         fields, sum band predictions.
#   :full_cascade_field — bandpass slow → mid → fast cross-scale at the same spatial
#                         blocks. Note: with overlap_mode=:exclude (default) and same
#                         coarse/fine grids, layer_dim collapses to 0, so cross-scale
#                         wiring is effectively absent. This matches the empirical
#                         finding that the result tied :no_xscale_field.
#
# Modes (multi-τ family — partition by memory timescale, NOT by frequency):
#   :multi_tau_2_field  — slow τ=30 → fast τ=2 cascade at SAME spatial blocks. Both
#                         reservoirs see the FULL broadband SST field. Cross-scale
#                         wiring uses overlap_mode=:include so layer_dim>0. Final SST
#                         forecast = fast reservoir's autonomous prediction.
#                         The 1D scalar analog (multi_tau_2 in main_enso_temporal_
#                         multiscale.jl) achieved 12-mo ACC=0.881 ± 0.002.
#   :multi_tau_3_field  — slow τ=30 → mid τ=8 → fast τ=2 cascade. Two run_multi_layer
#                         calls chained as in main_enso_monthly.jl:178–214 (1L→3L
#                         pattern). 1D scalar analog achieved 12-mo ACC=0.876,
#                         18-mo=0.74, 24-mo=0.56 — best long-lead skill seen.
#
# Comparison baselines:
#   E5 1L τ=30 (single spatial pipeline on full N3.4): 12-mo ACC=0.891 but
#     std_ratio=0.18 (amplitude-collapsed) and pc=0.11 (poor spatial).
#   E1/E12 best 3L: ACC=0.847, RMSE=0.406, pc=0.45, std_ratio=1.15.
#   B no_xscale_field: 12-mo ACC=0.890 ± 0.001, pc 3-mo=0.56 (best spatial yet).
#
# Usage:
#   ENSO_TM_MODE=no_xscale_field ENSO_SEED=42 julia --threads 4 --project=. main_enso_temporal_multiscale_field.jl

ENV["GKSwstype"] = "100"

using Pkg, Revise
Pkg.activate(".")

using CrossScaleRC
using JLD2, Dates, LinearAlgebra, Random, Statistics
using Plots, Measures

BLAS.set_num_threads(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

seed     = parse(Int, get(ENV, "ENSO_SEED", "42"))
mode     = Symbol(get(ENV, "ENSO_TM_MODE", "multi_tau_2_field"))
@assert mode in (:no_xscale_field, :full_cascade_field,
                 :multi_tau_2_field, :multi_tau_3_field) (
    "Unknown ENSO_TM_MODE=$(mode); expected one of " *
    "no_xscale_field, full_cascade_field, multi_tau_2_field, multi_tau_3_field")

is_multi_tau = mode in (:multi_tau_2_field, :multi_tau_3_field)
needs_bands  = mode in (:no_xscale_field, :full_cascade_field)

mode_tag = String(mode)
seed_tag = "seed$(seed)"
outdir   = get(ENV, "ENSO_OUTDIR",
               joinpath("results", "temporal_multiscale", mode_tag))

washout     = 12
train_len   = 288
predict_len = 96
warmup      = 12
dt          = 1.0

lon_range  = (126.0, 288.0)
lat_range  = (-36.0, 36.0)
start_date = Date(1982, 1, 1)

# Per-band reservoir hyperparameters
τ_slow     = parse(Float64, get(ENV, "ENSO_TM_TAU_SLOW",   "30.0"))
τ_mid      = parse(Float64, get(ENV, "ENSO_TM_TAU_MID",     "8.0"))
τ_fast     = parse(Float64, get(ENV, "ENSO_TM_TAU_FAST",    "2.0"))
ridge_slow = parse(Float64, get(ENV, "ENSO_TM_RIDGE_SLOW", "1e-3"))
ridge_mid  = parse(Float64, get(ENV, "ENSO_TM_RIDGE_MID",  "1e-2"))
ridge_fast = parse(Float64, get(ENV, "ENSO_TM_RIDGE_FAST", "1.0"))
N_slow     = parse(Int,     get(ENV, "ENSO_TM_NRES_SLOW", "500"))
N_mid      = parse(Int,     get(ENV, "ENSO_TM_NRES_MID",  "500"))
N_fast     = parse(Int,     get(ENV, "ENSO_TM_NRES_FAST", "500"))
ρ_slow     = parse(Float64, get(ENV, "ENSO_TM_RHO_SLOW",  "0.85"))
ρ_mid      = parse(Float64, get(ENV, "ENSO_TM_RHO_MID",   "0.75"))
ρ_fast     = parse(Float64, get(ENV, "ENSO_TM_RHO_FAST",  "0.55"))
glayer_mid_exp  = parse(Float64, get(ENV, "ENSO_TM_GLAYER_MID_EXP",  "-1.0"))
glayer_fast_exp = parse(Float64, get(ENV, "ENSO_TM_GLAYER_FAST_EXP", "-1.0"))

cutoffs_str = get(ENV, "ENSO_TM_CUTOFFS", "1/24,1/3")
cutoffs     = let parts = split(cutoffs_str, ',')
    Tuple(Float64(eval(Meta.parse(strip(s)))) for s in parts)
end
@assert length(cutoffs) == 2 "Expected two cutoffs"

# Spatial grid (single 2° resolution)
mixing = 2
grid_2d = (9, 4)

Random.seed!(seed)
mkpath(outdir)

println("="^60)
println("ENSO temporal multiscale FIELD — mode=$(mode_tag)  seed=$(seed)")
println("="^60)
println("  cutoffs   = $(cutoffs)")
println("  outdir    = $(outdir)")
println("  τ slow/mid/fast = $(τ_slow) / $(τ_mid) / $(τ_fast)")
println("  ρ slow/mid/fast = $(ρ_slow) / $(ρ_mid) / $(ρ_fast)")

# ---------------------------------------------------------------------------
# 1. Load 2° SST monthly cube
# ---------------------------------------------------------------------------

println("\nLoading 2° SST anomalies (daily) ...")
data_vec_daily, _ = load_data(
    [2.0]; show_data = false, refinement = 1,
    lon_range = lon_range, lat_range = lat_range,
    anomalies = true, train_indices = 1:(train_len * 30),
)

function daily_to_monthly(daily::AbstractArray{<:Real, 3}, start_date::Date)
    nlon, nlat, nt_daily = size(daily)
    dates = [start_date + Day(t - 1) for t in 1:nt_daily]
    ym    = [(year(d), month(d)) for d in dates]
    keys_ = unique(ym)
    M = Array{Float64, 3}(undef, nlon, nlat, length(keys_))
    for (k, target) in enumerate(keys_)
        mask = [x == target for x in ym]
        M[:, :, k] .= dropdims(mean(daily[:, :, mask]; dims = 3); dims = 3)
    end
    return M
end

println("Binning to monthly ...")
fine_2d = daily_to_monthly(data_vec_daily[1], start_date)
nlon_f, nlat_f, Ttot = size(fine_2d)
println("  monthly cube: $(nlon_f)×$(nlat_f) × $(Ttot)")
@assert train_len + predict_len ≤ Ttot

# Per-pixel z-score using training-window stats. NaN → 0 (land mask).
for i in 1:nlon_f, j in 1:nlat_f
    ts = @view fine_2d[i, j, :]
    if any(isnan, ts); ts .= 0.0; end
    train_view = @view ts[1:train_len]
    μ_p, σ_p = mean(train_view), std(train_view)
    if σ_p > 0
        ts .= (ts .- μ_p) ./ σ_p
    end
end

# ---------------------------------------------------------------------------
# 2. Bandpass-decompose each pixel's time series (only for bandpass modes)
# ---------------------------------------------------------------------------

bb_mat = reshape(fine_2d, nlon_f * nlat_f, Ttot)   # broadband per-pixel time series

slow_field = nothing; mid_field = nothing; fast_field = nothing
slow_mat   = nothing; mid_mat   = nothing; fast_mat   = nothing

if needs_bands
    println("\nBandpass-decomposing per pixel into 3 bands (cutoffs $(cutoffs)) ...")
    bands = bandpass_decompose_field(fine_2d, cutoffs; fs = 1.0, order = 4)
    slow_field = bands.slow
    mid_field  = bands.mid
    fast_field = bands.fast

    slow_mat = reshape(slow_field, nlon_f * nlat_f, Ttot)
    mid_mat  = reshape(mid_field,  nlon_f * nlat_f, Ttot)
    fast_mat = reshape(fast_field, nlon_f * nlat_f, Ttot)

    total_var = mean(var(bb_mat; dims = 2))
    println("  per-pixel variance fractions: slow=$(round(mean(var(slow_mat;dims=2))/total_var;digits=3))   mid=$(round(mean(var(mid_mat;dims=2))/total_var;digits=3))   fast=$(round(mean(var(fast_mat;dims=2))/total_var;digits=3))")
else
    println("\nMulti-τ mode: skipping bandpass decomposition (reservoirs see broadband signal).")
end

# ---------------------------------------------------------------------------
# 3. Spatial blocks (same 9×4 grid for all bands)
# ---------------------------------------------------------------------------

# Single-layer 2D blocks (used by :no_xscale_field and as the "coarse-only" block
# template for any mode that needs a non-cross-scale layer).
blocks_solo = make_blocks([fine_2d], [grid_2d], mixing)[1]
rec_dim, neigh_dim, _ = input_dimensions(blocks_solo)
println("  spatial blocks: $(length(blocks_solo)) blocks of ~$(rec_dim) pixels (with $(neigh_dim) neighbors)")

# Multi-layer block pair where the fine block points to the *same-spatial-location*
# coarse block (same grid, divisor = 1).
#   - For :full_cascade_field (legacy bandpass), the original code used the default
#     overlap_mode=:exclude, which with same grids drops layer_dim to 0 — i.e. NO
#     cross-scale wiring. Kept that legacy path for reproducibility of the negative
#     result in :full_cascade_field.
#   - For :multi_tau_*_field, we use overlap_mode=:include so layer_dim equals the
#     coarse block's rec_dim — i.e. the fine reservoir actually sees the coarse
#     reservoir's autonomous prediction at every pixel of its block.
blocks_pair_excl = make_blocks([fine_2d, fine_2d], [grid_2d, grid_2d], mixing)
blocks_band_coarse = blocks_pair_excl[1]
blocks_band_fine   = blocks_pair_excl[2]
_, _, layer_dim_excl = input_dimensions(blocks_band_fine)

blocks_pair_incl = make_blocks([fine_2d, fine_2d], [grid_2d, grid_2d], mixing; overlap_mode = :include)
blocks_xs_coarse = blocks_pair_incl[1]
blocks_xs_fine   = blocks_pair_incl[2]
_, _, layer_dim_incl = input_dimensions(blocks_xs_fine)
println("  cross-scale layer dim — exclude=$(layer_dim_excl)  include=$(layer_dim_incl)")

# Placeholder for run_single_layer's `data_layer` arg (matches broadband shape;
# bandpass band shapes are identical).
zero_layer = zeros(Float64, size(bb_mat))

# ---------------------------------------------------------------------------
# 4. Mode pipelines
# ---------------------------------------------------------------------------

local pred_slow_mat::Matrix{Float64}
local pred_mid_mat::Matrix{Float64}
local pred_fast_mat::Matrix{Float64}

if mode == :no_xscale_field
    println("\n--- :no_xscale_field — three independent spatial pipelines ---")

    g_in_solo(rec, neigh) = (10^(-0.5)/√rec, 10^(-0.5)/√max(neigh, 1))

    function _train_band_field(band_mat, τ, ridge_p, N, ρ)
        gr, gn = g_in_solo(rec_dim, neigh_dim)
        params = (N, ρ, 10, gr, gn, 0.0, τ, dt)
        preds, _, _, _, _ = run_single_layer(
            params, band_mat, zero_layer, train_len, predict_len, blocks_solo;
            washout = washout, warmup = warmup,
            ridge_parameter = ridge_p,
            show_progress = true, input_mode = :random,
            regression_mode = :quadratic,
        )
        return preds
    end

    println("  training SLOW band ...")
    pred_slow_mat = _train_band_field(slow_mat, τ_slow, ridge_slow, N_slow, ρ_slow)
    println("  training MID  band ...")
    pred_mid_mat  = _train_band_field(mid_mat,  τ_mid,  ridge_mid,  N_mid,  ρ_mid)
    println("  training FAST band ...")
    pred_fast_mat = _train_band_field(fast_mat, τ_fast, ridge_fast, N_fast, ρ_fast)

elseif mode == :full_cascade_field
    println("\n--- :full_cascade_field — slow → mid → fast cross-scale at SAME spatial blocks ---")

    rec_c, neigh_c, _       = input_dimensions(blocks_band_coarse)
    rec_f, neigh_f, layer_f = input_dimensions(blocks_band_fine)

    g_rec_A   = [10^(-0.5)/√rec_c,   10^(-0.5)/√rec_f]
    g_neigh_A = [10^(-0.5)/√max(neigh_c, 1), 10^(-0.5)/√max(neigh_f, 1)]
    g_layer_A = [0.0,                 10^(glayer_mid_exp)/√layer_f]
    params_A = ([N_slow, N_mid], [ρ_slow, ρ_mid], [10, 10],
                g_rec_A, g_neigh_A, g_layer_A,
                [τ_slow, τ_mid], [dt, dt])

    println("  Call A: slow → mid ...")
    preds_mid_A, preds_slow_A, train_pred_mid_A, train_pred_slow_A,
        _, _, _, _, _ = run_multi_layer(
            params_A, mid_mat, slow_mat, train_len, predict_len,
            [blocks_band_coarse, blocks_band_fine];
            washout = washout, warmup = warmup,
            ridge_parameter = [ridge_slow, ridge_mid],
            show_progress = true, input_mode = :random,
            regression_mode = [:quadratic, :quadratic])
    pred_slow_mat = preds_slow_A

    coarse_for_B = hcat(train_pred_mid_A, preds_mid_A[:, warmup+1:end])

    g_rec_B   = [10^(-0.5)/√rec_c,   10^(-0.5)/√rec_f]
    g_neigh_B = [10^(-0.5)/√max(neigh_c, 1), 10^(-0.5)/√max(neigh_f, 1)]
    g_layer_B = [0.0,                 10^(glayer_fast_exp)/√layer_f]
    params_B = ([N_mid, N_fast], [ρ_mid, ρ_fast], [10, 10],
                g_rec_B, g_neigh_B, g_layer_B,
                [τ_mid, τ_fast], [dt, dt])

    println("  Call B: mid → fast ...")
    preds_fast_B, preds_mid_B, _, _, _, _, _, _, _ = run_multi_layer(
        params_B, fast_mat, coarse_for_B, train_len, predict_len,
        [blocks_band_coarse, blocks_band_fine];
        washout = washout, warmup = warmup,
        ridge_parameter = [ridge_mid, ridge_fast],
        show_progress = true, input_mode = :random,
        regression_mode = [:quadratic, :quadratic])
    pred_mid_mat  = preds_mid_B
    pred_fast_mat = preds_fast_B

elseif mode == :multi_tau_2_field
    # Spatial multi-τ cascade ("double multi-scale" — 2-layer flavour).
    # Each spatial block hosts two reservoirs at different τ. Both see the FULL
    # broadband SST field (no bandpass decomposition). Cross-scale wiring lets
    # the fast reservoir consume the slow reservoir's autonomous prediction.
    # Final SST field forecast = fast reservoir's autonomous output.
    println("\n--- :multi_tau_2_field — slow τ=$(τ_slow) → fast τ=$(τ_fast) per spatial block (broadband) ---")

    rec_c, neigh_c, _       = input_dimensions(blocks_xs_coarse)
    rec_f, neigh_f, layer_f = input_dimensions(blocks_xs_fine)
    @assert layer_f > 0 "multi_tau requires overlap_mode=:include for nonzero layer_dim"

    g_rec_2   = [10^(-0.5)/√rec_c,         10^(-0.5)/√rec_f]
    g_neigh_2 = [10^(-0.5)/√max(neigh_c,1), 10^(-0.5)/√max(neigh_f,1)]
    g_layer_2 = [0.0,                       10^(glayer_fast_exp)/√layer_f]
    params_2 = ([N_slow, N_fast], [ρ_slow, ρ_fast], [10, 10],
                g_rec_2, g_neigh_2, g_layer_2,
                [τ_slow, τ_fast], [dt, dt])

    println("  block dims: rec=$(rec_f)  neigh=$(neigh_f)  layer=$(layer_f)")
    println("  hyperparams: ρ slow/fast=$(ρ_slow)/$(ρ_fast)  ridge slow/fast=$(ridge_slow)/$(ridge_fast)")
    println("  cross-scale gain (fast layer): 10^$(glayer_fast_exp)/√$(layer_f) = $(round(g_layer_2[2];digits=4))")

    preds_fast_mt, preds_slow_mt, _, _, _, _, _, _, _ = run_multi_layer(
        params_2, bb_mat, bb_mat, train_len, predict_len,
        [blocks_xs_coarse, blocks_xs_fine];
        washout = washout, warmup = warmup,
        ridge_parameter = [ridge_slow, ridge_fast],
        show_progress = true, input_mode = :random,
        regression_mode = [:quadratic, :quadratic])

    # In multi-τ mode the slow reservoir also predicts the BROADBAND signal —
    # both layers target the same data, only differing in τ. So we keep
    # pred_slow_mat for the diagnostic "what does the slow τ layer alone predict?"
    # but the FINAL field forecast is taken from the fast layer alone.
    pred_slow_mat = preds_slow_mt
    pred_mid_mat  = zeros(Float64, size(preds_fast_mt))     # unused
    pred_fast_mat = preds_fast_mt

elseif mode == :multi_tau_3_field
    # Spatial multi-τ cascade — 3-layer flavour. Two run_multi_layer calls chained
    # exactly like main_enso_monthly.jl:178–214. All three reservoirs see the FULL
    # broadband SST field (no bandpass decomposition).
    println("\n--- :multi_tau_3_field — slow τ=$(τ_slow) → mid τ=$(τ_mid) → fast τ=$(τ_fast) per spatial block (broadband) ---")

    rec_c, neigh_c, _       = input_dimensions(blocks_xs_coarse)
    rec_f, neigh_f, layer_f = input_dimensions(blocks_xs_fine)
    @assert layer_f > 0 "multi_tau requires overlap_mode=:include for nonzero layer_dim"

    # Call A: slow → mid (both see broadband)
    g_rec_A   = [10^(-0.5)/√rec_c,         10^(-0.5)/√rec_f]
    g_neigh_A = [10^(-0.5)/√max(neigh_c,1), 10^(-0.5)/√max(neigh_f,1)]
    g_layer_A = [0.0,                       10^(glayer_mid_exp)/√layer_f]
    params_A_3 = ([N_slow, N_mid], [ρ_slow, ρ_mid], [10, 10],
                  g_rec_A, g_neigh_A, g_layer_A,
                  [τ_slow, τ_mid], [dt, dt])

    println("  block dims: rec=$(rec_f)  neigh=$(neigh_f)  layer=$(layer_f)")
    println("  Call A: slow → mid (g_layer_mid=10^$(glayer_mid_exp)/√$(layer_f)) ...")
    preds_mid_A, preds_slow_A, train_pred_mid_A, train_pred_slow_A,
        _, _, _, _, _ = run_multi_layer(
            params_A_3, bb_mat, bb_mat, train_len, predict_len,
            [blocks_xs_coarse, blocks_xs_fine];
            washout = washout, warmup = warmup,
            ridge_parameter = [ridge_slow, ridge_mid],
            show_progress = true, input_mode = :random,
            regression_mode = [:quadratic, :quadratic])

    # Call B: mid → fast. Coarse for B = chain (train_pred_mid + autonomous mid pred)
    coarse_for_B = hcat(train_pred_mid_A, preds_mid_A[:, warmup+1:end])

    g_rec_B   = [10^(-0.5)/√rec_c,         10^(-0.5)/√rec_f]
    g_neigh_B = [10^(-0.5)/√max(neigh_c,1), 10^(-0.5)/√max(neigh_f,1)]
    g_layer_B = [0.0,                       10^(glayer_fast_exp)/√layer_f]
    params_B_3 = ([N_mid, N_fast], [ρ_mid, ρ_fast], [10, 10],
                  g_rec_B, g_neigh_B, g_layer_B,
                  [τ_mid, τ_fast], [dt, dt])

    println("  Call B: mid → fast (g_layer_fast=10^$(glayer_fast_exp)/√$(layer_f)) ...")
    preds_fast_B, preds_mid_B, _, _, _, _, _, _, _ = run_multi_layer(
        params_B_3, bb_mat, coarse_for_B, train_len, predict_len,
        [blocks_xs_coarse, blocks_xs_fine];
        washout = washout, warmup = warmup,
        ridge_parameter = [ridge_mid, ridge_fast],
        show_progress = true, input_mode = :random,
        regression_mode = [:quadratic, :quadratic])

    pred_slow_mat = preds_slow_A
    pred_mid_mat  = preds_mid_A
    pred_fast_mat = preds_fast_B
end

# ---------------------------------------------------------------------------
# 5. Reconstruct SST forecast field, compute Niño 3.4 + spatial metrics
# ---------------------------------------------------------------------------

# All preds_*_mat are shape (n_pixels, warmup + predict_len). Drop warmup.
test_window_cols = (warmup+1):(warmup+predict_len)
if is_multi_tau
    # Multi-τ: only the fast (final) layer's autonomous prediction is the field
    # forecast. Both layers target the same broadband signal — summing would
    # double-count.
    field_pred_mat = pred_fast_mat[:, test_window_cols]
else
    # Bandpass family: sum band predictions to reconstruct the full field.
    field_pred_mat = pred_slow_mat[:, test_window_cols] .+
                     pred_mid_mat[:,  test_window_cols] .+
                     pred_fast_mat[:, test_window_cols]
end
field_pred_3d = reshape(field_pred_mat, nlon_f, nlat_f, predict_len)

# Truth aligned to forecast window
t0 = train_len - warmup
truth_window_abs = (t0 + warmup + 1) : (t0 + warmup + predict_len)
field_true_3d = fine_2d[:, :, truth_window_abs]

# Per-layer / per-band truth/pred for diagnostics
slow_pred_3d = reshape(pred_slow_mat[:, test_window_cols], nlon_f, nlat_f, predict_len)
mid_pred_3d  = reshape(pred_mid_mat[:,  test_window_cols], nlon_f, nlat_f, predict_len)
fast_pred_3d = reshape(pred_fast_mat[:, test_window_cols], nlon_f, nlat_f, predict_len)
if needs_bands
    slow_true_3d = slow_field[:, :, truth_window_abs]
    mid_true_3d  = mid_field[:,  :, truth_window_abs]
    fast_true_3d = fast_field[:, :, truth_window_abs]
else
    # In multi-τ mode every layer targets the broadband field — diagnostics use
    # the broadband truth as the reference for each layer's autonomous skill.
    slow_true_3d = field_true_3d
    mid_true_3d  = field_true_3d
    fast_true_3d = field_true_3d
end

# Cropped grid coords for Niño 3.4
function cropped_lons_lats(res, lon_range, lat_range)
    lon_start = -180.0 + res / 2
    lat_start = -90.0  + res / 2
    lons_all  = lon_start .+ res .* (0 : round(Int, 360 / res) - 1)
    lats_all  = lat_start .+ res .* (0 : round(Int, 180 / res) - 1)
    lons360   = mod.(lons_all, 360.0)
    lon_mask  = lon_range[1] .<= lons360 .<= lon_range[2]
    lat_mask  = lat_range[1] .<= lats_all  .<= lat_range[2]
    return lons_all[lon_mask], lats_all[lat_mask]
end
lons_f, lats_f = cropped_lons_lats(2.0, lon_range, lat_range)

n34_true = nino34_index(field_true_3d, lons_f, lats_f)
n34_pred = nino34_index(field_pred_3d, lons_f, lats_f)

sc = skill_score(n34_true, n34_pred)
std_ratio_full = std(n34_pred) / std(n34_true)

println("\n=== Niño 3.4 forecast skill ===")
println("  full-window  ACC=$(round(sc.acc;digits=3))   RMSE=$(round(sc.rmse;digits=4))")
println("  RMSE skill   $(round(sc.rmse_skill;digits=3))")
println("  std_ratio    $(round(std_ratio_full;digits=3))")

println("\n=== Lead-time cumulative ACC / RMSE / std_ratio ===")
lead_months = [3, 6, 9, 12, 18, 24, 36, 48]
lead_accs       = Float64[]
lead_rmses      = Float64[]
lead_std_ratios = Float64[]
leads_used      = Int[]
for L in lead_months
    L > length(n34_true) && continue
    s_L = skill_score(n34_true[1:L], n34_pred[1:L])
    sr  = std(n34_pred[1:L]) / std(n34_true[1:L])
    push!(lead_accs, s_L.acc); push!(lead_rmses, s_L.rmse)
    push!(lead_std_ratios, sr); push!(leads_used, L)
    println("  $(lpad(L,2)) mo : ACC=$(round(s_L.acc;digits=3))   RMSE=$(round(s_L.rmse;digits=4))   std_ratio=$(round(sr;digits=3))")
end

# Spatial pattern correlation
function spatial_pc(A::AbstractMatrix, B::AbstractMatrix)
    a = vec(A); b = vec(B)
    am = a .- mean(a); bm = b .- mean(b)
    return dot(am, bm) / (norm(am) * norm(bm) + eps())
end
pcorr_per_t = [spatial_pc(field_true_3d[:, :, t], field_pred_3d[:, :, t])
               for t in 1:predict_len]

println("\n=== Spatial pattern correlation (truth vs prediction, per month) ===")
for L in [3, 6, 9, 12, 18, 24]
    L > length(pcorr_per_t) && continue
    println("  $(lpad(L,2)) mo : pc=$(round(pcorr_per_t[L];digits=3))")
end

# Per-band own-skill: how well does each band reservoir predict its own band?
function field_acc(true3d, pred3d)
    a = vec(true3d); b = vec(pred3d)
    am = a .- mean(a); bm = b .- mean(b)
    return dot(am, bm) / (std(a) * std(b) * length(a) + eps())
end

if is_multi_tau
    label_lhs = mode == :multi_tau_2_field ?
                ("slow τ=$(τ_slow) layer", nothing, "fast τ=$(τ_fast) layer (FINAL)") :
                ("slow τ=$(τ_slow) layer (Call A)", "mid τ=$(τ_mid) layer (Call A)",
                 "fast τ=$(τ_fast) layer (Call B, FINAL)")
    println("\n=== Per-layer autonomous skill on broadband field (full test window) ===")
    println("  $(rpad(label_lhs[1], 36)) ACC=$(round(field_acc(slow_true_3d, slow_pred_3d); digits=3))   std_ratio=$(round(std(vec(slow_pred_3d))/std(vec(slow_true_3d));digits=3))")
    if mode == :multi_tau_3_field
        println("  $(rpad(label_lhs[2], 36)) ACC=$(round(field_acc(mid_true_3d,  mid_pred_3d);  digits=3))   std_ratio=$(round(std(vec(mid_pred_3d)) /std(vec(mid_true_3d)); digits=3))")
    end
    println("  $(rpad(label_lhs[3], 36)) ACC=$(round(field_acc(fast_true_3d, fast_pred_3d); digits=3))   std_ratio=$(round(std(vec(fast_pred_3d))/std(vec(fast_true_3d));digits=3))")
else
    println("\n=== Per-band own-skill (full test window, vectorised across pixels and time) ===")
    println("  slow band: ACC=$(round(field_acc(slow_true_3d, slow_pred_3d); digits=3))   std_ratio=$(round(std(vec(slow_pred_3d))/std(vec(slow_true_3d));digits=3))")
    println("  mid  band: ACC=$(round(field_acc(mid_true_3d,  mid_pred_3d);  digits=3))   std_ratio=$(round(std(vec(mid_pred_3d)) /std(vec(mid_true_3d)); digits=3))")
    println("  fast band: ACC=$(round(field_acc(fast_true_3d, fast_pred_3d); digits=3))   std_ratio=$(round(std(vec(fast_pred_3d))/std(vec(fast_true_3d));digits=3))")
end

# ---------------------------------------------------------------------------
# 6. Plots
# ---------------------------------------------------------------------------

p1 = plot(n34_true; lw=2.5, color=:black, label="Observed",
          xlabel="month since forecast start", ylabel="Niño 3.4",
          title="Temporal MS field  $(mode_tag)  seed=$(seed)  ACC=$(round(sc.acc;digits=2))",
          legend=:topright)
plot!(p1, n34_pred; lw=2.0, color=:red, ls=:dash, label="Forecast")

p2 = plot(leads_used, lead_accs; marker=:circle, lw=2, color=:blue, label="N3.4 ACC",
          xlabel="Lead (months)", ylabel="cumulative ACC",
          title="Lead-time skill", ylim=(-1, 1), legend=:bottomleft)
hline!(p2, [0.5]; ls=:dash, color=:gray, label="0.5"); hline!(p2, [0.0]; ls=:dot, color=:black, label=false)

p3 = plot(1:length(pcorr_per_t), pcorr_per_t; lw=2, color=:purple, label="pc",
          xlabel="month", ylabel="spatial pattern correlation",
          title="Spatial pc per month", ylim=(-0.5, 1.0))
hline!(p3, [0.5]; ls=:dash, color=:gray, label="0.5")

p_summary = plot(p1, p2, p3; layout=(3, 1), size=(1200, 1000), left_margin=4mm)
out_png = joinpath(outdir, "enso_temporal_field_$(mode_tag)_$(seed_tag).png")
savefig(p_summary, out_png)
println("\nSaved: $(out_png)")

# Maps at selected leads (truth / forecast / err)
sc_map = 1.5
lons_plot = mod.(lons_f, 360.0); perm = sortperm(lons_plot); lons_plot = lons_plot[perm]
function map_panel(field2d, ttl; clim_=(-sc_map, sc_map), cmap=:RdBu)
    heatmap(lons_plot, lats_f, permutedims(field2d[perm, :]);
            clim=clim_, c=cmap, xlabel="lon (°E)", ylabel="lat (°N)",
            aspect_ratio=:equal,
            xlims=(minimum(lons_plot), maximum(lons_plot)),
            ylims=(minimum(lats_f), maximum(lats_f)),
            title=ttl, colorbar=true)
end

map_leads = filter(L -> L ≤ predict_len, [6, 12, 18, 24])
panels = Plots.Plot[]
for L in map_leads
    push!(panels, map_panel(field_true_3d[:, :, L], "Observed  $(L) mo"))
    pc_L = round(spatial_pc(field_true_3d[:, :, L], field_pred_3d[:, :, L]); digits=2)
    push!(panels, map_panel(field_pred_3d[:, :, L], "Forecast  $(L) mo  pc=$(pc_L)"))
    push!(panels, map_panel(field_true_3d[:, :, L] .- field_pred_3d[:, :, L], "Obs − Forecast  $(L) mo"))
end
p_maps = plot(panels...; layout=(length(map_leads), 3),
              size=(1800, 320 * length(map_leads)),
              plot_title="2° SST anomaly — $(mode_tag) seed=$(seed)",
              left_margin=4mm, bottom_margin=3mm)
savefig(p_maps, joinpath(outdir, "enso_temporal_field_maps_$(mode_tag)_$(seed_tag).png"))
println("Saved: $(joinpath(outdir, "enso_temporal_field_maps_$(mode_tag)_$(seed_tag).png"))")

# ---------------------------------------------------------------------------
# 7. Persist (Float32 + compress; reference once + per-seed slim file)
# ---------------------------------------------------------------------------

ref_path = joinpath(outdir, "enso_temporal_field_reference_$(mode_tag).jld2")
if !isfile(ref_path)
    if needs_bands
        jldsave(ref_path; compress = true,
                n34_true   = Float32.(n34_true),
                field_true = Float32.(field_true_3d),
                slow_true  = Float32.(slow_true_3d),
                mid_true   = Float32.(mid_true_3d),
                fast_true  = Float32.(fast_true_3d),
                lons = collect(lons_f), lats = collect(lats_f),
                cutoffs = collect(cutoffs),
                train_len = train_len, predict_len = predict_len, warmup = warmup,
                mode_tag = mode_tag)
    else
        jldsave(ref_path; compress = true,
                n34_true   = Float32.(n34_true),
                field_true = Float32.(field_true_3d),
                lons = collect(lons_f), lats = collect(lats_f),
                taus = mode == :multi_tau_2_field ? [τ_slow, τ_fast] : [τ_slow, τ_mid, τ_fast],
                train_len = train_len, predict_len = predict_len, warmup = warmup,
                mode_tag = mode_tag)
    end
    println("Saved reference: $(ref_path)")
end

save_kwargs = Dict{Symbol, Any}(
    :compress       => true,
    :n34_pred       => Float32.(n34_pred),
    :field_pred     => Float32.(field_pred_3d),
    :slow_pred      => Float32.(slow_pred_3d),
    :fast_pred      => Float32.(fast_pred_3d),
    :pcorr_per_t    => Float64.(pcorr_per_t),
    :lead_months    => collect(leads_used),
    :lead_accs      => Float64.(lead_accs),
    :lead_rmses     => Float64.(lead_rmses),
    :lead_std_ratios => Float64.(lead_std_ratios),
    :full_acc       => sc.acc, :full_rmse => sc.rmse, :full_rmse_skill => sc.rmse_skill,
    :std_ratio_full => std_ratio_full,
    :seed           => seed, :mode_tag => mode_tag,
)
# mid_pred only meaningful for 3-band / 3-τ modes
if needs_bands || mode == :multi_tau_3_field
    save_kwargs[:mid_pred] = Float32.(mid_pred_3d)
end
jldsave(joinpath(outdir, "enso_temporal_field_preds_$(mode_tag)_$(seed_tag).jld2"); save_kwargs...)

println("\n" * "="^60)
println("SUMMARY  mode=$(mode_tag)  seed=$(seed)")
println("="^60)
println("  N3.4 12-mo ACC=$(round(lead_accs[findfirst(==(12), leads_used)];digits=3))   RMSE=$(round(lead_rmses[findfirst(==(12), leads_used)];digits=4))   std_ratio=$(round(lead_std_ratios[findfirst(==(12), leads_used)];digits=3))")
println("  Spatial pc 12-mo: $(round(pcorr_per_t[12];digits=3))")
println("="^60)
