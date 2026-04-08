# ENSO forecasting via cross-scale reservoir computing
#
# Strategy: subtract the seasonal climatology from tropical Pacific SST to produce
# SST *anomalies*, then apply the multi-layer ESN framework to learn and predict
# the quasi-periodic ENSO signal (2-7 year cycle).
#
# Three-layer architecture:
#   Layer 1 (coarse, 18°): basin-scale SST patterns / teleconnections
#   Layer 2 (medium,  6°): Walker Circulation / thermocline tilt
#   Layer 3 (fine,    2°): sharp SST gradients, cold tongue / warm pool boundary
#
# The 3-layer pipeline is implemented by chaining two run_multi_layer calls:
#   Call A: coarse (18°) → medium (6°)
#   Call B: medium (6°, independently retrained as standalone coarse) → fine (2°)
#
# Usage (remote CPU machine):
#   julia --threads auto main_enso.jl
# Results are saved to results/ as PNG files.

using Pkg, Revise
Pkg.activate(".")
Pkg.instantiate()

using CrossScaleRC
using LinearAlgebra
using Statistics
using Plots, Measures, LaTeXStrings

BLAS.set_num_threads(1)   # prevent BLAS/Julia-thread contention

# ---------------------------------------------------------------------------
# 1. Experiment configuration
# ---------------------------------------------------------------------------

sampling_rate = 4          # upsample daily SST 4× → dt = 0.25 day/step
                           # 4 steps ≈ 1 day, 120 steps ≈ 1 month

washout     = 1_000        # transient steps discarded before ridge regression
train_len   = 30_000       # training window: 30 000 × 0.25 day = 7 500 days ≈ 20.5 yr
predict_len = 3_000        # closed-loop prediction: 3 000 × 0.25 day = 750 days ≈ 2 yr
warmup      = 1_000        # state warm-up from true data before closed-loop begins

# Tropical Pacific domain — boundaries aligned to 18° cell edges so that
# nlon and nlat divide evenly across all three resolutions (18°→6°→2°, ×3 at each step).
# This is required by add_cross_layer! in make_blocks.
#   18° domain:  9 lon ×  4 lat cells  (162° lon span, 72° lat span)
#    6° domain: 27 lon × 12 lat cells  (ratios 27/9=3, 12/4=3 ✓)
#    2° domain: 81 lon × 36 lat cells  (ratios 81/27=3, 36/12=3 ✓)
lon_range = (126.0, 288.0)   # 126°E → 72°W  (multiples of 18°)
lat_range = (-36.0, 36.0)    # 36°S  → 36°N  (multiples of 18°)

# ---------------------------------------------------------------------------
# 2. Load data  (3 resolutions, SST anomalies)
# ---------------------------------------------------------------------------

# Climatology is computed over the TRAINING window expressed in DAILY units
# (pre-upsampling). load_sst_data removes the seasonal mean before applying
# the cubic-spline time refinement, so mod1(t, 365) maps correctly.
train_days_for_clim = (washout + train_len) ÷ sampling_rate   # = 7_750 days
train_indices       = 1:train_days_for_clim

println("Loading SST anomalies (3 resolutions)...")
data_vec, dt = load_data(
    [18.0, 6.0, 2.0];
    show_data     = false,
    refinement    = sampling_rate,
    lon_range     = lon_range,
    lat_range     = lat_range,
    anomalies     = true,
    train_indices = train_indices
)

very_coarse = data_vec[1]   # (9  × 4  × Ttot) — 18° resolution
medium      = data_vec[2]   # (27 × 12 × Ttot) —  6° resolution
fine        = data_vec[3]   # (81 × 36 × Ttot) —  2° resolution

nlon_vc, nlat_vc, Ttot = size(very_coarse)
nlon_m,  nlat_m,  _    = size(medium)
nlon_f,  nlat_f,  _    = size(fine)

println("Domain sizes (lon × lat × T):")
println("  18° coarse : $(nlon_vc)×$(nlat_vc) = $(nlon_vc*nlat_vc) cells")
println("   6° medium : $(nlon_m)×$(nlat_m) = $(nlon_m*nlat_m) cells")
println("    2° fine  : $(nlon_f)×$(nlat_f) = $(nlon_f*nlat_f) cells")
println("  Total steps: $Ttot  (need ≥ $(train_len + predict_len))")

train_len + predict_len ≤ Ttot ||
    error("Not enough data: have $Ttot, need $(train_len + predict_len)")

# Flatten spatial dims → (ncells × T) for the reservoir
vc_mat = reshape(very_coarse, nlon_vc * nlat_vc, Ttot)
m_mat  = reshape(medium,      nlon_m  * nlat_m,  Ttot)
f_mat  = reshape(fine,        nlon_f  * nlat_f,  Ttot)

# ---------------------------------------------------------------------------
# 3. Spatial block construction
# ---------------------------------------------------------------------------

# Grid divisions (div_lon, div_lat) must divide (nlon, nlat) of each layer exactly.
# Call A: coarse (3,1)→3 blocks; medium (9,3)→27 blocks.  Cross-layer: 27÷9=3✓ 12÷4=3✓
# Call B: medium (9,3)→27 blocks; fine  (9,4)→36 blocks.  Cross-layer: 81÷27=3✓ 36÷12=3✓
grids_AB = [(3, 1), (9, 3)]
grids_BC = [(9, 3), (9, 4)]

mixing = 2

blocks_AB = make_blocks(data_vec[1:2], grids_AB, mixing)   # coarse → medium
blocks_BC = make_blocks(data_vec[2:3], grids_BC, mixing)   # medium → fine

blocks_vc   = blocks_AB[1]   # 18° coarse — coarse layer for Call A (rows_layer = [])
blocks_m_AB = blocks_AB[2]   # 6°  medium — fine   layer for Call A (rows_layer → 18°)
blocks_m_BC = blocks_BC[1]   # 6°  medium — coarse layer for Call B (rows_layer = [])
blocks_f    = blocks_BC[2]   # 2°  fine   — fine   layer for Call B (rows_layer → 6°)

# ---------------------------------------------------------------------------
# 4. Hyperparameters
# ---------------------------------------------------------------------------

rec_vc, neigh_vc, _       = input_dimensions(blocks_vc)
rec_m,  neigh_m,  layer_m = input_dimensions(blocks_m_AB)  # layer_m > 0 (← 18° cells)
rec_f,  neigh_f,  layer_f = input_dimensions(blocks_f)     # layer_f > 0 (← 6°  cells)

println("\nInput dimensions per block:")
println("  18° coarse : rec=$rec_vc, neigh=$neigh_vc, layer=0")
println("   6° medium : rec=$rec_m,  neigh=$neigh_m,  layer=$layer_m")
println("    2° fine  : rec=$rec_f,  neigh=$neigh_f,  layer=$layer_f")

# ── Call A: coarse (18°) → medium (6°) ──────────────────────────────────────
res_size_A = [800,   1_000]
res_rad_A  = [0.85,  0.75]
degree_A   = [10,    10]
g_rec_A    = [10^(-1.5)/√rec_vc,   10^(-0.5)/√rec_m]
g_neigh_A  = [10^(-1.5)/√neigh_vc, 10^(-0.5)/√neigh_m]
g_layer_A  = [0.0,                  10^(-1.0)/√layer_m]
ridge_A    = [1e-2,  1e-1]
τ_A        = [2.5,   2.5]
dt_A       = [dt,    dt]
params_A   = (res_size_A, res_rad_A, degree_A, g_rec_A, g_neigh_A, g_layer_A, τ_A, dt_A)

# ── Call B: medium (6°) → fine (2°) ─────────────────────────────────────────
res_size_B = [1_000, 1_000]
res_rad_B  = [0.75,  0.55]
degree_B   = [10,    10]
g_rec_B    = [10^(-0.5)/√rec_m,   10^(-0.5)/√rec_f]
g_neigh_B  = [10^(-0.5)/√neigh_m, 10^(-0.5)/√neigh_f]
g_layer_B  = [0.0,                 10^(-1.0)/√layer_f]
ridge_B    = [1e-1,  10^(1.0)]
τ_B        = [2.5,   2.5]
dt_B       = [dt,    dt]
params_B   = (res_size_B, res_rad_B, degree_B, g_rec_B, g_neigh_B, g_layer_B, τ_B, dt_B)

# ---------------------------------------------------------------------------
# 5. Run 3-layer pipeline (chained)
# ---------------------------------------------------------------------------

println("\n--- Call A: coarse (18°) → medium (6°) ---")
preds_m, preds_vc,
    train_pred_m, train_pred_vc,
    train_data_vc, train_data_m,
    _, X_vc, X_m = run_multi_layer(
        params_A,
        m_mat, vc_mat,
        train_len, predict_len,
        [blocks_vc, blocks_m_AB];
        washout         = washout,
        warmup          = warmup,
        ridge_parameter = ridge_A,
        show_progress   = true,
        input_mode      = :random,
        regression_mode = [:linear, :linear]
    )

println("\n--- Call B: medium (6°) → fine (2°) ---")
preds_f, _,
    train_pred_f, _,
    _, train_data_f,
    _, _, X_f = run_multi_layer(
        params_B,
        f_mat, m_mat,
        train_len, predict_len,
        [blocks_m_BC, blocks_f];
        washout         = washout,
        warmup          = warmup,
        ridge_parameter = ridge_B,
        show_progress   = true,
        input_mode      = :random,
        regression_mode = [:linear, :linear]
    )

# ---------------------------------------------------------------------------
# 6. Align test windows
#    test_* spans the same time range as preds_* (warmup + predict_len steps)
# ---------------------------------------------------------------------------

t0 = train_len - warmup   # first time index of the test window (1-indexed into *_mat)

test_f  = f_mat[:,  t0 + 1 : t0 + size(preds_f,  2)]
test_m  = m_mat[:,  t0 + 1 : t0 + size(preds_m,  2)]
test_vc = vc_mat[:, t0 + 1 : t0 + size(preds_vc, 2)]

# ---------------------------------------------------------------------------
# 7. Niño 3.4 evaluation — single-window summary
# ---------------------------------------------------------------------------

# Re-derive cropped lon/lat vectors for the fine (2°) grid, consistent with the
# same mask applied inside _domain_indices in loading.jl.
function cropped_lons_lats(nlon::Int, nlat::Int, res::Real, lon_range, lat_range)
    lon_start = -180.0 + res / 2
    lat_start = -90.0  + res / 2
    lons_all  = lon_start .+ res .* (0 : round(Int, 360 / res) - 1)
    lats_all  = lat_start .+ res .* (0 : round(Int, 180 / res) - 1)
    lons360   = mod.(lons_all, 360.0)
    lo_l, hi_l = lon_range
    lon_mask  = lo_l .<= lons360 .<= hi_l
    lat_mask  = lat_range[1] .<= lats_all .<= lat_range[2]
    return lons_all[lon_mask], lats_all[lat_mask]
end

lons_f_crop, lats_f_crop = cropped_lons_lats(nlon_f, nlat_f, 2.0, lon_range, lat_range)

# Reshape to 3D (lon × lat × time) for nino34_index
test_f_3d  = reshape(test_f,  nlon_f, nlat_f, size(test_f,  2))
preds_f_3d = reshape(preds_f, nlon_f, nlat_f, size(preds_f, 2))

# Skip the warmup phase; keep only the autonomous prediction columns
n34_true = nino34_index(test_f_3d[:,  :, warmup + 1 : end], lons_f_crop, lats_f_crop)
n34_pred = nino34_index(preds_f_3d[:, :, warmup + 1 : end], lons_f_crop, lats_f_crop)

sc = skill_score(n34_true, n34_pred)
println("\n=== ENSO Forecast Skill (full prediction window) ===")
println("  Niño 3.4 ACC        : $(round(sc.acc;        digits=3))")
println("  Niño 3.4 RMSE       : $(round(sc.rmse;       digits=4))  [normalised units]")
println("  RMSE skill vs pers. : $(round(sc.rmse_skill; digits=3))")

# Spatial RMSE error curves (post-warmup)
_pw = warmup + 1   # post-warmup start index
error_curve_f  = [rmse_upto(test_f[:,  _pw:end], preds_f[:,  _pw:end]; T=t)
                  for t in 1 : size(preds_f[:,  _pw:end], 2)]
error_curve_m  = [rmse_upto(test_m[:,  _pw:end], preds_m[:,  _pw:end]; T=t)
                  for t in 1 : size(preds_m[:,  _pw:end], 2)]
error_curve_vc = [rmse_upto(test_vc[:, _pw:end], preds_vc[:, _pw:end]; T=t)
                  for t in 1 : size(preds_vc[:, _pw:end], 2)]

# ---------------------------------------------------------------------------
# 8. Multi-window skill evaluation
#    Divide the post-warmup prediction into N_windows non-overlapping segments
#    and report per-window Niño 3.4 skill to diagnose temporal consistency
#    (i.e. whether skill degrades monotonically or varies across the period).
# ---------------------------------------------------------------------------

N_windows   = 3
T_post      = length(n34_true)          # number of post-warmup steps
window_size = T_post ÷ N_windows        # steps per window

days_per_step  = dt                     # dt = 1/sampling_rate days/step = 0.25 days
days_per_window = round(Int, window_size * days_per_step)

println("\n=== Multi-window Niño 3.4 Skill  ($N_windows × $window_size steps ≈ $(days_per_window) days each) ===")

window_accs  = Float64[]
window_rmses = Float64[]

for w in 1:N_windows
    i0 = (w - 1) * window_size + 1
    i1 = w * window_size
    sc_w = skill_score(n34_true[i0:i1], n34_pred[i0:i1])
    push!(window_accs,  sc_w.acc)
    push!(window_rmses, sc_w.rmse)
    day0 = round(Int, (i0 - 1) * days_per_step)
    day1 = round(Int,  i1      * days_per_step)
    println("  Window $w  (days $day0 – $day1):  ACC=$(round(sc_w.acc; digits=3))" *
            "  RMSE=$(round(sc_w.rmse; digits=4))" *
            "  RMSE-skill=$(round(sc_w.rmse_skill; digits=3))")
end

# ---------------------------------------------------------------------------
# 9. Lead-time Niño 3.4 skill vs forecast horizon
#
#    For each horizon H, compute cumulative ACC and RMSE using the first H
#    post-warmup steps of the prediction.  This gives a lower bound on skill
#    as a function of lead time from a SINGLE forecast trajectory.
#
#    A proper multi-start lead-time curve (computing ACC across an ensemble of
#    re-initialised test runs) would require retaining the trained block_models
#    and calling test_parallel_reservoir multiple times — this is left as a
#    future extension.
# ---------------------------------------------------------------------------

# Lead-time horizons expressed in forecast steps:
#   1 step = dt = 0.25 days  →  1 month ≈ 30 / 0.25 = 120 steps
month_steps   = round(Int, 30.0 / dt)          # 120 steps ≈ 30 days
raw_horizons  = [1, 2, 3, 4, 6, 8] .* month_steps
lead_horizons = filter(H -> H ≤ T_post, raw_horizons)

println("\n=== Cumulative Lead-time Niño 3.4 Skill  (1 step = $(dt) day, month ≈ $(month_steps) steps) ===")

lead_accs  = Float64[]
lead_rmses = Float64[]

for H in lead_horizons
    sc_h = skill_score(n34_true[1:H], n34_pred[1:H])
    push!(lead_accs,  sc_h.acc)
    push!(lead_rmses, sc_h.rmse)
    months = H / month_steps
    println("  H = $H steps  (~$(round(months; digits=1)) mo):  ACC=$(round(sc_h.acc; digits=3))" *
            "  RMSE=$(round(sc_h.rmse; digits=4))")
end

lead_months = lead_horizons ./ month_steps

# ---------------------------------------------------------------------------
# 10. Plots — save to results/
# ---------------------------------------------------------------------------

mkpath("results")

# ── Helper: three-panel heatmap per layer ───────────────────────────────────
function layer_heatmaps(test_mat, pred_mat, scale, title_str)
    h1 = heatmap(test_mat[:, _pw:end], clim=(-scale, scale), c=:RdBu,
                 xlabel="step", title="Observed", colorbar=true)
    h2 = heatmap(pred_mat[:, _pw:end], clim=(-scale, scale), c=:RdBu,
                 xlabel="step", title="Forecast", colorbar=true)
    h3 = heatmap(abs.(test_mat[:, _pw:end] .- pred_mat[:, _pw:end]),
                 clim=(0, scale), c=:Reds,
                 xlabel="step", title="|Error|", colorbar=true)
    return plot(h1, h2, h3, size=(900, 300), plot_title=title_str, left_margin=2mm)
end

sc_vc = 0.5 * maximum(abs.(very_coarse))
sc_m  = 0.5 * maximum(abs.(medium))
sc_f  = 0.5 * maximum(abs.(fine))

p_vc = layer_heatmaps(test_vc, preds_vc, sc_vc, "Layer 1 — 18° coarse")
p_m  = layer_heatmaps(test_m,  preds_m,  sc_m,  "Layer 2 — 6° medium")
p_f  = layer_heatmaps(test_f,  preds_f,  sc_f,  "Layer 3 — 2° fine")

p_layers = plot(p_vc, p_m, p_f, layout=(3, 1), size=(1_000, 900), left_margin=5mm)
savefig(p_layers, "results/enso_layer_heatmaps.png")
println("Saved: results/enso_layer_heatmaps.png")

# ── RMSE error curves ───────────────────────────────────────────────────────
p_rmse = plot(error_curve_vc, label="18° coarse", lw=2, color=:blue,
              xlabel="step (post-warmup)", ylabel="Cumulative RMSE",
              title="ENSO forecast error", grid=true, legend=:topleft)
plot!(p_rmse, error_curve_m,  label="6° medium",  lw=2, color=:orange)
plot!(p_rmse, error_curve_f,  label="2° fine",    lw=2, color=:red)

# ── Niño 3.4 time series ────────────────────────────────────────────────────
p_n34 = plot(n34_true, label="Observed", lw=2, color=:black,
             xlabel="step (post-warmup)", ylabel="Niño 3.4 (norm.)",
             title="Niño 3.4  |  ACC = $(round(sc.acc; digits=2))",
             legend=:topright)
plot!(p_n34, n34_pred, label="Forecast", lw=2, color=:red, linestyle=:dash)

# ── Multi-window ACC bar ─────────────────────────────────────────────────────
p_windows = bar(1:N_windows, window_accs,
                xlabel="Window (~$(days_per_window) d each)", ylabel="Niño 3.4 ACC",
                title="Skill across prediction segments",
                fillcolor=:steelblue, linecolor=:steelblue,
                ylim=(-0.5, 1.0), legend=false)
hline!(p_windows, [0.5], lw=1.5, linestyle=:dash, color=:gray)
hline!(p_windows, [0.0], lw=1.0, linestyle=:dot,  color=:black)

# ── Lead-time ACC ──────────────────────────────────────────────────────────
p_lead = plot(lead_months, lead_accs,
              marker=:circle, lw=2, color=:blue,
              xlabel="Lead (months)", ylabel="Niño 3.4 ACC",
              title="Cumulative skill vs lead time",
              label="ACC", ylim=(-0.5, 1.0), grid=true, legend=:topright)
hline!(p_lead, [0.5], lw=1.5, linestyle=:dash, color=:gray,  label="ACC = 0.5 (useful)")
hline!(p_lead, [0.0], lw=1.0, linestyle=:dot,  color=:black, label="no skill")

p_skill = plot(p_rmse, p_n34, p_windows, p_lead,
               layout=(2, 2), size=(1_200, 800), left_margin=3mm, bottom_margin=3mm)
savefig(p_skill, "results/enso_skill_summary.png")
println("Saved: results/enso_skill_summary.png")

try display(p_layers); catch end
try display(p_skill);  catch end

# ---------------------------------------------------------------------------
# 11. Summary table
# ---------------------------------------------------------------------------

println("\n" * "="^55)
println("ENSO EXPERIMENT SUMMARY")
println("="^55)
println("Train   : $train_len steps  ($(round(train_len * dt / 365; digits=1)) yr)")
println("Predict : $predict_len steps  ($(round(predict_len * dt / 365; digits=1)) yr)")
println("Domain  : lon $(lon_range[1])°–$(lon_range[2])°  lat $(lat_range[1])°–$(lat_range[2])°")
println("")
println("Full-window Niño 3.4 skill:")
println("  ACC            = $(round(sc.acc;        digits=3))")
println("  RMSE           = $(round(sc.rmse;       digits=4))")
println("  RMSE vs pers.  = $(round(sc.rmse_skill; digits=3))")
println("")
println("Multi-window ACCs  : " *
    join(["w$w: $(round(window_accs[w]; digits=3))" for w in 1:N_windows], " | "))
println("Lead-time ACCs     : " *
    join(["$(round(m; digits=0))mo→$(round(a; digits=2))"
          for (m, a) in zip(lead_months, lead_accs)], " | "))
println("="^55)
