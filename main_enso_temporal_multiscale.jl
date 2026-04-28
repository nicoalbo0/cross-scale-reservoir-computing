# ENSO temporal cross-scale reservoir on the 1D Niño 3.4 monthly index.
#
# Architecture (motivation, recharge oscillator analog):
#   Decompose the scalar Niño 3.4(t) into temporal frequency bands and run
#   one reservoir per band. The slow band carries the ENSO oscillator and
#   decadal modulation; the mid band carries seasonal residuals + Walker-cell
#   variability; the fast band carries intraseasonal noise (MJO, wind bursts).
#   Cross-scale wiring (slow → mid → fast) mirrors Jin's "slow heat content
#   drives fast SST response" structure, but parametric content is learned by
#   the reservoirs from data — the architecture is a prior, not a constraint.
#
# Modes (ENSO_TM_MODE):
#   :single_reservoir — un-decomposed Niño 3.4 → one reservoir (canary baseline).
#   :no_xscale        — decompose, three INDEPENDENT reservoirs, sum predictions.
#   :two_band_cascade — slow band feeds mid band via run_multi_layer.
#   :full_cascade     — slow → mid → fast cross-scale chain.
#
# Usage:
#   ENSO_TM_MODE=single_reservoir ENSO_SEED=42 julia --threads 4 --project=. main_enso_temporal_multiscale.jl

ENV["GKSwstype"] = "100"

using Pkg, Revise
Pkg.activate(".")

using CrossScaleRC
using JLD2, Dates, LinearAlgebra, Random, Statistics
using Plots, Measures

BLAS.set_num_threads(1)

# ---------------------------------------------------------------------------
# Configuration (env-var driven, mirrors main_enso_monthly.jl)
# ---------------------------------------------------------------------------

seed     = parse(Int, get(ENV, "ENSO_SEED", "42"))
mode     = Symbol(get(ENV, "ENSO_TM_MODE", "single_reservoir"))
@assert mode in (:single_reservoir, :no_xscale, :two_band_cascade, :full_cascade) (
    "Unknown ENSO_TM_MODE=$(mode); expected one of " *
    "single_reservoir, no_xscale, two_band_cascade, full_cascade")

mode_tag = String(mode)
seed_tag = "seed$(seed)"
outdir   = get(ENV, "ENSO_OUTDIR",
               joinpath("results", "temporal_multiscale", mode_tag))

# Pipeline timing (1 reservoir step = 1 calendar month)
washout     = 12
train_len   = 288     # 24 yr training
predict_len = 96      # 8 yr forecast
warmup      = 12
dt          = 1.0

lon_range  = (126.0, 288.0)
lat_range  = (-36.0, 36.0)
start_date = Date(1982, 1, 1)

# Per-band reservoir hyperparameters (defaults; env-overridable)
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
g_in_rec   = parse(Float64, get(ENV, "ENSO_TM_GIN_REC",    "0.5"))
glayer_mid_exp  = parse(Float64, get(ENV, "ENSO_TM_GLAYER_MID_EXP",  "-1.0"))
glayer_fast_exp = parse(Float64, get(ENV, "ENSO_TM_GLAYER_FAST_EXP", "-1.0"))

# Cutoff frequencies (cycles/month); two cutoffs → three bands.
cutoffs_str = get(ENV, "ENSO_TM_CUTOFFS", "1/24,1/3")
cutoffs     = let parts = split(cutoffs_str, ',')
    Tuple(Float64(eval(Meta.parse(strip(s)))) for s in parts)
end
@assert length(cutoffs) == 2 "Expected two cutoffs (slow|mid|fast); got $(length(cutoffs))"

# Canary mode τ override (used only by :single_reservoir).
τ_single = parse(Float64, get(ENV, "ENSO_TM_TAU_SINGLE", string(τ_slow)))

Random.seed!(seed)
mkpath(outdir)

println("="^60)
println("ENSO temporal multiscale — mode=$(mode_tag)  seed=$(seed)")
println("="^60)
println("  cutoffs   = $(cutoffs)  (cycles/month)")
println("  outdir    = $(outdir)")
println("  τ slow/mid/fast/single = $(τ_slow) / $(τ_mid) / $(τ_fast) / $(τ_single)")

# ---------------------------------------------------------------------------
# 1. Load 2° SST anomalies → monthly bin → Niño 3.4 series
# ---------------------------------------------------------------------------

println("\nLoading 2° SST anomalies (daily) ...")
train_days_for_clim = train_len * 30
data_vec_daily, _dt = load_data(
    [2.0];
    show_data     = false,
    refinement    = 1,
    lon_range     = lon_range,
    lat_range     = lat_range,
    anomalies     = true,
    train_indices = 1:train_days_for_clim,
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

# Cropped grid coordinates for Niño 3.4 spatial average. Note: we generate the
# GLOBAL grid (size determined from `res`, not from the cropped data dims) and
# then apply the same lon/lat mask that load_data used. Passing the cropped
# data dims here would produce incorrect coordinates.
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

println("Computing Niño 3.4 monthly index ...")
n34_raw = nino34_index(fine_2d, lons_f, lats_f)         # length Ttot

# Z-score the full series (using the mean/std over the *training* window only,
# to avoid test leakage). The whole pipeline operates on this normalised signal.
n34_train_view = @view n34_raw[1:train_len]
μ_n34 = mean(n34_train_view)
σ_n34 = std(n34_train_view)
@assert σ_n34 > 0 "Niño 3.4 std ≈ 0 over training window — bad data?"
n34 = (n34_raw .- μ_n34) ./ σ_n34
println("  series length = $(length(n34))   train μ=$(round(μ_n34;digits=4)) σ=$(round(σ_n34;digits=4))")

# ---------------------------------------------------------------------------
# 2. Mode-specific pipeline
# ---------------------------------------------------------------------------

# All modes use a trivial 1D block topology for the per-band scalar reservoir.
blocks_solo = make_blocks(1, 1, 0)   # rec_dim=1, neigh_dim=0, layer_dim=0

n34_mat = reshape(n34, 1, Ttot)                          # (1, Ttot) data matrix
zero_layer = zeros(eltype(n34_mat), 1, Ttot)             # placeholder when no cross-scale

n34_pred       = Float64[]                 # forecast over the test window
bands_used     = nothing                   # band signals if decomposed
preds_per_band = Dict{Symbol, Vector{Float64}}()   # per-band forecast (test window)

if mode == :single_reservoir
    # Canary: classic low-dim ESN regime (Mackey-Glass-style). For a 1D scalar
    # input, the leaky-integrator hyperparameters that work for the spatial
    # 1L τ=30 (ρ=0.55, ridge=1.0, big input dim) collapse to amplitude-zero in
    # closed loop because the readout is over-regularised. Switching to ρ=0.95
    # (edge-of-chaos memory), no leak (τ=1), and very light ridge gives the
    # standard ESN regime that handles 1D scalar forecasting (Mackey-Glass,
    # Lorenz, etc.). Note: this is fundamentally different from the spatial
    # canary — it tests "what's achievable on a 1D ENSO time series" not
    # "match the spatial 0.891 number." Establishes the baseline for stages
    # A.2/A.3/A.4 to compare against.
    println("\n--- :single_reservoir — one reservoir on un-decomposed Niño 3.4 ---")
    N_canary     = parse(Int,     get(ENV, "ENSO_TM_NRES_SINGLE",  "1000"))
    ρ_canary     = parse(Float64, get(ENV, "ENSO_TM_RHO_SINGLE",   "0.95"))
    ridge_canary = parse(Float64, get(ENV, "ENSO_TM_RIDGE_SINGLE", "1e-7"))
    τ_canary     = parse(Float64, get(ENV, "ENSO_TM_TAU_SINGLE",   "1.0"))
    g_in_canary  = 10^(-0.5) / √1.0
    println("  N=$(N_canary)  ρ=$(ρ_canary)  τ=$(τ_canary)  g_in=$(round(g_in_canary;digits=4))  ridge=$(ridge_canary)")
    params = (N_canary, ρ_canary, 10, g_in_canary, 0.0, 0.0, τ_canary, dt)
    preds, _, _, _, _ = run_single_layer(
        params, n34_mat, zero_layer, train_len, predict_len, blocks_solo;
        washout = washout, warmup = warmup,
        ridge_parameter = ridge_canary,

        show_progress = true, input_mode = :random,
        regression_mode = :quadratic,
    )
    # preds shape: (1, warmup + predict_len). Drop warmup window.
    n34_pred = vec(preds[1, warmup+1:end])
elseif mode == :no_xscale
    # A.2 ablation: decompose Niño 3.4 into 3 bands; train ONE reservoir per band
    # independently (no cross-scale wiring); reconstruct n34_pred = Σ band preds.
    # Tests whether decomposition itself adds value, separable from cross-scale.
    println("\n--- :no_xscale — three independent per-band reservoirs ---")
    bands = bandpass_decompose(n34, cutoffs; fs = 1.0, order = 4)
    bands_used = bands
    println("  band variance fractions: slow=$(round(var(bands.slow)/var(n34);digits=3))   mid=$(round(var(bands.mid)/var(n34);digits=3))   fast=$(round(var(bands.fast)/var(n34);digits=3))")

    # Each band's own reservoir hyperparameters. Defaults are physics-motivated:
    # slow band has long memory (τ_slow), mid faster (τ_mid), fast fastest (τ_fast).
    # Use the same ESN regime that worked for the canary (ρ=0.95-style) since
    # each band reservoir is also 1D scalar.
    function _train_band(band_signal, τ, ridge_p, N, ρ)
        mat = reshape(Float64.(band_signal), 1, Ttot)
        params = (N, ρ, 10, 10^(-0.5)/√1.0, 0.0, 0.0, τ, dt)
        preds, _, _, _, _ = run_single_layer(
            params, mat, zero_layer, train_len, predict_len, blocks_solo;
            washout = washout, warmup = warmup,
            ridge_parameter = ridge_p,
            show_progress = false, input_mode = :random,
            regression_mode = :quadratic,
        )
        return vec(preds[1, warmup+1:end])
    end

    println("  training slow reservoir  (τ=$(τ_slow), ρ=$(ρ_slow), ridge=$(ridge_slow), N=$(N_slow)) ...")
    pred_slow = _train_band(bands.slow, τ_slow, ridge_slow, N_slow, ρ_slow)
    println("  training mid reservoir   (τ=$(τ_mid),  ρ=$(ρ_mid),  ridge=$(ridge_mid),  N=$(N_mid)) ...")
    pred_mid  = _train_band(bands.mid,  τ_mid,  ridge_mid,  N_mid,  ρ_mid)
    println("  training fast reservoir  (τ=$(τ_fast), ρ=$(ρ_fast), ridge=$(ridge_fast), N=$(N_fast)) ...")
    pred_fast = _train_band(bands.fast, τ_fast, ridge_fast, N_fast, ρ_fast)

    n34_pred = pred_slow .+ pred_mid .+ pred_fast
    preds_per_band = Dict(:slow => pred_slow, :mid => pred_mid, :fast => pred_fast)

    # Per-band skill on the band's OWN ground truth (truncated to test window)
    t_truth = (train_len + 1) : (train_len + length(n34_pred))
    for (b, p) in preds_per_band
        truth_band = bands[b][t_truth]
        s = skill_score(truth_band, p)
        sr = std(p) / std(truth_band)
        println("    $(rpad(string(b),5)) band own-skill ACC=$(round(s.acc;digits=3))   RMSE=$(round(s.rmse;digits=4))   std_ratio=$(round(sr;digits=3))")
    end
else
    error("Mode $(mode) not yet implemented in this stage. ",
          "Add :two_band_cascade, :full_cascade in subsequent stages.")
end

# Truth aligned with prediction window
t0 = train_len - warmup
n34_true = n34[t0 + warmup + 1 : t0 + warmup + length(n34_pred)]
@assert length(n34_pred) == length(n34_true)

# ---------------------------------------------------------------------------
# 3. Skill metrics — pair ACC with RMSE and std_ratio (per project memory:
#     ACC alone hides amplitude-collapsed forecasts).
# ---------------------------------------------------------------------------

sc = skill_score(n34_true, n34_pred)
std_ratio_full = std(n34_pred) / std(n34_true)

println("\n=== ENSO temporal multiscale forecast skill ===")
println("  mode               : $(mode_tag)")
println("  full-window ACC    : $(round(sc.acc;        digits=3))")
println("  full-window RMSE   : $(round(sc.rmse;       digits=4))")
println("  RMSE skill (vs pers): $(round(sc.rmse_skill; digits=3))")
println("  std_ratio (pred/true): $(round(std_ratio_full; digits=3))")

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

# ---------------------------------------------------------------------------
# 4. Plots: Niño 3.4 trajectory + lead-time skill
# ---------------------------------------------------------------------------

p1 = plot(n34_true; lw=2.5, color=:black, label="Observed",
          xlabel="month since forecast start", ylabel="Niño 3.4 (z-score)",
          title="Temporal multiscale  mode=$(mode_tag)  seed=$(seed)  ACC=$(round(sc.acc;digits=2))",
          legend=:topright)
plot!(p1, n34_pred; lw=2.0, color=:red, ls=:dash, label="Forecast")

p2 = plot(leads_used, lead_accs; marker=:circle, lw=2, color=:blue, label="ACC",
          xlabel="Lead (months)", ylabel="cumulative ACC",
          title="Lead-time skill", ylim=(-1, 1), legend=:bottomleft)
hline!(p2, [0.5]; ls=:dash, color=:gray, label="ACC=0.5")
hline!(p2, [0.0]; ls=:dot,  color=:black, label=false)

p_summary = plot(p1, p2; layout=(2, 1), size=(1200, 700), left_margin=4mm)
out_png = joinpath(outdir, "enso_temporal_$(mode_tag)_$(seed_tag).png")
savefig(p_summary, out_png)
println("\nSaved: $(out_png)")

# ---------------------------------------------------------------------------
# 5. Persist (Float32 + compress; reference once + per-seed slim file)
# ---------------------------------------------------------------------------

ref_path = joinpath(outdir, "enso_temporal_reference_$(mode_tag).jld2")
if !isfile(ref_path)
    jldsave(ref_path; compress = true,
            n34_true = Float32.(n34_true),
            n34_full = Float32.(n34),
            mu_n34 = Float32(μ_n34), sigma_n34 = Float32(σ_n34),
            cutoffs = collect(cutoffs),
            train_len = train_len, predict_len = predict_len, warmup = warmup,
            mode_tag = mode_tag)
    println("Saved reference: $(ref_path)")
end

per_seed = Dict{Symbol, Any}(
    :n34_pred       => Float32.(n34_pred),
    :lead_months    => collect(leads_used),
    :lead_accs      => Float64.(lead_accs),
    :lead_rmses     => Float64.(lead_rmses),
    :lead_std_ratios => Float64.(lead_std_ratios),
    :full_acc       => sc.acc,
    :full_rmse      => sc.rmse,
    :full_rmse_skill => sc.rmse_skill,
    :std_ratio_full => std_ratio_full,
    :seed           => seed,
    :mode_tag       => mode_tag,
)
# bands_used / preds_per_band attached when the mode produces them (later stages)
if bands_used !== nothing
    per_seed[:bands_keys] = collect(keys(bands_used))
end
for (b, p) in preds_per_band
    per_seed[Symbol("pred_", b)] = Float32.(p)
end

jldsave(joinpath(outdir, "enso_temporal_preds_$(mode_tag)_$(seed_tag).jld2"); compress = true,
        per_seed...)

println("\n" * "="^60)
println("SUMMARY  mode=$(mode_tag)  seed=$(seed)")
println("="^60)
println("  full-window ACC : $(round(sc.acc; digits=3))   RMSE: $(round(sc.rmse; digits=4))")
println("  std_ratio       : $(round(std_ratio_full; digits=3))")
println("="^60)
