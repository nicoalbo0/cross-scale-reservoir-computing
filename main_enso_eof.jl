# EOF-based ENSO forecast.
#
# Idea: instead of trying to predict SST at every pixel, project the field
# onto a small basis of spatial EOFs (Empirical Orthogonal Functions). The
# top K modes are physically meaningful (EOF1 ≈ canonical ENSO, EOF2 ≈ ENSO
# Modoki / quadrupole, etc). The reservoir then forecasts a K-dim time
# series; the spatial field is reconstructed by combining EOF patterns
# weighted by predicted coefficients. Spatial structure is built in by
# construction.
#
# Pipeline:
#   1. Load daily 2° SST anomalies, bin to monthly, normalise each pixel.
#   2. SVD decompose training-period field into U Σ V'.
#   3. Top-K U columns = EOF patterns (n_pix × K). Project full series:
#      coefficients = U_K' * X  (K × T_total).
#   4. Train ONE small reservoir (no spatial blocking) to forecast the
#      K-dim coefficient time series.
#   5. Reconstruct SST = U_K * coefficients_predicted.
#   6. Compute Niño 3.4 from reconstruction; pattern correlation vs truth.

ENV["GKSwstype"] = "100"   # suppress GKS preview window (no popups during runs)

using Pkg, Revise
Pkg.activate(".")

using CrossScaleRC
using JLD2, Dates, LinearAlgebra, Random, Statistics, Plots, Measures

BLAS.set_num_threads(1)

seed = parse(Int, get(ENV, "ENSO_SEED", "42"))
Random.seed!(seed)
seed_tag = "seed$(seed)"

K           = parse(Int, get(ENV, "ENSO_K", "20"))   # number of EOFs to keep
N_res       = parse(Int, get(ENV, "ENSO_N", "500"))  # reservoir size
washout     = 12
train_len   = 288
predict_len = 96
warmup      = 12
dt          = 1.0
lon_range   = (126.0, 288.0)
lat_range   = (-36.0, 36.0)
start_date  = Date(1982, 1, 1)

# ------------------------------------------------------------------
# Data: daily anomalies → monthly means → normalise
# ------------------------------------------------------------------
println("Loading 2° SST anomalies (daily) ...")
data_vec_daily, _ = load_data(
    [2.0]; refinement = 1,
    lon_range = lon_range, lat_range = lat_range,
    anomalies = true,
    train_indices = 1:(train_len * 30),
)
daily = data_vec_daily[1]
nlon, nlat, nt_daily = size(daily)

println("Binning to monthly ...")
function daily_to_monthly(daily, start_date::Date)
    nlon, nlat, nt = size(daily)
    dates = [start_date + Day(t - 1) for t in 1:nt]
    ym    = [(year(d), month(d)) for d in dates]
    keys  = unique(ym)
    M = Array{Float64, 3}(undef, nlon, nlat, length(keys))
    for (k, target) in enumerate(keys)
        mask = [x == target for x in ym]
        M[:, :, k] .= dropdims(mean(daily[:, :, mask]; dims=3); dims=3)
    end
    return M
end
monthly = daily_to_monthly(daily, start_date)

# Normalise per pixel (zero mean, unit std). Replace NaNs with 0 (land mask).
for i in 1:nlon, j in 1:nlat
    ts = @view monthly[i, j, :]
    any(isnan, ts) && (ts .= 0.0)
    μ, σ = mean(ts), std(ts)
    σ > 0 && (ts .= (ts .- μ) ./ σ)
end

Ttot = size(monthly, 3)
println("Spatial domain: $(nlon)×$(nlat) = $(nlon*nlat) pixels, $(Ttot) months")
@assert train_len + predict_len ≤ Ttot

# Flatten spatial dims: X[pixel, time]
X = reshape(monthly, nlon * nlat, Ttot)

# ------------------------------------------------------------------
# EOF decomposition over the TRAINING window
# ------------------------------------------------------------------
println("Computing EOFs over training window (1:$train_len) ...")
X_train = X[:, 1:train_len]            # (n_pix × train_len)
# already mean-zero from per-pixel normalisation, but recenter to be safe
pixel_means = mean(X_train; dims=2)
X_train_c = X_train .- pixel_means

# SVD: X_train_c = U * S * V'
F = svd(X_train_c)
U_K = F.U[:, 1:K]                      # (n_pix × K) — EOF spatial patterns
println("Top $(K) EOFs explain $(round(100 * sum(F.S[1:K] .^ 2) / sum(F.S .^ 2); digits=2))% of training variance")

# Project ENTIRE series (including test) onto EOFs.
PC = U_K' * (X .- pixel_means)         # (K × Ttot)
println("PC shape: $(size(PC))")

# ------------------------------------------------------------------
# Train one reservoir on the K-dim PC time series.
# ------------------------------------------------------------------
res_size = N_res
res_rad  = 0.85
degree   = 10
g_in     = 0.5
τ        = 30.0
ridge    = 1e-2

# Build a single-layer block (no spatial partitioning — one block covers all K modes).
blocks = make_blocks(K, 1, 0)            # 1D block: 1 block covering all K dims, no neighbor mixing
rec_dim, neigh_dim, layer_dim = input_dimensions(blocks)

params_solo = (res_size, res_rad, degree, g_in, 0.0, 0.0, τ, dt)

println("\nTraining reservoir on PCs (N=$(res_size), τ=$(τ)) ...")
preds_PC, train_pred_PC, train_data_PC, X_states, _ = run_single_layer(
    params_solo,
    PC,
    zeros(eltype(PC), size(PC)),       # no layer input
    train_len, predict_len, blocks;
    washout = washout, warmup = warmup,
    ridge_parameter = ridge,
    show_progress = true, input_mode = :random,
    regression_mode = :quadratic,
)

# Align test window
t0 = train_len - warmup
PC_test = PC[:, t0 + 1 : t0 + size(preds_PC, 2)]

# ------------------------------------------------------------------
# Reconstruct spatial field from PCs
# ------------------------------------------------------------------
function reconstruct(PCs, U_K, pixel_means, nlon, nlat)
    Xrec = U_K * PCs .+ pixel_means     # (n_pix × T)
    return reshape(Xrec, nlon, nlat, size(PCs, 2))
end
field_obs  = reconstruct(PC_test,  U_K, pixel_means, nlon, nlat)
field_pred = reconstruct(preds_PC, U_K, pixel_means, nlon, nlat)
# Note field_obs is the EOF-projected truth (low-rank); for fairness we
# also keep the raw truth for spatial pattern correlation against pred.
truth_test = reshape(X[:, t0 + 1 : t0 + size(preds_PC, 2)], nlon, nlat, size(preds_PC, 2))

# ------------------------------------------------------------------
# Niño 3.4 from reconstructed field
# ------------------------------------------------------------------
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
lons, lats = cropped_lons_lats(nlon, nlat, 2.0, lon_range, lat_range)

n34_true = nino34_index(truth_test[:, :, warmup + 1 : end], lons, lats)
n34_pred = nino34_index(field_pred[:,  :, warmup + 1 : end], lons, lats)

sc = skill_score(n34_true, n34_pred)
println("\n=== EOF-ENSO Forecast ===")
println("  K (modes)        : $K")
println("  Niño 3.4 ACC     : $(round(sc.acc; digits=3))")
println("  Niño 3.4 RMSE    : $(round(sc.rmse; digits=4))")

# Spatial pattern correlation per time-step against RAW truth
function spatial_pc(A, B)
    a = vec(A); b = vec(B)
    am = a .- mean(a); bm = b .- mean(b)
    return dot(am, bm) / (norm(am) * norm(bm) + eps())
end
T = size(truth_test, 3)
pcorr_per_t = [spatial_pc(truth_test[:, :, t], field_pred[:, :, t]) for t in (warmup + 1):T]
println("Spatial pattern correlation (truth vs prediction):")
for L in [3, 6, 9, 12, 18, 24, 36, 48]
    L > length(pcorr_per_t) && continue
    println("  $(L) mo: pc=$(round(pcorr_per_t[L]; digits=3))")
end

# Lead-time Niño 3.4 cumulative ACC
println("Niño 3.4 cumulative ACC:")
for L in [3, 6, 9, 12, 18, 24, 36, 48]
    L > length(n34_true) && continue
    s = skill_score(n34_true[1:L], n34_pred[1:L])
    println("  $(L) mo: ACC=$(round(s.acc; digits=3))")
end

# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------
mkpath("results")
seed_tag_full = "K$(K)_N$(N_res)_$(seed_tag)"

p1 = plot(n34_true; lw=2.5, color=:black, label="Observed",
          xlabel="month", ylabel="Niño 3.4 (norm.)",
          title="EOF-ENSO  K=$(K) N=$(N_res) seed=$(seed)  ACC=$(round(sc.acc; digits=2))")
plot!(p1, n34_pred; lw=2, color=:red, ls=:dash, label="Forecast")

p2 = plot(1:length(pcorr_per_t), pcorr_per_t; lw=2, color=:red,
          xlabel="month", ylabel="spatial pattern correlation",
          title="Pattern correlation (truth vs prediction)",
          ylim=(-0.5, 1.0))
hline!(p2, [0.5]; ls=:dash, color=:gray, label="0.5")
hline!(p2, [0.0]; ls=:dot,  color=:black, label=false)

p_top = plot(p1, p2; layout=(2, 1), size=(1200, 700), left_margin=4mm)
savefig(p_top, "results/enso_eof_skill_$(seed_tag_full).png")
println("Saved: results/enso_eof_skill_$(seed_tag_full).png")

# Maps at selected leads (truth vs prediction vs |error|)
snap_leads = filter(L -> L ≤ size(field_pred, 3) - warmup, [3, 6, 9, 12, 18, 24])
sc_map = 1.5
lons_plot = mod.(lons, 360.0); perm = sortperm(lons_plot); lons_plot = lons_plot[perm]
function map_panel(field2d, ttl; clim_=(-sc_map, sc_map), cmap=:RdBu)
    heatmap(lons_plot, lats, permutedims(field2d[perm, :]);
            clim=clim_, c=cmap, xlabel="lon (°E)", ylabel="lat (°N)",
            aspect_ratio=:equal,
            xlims=(minimum(lons_plot), maximum(lons_plot)),
            ylims=(minimum(lats), maximum(lats)),
            title=ttl, colorbar=true)
end
panels = Plots.Plot[]
for L in snap_leads
    obs = truth_test[:, :, warmup + L]
    pred = field_pred[:, :, warmup + L]
    err = obs .- pred
    pc = round(spatial_pc(obs, pred); digits=2)
    push!(panels, map_panel(obs,  "Observed ($(L) mo)"))
    push!(panels, map_panel(pred, "EOF Forecast ($(L) mo)  pc=$(pc)"))
    push!(panels, map_panel(err,  "Obs − Forecast ($(L) mo)"))
end
p_maps = plot(panels...; layout=(length(snap_leads), 3),
              size=(1800, 320 * length(snap_leads)),
              plot_title="EOF-based 2° SST forecast — K=$(K) N=$(N_res) seed=$(seed)",
              left_margin=4mm, bottom_margin=3mm)
savefig(p_maps, "results/enso_eof_maps_$(seed_tag_full).png")
println("Saved: results/enso_eof_maps_$(seed_tag_full).png")

# Save full PCs and field for ensemble aggregation later
# Save compactly: Float32 for big arrays, compression on. Reference data
# (truth_test, EOFs, pixel_means, lons/lats) is identical across seeds — write
# it ONCE to results/enso_eof_reference.jld2; per-seed file holds only the
# seed-specific forecast.
ref_path = "results/enso_eof_reference.jld2"
if !isfile(ref_path)
    jldsave(ref_path; compress=true,
            truth_test = Float32.(truth_test),
            U_K        = Float32.(U_K),
            pixel_means = Float32.(pixel_means),
            lons = collect(lons), lats = collect(lats),
            K = K, N_res = N_res)
    println("Saved reference: $(ref_path)")
end

jldsave("results/enso_eof_preds_$(seed_tag_full).jld2"; compress=true,
        n34_true   = Float32.(n34_true),
        n34_pred   = Float32.(n34_pred),
        field_pred = Float32.(field_pred),
        preds_PC   = Float32.(preds_PC),
        PC_test    = Float32.(PC_test),
        K = K, N_res = N_res, seed = seed,
        full_acc   = sc.acc, full_rmse = sc.rmse)
