# Run climatology, persistence, and damped-persistence baselines on W1.
# Produces per-baseline JLD2 files in
# results/temporal_multiscale_compare/tier4_head_to_head/<baseline>_W1/
# in the SAME format as the reservoir runs (n34_pred, field_pred, seed, etc.)
# so that downstream scripts (event_skill_table.jl, plot_event_aligned.jl)
# can compare apples to apples.
#
# Each baseline gets a single dummy seed=0 since it has no random component
# (purely deterministic from training data). We write 1 file per baseline.

ENV["GKSwstype"] = "100"
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using JLD2, Statistics, LinearAlgebra, Dates
using CrossScaleRC

# ----------------------------------------------------------------------------
# Mirror main_enso_temporal_multiscale_field.jl loading + windowing
# ----------------------------------------------------------------------------
washout      = 12
train_start  = 1
train_len    = 288
predict_len  = 96
warmup       = 12
lon_range    = (126.0, 288.0)
lat_range    = (-36.0, 36.0)
start_date   = Date(1982, 1, 1)

println("Loading 2° SST anomalies (daily)...")
clim_day_lo = (train_start - 1) * 30 + 1
clim_day_hi = (train_start - 1 + train_len) * 30
data_vec_daily, _ = load_data(
    [2.0]; show_data = false, refinement = 1,
    lon_range = lon_range, lat_range = lat_range,
    anomalies = true, train_indices = clim_day_lo:clim_day_hi,
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

println("Binning to monthly...")
fine_2d_full = daily_to_monthly(data_vec_daily[1], start_date)
nlon_f, nlat_f, Ttot_full = size(fine_2d_full)
fine_2d = fine_2d_full[:, :, train_start:end]
Ttot    = size(fine_2d, 3)

# Per-pixel z-score using training-window stats.
for i in 1:nlon_f, j in 1:nlat_f
    ts = @view fine_2d[i, j, :]
    if any(isnan, ts); ts .= 0.0; end
    train_view = @view ts[1:train_len]
    μ_p, σ_p = mean(train_view), std(train_view)
    if σ_p > 0
        ts .= (ts .- μ_p) ./ σ_p
    end
end

# Train/test split
field_train = fine_2d[:, :, 1:train_len]
# Test window aligned with reservoir scripts: starts after warmup at month
# train_len - warmup + warmup + 1 = train_len + 1 of the original data;
# but reservoir's reference uses `truth_window_abs = (t0+warmup+1):(t0+warmup+predict_len)`
# with t0 = train_len - warmup → window = (train_len+1):(train_len+predict_len).
truth_window = (train_len + 1) : (train_len + predict_len)
field_truth_3d = fine_2d[:, :, truth_window]

# Cropped lon/lat for Niño 3.4
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

# Truth Niño 3.4 (matches the reservoir scripts)
n34_true = nino34_index(field_truth_3d, lons_f, lats_f)

# ----------------------------------------------------------------------------
# Run baselines
# ----------------------------------------------------------------------------
ROOT = "results/temporal_multiscale_compare/tier4_head_to_head"

baselines = [
    ("climatology",          forecast_climatology(field_train, predict_len)),
    ("persistence",          forecast_persistence(field_train, predict_len)),
    ("damped_persistence_3", forecast_damped_persistence(field_train, predict_len; tau_decay = 3.0)),
    ("damped_persistence_6", forecast_damped_persistence(field_train, predict_len; tau_decay = 6.0)),
    ("damped_persistence_12", forecast_damped_persistence(field_train, predict_len; tau_decay = 12.0)),
]

for (name, field_pred_3d) in baselines
    outdir = joinpath(ROOT, "$(name)_W1")
    mkpath(outdir)

    n34_pred = nino34_index(field_pred_3d, lons_f, lats_f)

    # Match the reference file used by event_skill_table.jl
    ref_path = joinpath(outdir, "enso_temporal_field_reference_$(name).jld2")
    if !isfile(ref_path)
        jldsave(ref_path; compress = true,
            n34_true = Float32.(n34_true),
            field_true = Float32.(field_truth_3d),
            lons = collect(lons_f), lats = collect(lats_f),
            train_len = train_len, predict_len = predict_len, warmup = warmup,
            mode_tag = name)
    end

    # Compute headline metrics for this baseline (uses event-relevant lead window)
    headline = forecast_headline(n34_true, n34_pred, field_truth_3d, field_pred_3d,
                                 lons_f, lats_f)

    # Single per-seed file (seed=0 — deterministic baseline)
    save_kwargs = Dict{Symbol, Any}(
        :compress           => true,
        :n34_pred           => Float32.(n34_pred),
        :field_pred         => Float32.(field_pred_3d),
        :acc12              => headline.acc12,
        :rmse12             => headline.rmse12,
        :std_ratio12        => headline.std_ratio12,
        :pc3                => headline.pc3,
        :pc12               => headline.pc12,
        :ppacc_n34_mean     => headline.ppacc_n34_mean,
        :ppacc_global_mean  => headline.ppacc_global_mean,
        :ppacc_map          => Float32.(headline.ppacc_map),
        :seed               => 0,
        :mode_tag           => name,
        :train_start        => train_start,
        :train_len          => train_len,
        :predict_len        => predict_len,
        :window_label       => "W1",
    )
    jldsave(joinpath(outdir, "enso_temporal_field_preds_$(name)_seed0.jld2");
        save_kwargs...)

    println("$(rpad(name, 24))  acc12=$(round(headline.acc12; digits=3))  pc3=$(round(headline.pc3; digits=3))  std_ratio12=$(round(headline.std_ratio12; digits=3))")
end

println("\nDone. Per-seed files saved under $(ROOT)/<baseline>_W1/")
println("Now run: julia --project=. scripts/event_skill_table.jl")
