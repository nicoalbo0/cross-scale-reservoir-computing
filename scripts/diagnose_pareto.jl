# Diagnostic for the Stage E per-pixel pareto:
# A wins acc12 at the Niño 3.4 INDEX (0.873) but mean per-pixel ACC inside
# the box is -0.058. Why? Hypothesis: A predictions carry weak per-pixel
# signal but spatially-uncorrelated noise; spatial averaging (the index)
# improves SNR by √N. Diagnose by:
# (1) Computing the cross-pixel error correlation matrix.
# (2) Computing per-pixel correlation distribution for A vs B vs D.
# (3) Computing the index ACC implied by the per-pixel signal+noise model.
# (4) Plotting region-by-region per-pixel ACC (cold tongue, warm pool, off-equator).

ENV["GKSwstype"] = "100"
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using JLD2, Statistics, Glob, Plots, Measures, LinearAlgebra, Printf

ROOT = "results/temporal_multiscale_compare/tier4_head_to_head"
ARCHS = [("A_no_xscale",   joinpath(ROOT, "no_xscale_field_W1"),   "no_xscale_field"),
         ("B_multi_tau",   joinpath(ROOT, "multi_tau_3_field_W1"), "multi_tau_3_field"),
         ("D_full_cascade", joinpath(ROOT, "full_cascade_field_W1"), "full_cascade_field")]

# Load truth (lons, lats, field_true)
ref_path = joinpath(ARCHS[1][2], "enso_temporal_field_reference_no_xscale_field.jld2")
lons, lats, field_true = jldopen(ref_path, "r") do f
    Float64.(f["lons"]), Float64.(f["lats"]), Float64.(f["field_true"])
end
nlon, nlat, nt = size(field_true)
lons360 = mod.(lons, 360.0)
lon_mask = 190.0 .<= lons360 .<= 240.0
lat_mask = -5.0  .<= lats   .<= 5.0
println("Niño 3.4 box pixels: $(sum(lon_mask) * sum(lat_mask)) ($(sum(lon_mask)) × $(sum(lat_mask)))")

function load_ensemble_pred(dir, mode)
    paths = sort(glob("enso_temporal_field_preds_$(mode)_seed*.jld2", dir))
    isempty(paths) && error("No per-seed in $dir")
    preds = Vector{Array{Float64, 3}}()
    for p in paths
        jldopen(p, "r") do f
            push!(preds, Float64.(f["field_pred"]))
        end
    end
    stk = cat(preds...; dims = 4)
    return dropdims(mean(stk; dims = 4); dims = 4)  # ensemble-mean field (nlon,nlat,nt)
end

function pearson_corr(a, b)
    am = a .- mean(a); bm = b .- mean(b)
    σa = std(a); σb = std(b)
    (σa > 0 && σb > 0) ? mean(am .* bm) / (σa * σb) : NaN
end

println("\n" * "="^72)
println("Per-pixel ACC distribution (ensemble-mean prediction across 8 seeds)")
println("="^72)

box_results = Dict{String, NamedTuple}()
for (label, dir, mode) in ARCHS
    pred = load_ensemble_pred(dir, mode)
    @assert size(pred) == size(field_true)

    # Index = spatial mean inside Niño 3.4 box
    truth_box = field_true[lon_mask, lat_mask, :]
    pred_box  = pred[lon_mask, lat_mask, :]
    nbox = size(truth_box, 1) * size(truth_box, 2)

    truth_box_flat = reshape(truth_box, nbox, nt)
    pred_box_flat  = reshape(pred_box,  nbox, nt)
    valid_pix = vec([!any(isnan, truth_box_flat[i, :]) && !any(isnan, pred_box_flat[i, :])
                     for i in 1:nbox])

    truth_v = truth_box_flat[valid_pix, :]
    pred_v  = pred_box_flat[valid_pix, :]
    npix = size(truth_v, 1)

    # Niño 3.4 index = spatial mean
    n34_truth = vec(mean(truth_v; dims = 1))
    n34_pred  = vec(mean(pred_v;  dims = 1))
    acc_index = pearson_corr(n34_truth[1:12], n34_pred[1:12])
    acc_full  = pearson_corr(n34_truth, n34_pred)

    # Per-pixel ACC: compute on first 12 mo too
    ppacc_12 = [pearson_corr(truth_v[i, 1:12], pred_v[i, 1:12]) for i in 1:npix]
    ppacc_full = [pearson_corr(truth_v[i, :], pred_v[i, :]) for i in 1:npix]

    # Error decomposition per pixel: e_i(t) = pred_i(t) - truth_i(t)
    err = pred_v .- truth_v
    err_var = vec(var(err; dims = 2))
    truth_var = vec(var(truth_v; dims = 2))
    snr = truth_var ./ (err_var .+ eps())

    # Cross-pixel error correlation (mean off-diagonal of correlation matrix of errors)
    err_norm = (err .- mean(err; dims = 2)) ./ (std(err; dims = 2) .+ eps())
    err_corr_mat = (err_norm * err_norm') ./ size(err_norm, 2)
    n_pixels = size(err_corr_mat, 1)
    off_mask = .!Matrix{Bool}(I, n_pixels, n_pixels)
    err_corr_mean = mean(err_corr_mat[off_mask])

    println("\n--- $(label) ---")
    @printf("  N3.4 INDEX ACC    : 12mo=%.3f   full=%.3f\n", acc_index, acc_full)
    @printf("  per-pixel ACC 12mo: median=%.3f  mean=%.3f  Q25=%.3f Q75=%.3f\n",
            median(ppacc_12), mean(ppacc_12),
            quantile(ppacc_12, 0.25), quantile(ppacc_12, 0.75))
    @printf("  per-pixel ACC full: median=%.3f  mean=%.3f  Q25=%.3f Q75=%.3f\n",
            median(ppacc_full), mean(ppacc_full),
            quantile(ppacc_full, 0.25), quantile(ppacc_full, 0.75))
    @printf("  per-pixel SNR     : median=%.3f  Q25=%.3f Q75=%.3f\n",
            median(snr), quantile(snr, 0.25), quantile(snr, 0.75))
    @printf("  cross-pixel error correlation (mean off-diag): %.3f\n", err_corr_mean)

    box_results[label] = (ppacc_12 = ppacc_12, ppacc_full = ppacc_full,
                          npix = npix, err_corr = err_corr_mean,
                          acc_index = acc_index)
end

# Plot per-pixel ACC histograms
p_hists = plot(layout = (1, 3), size = (1500, 400), left_margin = 4mm, bottom_margin = 4mm)
for (i, (label, _, _)) in enumerate(ARCHS)
    r = box_results[label]
    histogram!(p_hists[i], r.ppacc_12;
        bins = -1:0.05:1, normalize = :probability, label = false,
        title = "$(label) — pp-ACC@12mo (npix=$(r.npix))",
        xlabel = "per-pixel ACC", ylabel = "fraction",
        color = :steelblue, alpha = 0.7)
    vline!(p_hists[i], [median(r.ppacc_12)]; lw = 2, color = :red,
        label = "median=$(round(median(r.ppacc_12); digits=3))")
end
savefig(p_hists, joinpath(ROOT, "ppacc_distribution.png"))
println("\nSaved ppacc_distribution.png")

# ----------------------------------------------------------------------------
# Hypothesis test: with low per-pixel ACC + low cross-pixel error corr,
# does the index-ACC equal what is predicted by the noise-cancellation model?
# Index ACC ≈ corr(<truth>, <pred>) where < > = spatial mean over npix pixels.
# If errors are independent across pixels with mean per-pixel ACC of ρ_pp,
# the index ACC is bounded by Cov / sqrt(Var) which depends on whether the
# truth signal is spatially coherent (which it IS for Niño 3.4 — it IS a
# coherent SST anomaly).
# ----------------------------------------------------------------------------
println("\n" * "="^72)
println("Sanity: index ACC vs per-pixel-ACC + cross-pixel-error-correlation")
println("="^72)
for (label, _, _) in ARCHS
    r = box_results[label]
    @printf("  %-15s  index_ACC=%.3f   per-pixel ACC median=%.3f   err_corr=%.3f\n",
            label, r.acc_index, median(r.ppacc_12), r.err_corr)
end
