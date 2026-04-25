# Sanity check for the ENSO climatology fix.
#
# Plots, for a single Niño 3.4 pixel, the raw SST with (a) the fixed
# climatology computed via Dates.dayofyear and (b) the old biased
# climatology computed via mod1(t, 365). Also plots the resulting
# anomalies and the seasonal-cycle curves themselves. A correct
# climatology should sit on the seasonal envelope of the raw SST and
# produce a zero-mean anomaly series with no residual annual cycle.

ENV["GKSwstype"] = "100"
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JLD2, Dates, Statistics, Plots, Measures

println("Loading sst_final_2.0.jld2 ...")
f = jldopen(joinpath(@__DIR__, "..", "data", "sst", "sst_final_2.0.jld2"))
sst = f["sst"]
close(f)

nlon, nlat, nt = size(sst)
println("Shape: ($nlon × $nlat × $nt)")

# 2° grid centres
lons_all = -179.0 .+ 2.0 .* (0:nlon-1)
lats_all = -89.0  .+ 2.0 .* (0:nlat-1)

# Pick a pixel in the Niño 3.4 region (5°N-5°S, 170°W-120°W).
# Target: 150°W (= 210° in [0,360)), 0°N. Find nearest grid centre that is
# not land (non-NaN in the first time slice).
i_target = argmin(abs.(mod.(lons_all, 360.0) .- 210.0))
j_target = argmin(abs.(lats_all .- 0.0))

function first_valid_pixel(sst, i0, j0, nlon, nlat)
    for δ in 0:5, si in (-1, 1), sj in (-1, 1), δi in 0:δ, δj in 0:δ
        ii = clamp(i0 + si*δi, 1, nlon)
        jj = clamp(j0 + sj*δj, 1, nlat)
        !isnan(sst[ii, jj, 1]) && return (ii, jj)
    end
    return (i0, j0)
end
i_pix, j_pix = first_valid_pixel(sst, i_target, j_target, nlon, nlat)
println("Pixel: lon=$(lons_all[i_pix])°E   lat=$(lats_all[j_pix])°N")

pixel = Float64.(sst[i_pix, j_pix, :])

train_indices = 1:7500
start_date    = Date(1982, 1, 1)

# --- fixed climatology (Dates.dayofyear; 366 bins) ---
clim_fixed = zeros(366)
cnt_fixed  = zeros(Int, 366)
for t in train_indices
    doy = dayofyear(start_date + Day(t - 1))
    if !isnan(pixel[t])
        clim_fixed[doy] += pixel[t]
        cnt_fixed[doy]  += 1
    end
end
clim_fixed ./= max.(cnt_fixed, 1)
if cnt_fixed[366] == 0 && cnt_fixed[365] > 0
    clim_fixed[366] = clim_fixed[365]
end

# --- old biased climatology (mod1(t, 365)) ---
clim_old = zeros(365)
cnt_old  = zeros(Int, 365)
for t in train_indices
    doy = mod1(t, 365)
    if !isnan(pixel[t])
        clim_old[doy] += pixel[t]
        cnt_old[doy]  += 1
    end
end
clim_old ./= max.(cnt_old, 1)

# --- anomaly series both ways ---
anom_fixed = similar(pixel)
anom_old   = similar(pixel)
for t in 1:nt
    anom_fixed[t] = pixel[t] - clim_fixed[dayofyear(start_date + Day(t - 1))]
    anom_old[t]   = pixel[t] - clim_old[mod1(t, 365)]
end

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
window = 1:1100   # ~3 years, spans a leap-day cycle

p1 = plot(window, pixel[window], label="Raw SST", color=:black, lw=1.5,
          xlabel="day since 1982-01-01", ylabel="SST (K)",
          title="Seasonal cycle fit — lon=$(lons_all[i_pix])°E, lat=$(lats_all[j_pix])°N")
fitted_fixed = [clim_fixed[dayofyear(start_date + Day(t - 1))] for t in window]
fitted_old   = [clim_old[mod1(t, 365)] for t in window]
plot!(p1, window, fitted_fixed, label="Climatology (fixed)",     color=:green, lw=2)
plot!(p1, window, fitted_old,   label="Climatology (old mod1)",  color=:red,   lw=1.5, ls=:dash)

p2 = plot(1:365, clim_fixed[1:365], label="Fixed (Dates.dayofyear)", color=:green, lw=2,
          xlabel="day of year (1-365)", ylabel="SST (K)",
          title="Seasonal cycle — mean per doy")
plot!(p2, 1:365, clim_old, label="Old (mod1)", color=:red, ls=:dash, lw=1.5)

p3 = plot(window, anom_fixed[window], label="Anomaly (fixed)", color=:green, lw=1.5,
          xlabel="day", ylabel="SST anomaly (K)",
          title="Residual after removing seasonal cycle  (first 3 yr)")
plot!(p3, window, anom_old[window], label="Anomaly (old)", color=:red, ls=:dash, lw=1)
hline!(p3, [0.0], color=:black, lw=0.5, label=false)

p4 = plot(1:nt, anom_fixed, label="Fixed", color=:green, alpha=0.8, lw=0.5,
          xlabel="day since 1982-01-01", ylabel="SST anomaly (K)",
          title="Full anomaly series 1982–2015")
plot!(p4, 1:nt, anom_old, label="Old (mod1)", color=:red, alpha=0.5, lw=0.5)
vline!(p4, [train_indices[end]], color=:gray, ls=:dot, lw=1, label="end of train window")

final = plot(p1, p2, p3, p4, layout=(2, 2), size=(1400, 900),
             plot_title="Climatology sanity check — $(lons_all[i_pix])°E, $(lats_all[j_pix])°N",
             left_margin=3mm, bottom_margin=3mm)
mkpath(joinpath(@__DIR__, "..", "results"))
out = joinpath(@__DIR__, "..", "results", "climatology_sanity.png")
savefig(final, out)
println("Saved: $out")

# Numerical sanity: on the training window the fixed anomaly mean should be ≈ 0
# (exactly 0 if every doy has equal sample count, near 0 otherwise).
mean_fixed = mean(anom_fixed[train_indices])
mean_old   = mean(anom_old[train_indices])
std_fixed  = std(anom_fixed[train_indices])
std_old    = std(anom_old[train_indices])

# Also: zero-lag autocorrelation at 365 days (should be near 0 if seasonal
# cycle is cleanly removed; residual cycle produces a strong peak).
function ac(x, lag)
    n = length(x)
    x̄ = mean(x)
    num = sum((x[1:n-lag] .- x̄) .* (x[lag+1:n] .- x̄))
    den = sum((x .- x̄) .^ 2)
    return num / den
end
ac_fixed_365 = ac(anom_fixed[1:7500], 365)
ac_old_365   = ac(anom_old[1:7500], 365)

println()
println("Train-window anomaly stats:")
println("  Fixed : mean=$(round(mean_fixed; digits=5))   std=$(round(std_fixed; digits=3))   AC(365)=$(round(ac_fixed_365; digits=3))")
println("  Old   : mean=$(round(mean_old;   digits=5))   std=$(round(std_old;   digits=3))   AC(365)=$(round(ac_old_365;   digits=3))")
println()
println("A cleanly-removed seasonal cycle has AC(365) close to ENSO autocorrelation (~0.5)")
println("but without the residual annual peak. A biased climatology leaves a strong AC(365).")
