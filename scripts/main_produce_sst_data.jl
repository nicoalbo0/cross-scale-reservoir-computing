# Activate environment
using Pkg, Revise
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using CrossScaleRC, JLD2
using ProgressMeter
using Dates

dir = pwd()

# Native grid 1440 (lon) × 720 (lat) at 0.25°.
# step_regrid ∈ {8, 24, 72}  →  resolution ∈ {2.0°, 6.0°, 18.0°}.
for step_regrid in [8, 24, 72]

    res_deg    = 0.25 * step_regrid
    out_path   = joinpath(dir, "data", "sst", "sst_final_$(res_deg).jld2")

    dates = Date(1982, 1, 1) : Day(1) : Date(2015, 12, 31)

    # First pass: count available per-day regridded files.
    counter = 0
    for d in dates
        yy = lpad(Dates.year(d),  4, '0')
        mm = lpad(Dates.month(d), 2, '0')
        dd = lpad(Dates.day(d),   2, '0')
        isfile(joinpath(dir, "data", "sst", yy, mm, dd, "sst_regridded_$(res_deg).jld2")) && (counter += 1)
    end

    println("\n=== $(res_deg)°:  $(counter) / $(length(dates)) days available ===")
    p = Progress(counter, "Concat $(res_deg)°", 50)

    nlon = convert(Int64, 1440 / step_regrid)
    nlat = convert(Int64, 720  / step_regrid)
    sst  = Array{Float32}(undef, nlon, nlat, counter)

    idx = 1
    for d in dates
        yy = lpad(Dates.year(d),  4, '0')
        mm = lpad(Dates.month(d), 2, '0')
        dd = lpad(Dates.day(d),   2, '0')
        in_path = joinpath(dir, "data", "sst", yy, mm, dd, "sst_regridded_$(res_deg).jld2")
        isfile(in_path) || continue
        f = jldopen(in_path)
        sst[:, :, idx] = f["sst_regridded"]
        close(f)
        idx += 1
        next!(p)
    end

    jldsave(out_path; sst = sst)
    println("Saved: $(out_path)  size=$(size(sst))")
end
