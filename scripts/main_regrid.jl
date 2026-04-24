# Activate environment
using Pkg, Revise
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using CrossScaleRC
using ProgressMeter
using Dates

dir = pwd()

scale_factor = 0.01
add_offset   = 273.15
valid_min    = -300
valid_max    = 4500
range_phys   = (valid_min*scale_factor + add_offset, valid_max*scale_factor + add_offset)   # (270.15, 318.15) K

# Native grid 1440 × 720 at 0.25°.
# step_regrid ∈ {8, 24, 72}  →  resolution ∈ {2.0°, 6.0°, 18.0°}
# All three are required by main_enso.jl.
for step_regrid in [8, 24, 72]

    res_deg = 0.25 * step_regrid
    println("\n=== Regridding to $(res_deg)°  (step=$(step_regrid)) ===")

    # enumerate calendar days 1982-01-01 .. 2015-12-31 honouring month lengths
    dates = Date(1982, 1, 1) : Day(1) : Date(2015, 12, 31)
    p = Progress(length(dates), "Regridding $(res_deg)°", 50)

    for d in dates
        year  = lpad(Dates.year(d),  4, '0')
        month = lpad(Dates.month(d), 2, '0')
        day   = lpad(Dates.day(d),   2, '0')
        read_and_regrid(dir, step_regrid, day, month, year,
                        scale_factor, add_offset, range_phys)
        next!(p)
    end
end
