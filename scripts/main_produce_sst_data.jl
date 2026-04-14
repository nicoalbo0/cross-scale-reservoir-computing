# Activate environment
using Pkg, Revise
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using CrossScaleRC, JLD2
using ProgressMeter

dir = pwd()

years_vec = ["$(i)" for i in 1982:2016]
months_vec = ["$(i)" for i in 1:12]
months_vec[1:9] = ["0$(i)" for i in 1:9]
days_vec = ["$(i)" for i in 1:31]
days_vec[1:9] = ["0$(i)" for i in 1:9]


scale_factor = 0.01
add_offset = 273.15
valid_min = -300
valid_max = 4500
range = (valid_min*scale_factor+add_offset, valid_max*scale_factor+add_offset)

#fill_value = -32768
# lon (0.25°) x lat (0.25°) x day
# lon: -179.875, -179.625, ..., 179.625, 179.875
# lat: -89.875, -89.625,..., 89.625, 89.875
# time: 1
# 1440 x 720 x 1 -> 240 x 120 x 1
# lon x lat x time

for step_regrid in [8; 72] #[1; 2; 4; 6; 12; 18; 24; 36; 48; 60]

    counter = 0
    for year in years_vec
        for month in months_vec
            for day in days_vec
                if isfile(dir*"/data/sst/$(year)/$(month)/$(day)/sst_regridded_$(0.25*step_regrid).jld2")
                    counter += 1
                end
            end
        end
    end

    println("$(counter) of days in dataset.")
    p = Progress(counter, "Concat data $(step_regrid*0.25)...", 50)

    lat = convert(Int64, 1440 / step_regrid)
    lon = convert(Int64, 720 / step_regrid)
    sst = Array{Float32}(undef, lat, lon, counter)

    counter_start = 1
    for year in years_vec
        for month in months_vec
            for day in days_vec
                if isfile(dir*"/data/sst/$(year)/$(month)/$(day)/sst_regridded_$(0.25*step_regrid).jld2")
                    f = jldopen(dir*"/data/sst/$(year)/$(month)/$(day)/sst_regridded_$(0.25*step_regrid).jld2")
                    sst_tmp = f["sst_regridded"]
                    sst[:,:,counter_start] = sst_tmp
                    counter_start += 1
                    close(f)
                    next!(p)
                end
            end
        end
    end

    jldsave(dir*"/data/sst/sst_final_$(0.25*step_regrid).jld2", sst=sst)
end

