# Activate environment
using Pkg, Revise
Pkg.activate("/../.")
Pkg.instantiate()

using CrossScaleRC
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
range = (valid_min*scale_factor+add_offset, valid_max*scale_factor+add_offset) #(270.15, 318.15)

# fill_value = -32768
# lon (0.25°) x lat (0.25°) x day
# lon: -179.875, -179.625, ..., 179.625, 179.875
# lat: -89.875, -89.625,..., 89.625, 89.875
# time: 1
# 1440 x 720 x 1
# lon x lat x time

for step_regrid in [8; 72] #[1; 2; 4; 6; 12; 18; 24; 36; 48; 60] 

    p = Progress(length(years_vec)*length(months_vec)*length(days_vec), "Regridding $(step_regrid*0.25)°...", 50)

    for year in years_vec
        for month in months_vec
            for day in days_vec
                read_and_regrid(dir, step_regrid, day, month, year, scale_factor, add_offset, range)
                next!(p)
            end
        end
    end

end

