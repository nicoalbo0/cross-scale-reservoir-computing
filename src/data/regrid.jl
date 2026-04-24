"""
    read_and_regrid(base_dir, step_regrid, day, month, year, scale_factor, add_offset, valid_range)

Read the daily ESA-CCI GMPE SST NetCDF under
`{base_dir}/data/sst/{year}/{month}/{day}/` and write a block-averaged
array to `sst_regridded_{0.25*step_regrid}.jld2` (key `sst_regridded`).

Spatial grid: 1440×720 at 0.25° → (1440/step × 720/step). Invalid pixels
(outside `valid_range` in Kelvin, or NetCDF `missing`) become `NaN`; each
coarse cell is the mean of the valid (non-NaN) fine cells it contains (all-NaN
cell → NaN). Skips silently if the output file already exists.

`scale_factor`, `add_offset`, and `valid_range` are accepted for call-site
compatibility — NCDatasets already applies `scale_factor`/`add_offset` on
read, and `valid_range` is used for range masking in physical (Kelvin) units.
"""
function read_and_regrid(base_dir::AbstractString, step_regrid::Int,
                         day::AbstractString, month::AbstractString, year::AbstractString,
                         scale_factor::Real, add_offset::Real, valid_range::Tuple)

    res_deg = 0.25 * step_regrid
    day_dir = joinpath(base_dir, "data", "sst", year, month, day)
    out_path = joinpath(day_dir, "sst_regridded_$(res_deg).jld2")

    isfile(out_path) && return nothing
    isdir(day_dir)   || return nothing

    ncs = filter(f -> endswith(f, ".nc"), readdir(day_dir))
    isempty(ncs) && return nothing
    nc_path = joinpath(day_dir, ncs[1])

    raw = NCDataset(nc_path) do ds
        Array(ds["analysed_sst"][:, :, 1])   # (1440, 720) in Kelvin, missing → `missing`
    end

    lo, hi = valid_range
    data = Matrix{Float64}(undef, size(raw)...)
    @inbounds for i in eachindex(raw)
        v = raw[i]
        data[i] = (ismissing(v) || v < lo || v > hi) ? NaN : Float64(v)
    end

    nlon_f, nlat_f = size(data)
    @assert nlon_f % step_regrid == 0 && nlat_f % step_regrid == 0
    nlon_c = nlon_f ÷ step_regrid
    nlat_c = nlat_f ÷ step_regrid

    coarse = Array{Float32}(undef, nlon_c, nlat_c)
    @inbounds for j in 1:nlat_c, i in 1:nlon_c
        i0 = (i - 1) * step_regrid + 1
        j0 = (j - 1) * step_regrid + 1
        acc = 0.0
        n   = 0
        for jj in j0 : j0 + step_regrid - 1, ii in i0 : i0 + step_regrid - 1
            x = data[ii, jj]
            if !isnan(x)
                acc += x
                n   += 1
            end
        end
        coarse[i, j] = n > 0 ? Float32(acc / n) : NaN32
    end

    jldsave(out_path; sst_regridded = coarse)
    return nothing
end
