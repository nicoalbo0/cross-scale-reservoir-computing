"""
    load_data(Q, L, μ; show_data=false, refinement)

Load Kuramoto–Sivashinsky (KS) data. Dispatches to `load_kuramoto_data`.
"""
load_data(Q::Int, L::Int, μ::Real; show_data::Bool=false, refinement::Int) = load_kuramoto_data(Q, L, μ; show_data=show_data, refinement=refinement)

"""
    load_data(res; show_data=false, refinement, lon_range=nothing, lat_range=nothing, anomalies=false, train_indices=nothing)

Load SST (sea surface temperature) data for given resolution(s). Dispatches to `load_sst_data`.
"""
load_data(res::Vector{Float64}; show_data::Bool=false, refinement::Int,
          lon_range=nothing, lat_range=nothing,
          anomalies::Bool=false, train_indices=nothing) =
    load_sst_data(res; show_data=show_data, refinement=refinement,
                  lon_range=lon_range, lat_range=lat_range,
                  anomalies=anomalies, train_indices=train_indices)

"""
    load_kuramoto_data(Q, L, μ; show_data=false, refinement)

Load Kuramoto–Sivashinsky data from CSV. Data is normalized (zero mean, unit variance).
Optionally interpolated in time by `refinement`. Returns `(data, dt)` where `data` is (L×T).
"""
function load_kuramoto_data(Q::Int, L::Int, μ::T; show_data::Bool=false, refinement::Int) where T<:Real

    filepath = pwd()*"/data/kuramoto/Q$(Q)_L$(L)_mu$(μ)_ks_data.csv"
    dt = 0.25

    f = Matrix(CSV.read("$filepath", DataFrame))';
    data = (f .- mean(f,dims=2)) ./ std(f, dims=2)

    if refinement > 1
        data = cubic_time_interpolate(data, dt, refinement)
    end
        
    if show_data
        p = heatmap(data[:, 1:5000]; title="KS data sample", xlabel="t", ylabel="row (state)")
        display(p)
    end
    
    return data, dt / refinement

end

"""
    load_sst_data(resolutions_vec; show_data=false, refinement, lon_range=nothing, lat_range=nothing, anomalies=false, train_indices=nothing)

Load SST data for each resolution in `resolutions_vec`. Uses cleaned JLD2 files
(`sst_clean_<res>.jld2`); creates them from `sst_final_<res>.jld2` if needed.
Temporal/spatial cleaning and normalization applied. Returns `(data_vec, dt)` where
`data_vec` is a vector of (nlon×nlat×nt) arrays.

# Keyword arguments
- `lon_range`: `(lon_min, lon_max)` in [0, 360) to crop longitude. `nothing` = global.
- `lat_range`: `(lat_min, lat_max)` in [-90, 90] to crop latitude. `nothing` = global.
- `anomalies`: if `true`, subtract the seasonal climatology (day-of-year mean computed
  over `train_indices`) from the full time series before normalization.
- `train_indices`: index range used to compute the climatology (required when `anomalies=true`).
"""
function load_sst_data(resolutions_vec::Vector{T};
                       show_data::Bool=false,
                       refinement::Int,
                       lon_range=nothing,
                       lat_range=nothing,
                       anomalies::Bool=false,
                       train_indices=nothing) where T<:Real

    if anomalies && isnothing(train_indices)
        error("train_indices must be provided when anomalies=true")
    end

    dir = pwd()*"/data/sst"

    data_vec = Vector{Array{Float64, 3}}(undef, length(resolutions_vec))

    for (res_i, res) in enumerate(resolutions_vec)

        if !isfile(dir*"/sst_clean_$(res).jld2")

            f = jldopen(dir*"/sst_final_$(res).jld2")
            data = f["sst"]
            close(f)

            nlon, nlat, nt = size(data)
            nan_threshold = nt * 0.15

            # temporal cleaning
            for i in axes(data,1), j in axes(data,2)

                row = @view data[i, j, :]

                if count(isnan, row) > nan_threshold
                    row .= NaN
                else
                    times = findall(!isnan, row)

                    if !isempty(times)
                        t = LinRange(0, 1, length(times))
                        vals = row[times]

                        itp = Interpolations.scale(
                            interpolate(hcat(times, vals),
                                (BSpline(Cubic(Natural(OnGrid()))), NoInterp())),
                            t, 1:2
                        )

                        ti = LinRange(0,1,nt)
                        row .= [itp(ti[k], 2) for k in eachindex(ti)]
                    end
                end
            end

            # spatial cleaning ----
            for i in 2:nlon-1, j in 2:nlat-1

                counter = 0
                for ii in i-1:i+1, jj in j-1:j+1
                    if isnan(data[ii, jj, 1])
                        counter += 1
                    end
                end

                if counter >= 7
                    data[i,j,:] .= NaN
                end
            end

            @assert size(data) == (nlon, nlat, nt)

            jldsave(dir*"/sst_clean_$(res).jld2", sst=data)

        else

            f = jldopen(dir*"/sst_clean_$(res).jld2")
            data = f["sst"]
            close(f)
        end

        nlon, nlat, nt = size(data)

        # --- geographic crop ---
        lon_idxs, lat_idxs = _domain_indices(nlon, nlat, res, lon_range, lat_range)
        data = data[lon_idxs, lat_idxs, :]
        nlon, nlat, nt = size(data)

        # --- seasonal anomaly removal (on native daily data, before time interpolation) ---
        # train_indices must be expressed in units of the daily (pre-refinement) time series.
        if anomalies
            data = _remove_climatology(data, train_indices)
        end

        if show_data

            p = heatmap(data[:, :, 1]', title="Sample data (res=$(res)°)",
            framestyle=:box, colorbar_title="\nCounts",
            xlabel="Longitude", ylabel="Latitude",
            right_margin=5mm)
            display(p)

        end

        if refinement > 1
            data = reshape(data, nlon * nlat, nt)
            data = cubic_time_interpolate(data, 1.0, refinement)
            nt = size(data, 2)
            data = reshape(data, nlon, nlat, nt)
        end

        data = reshape(data, nlon * nlat, size(data, 3))
        data = (data .- mean(data, dims=2)) ./ (std(data, dims=2))
        data = reshape(data, nlon, nlat, size(data, 2))
        data[isnan.(data)] .= 0.0

        data_vec[res_i] = data
    end

    return data_vec, 1 / refinement

end

"""
    _domain_indices(nlon, nlat, res, lon_range, lat_range) -> (lon_idxs, lat_idxs)

Compute integer index ranges into the global SST grid (0.25° origin at -179.875° lon,
-89.875° lat, regridded to `res`°) corresponding to `lon_range` (in [0,360)) and
`lat_range` (in [-90,90]). Returns `(:, :)` (full range) when the respective range
argument is `nothing`.
"""
function _domain_indices(nlon::Int, nlat::Int, res::Real, lon_range, lat_range)
    # Grid origin: first cell centred at -179.875° lon, -89.875° lat (0.25° native grid)
    # At resolution `res`, cell centres start at (-180 + res/2) and step by `res`.
    lon_start = -180.0 + res / 2
    lat_start = -90.0  + res / 2

    lons = lon_start .+ res .* (0:nlon-1)   # in [-180, 180)
    lats = lat_start .+ res .* (0:nlat-1)   # in [-90, 90)

    # Convert lons to [0, 360) for comparison with lon_range
    lons360 = mod.(lons, 360.0)

    if isnothing(lon_range)
        lon_idxs = 1:nlon
    else
        lo, hi = lon_range
        if lo <= hi
            lon_idxs = findall(lo .<= lons360 .<= hi)
        else
            # wraps around 0°, e.g. (300, 60)
            lon_idxs = findall(lons360 .>= lo .|| lons360 .<= hi)
        end
        isempty(lon_idxs) && error("lon_range=$(lon_range) produced no grid cells at res=$(res)°")
    end

    if isnothing(lat_range)
        lat_idxs = 1:nlat
    else
        lo, hi = lat_range
        lat_idxs = findall(lo .<= lats .<= hi)
        isempty(lat_idxs) && error("lat_range=$(lat_range) produced no grid cells at res=$(res)°")
    end

    return lon_idxs, lat_idxs
end

"""
    _remove_climatology(data, train_indices) -> Array{Float64,3}

Subtract the seasonal climatology from `data` (nlon × nlat × nt).

The climatology is the mean SST for each day-of-year (1–365), computed using only
the time steps in `train_indices`. Leap-day handling: day 366 is mapped to day 365.
"""
function _remove_climatology(data::Array{<:Real, 3}, train_indices)
    nlon, nlat, nt = size(data)
    out = copy(data)

    # Accumulate sum and count per doy (1-365) over training period
    clim_sum   = zeros(Float64, nlon, nlat, 365)
    clim_count = zeros(Int,     nlon, nlat, 365)

    for t in train_indices
        doy = min(mod1(t, 365), 365)   # maps t → 1:365 (simple cyclic; works for daily data)
        @inbounds for j in 1:nlat, i in 1:nlon
            if !isnan(data[i, j, t])
                clim_sum[i, j, doy]   += data[i, j, t]
                clim_count[i, j, doy] += 1
            end
        end
    end

    clim = clim_sum ./ max.(clim_count, 1)   # mean; avoid div-by-zero

    # Subtract climatology from all time steps
    for t in 1:nt
        doy = min(mod1(t, 365), 365)
        @inbounds out[:, :, t] .-= clim[:, :, doy]
    end

    return out
end