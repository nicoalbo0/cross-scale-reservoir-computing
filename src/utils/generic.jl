"""
    input_dimensions(blocks) -> (rec_dim, neigh_dim, layer_dim)

Return consistent input dimensions from the first block; assert all blocks have the same
`length(rows_rec)`, `length(rows_neigh)`, and `length(rows_layer)`.
"""
function input_dimensions(blocks::Vector{<:NamedTuple})
    isempty(blocks) && error("blocks must not be empty")

    rec_dim   = length(blocks[1].rows_rec)
    neigh_dim = length(blocks[1].rows_neigh)
    layer_dim = length(blocks[1].rows_layer)

    @inbounds for (i, b) in pairs(blocks)
        length(b.rows_rec)   == rec_dim   || error("Inconsistent rec_dim at block $i")
        length(b.rows_neigh) == neigh_dim || error("Inconsistent neigh_dim at block $i")
        length(b.rows_layer) == layer_dim || error("Inconsistent layer_dim at block $i")
    end

    return (rec_dim, neigh_dim, layer_dim)
end

"""
    cubic_time_interpolate(data, dt, refinement)

Temporally interpolate `data` (L×T) with cubic splines. New time step is `dt/refinement`;
returns (L×T_new) matrix.
"""
function cubic_time_interpolate(data::Matrix{T}, dt::T, refinement::Int) where T<:Real
    L, t = size(data)

    t_coarse = (0:t-1) .* dt

    dt_fine = dt / refinement
    t_fine = collect(0:dt_fine:(t-1)*dt)

    data_interp = zeros(T, L, length(t_fine))

    for i in 1:L
        itp = cubic_spline_interpolation(
            t_coarse,
            data[i, :];
            extrapolation_bc = Line()  # safe for edges
        )

        @inbounds data_interp[i, :] .= itp.(t_fine)
    end

    return data_interp
end

"""
    regrid_average(data, divisor) -> Matrix

Spatially coarse-grain `data` by averaging over blocks of size `divisor`
along the first (spatial) dimension.

- `data`: (L × T) matrix.
- `divisor`: integer divisor of L.
- **Returns**: (L/divisor × T) matrix.

Each coarse cell is the mean of `divisor` contiguous fine cells.
"""
function regrid_average(data::AbstractMatrix{T}, divisor::Int) where T<:Real
    L, Tlen = size(data)

    @assert L % divisor == 0 "div must divide spatial dimension L exactly"

    Lc = div(L, divisor)
    out = zeros(T, Lc, Tlen)

    @inbounds for i in 1:Lc
        r = (i-1)*divisor + 1 : i*divisor
        out[i, :] .= vec(sum(@views(data[r, :]), dims=1)) ./ divisor
    end

    return out
end

"""
    layer_params(params, l)

Extract layer-`l` parameters from a tuple of parameter vectors (e.g. for multi-layer RC,
where `params` is an 8-tuple of vectors: N, g, degree, g_in_rec, g_in_neigh, g_in_layer, τ, dt).

# Example
```julia
params = (res_size, radius, degree, g_in_rec, g_in_neigh, g_in_layer, τ, dt)  # each a vector per layer
layer_params(params, 2)  # 8-tuple of scalars for layer 2
```
"""
@inline function layer_params(params::Tuple, l::Int)
    return ntuple(i -> params[i][l], length(params))
end

"""
    square_even_rows(XX)

Return a copy of `XX` with even rows (2, 4, ...) squared. Used for quadratic readout features.
"""
function square_even_rows(XX::Matrix{T}) where T<:Real

    X = copy(XX)

    for i in 2:2:size(X, 1)
        for j in axes(X, 2)
            X[i, j] = X[i, j]^2
        end
    end
    return X
end

"""
    nino34_index(sst_3d, lons, lats) -> Vector{Float64}

Compute the Niño 3.4 index: spatial mean SST (anomaly) over 5°S–5°N, 170°W–120°W
(190°–240° in [0,360) convention) for each time step.

- `sst_3d`: (nlon × nlat × nt) array of SST or SST anomalies.
- `lons`: vector of length nlon with cell-centre longitudes in [-180, 180) or [0, 360).
- `lats`: vector of length nlat with cell-centre latitudes in [-90, 90].

Returns a vector of length nt.
"""
function nino34_index(sst_3d::Array{<:Real, 3}, lons::AbstractVector, lats::AbstractVector)
    # Niño 3.4 box: 170°W–120°W, 5°S–5°N
    # In [0,360): 190°–240°; in [-180,180): -170°–-120°
    lons360 = mod.(lons, 360.0)
    lon_mask = 190.0 .<= lons360 .<= 240.0
    lat_mask = -5.0  .<= lats   .<= 5.0

    any(lon_mask) || error("No longitude grid cells fall in the Niño 3.4 box (170W–120W). Check lons.")
    any(lat_mask) || error("No latitude grid cells fall in the Niño 3.4 box (5S–5N). Check lats.")

    region = sst_3d[lon_mask, lat_mask, :]           # (n_lo × n_la × nt)
    n_lo, n_la, nt = size(region)
    index = Vector{Float64}(undef, nt)
    for t in 1:nt
        slice = @view region[:, :, t]
        valid = filter(!isnan, vec(slice))
        index[t] = isempty(valid) ? NaN : mean(valid)
    end
    return index
end

"""
    sst_grid_coords(nlon, nlat, res) -> (lons, lats)

Return the cell-centre longitude and latitude vectors for a global SST grid at
resolution `res`°, consistent with the Copernicus 0.25° native grid (first centre at
-179.875° lon, -89.875° lat).
"""
function sst_grid_coords(nlon::Int, nlat::Int, res::Real)
    lon_start = -180.0 + res / 2
    lat_start = -90.0  + res / 2
    lons = lon_start .+ res .* (0:nlon-1)
    lats = lat_start .+ res .* (0:nlat-1)
    return lons, lats
end

"""
    skill_score(index_true, index_pred) -> NamedTuple

Compute ENSO forecast skill scores for the Niño 3.4 index (or any scalar time series).

Returns a NamedTuple with:
- `acc`: Anomaly Correlation Coefficient (Pearson r between true and predicted).
- `rmse`: Root mean square error.
- `rmse_skill`: RMSE skill score relative to a persistence baseline (lag-1):
  `1 - RMSE_pred / RMSE_persistence`. Positive = better than persistence.
"""
function skill_score(index_true::AbstractVector, index_pred::AbstractVector)
    @assert length(index_true) == length(index_pred) "index_true and index_pred must have the same length"
    n = length(index_true)
    n >= 2 || error("Need at least 2 time steps for skill_score")

    μ_t = mean(index_true)
    μ_p = mean(index_pred)
    σ_t = std(index_true)
    σ_p = std(index_pred)

    acc  = (σ_t > 0 && σ_p > 0) ?
           mean((index_true .- μ_t) .* (index_pred .- μ_p)) / (σ_t * σ_p) : NaN

    rmse_val = sqrt(mean((index_true .- index_pred).^2))

    # Persistence baseline: predict index_true[t] = index_true[t-1]
    rmse_pers = sqrt(mean((index_true[2:end] .- index_true[1:end-1]).^2))
    rmse_skill = 1.0 - rmse_val / rmse_pers

    return (acc=acc, rmse=rmse_val, rmse_skill=rmse_skill)
end

"""
    per_pixel_acc(field_true_3d, field_pred_3d) -> Matrix{Float64}

Anomaly correlation coefficient at each (lon, lat) pixel between the truth and
forecast 3D arrays of shape `(nlon, nlat, nt)`. Returns an `(nlon × nlat)` matrix.

NaN at pixels with zero temporal variance in either input (e.g. land mask after
zero-fill); callers should mask those out before averaging.
"""
function per_pixel_acc(field_true::AbstractArray{<:Real, 3},
                       field_pred::AbstractArray{<:Real, 3})
    @assert size(field_true) == size(field_pred) "field_true and field_pred must match"
    nlon, nlat, _ = size(field_true)
    out = fill(NaN, nlon, nlat)
    @inbounds for i in 1:nlon, j in 1:nlat
        t = @view field_true[i, j, :]
        p = @view field_pred[i, j, :]
        σt = std(t); σp = std(p)
        if σt > 0 && σp > 0
            μt = mean(t); μp = mean(p)
            out[i, j] = mean((t .- μt) .* (p .- μp)) / (σt * σp)
        end
    end
    return out
end

"""
    forecast_headline(n34_true, n34_pred, field_true_3d, field_pred_3d, lons, lats; nino_box_only=false)
        -> NamedTuple

Compact 7-tuple summary used by the Stage E comparison pipeline:
- `acc12`, `rmse12`, `std_ratio12` — 12-month-cumulative Niño 3.4 metrics
- `pc3`, `pc12` — spatial pattern correlations at 3 and 12 months
- `ppacc_n34_mean` — per-pixel ACC averaged inside the Niño 3.4 box (190°–240°E, ±5°)
- `ppacc_global_mean` — per-pixel ACC averaged over all non-NaN pixels

NaN pixels (land or zero-variance) are excluded from the means.
"""
function forecast_headline(n34_true::AbstractVector,
                           n34_pred::AbstractVector,
                           field_true::AbstractArray{<:Real, 3},
                           field_pred::AbstractArray{<:Real, 3},
                           lons::AbstractVector,
                           lats::AbstractVector)
    L12 = min(12, length(n34_true))
    L3  = min(3,  length(n34_true))
    s12 = skill_score(n34_true[1:L12], n34_pred[1:L12])
    sr12 = std(n34_pred[1:L12]) / std(n34_true[1:L12])

    function pc_at(t)
        a = vec(@view field_true[:, :, t])
        b = vec(@view field_pred[:, :, t])
        am = a .- mean(a); bm = b .- mean(b)
        return dot(am, bm) / (norm(am) * norm(bm) + eps())
    end
    pc3  = L3  > 0 && L3  ≤ size(field_true, 3) ? pc_at(L3)  : NaN
    pc12 = L12 > 0 && L12 ≤ size(field_true, 3) ? pc_at(L12) : NaN

    ppacc = per_pixel_acc(field_true, field_pred)

    lons360 = mod.(lons, 360.0)
    lon_mask = 190.0 .<= lons360 .<= 240.0
    lat_mask = -5.0  .<= lats   .<= 5.0
    n34_box = ppacc[lon_mask, lat_mask]
    n34_valid = filter(!isnan, vec(n34_box))
    global_valid = filter(!isnan, vec(ppacc))

    return (acc12 = s12.acc, rmse12 = s12.rmse, std_ratio12 = sr12,
            pc3 = pc3, pc12 = pc12,
            ppacc_n34_mean    = isempty(n34_valid)    ? NaN : mean(n34_valid),
            ppacc_global_mean = isempty(global_valid) ? NaN : mean(global_valid),
            ppacc_map = ppacc)
end

"""
    rmse_upto(data, pred; T=size(data, 2), coords=axes(data, 1))

Compute RMSE over the first `T` time steps and over rows indexed by `coords`.
`data` and `pred` must have the same size (N × Tfull); rows are coordinates, columns are time.

- `T`: number of time samples (from the start) to include in the RMSE.
- `coords`: row indices (coordinates) to include.
"""
function rmse_upto(data::AbstractMatrix, pred::AbstractMatrix;
                   T::Integer = size(data, 2),
                   coords = axes(data, 1))

    @assert size(data) == size(pred)
    T′ = min(T, size(data, 2))
    ncoords = length(coords)
    @assert ncoords > 0 && T′ > 0

    S = promote_type(eltype(data), eltype(pred))
    sse = zero(S)

    @inbounds for t in 1:T′
        for i in coords
            δ = data[i, t] - pred[i, t]
            sse += δ * δ
        end
    end

    return sqrt(sse / (ncoords * T′))
end