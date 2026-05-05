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

    # Per-pixel ACC over the first 12 months (standard ENSO lead). Beyond ~18 mo
    # individual pixels diverge from truth even when the spatial-mean index
    # remains correlated, so a full-window ppacc is uninformative.
    L12_field = min(12, size(field_true, 3))
    ppacc = per_pixel_acc(@view(field_true[:, :, 1:L12_field]),
                          @view(field_pred[:, :, 1:L12_field]))

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

# ============================================================================
# Stage G — event-pattern skill metrics
# ============================================================================

"""
    find_enso_events(n34::AbstractVector;
                     n_events = 6, threshold = 1.0, min_separation = 6) -> Vector{Int}

Greedy event detector for the ENSO Niño 3.4 index. Picks the strongest
|N3.4| local maxima with at least `min_separation` months between them, up to
`n_events` total. Only events with |n34| ≥ `threshold` are considered.

Reproduces the post-hoc detection rule from `scripts/plot_event_aligned.jl`,
formalized for use in `event_skill`.
"""
function find_enso_events(n34::AbstractVector{<:Real};
                          n_events::Integer = 6,
                          threshold::Real = 1.0,
                          min_separation::Integer = 6)
    order = sortperm(abs.(n34); rev = true)
    events = Int[]
    for i in order
        abs(n34[i]) ≥ threshold || continue
        all(abs(i - e) > min_separation for e in events) || continue
        push!(events, i)
        length(events) ≥ n_events && break
    end
    sort!(events)
    return events
end

"""
    phase_aligned_pc(field_true_3d, field_pred_3d, t_event;
                     phase_tol = 3) -> (best_pc, best_offset)

Pattern correlation between `field_true_3d[:, :, t_event]` and the
forecast at any time `t_event + δ` for `δ ∈ -phase_tol:phase_tol`,
returning the maximum (best_pc) and the offset that achieves it. Allows
the forecast a phase tolerance of ±`phase_tol` months — a forecast that
predicts the event a quarter early or late still scores as a correct
spatial pattern, but a year off does not.
"""
function phase_aligned_pc(field_true::AbstractArray{<:Real, 3},
                          field_pred::AbstractArray{<:Real, 3},
                          t_event::Integer;
                          phase_tol::Integer = 3)
    nt = size(field_true, 3)
    @assert size(field_pred, 3) == nt "true / pred time-axis mismatch"
    truth_t = vec(@view field_true[:, :, t_event])
    am = truth_t .- mean(truth_t)
    norm_am = norm(am)
    best_pc = -Inf
    best_off = 0
    for δ in -phase_tol:phase_tol
        t = t_event + δ
        (1 ≤ t ≤ nt) || continue
        pred_t = vec(@view field_pred[:, :, t])
        bm = pred_t .- mean(pred_t)
        denom = norm_am * norm(bm) + eps()
        pc = dot(am, bm) / denom
        if pc > best_pc
            best_pc = pc
            best_off = δ
        end
    end
    return (best_pc = best_pc, best_offset = best_off)
end

"""
    event_skill(n34_true, field_true, n34_pred, field_pred;
                lead_window=(10, 20), phase_tol=3,
                event_threshold=1.0, min_separation=6,
                false_alarm_threshold=1.0) -> NamedTuple

Event-pattern skill for an ENSO forecast. Identifies events in `n34_true`
inside the lead window, then for each event computes phase-aligned pattern
correlation, sign correctness, and counts forecast peaks that don't
correspond to real events (false alarms). Returns a NamedTuple:

- `events_true`        : event months (within `lead_window`) detected in truth
- `sign_correct`       : per-event Bool (forecast n34 has correct sign at the
                         best phase-aligned offset)
- `best_pc`            : per-event max pattern correlation within ±phase_tol
- `best_offset`        : per-event phase offset where pc maximum
- `false_alarms`       : count of forecast |n34| peaks > false_alarm_threshold
                         within `lead_window` that aren't within phase_tol of a
                         truth event
- `sign_accuracy`      : fraction of events with correct sign
- `mean_event_pc`      : mean of `best_pc`
- `weighted_event_pc`  : `best_pc` weighted by |n34_true[event]|
- `phase_bias_mean`    : mean signed offset (negative = forecast leads truth)

Always pair `mean_event_pc` with `sign_accuracy` and `false_alarms` —
high pc with low sign accuracy means the model has the spatial pattern
flipped at events; low false alarms with low pc means the model is too
quiet to forecast anything.
"""
function event_skill(n34_true::AbstractVector{<:Real},
                     field_true::AbstractArray{<:Real, 3},
                     n34_pred::AbstractVector{<:Real},
                     field_pred::AbstractArray{<:Real, 3};
                     lead_window::Tuple{Int, Int} = (10, 20),
                     phase_tol::Integer = 3,
                     event_threshold::Real = 1.0,
                     min_separation::Integer = 6,
                     false_alarm_threshold::Real = 1.0)
    nt = length(n34_true)
    @assert length(n34_pred) == nt
    @assert size(field_true, 3) == nt
    @assert size(field_pred, 3) == nt

    lo, hi = lead_window
    @assert lo ≥ 1 && hi ≤ nt

    # Detect events in the truth restricted to the lead window
    candidate_events = find_enso_events(n34_true; n_events = nt,
                                         threshold = event_threshold,
                                         min_separation = min_separation)
    events = filter(t -> lo ≤ t ≤ hi, candidate_events)

    n_events = length(events)
    sign_correct = falses(n_events)
    best_pc       = fill(NaN, n_events)
    best_offset   = zeros(Int, n_events)

    for (k, t_event) in enumerate(events)
        r = phase_aligned_pc(field_true, field_pred, t_event; phase_tol = phase_tol)
        best_pc[k]     = r.best_pc
        best_offset[k] = r.best_offset
        # Sign correctness: forecast n34 at the best-offset month has same sign
        # as truth at the event.
        t_pred = t_event + r.best_offset
        sign_correct[k] = sign(n34_pred[t_pred]) == sign(n34_true[t_event])
    end

    # False alarms: count forecast n34 peaks (local maxima of |n34_pred|)
    # within `lead_window` whose nearest truth event is more than `phase_tol`
    # months away. Implementation: greedy, using the same threshold rule.
    forecast_peaks = find_enso_events(n34_pred; n_events = nt,
                                       threshold = false_alarm_threshold,
                                       min_separation = min_separation)
    forecast_peaks_in_window = filter(t -> lo ≤ t ≤ hi, forecast_peaks)
    false_alarms = 0
    for fp in forecast_peaks_in_window
        if isempty(events) || minimum(abs.(events .- fp)) > phase_tol
            false_alarms += 1
        end
    end

    sign_accuracy   = n_events > 0 ? sum(sign_correct) / n_events : NaN
    mean_event_pc   = n_events > 0 ? mean(best_pc) : NaN
    weights = n_events > 0 ? abs.(n34_true[events]) : Float64[]
    weighted_event_pc = if n_events > 0 && sum(weights) > 0
        sum(best_pc .* weights) / sum(weights)
    else
        NaN
    end
    phase_bias_mean = n_events > 0 ? mean(best_offset) : NaN

    return (events_true       = events,
            sign_correct       = sign_correct,
            best_pc            = best_pc,
            best_offset        = best_offset,
            false_alarms       = false_alarms,
            sign_accuracy      = sign_accuracy,
            mean_event_pc      = mean_event_pc,
            weighted_event_pc  = weighted_event_pc,
            phase_bias_mean    = phase_bias_mean,
            n_events           = n_events,
            n_forecast_peaks   = length(forecast_peaks_in_window))
end