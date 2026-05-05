"""
    forecast_persistence(field_train, predict_len; warmup = 0)

Persistence baseline: forecast = last training-window state, held constant for
`predict_len` months (after `warmup` months of buffer that mirrors the reservoir
test pipeline). Returns `(nlon × nlat × predict_len)` field.

The "last state" is `field_train[:, :, end]`. For an anomaly field this means
the forecast is "this month's anomaly stays the same forever".
"""
function forecast_persistence(field_train::AbstractArray{<:Real, 3},
                              predict_len::Integer)
    nlon, nlat, _ = size(field_train)
    last_state = field_train[:, :, end]
    out = zeros(Float64, nlon, nlat, predict_len)
    for t in 1:predict_len
        out[:, :, t] .= last_state
    end
    return out
end

"""
    forecast_damped_persistence(field_train, predict_len; tau_decay = 6.0)

Damped persistence: forecast(t+l) = last_anomaly · exp(−l / tau_decay).
The anomaly decays exponentially toward 0 (climatology) with e-folding time
`tau_decay` months. Standard ENSO operational baseline.
"""
function forecast_damped_persistence(field_train::AbstractArray{<:Real, 3},
                                     predict_len::Integer;
                                     tau_decay::Real = 6.0)
    nlon, nlat, _ = size(field_train)
    last_state = field_train[:, :, end]
    out = zeros(Float64, nlon, nlat, predict_len)
    for l in 1:predict_len
        out[:, :, l] .= last_state .* exp(-l / tau_decay)
    end
    return out
end

"""
    forecast_climatology(field_train, predict_len)

Climatology baseline: forecast = zero anomaly everywhere for all leads.
After per-pixel z-score (which we apply upstream), this is mathematically the
expected-value forecast given training stats. Reported separately as a sanity
floor — any model whose event-skill is worse than climatology is genuinely
broken.
"""
function forecast_climatology(field_train::AbstractArray{<:Real, 3},
                              predict_len::Integer)
    nlon, nlat, _ = size(field_train)
    return zeros(Float64, nlon, nlat, predict_len)
end
