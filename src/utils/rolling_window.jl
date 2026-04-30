"""
    rolling_windows(Ttot; train_len, predict_len, stride=12, n_windows=nothing)

Return a vector of NamedTuples `(train_start, train_end, predict_start, predict_end, label)`
for non-overlapping or stride-spaced rolling cross-validation windows over a series
of total length `Ttot`.

- `train_len`, `predict_len`: window sizes in samples
- `stride`: shift in samples between successive windows (default 12 months)
- `n_windows`: cap on the number of windows produced; `nothing` = as many as fit

Each window satisfies `train_end + predict_len ≤ Ttot`. Indices are 1-based.
"""
function rolling_windows(Ttot::Integer;
                         train_len::Integer,
                         predict_len::Integer,
                         stride::Integer = 12,
                         n_windows::Union{Nothing, Int} = nothing)
    @assert train_len > 0 && predict_len > 0 && stride > 0
    out = NamedTuple{(:train_start, :train_end, :predict_start, :predict_end, :label),
                     Tuple{Int, Int, Int, Int, String}}[]
    k = 0
    train_start = 1
    while train_start + train_len + predict_len - 1 ≤ Ttot
        k += 1
        train_end     = train_start + train_len - 1
        predict_start = train_end + 1
        predict_end   = predict_start + predict_len - 1
        push!(out, (train_start = train_start,
                    train_end = train_end,
                    predict_start = predict_start,
                    predict_end = predict_end,
                    label = "W$(k)"))
        n_windows !== nothing && k ≥ n_windows && break
        train_start += stride
    end
    return out
end
