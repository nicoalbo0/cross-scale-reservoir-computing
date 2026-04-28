"""
    bandpass_decompose(signal, cutoffs; fs=1.0, order=4)

Decompose a 1D real-valued time series into `length(cutoffs)+1` non-overlapping
frequency bands using Butterworth filters and zero-phase `filtfilt`.

# Arguments
- `signal::AbstractVector{<:Real}`: input time series (any length).
- `cutoffs::NTuple{N,<:Real}`: filter corner frequencies in **cycles per sample**
  (i.e. Hz when `fs=1`). Must be sorted ascending and strictly between 0 and
  `fs/2` (Nyquist).
- `fs::Real=1.0`: sampling frequency. For monthly data, use `fs=1.0` so cutoffs
  are interpreted as cycles per month.
- `order::Int=4`: Butterworth order. 4 balances cutoff sharpness vs. boundary
  transient length. Use 3 if rings appear; 5 if leakage between bands is too
  large.

# Returns
A `NamedTuple`:
- For 1 cutoff (2 bands): `(slow, fast)`.
- For 2 cutoffs (3 bands): `(slow, mid, fast)`.
- For more cutoffs: `(slow=…, mids=Vector, fast=…)`.

Each band is the same length and element type (`Float64`-promoted) as the
input.

# Notes
- Uses `filtfilt` (forward-backward), so the bands are **zero phase** —
  features in the input are not time-shifted in any band.
- **Reconstruction is exact** by construction: `signal == sum(bands)` to
  floating-point precision in the interior. The implementation uses cumulative
  lowpass differences (slow = LP(f_1); mid_k = LP(f_{k+1}) - LP(f_k);
  fast = signal - LP(f_N)), which is linear and unitary at every frequency.
  Naively summing independent BP/HP/LP Butterworth filters does NOT
  reconstruct exactly — that's why we don't.
- **Boundary transient.** `filtfilt` halves but does not eliminate edge
  effects. With order=4 and a low cutoff like 1/24 cyc/sample, expect roughly
  `3*order/cutoff ≈ 30` samples of transient at each end. For ENSO (408
  monthly samples, train=288, test=96, washout=12), this is acceptable but
  the very first/last months of the series are degraded; trim if reporting
  edge metrics.

# Example
```julia
using CrossScaleRC
n34 = nino34_index(sst_field, lons, lats)              # 408 monthly samples
bands = bandpass_decompose(n34, (1/24, 1/3); fs=1.0)   # slow, mid, fast
recon = bands.slow .+ bands.mid .+ bands.fast          # ≈ n34 (interior)
```
"""
function bandpass_decompose(signal::AbstractVector{<:Real},
                            cutoffs::NTuple{N,<:Real};
                            fs::Real = 1.0, order::Int = 4) where {N}
    @assert N ≥ 1 "need at least one cutoff"
    @assert issorted(cutoffs) "cutoffs must be sorted ascending"
    @assert all(c -> 0 < c < fs / 2, cutoffs) "cutoffs must lie in (0, fs/2)"

    s = float.(signal)

    # Decomposition strategy: cumulative lowpass differences. Define
    #   B_0 = LP(f_1)
    #   B_k = LP(f_{k+1}) - LP(f_k)  for k = 1, …, N-1
    #   B_N = signal - LP(f_N)
    # which guarantees `signal == sum(B_k)` exactly (modulo floating-point) by
    # linearity of `filtfilt`. The B_k for 1 ≤ k < N are bandpass-shaped; B_0
    # is the lowpass slow band; B_N is the highpass-style residual fast band.
    # DSP.jl v0.8: `fs` is a kwarg of `digitalfilter`, not of `Lowpass/...`.
    lp_outputs = [filtfilt(digitalfilter(Lowpass(c), Butterworth(order); fs = fs), s)
                  for c in cutoffs]

    slow = lp_outputs[1]
    fast = s .- lp_outputs[end]

    if N == 1
        return (slow = slow, fast = fast)
    elseif N == 2
        mid = lp_outputs[2] .- lp_outputs[1]
        return (slow = slow, mid = mid, fast = fast)
    else
        mids = [lp_outputs[i+1] .- lp_outputs[i] for i in 1:(N-1)]
        return (slow = slow, mids = mids, fast = fast)
    end
end

"""
    reconstruct_bands(bands)

Sum the band outputs of `bandpass_decompose` to recover the original signal.
Linearity of `filtfilt` makes this exact except for boundary transients (see
notes in `bandpass_decompose`).
"""
function reconstruct_bands(bands::NamedTuple)
    if haskey(bands, :mids)
        return bands.slow .+ sum(bands.mids) .+ bands.fast
    elseif haskey(bands, :mid)
        return bands.slow .+ bands.mid .+ bands.fast
    else
        return bands.slow .+ bands.fast
    end
end

"""
    bandpass_decompose_field(field, cutoffs; fs=1.0, order=4)

Apply `bandpass_decompose` to each pixel's time series independently for a
spatiotemporal `field` of shape `(nlon × nlat × nt)`. Returns a `NamedTuple`
of bands, each the same shape as the input.

This is the field analog used for the per-pixel temporal cross-scale
architecture. Each pixel's time series is decomposed independently — no
cross-pixel filter coupling. NaN pixels (land) are replaced by zero before
filtering; the corresponding band entries are also zero (the filter of a
zero series is a zero series).
"""
function bandpass_decompose_field(field::AbstractArray{<:Real, 3},
                                  cutoffs::NTuple{N, <:Real};
                                  fs::Real = 1.0, order::Int = 4) where {N}
    nlon, nlat, nt = size(field)

    # Allocate output cubes (one per band)
    n_bands = N == 1 ? 2 : (N == 2 ? 3 : N + 1)
    out = [Array{Float64, 3}(undef, nlon, nlat, nt) for _ in 1:n_bands]

    @inbounds for i in 1:nlon, j in 1:nlat
        ts = view(field, i, j, :)
        # Replace NaN with 0 in a copy (filter requires finite real values).
        ts_clean = if any(isnan, ts)
            replace(Float64.(ts), NaN => 0.0)
        else
            Float64.(ts)
        end
        bands = bandpass_decompose(ts_clean, cutoffs; fs = fs, order = order)
        if N == 1
            out[1][i, j, :] .= bands.slow
            out[2][i, j, :] .= bands.fast
        elseif N == 2
            out[1][i, j, :] .= bands.slow
            out[2][i, j, :] .= bands.mid
            out[3][i, j, :] .= bands.fast
        else
            out[1][i, j, :] .= bands.slow
            for k in 1:length(bands.mids)
                out[k+1][i, j, :] .= bands.mids[k]
            end
            out[end][i, j, :] .= bands.fast
        end
    end

    if N == 1
        return (slow = out[1], fast = out[2])
    elseif N == 2
        return (slow = out[1], mid = out[2], fast = out[3])
    else
        return (slow = out[1], mids = out[2:end-1], fast = out[end])
    end
end
