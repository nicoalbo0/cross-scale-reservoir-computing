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
        itp = CubicSplineInterpolation(
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