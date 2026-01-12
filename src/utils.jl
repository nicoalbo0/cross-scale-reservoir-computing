function load_data(Q::Int, L::Int, μ::T; show_data::Bool=false, interpolate_data::Bool=false) where T<:Real

    filepath = pwd()*"/data/Q$(Q)_L$(L)_mu$(μ)_ks_data.csv"
    dt, refinement = 0.25, 1

    f = Matrix(CSV.read("$filepath", DataFrame))';
    data = (f .- mean(f,dims=2)) ./ std(f, dims=2)

    if interpolate_data
        dt, refinement = 0.25, 4
        data = cubic_time_interpolate(data, dt, refinement)
    end
        
    if show_data
        p = heatmap(data[:, 1:5000]; title="KS data sample", xlabel="t", ylabel="row (state)")
        display(p)
    end
    
    return data, dt / refinement

end

function make_blocks_single_layer(L::Int, nblocks::Int, mixing::Int)

    base, rem = divrem(L, nblocks)

    if rem != 0
        error("L must be divisible by nblocks (got L=$L, nblocks=$nblocks).")
    end
    ranges = [base*(i-1) + 1 : (base*i) for i in 1:nblocks]

    blocks = Vector{
        NamedTuple{(:rows_rec, :rows_neigh, :rows_layer),
                   Tuple{UnitRange{Int}, Vector{Int}, Vector{Int}}}
    }(undef, nblocks)

    for (i, rows_rec) in enumerate(ranges)

        idxs = (first(rows_rec)-mixing):(last(rows_rec)+mixing)
        rows_extended = mod.(idxs .- 1, L) .+ 1

        rows_neigh = setdiff(rows_extended, rows_rec)

        # ensure no neighbor goes missing
        @assert length(rows_neigh) == mixing * 2

        blocks[i] = (
            rows_rec   = rows_rec,
            rows_neigh = rows_neigh,
            rows_layer = Int[]
        )
    end

    return blocks
end

function make_blocks_multilayer(L_f::Int, divisor::Int, n_fine::Int, mixing::Int, n_coarse::Int; overlap_mode::Symbol)
    # --- sanity ---
    @assert L_f % divisor == 0
    L_c = div(L_f, divisor)
    @assert n_fine % n_coarse == 0

    # fine blocks (fine index space)
    fine_blocks = make_blocks_single_layer(L_f, n_fine, mixing)

    # coarse block geometry
    base_c, rem = divrem(L_c, n_coarse)
    rem == 0 || error("L_c must be divisible by n_coarse")

    coarse_ranges = [
        base_c*(i-1)+1 : base_c*i for i in 1:n_coarse
    ]

    # mapping fine index -> coarse index
    fine_to_coarse(i) = div(i-1, divisor) + 1

    k = n_fine ÷ n_coarse
    blocks = similar(fine_blocks)

    for i in 1:n_fine
        rows_rec   = fine_blocks[i].rows_rec
        rows_neigh = fine_blocks[i].rows_neigh

        # parent coarse block
        parent = cld(i, k)
        parent_coarse = coarse_ranges[parent]

        # coarse indices overlapping with this fine block
        overlap = unique(fine_to_coarse.(rows_rec))

        if overlap_mode === :exclude
            # remove overlap → non-redundant cross-scale input
            rows_layer = setdiff(parent_coarse, overlap)

        elseif overlap_mode === :include
            # keep overlap → redundant cross-scale input
            rows_layer = overlap

        end

        blocks[i] = (
            rows_rec   = rows_rec,
            rows_neigh = rows_neigh,
            rows_layer = collect(rows_layer)
        )
    end

    return blocks
end

"""
Single-layer block construction.
"""
make_blocks(L::Int, nblocks::Int, mixing::Int) = make_blocks_single_layer(L, nblocks, mixing)


"""
Hard multi-layer block construction (coarse → fine).

Arguments:
- L_f          : fine spatial size
- div          : regridding factor (L_f / L_c)
- n_fine       : number of fine blocks
- mixing       : neighbor width at fine scale
- n_coarse     : number of coarse blocks
"""
make_blocks(L_f::Int, div::Int, n_fine::Int, mixing::Int, n_coarse::Int; overlap_mode::Symbol) = make_blocks_multilayer(L_f, div, n_fine, mixing, n_coarse; overlap_mode = overlap_mode)

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
    regrid_average(data::AbstractMatrix{T}, div::Int) -> Matrix{T}

Spatially coarse-grain `data` by averaging over blocks of size `div`
along the first (spatial) dimension.

- data: (L × T)
- divisor: integer divisor of L
- output: (L/divisor × T)

Each coarse cell is the mean of `div` contiguous fine cells.
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
Extract layer-l parameters from a tuple of parameter vectors.

Example:
    params = (res_size, radius, degree, g_in_rec, g_in_neigh, g_in_layer)
    layer_params(params, 2)
"""
@inline function layer_params(params::Tuple, l::Int)
    return ntuple(i -> params[i][l], length(params))
end

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

Compute RMSE_{t ≤ T} for `data` and `pred` shaped (N × Tfull), where rows are
coordinates and columns are time samples.

- `T`: prediction horizon (number of time samples included from the start).
- `coords`: indices of observed coordinates (rows) to include.
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