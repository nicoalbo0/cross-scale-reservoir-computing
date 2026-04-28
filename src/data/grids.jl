"""
    make_blocks_single_layer(L, nblocks, mixing)

Build 1D single-layer blocks: divide `L` into `nblocks` contiguous ranges; each block has
`rows_rec` (recurrent), `rows_neigh` (neighbors within `mixing`), and empty `rows_layer`.
`L` must be divisible by `nblocks`. Neighbors wrap periodically.
"""
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

"""
    make_blocks_multi_layer(L_f, divisor, n_fine, mixing, n_coarse; overlap_mode)

Build two-scale 1D blocks: fine blocks from `make_blocks_single_layer(L_f, n_fine, mixing)`,
each assigned a parent coarse block. `rows_layer` is filled from the coarse block;
`overlap_mode` is `:exclude` (non-overlapping with fine) or `:include` (overlapping).
"""
function make_blocks_multi_layer(L_f::Int, divisor::Int, n_fine::Int, mixing::Int, n_coarse::Int; overlap_mode::Symbol)
    # --- sanity ---
    @assert L_f % divisor == 0
    L_c = div(L_f, divisor)
    @assert n_fine % n_coarse == 0

    @assert overlap_mode === :exclude || overlap_mode === :include

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
    make_blocks(L, nblocks, mixing)

Single-layer 1D block construction. See `make_blocks_single_layer`.
"""
make_blocks(L::Int, nblocks::Int, mixing::Int) = make_blocks_single_layer(L, nblocks, mixing)

"""
    make_blocks(L_f, div, n_fine, mixing, n_coarse; overlap_mode)

Multi-layer 1D block construction (coarse → fine).

# Arguments
- `L_f`: Fine spatial size.
- `div`: Regridding factor (L_f / L_c).
- `n_fine`: Number of fine blocks.
- `mixing`: Neighbor width at fine scale.
- `n_coarse`: Number of coarse blocks.
- `overlap_mode`: `:exclude` or `:include` for cross-scale overlap.
"""
make_blocks(L_f::Int, div::Int, n_fine::Int, mixing::Int, n_coarse::Int; overlap_mode::Symbol) = make_blocks_multi_layer(L_f, div, n_fine, mixing, n_coarse; overlap_mode = overlap_mode)

"""
    linear_index(i, j, nlat)

Map 2D grid indices (i, j) to linear index. Layout is column-major with first index i (e.g. lon),
second j (e.g. lat), so j varies fastest: index = (i-1)*nlat + j.
"""
@inline linear_index(i::Int, j::Int, nlat::Int) = (i - 1) * nlat + j

"""
    make_blocks_single_layer_2d(data, grid; mixing=0)

Build 2D single-layer blocks. `data` is (nlon×nlat×nt); `grid` is (div_lon, div_lat).
Domain is split into div_lon×div_lat blocks; each has `rows_rec`, `rows_neigh` (with
`mixing` in lon/lat), and empty `rows_layer`. Longitude is periodic; latitude is clamped.
"""
function make_blocks_single_layer_2d(
    data::AbstractArray{T,3},
    grid::Tuple{Int,Int};
    mixing::Int = 0
) where T

    nlon, nlat, _ = size(data)
    div_lon, div_lat = grid

    @assert nlon % div_lon == 0 "nlon must be divisible by div_lon"
    @assert nlat % div_lat == 0 "nlat must be divisible by div_lat"

    lon_block = div(nlon, div_lon)
    lat_block = div(nlat, div_lat)

    blocks = Vector{
        NamedTuple{(:rows_rec, :rows_neigh, :rows_layer),
                   Tuple{Vector{Int}, Vector{Int}, Vector{Int}}}
    }()

    for bi in 1:div_lon, bj in 1:div_lat

        lon_range = (bi-1)*lon_block + 1 : bi*lon_block
        lat_range = (bj-1)*lat_block + 1 : bj*lat_block

        rows_rec = Int[]
        rows_rec_set = Set{Int}()

        # recurrent indices
        for i in lon_range, j in lat_range
            idx = linear_index(i, j, nlat)
            push!(rows_rec, idx)
            push!(rows_rec_set, idx)
        end

        neigh_set = Set{Int}()

        for i in lon_range, j in lat_range
            for di in -mixing:mixing, dj in -mixing:mixing

                # periodic longitude
                ii = mod1(i + di, nlon)

                # non-periodic latitude (clamped)
                jj = mod1(j + dj, nlat)

                idx = linear_index(ii, jj, nlat)

                if !(idx in rows_rec_set)
                    push!(neigh_set, idx)
                end
            end
        end

        push!(blocks, (
            rows_rec   = rows_rec,
            rows_neigh = collect(neigh_set),
            rows_layer = Int[]
        ))
    end

    return blocks
end

"""
    add_cross_layer!(fine_blocks, coarse_blocks, nlon_f, nlat_f, nlon_c, nlat_c; overlap_mode=:exclude)

In-place: set each fine block's `rows_layer` to indices in the parent coarse
block's recurrent set, using 2D index mapping. Fine grid dimensions must be
multiples of coarse.

`overlap_mode`:
- `:exclude` (paper-faithful, default): drop the coarse cells that coincide
  with the fine block's own spatial support, so the cross-scale input is
  strictly *non-redundant* with the fine block's recurrent input.
  Matches arXiv:2510.11209 §1.3: "only the nonoverlapping portion of the
  coarser reservoir output is included."
- `:include`: pass the entire parent coarse block — overlap retained.
  (Pre-paper behavior; left available for ablation.)
"""
function add_cross_layer!(
    fine_blocks,
    coarse_blocks,
    nlon_f, nlat_f,
    nlon_c, nlat_c;
    overlap_mode::Symbol = :exclude
)

    @assert nlon_f % nlon_c == 0
    @assert nlat_f % nlat_c == 0
    @assert overlap_mode === :exclude || overlap_mode === :include

    scale_lon = div(nlon_f, nlon_c)
    scale_lat = div(nlat_f, nlat_c)

    # map a fine pixel's flat index to the coarse cell's flat index
    function fine_to_coarse_idx(idx)
        i_f = div(idx-1, nlat_f) + 1
        j_f = mod(idx-1, nlat_f) + 1
        i_c = cld(i_f, scale_lon)
        j_c = cld(j_f, scale_lat)
        return linear_index(i_c, j_c, nlat_c)
    end

    for k in eachindex(fine_blocks)

        fb = fine_blocks[k]

        # parent coarse block: the one that contains the coarse cell
        # corresponding to the fine block's first recurrent pixel
        first_coarse_idx = fine_to_coarse_idx(fb.rows_rec[1])
        parent_block_id = nothing
        for (b, cb) in enumerate(coarse_blocks)
            if first_coarse_idx in cb.rows_rec
                parent_block_id = b
                break
            end
        end
        @assert parent_block_id !== nothing

        parent_rec = coarse_blocks[parent_block_id].rows_rec

        if overlap_mode === :exclude
            # Coarse cells that the fine block "shadows" (its rows_rec
            # mapped down to the coarse grid). Subtract from parent_rec.
            shadow = unique(fine_to_coarse_idx.(fb.rows_rec))
            rows_layer = collect(setdiff(parent_rec, shadow))
        else  # :include
            rows_layer = parent_rec
        end

        fine_blocks[k] = (
            rows_rec   = fb.rows_rec,
            rows_neigh = fb.rows_neigh,
            rows_layer = rows_layer
        )
    end
end

"""
    make_blocks_multi_layer_2d(data_vec, grids, mixing; overlap_mode=:exclude)

Build multi-layer 2D blocks. `data_vec` and `grids` have one entry per layer.
Each layer is built with `make_blocks_single_layer_2d`; then `add_cross_layer!` fills
`rows_layer` from the coarser layer. Returns a vector of block vectors (one per layer).

`overlap_mode` is forwarded to `add_cross_layer!`. Default `:exclude` matches
arXiv:2510.11209.
"""
function make_blocks_multi_layer_2d(
    data_vec::Vector{<:AbstractArray{Float64,3}},
    grids::Vector{Tuple{Int,Int}},
    mixing::Int;
    overlap_mode::Symbol = :exclude
)

    @assert length(data_vec) == length(grids)

    nlayers = length(data_vec)

    # --- build independent layers first
    layers = [
        make_blocks_single_layer_2d(
            data_vec[i],
            grids[i];
            mixing = mixing
        )
        for i in 1:nlayers
    ]

    # --- add cross-scale connections
    for l in 2:nlayers

        nlon_f, nlat_f, _ = size(data_vec[l])
        nlon_c, nlat_c, _ = size(data_vec[l-1])

        add_cross_layer!(
            layers[l],
            layers[l-1],
            size(data_vec[l],1), size(data_vec[l],2),
            size(data_vec[l-1],1), size(data_vec[l-1],2);
            overlap_mode = overlap_mode
        )
    end

    return layers
end

"""
    make_blocks(data_vec, grids, mixing)

Multi-layer 2D block construction. Dispatches to `make_blocks_multi_layer_2d`.
"""
make_blocks(data_vec::Vector{Array{Float64,3}}, grids::Vector{Tuple{Int,Int}}, mixing::Int; overlap_mode::Symbol=:exclude) = make_blocks_multi_layer_2d(data_vec, grids, mixing; overlap_mode=overlap_mode)