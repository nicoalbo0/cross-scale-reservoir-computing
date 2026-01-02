    #=
    W_in_rec   = zeros(T, N, rec_dimensions)
    W_in_neigh = zeros(T, N, neigh_dimensions)
    W_in_layer = zeros(T, N, layer_dimensions)

    
    q = div(N, rec_dimensions)
    for i in 1:rec_dimensions
        W_in_rec[(i-1)*q+1:i*q, i] .= g_in_rec .* (-1 .+ 2 .* rand(q))
    end

    if neigh_dimensions > 0
        q = div(N, neigh_dimensions)
        for i in 1:neigh_dimensions
            W_in_neigh[(i-1)*q+1:i*q, i] .= g_in_neigh .* (-1 .+ 2 .* rand(q))
        end
    end

    if layer_dimensions > 0
        q = div(N, layer_dimensions)
        for i in 1:layer_dimensions
            W_in_layer[(i-1)*q+1:i*q, i] .= g_in_layer .* (-1 .+ 2 .* rand(q))
        end
    end

    # input weight matrix
    
    #=W_in_rec   = 2 .* g_in_rec .*rand(N, rec_dimensions) .- 1
    W_in_neigh = 2 .* g_in_neigh .*rand(N, neigh_dimensions) .- 1
    W_in_layer = 2 .* g_in_layer .*rand(N, layer_dimensions) .- 1
    

    W_in_rec   = build_W_in(N, rec_dimensions,   g_in_rec)
    W_in_neigh = build_W_in(N, neigh_dimensions, g_in_neigh)
    W_in_layer = build_W_in(N, layer_dimensions, g_in_layer)
    =#

    W_in_rec = build_W_in(
        N, rec_dimensions, g_in_rec;
        mode = :structured
    )

    W_in_neigh = build_W_in(
        N, neigh_dimensions, g_in_neigh;
        mode = :structured
    )

    W_in_layer = build_W_in(
        N, layer_dimensions, g_in_layer;
        mode = :structured
    )
    =#

    W_in_rec, W_in_neigh, W_in_layer =
        build_W_in(
            N,
            rec_dimensions,
            neigh_dimensions,
            layer_dimensions,
            g_in_rec,
            g_in_neigh,
            g_in_layer;
            mode = input_mode
        )


#=
function build_W_in(
    N::Int,
    Din::Int,
    gain::Real;
    mode::Symbol
)
    Din == 0 && return zeros(N, 0)

    @assert mode === :structured || mode === :random

    W = zeros(N, Din)
    q = div(N, Din)

    if mode === :structured
        # contiguous blocks (locality-preserving)
        for j in 1:Din
            W[(j-1)*q+1:j*q, j] .= gain .* (2 .* rand(q) .- 1)
        end

    elseif mode === :random
        # permutation-based (symmetry-breaking)
        perm = randperm(N)
        for j in 1:Din
            idx = perm[(j-1)*q+1:j*q]
            W[idx, j] .= gain .* (2 .* rand(q) .- 1)
        end
    end

    return W
end
=#

function build_W_in(
    N::Int,
    rec_dim::Int,
    neigh_dim::Int,
    layer_dim::Int,
    g_rec::Real,
    g_neigh::Real,
    g_layer::Real;
    mode::Symbol = :structured
)
    D = rec_dim + neigh_dim + layer_dim
    D == 0 && return (
        zeros(N, 0),
        zeros(N, 0),
        zeros(N, 0)
    )

    @assert mode === :structured || mode === :random

    # global matrix
    W = zeros(N, D)

    # column-wise gains
    gains = vcat(
        fill(g_rec,   rec_dim),
        fill(g_neigh, neigh_dim),
        fill(g_layer, layer_dim)
    )

    # neuron assignment
    q = div(N, D)

    if mode === :structured
        # contiguous neuron blocks
        for j in 1:D
            rows = (j-1)*q+1 : j*q
            W[rows, j] .= gains[j] .* (2 .* rand(q) .- 1)
        end

    else
        # random permutation of neurons
        perm = randperm(N)
        for j in 1:D
            rows = perm[(j-1)*q+1 : j*q]
            W[rows, j] .= gains[j] .* (2 .* rand(q) .- 1)
        end
    end

    # split by columns (cheap views)
    i1 = rec_dim
    i2 = i1 + neigh_dim

    W_rec   = W[:, 1:i1]
    W_neigh = W[:, i1+1:i2]
    W_layer = W[:, i2+1:end]

    return W_rec, W_neigh, W_layer
end

