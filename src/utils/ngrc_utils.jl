# Combinations with repetition as a dense matrix (k × ncomb)
function combs_with_repetition_mat(n::Int, k::Int)
    n ≥ 1 || throw(ArgumentError("n must be ≥ 1"))
    k ≥ 0 || throw(ArgumentError("k must be ≥ 0"))

    ncomb = binomial(n + k - 1, k)
    out = Matrix{Int}(undef, k, ncomb)
    buf = Vector{Int}(undef, k)
    col = Ref(1)

    function rec(pos::Int, start::Int)
        if pos > k
            c = col[]
            @inbounds for r in 1:k
                out[r, c] = buf[r]
            end
            col[] = c + 1
            return
        end
        @inbounds for i in start:n
            buf[pos] = i
            rec(pos + 1, i)
        end
        return
    end

    if k == 0
        # single "empty" combination: 0×1 matrix
        return Matrix{Int}(undef, 0, 1)
    end

    rec(1, 1)
    return out
end

function feature_length(nlin::Int, degree::Int)
    f = 1 + nlin
    @inbounds for p in 2:degree
        f += binomial(nlin + p - 1, p)
    end
    return f
end

# Map linear-stack index 1:nlin -> (i,j) in xhist (M×past) where idx=(j-1)*M+i.
function linear_index_maps(M::Int, past::Int)
    nlin = M * past
    lin_i = Vector{Int}(undef, nlin)
    lin_j = Vector{Int}(undef, nlin)
    @inbounds for idx in 1:nlin
        lin_i[idx] = (idx - 1) % M + 1
        lin_j[idx] = (idx - 1) ÷ M + 1
    end
    return lin_i, lin_j
end

# Full feature fill (all features), using dense combo matrices.
function fill_features_full!(
    ϕ::Vector{Float64},
    xhist::Matrix{Float64},                 # (M, past), col1=current
    poly_combos::Vector{Matrix{Int}}        # for p=2..degree: (p × ncomb)
)
    M, past = size(xhist)
    nlin = M * past

    @inbounds ϕ[1] = 1.0

    # linear stack
    pos = 2
    @inbounds for j in 1:past
        for i in 1:M
            ϕ[pos] = xhist[i, j]
            pos += 1
        end
    end

    lin = @view ϕ[2:(1 + nlin)]

    # polynomial monomials
    @inbounds for combmat in poly_combos
        k = size(combmat, 1)
        ncomb = size(combmat, 2)
        for c in 1:ncomb
            prod = 1.0
            for r in 1:k
                prod *= lin[combmat[r, c]]
            end
            ϕ[pos] = prod
            pos += 1
        end
    end

    return ϕ
end

# Sparse-feature computation: compute only selected features.
struct FeatureMap
    kind::Vector{UInt8}     # 0=constant, 1=linear, 2=poly
    offset::Vector{Int}     # linear: linear-idx; poly: start in idxpool
    len::Vector{UInt8}      # poly degree (k), else 0/1
    idxpool::Vector{Int}    # concatenated poly index lists
end

function build_feature_map(nlin::Int, poly_combos::Vector{Matrix{Int}})
    F = 1 + nlin
    total_pool = 0
    @inbounds for combmat in poly_combos
        total_pool += size(combmat, 1) * size(combmat, 2)
        F += size(combmat, 2)
    end

    kind = Vector{UInt8}(undef, F)
    offset = Vector{Int}(undef, F)
    len = Vector{UInt8}(undef, F)
    idxpool = Vector{Int}(undef, total_pool)

    # constant
    kind[1] = 0x00
    offset[1] = 0
    len[1] = 0x00

    # linear features
    @inbounds for idx in 1:nlin
        f = 1 + idx
        kind[f] = 0x01
        offset[f] = idx
        len[f] = 0x01
    end

    # poly features
    pos = 2 + nlin
    poolpos = 1
    @inbounds for combmat in poly_combos
        k = size(combmat, 1)
        ncomb = size(combmat, 2)
        k_u8 = UInt8(k)
        for c in 1:ncomb
            kind[pos] = 0x02
            offset[pos] = poolpos
            len[pos] = k_u8
            for r in 1:k
                idxpool[poolpos] = combmat[r, c]
                poolpos += 1
            end
            pos += 1
        end
    end

    return FeatureMap(kind, offset, len, idxpool)
end

function fill_features_selected!(
    ϕsub::AbstractVector{Float64},
    xhist::Matrix{Float64},
    fmap::FeatureMap,
    feat_idx::Vector{Int},
    lin_i::Vector{Int},
    lin_j::Vector{Int}
)
    @inbounds for k in eachindex(feat_idx)
        f = feat_idx[k]
        kd = fmap.kind[f]
        if kd == 0x00
            ϕsub[k] = 1.0
        elseif kd == 0x01
            idx = fmap.offset[f]
            ϕsub[k] = xhist[lin_i[idx], lin_j[idx]]
        else
            start = fmap.offset[f]
            l = Int(fmap.len[f])
            prod = 1.0
            for q in 0:(l - 1)
                idx = fmap.idxpool[start + q]
                prod *= xhist[lin_i[idx], lin_j[idx]]
            end
            ϕsub[k] = prod
        end
    end
    return ϕsub
end

# Pick a sparse subset of features (always include constant feature 1).
function select_feature_indices(F::Int, nfeat::Int, rng::AbstractRNG)
    if nfeat <= 0 || nfeat >= F
        return collect(1:F)
    end

    nsel = max(1, nfeat)
    if nsel == 1
        return [1]
    end

    perm = randperm(rng, F - 1)  # 1:(F-1)
    idx = Vector{Int}(undef, nsel)
    idx[1] = 1
    @inbounds for k in 2:nsel
        idx[k] = 1 + perm[k - 1] # -> 2:F
    end
    sort!(idx)
    return idx
end

function nextgen_closedloop(
    data::AbstractMatrix{<:Real},
    train_len::Int,
    predict_len::Int;
    washout::Int       = 0,
    past::Int          = 2,
    delay::Int         = 1,          # stride between consecutive lags ≥ 1
    degree::Int        = 2,
    ridge::Float64     = 1e-6,
    noise_std::Float64 = 0.0,        # std of Gaussian input-noise regularizer
    nfeat::Int         = 0,
    seed::Union{Nothing,Int} = nothing
)
    X = Matrix{Float64}(data)
    M, T = size(X)

    t_start = washout + 1 + (past - 1) * delay   # first t with full lag history
    t_end   = t_start + train_len - 1
    t_end + 1 ≤ T || error("Not enough samples for train_len given washout/past/delay.")

    nlin        = M * past
    poly_combos = [combs_with_repetition_mat(nlin, p) for p in 2:degree]
    F           = feature_length(nlin, degree)

    rng      = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    feat_idx = select_feature_indices(F, nfeat, rng)
    Fsub     = length(feat_idx)

    lin_i, lin_j = linear_index_maps(M, past)
    fmap  = Fsub < F ? build_feature_map(nlin, poly_combos) : nothing

    xhist = Matrix{Float64}(undef, M, past)
    ϕfull = Vector{Float64}(undef, F)
    ϕcol  = Vector{Float64}(undef, Fsub)

    # ---- Build Φsub (Fsub × train_len) and Y (M × train_len) ---------------
    Φsub = Matrix{Float64}(undef, Fsub, train_len)
    Y    = Matrix{Float64}(undef, M,    train_len)

    for col in 1:train_len
        t = t_start + col - 1
        for j in 1:past
            xhist[:, j] .= X[:, t - (j - 1) * delay]   # strided lags
        end
        if Fsub == F
            fill_features_full!(ϕfull, xhist, poly_combos)
            Φsub[:, col] .= ϕfull
        else
            fill_features_selected!(ϕcol, xhist, fmap::FeatureMap, feat_idx, lin_i, lin_j)
            Φsub[:, col] .= ϕcol
        end
        Y[:, col] .= X[:, t + 1]    # next-step target
    end

    # Gaussian input-noise regularization
    noise_std > 0 && (Φsub .+= noise_std .* randn(rng, Fsub, train_len))

    # ---- Ridge regression ---------------------------------------------------
    K = Φsub * Φsub'
    for i in 1:Fsub; K[i, i] += ridge; end
    Wout = Matrix((cholesky!(Symmetric(K)) \ (Φsub * Y'))')  # M × Fsub

    preds_train = Wout * Φsub   # M × train_len

    # ---- Closed-loop prediction ---------------------------------------------
    # Ring buffer of depth (1 + (past-1)*delay): newest state at column 1.
    buf_size = 1 + (past - 1) * delay
    buf = Matrix{Float64}(undef, M, buf_size)
    for s in 1:buf_size
        buf[:, s] .= X[:, t_end + 1 - (s - 1)]
    end

    preds_test = Matrix{Float64}(undef, M, predict_len)

    for k in 1:predict_len
        for j in 1:past
            xhist[:, j] .= buf[:, 1 + (j - 1) * delay]   # sample at stride
        end
        if Fsub == F
            fill_features_full!(ϕfull, xhist, poly_combos)
            ϕcol .= ϕfull
        else
            fill_features_selected!(ϕcol, xhist, fmap::FeatureMap, feat_idx, lin_i, lin_j)
        end

        xnext             = Wout * ϕcol
        preds_test[:, k] .= xnext

        # Shift ring buffer right and insert new state at front
        for s in buf_size:-1:2; buf[:, s] .= buf[:, s - 1]; end
        buf[:, 1] .= xnext
    end

    return preds_test, preds_train, Wout
end