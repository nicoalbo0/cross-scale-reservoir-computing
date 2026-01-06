function generate_reservoir(params::Tuple{Int, T, Int, T, T, T}, input_dimensions::Tuple{Int, Int, Int}; input_mode::Symbol) where T<:Real
    N, g, degree, g_in_rec, g_in_neigh, g_in_layer = params
    rec_dimensions, neigh_dimensions, layer_dimensions = input_dimensions
    sparsity = degree / N

    W = sprand(N, N, sparsity)
    ρ = maximum(abs.(eigvals(Matrix(W))))
    W .*= g / ρ

    # input weight matrix
    W_in_rec   = g_in_rec .* (2 .* rand(N, rec_dimensions) .- 1)
    W_in_neigh = g_in_neigh .* (2 .* rand(N, neigh_dimensions) .- 1)
    W_in_layer = g_in_layer .* (2 .* rand(N, layer_dimensions) .- 1)
    
    return Reservoir{T}(W, W_in_rec, W_in_neigh, W_in_layer)
end

function fit_ridge_regression(X::Matrix{T}, Y::Matrix{T}, ridge::T, washout::Int) where T<:AbstractFloat
    X_cut = X[:, washout+1:end]
    Y = Y[:, washout+1:end]
    N = size(X, 1)

    X = square_even_rows(X_cut)

    I_N = Matrix{T}(I, N, N)
    XX = Symmetric(X * X' + ridge * I_N)
    #W_out = (Y * X') * LinearAlgebra.inv!(cholesky(XX))
    W_out = (Y * X') / cholesky(XX)

    return W_out
end

function train_parallel_reservoir(
    res::Reservoir{T},
    data::Matrix{T},
    data_layer::Matrix{T},
    train_time::Int,
    blocks::Vector{BlockType};
    washout::Int,
    ridge_parameter::T,
    show_progress::Bool = false
) where T<:AbstractFloat

    L, _        = size(data)
    num_blocks  = length(blocks)

    N           = size(res.W, 1)

    models = Vector{BlockModel{T}}(undef, num_blocks)
    prediction = zeros(T, L, train_time)

    prog = show_progress ? Progress(num_blocks; desc="Training", barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',), barlen=10, color=:white) : nothing

    Threads.@threads for i in 1:num_blocks
        rows_rec, rows_neigh, rows_layer = blocks[i]
        x = zeros(T, N)

        X = Matrix{T}(undef, N, train_time)
        Y = Matrix{T}(undef, length(rows_rec), train_time)

        W_x         = zeros(T, N)
        Win_rec_u   = zeros(T, N)
        Win_neigh_u = zeros(T, N)
        Win_layer_u = zeros(T, N)

        for t in 1:train_time

            X[:, t] .= x

            # teacher forcing at time t
            u_rec   = data[rows_rec, t]
            u_neigh = data[rows_neigh, t]
            u_layer = data_layer[rows_layer, t]

            mul!(W_x,          res.W,                x)
            mul!(Win_rec_u,    res.W_in_rec,     u_rec)
            mul!(Win_neigh_u,  res.W_in_neigh, u_neigh)
            mul!(Win_layer_u,  res.W_in_layer, u_layer)

            @inbounds for k in eachindex(x)
                x[k] = tanh(W_x[k] + Win_rec_u[k] + Win_neigh_u[k] + Win_layer_u[k])
            end

            # target is u_rec at time t+1
            Y[:, t] .= @views data[rows_rec, t]
        end

        W_out = fit_ridge_regression(X, Y, ridge_parameter, washout)

        models[i] = BlockModel{T}(W_out, x, rows_rec, rows_neigh, rows_layer)
        prediction[rows_rec, :] .= W_out * square_even_rows(X)

        if show_progress
            next!(prog)
        end
    end

    return models, prediction, data[:, 1:train_time]
end

function test_parallel_reservoir(
    res::Reservoir{T},
    block_models::Vector{BlockModel{T}},
    data::Matrix{T},
    data_layer::Matrix{T},
    train_time::Int,
    test_time::Int;
    warmup::Int
) where T<:Real

    L, Ttot = size(data)
    num_blocks = length(block_models)
    N = size(res.W, 1)

    W_x         = zeros(T, N)
    Win_rec_u   = zeros(T, N)
    Win_neigh_u = zeros(T, N)
    Win_layer_u = zeros(T, N)

    t0 = train_time - warmup
    @assert train_time + test_time ≤ Ttot

    for bm in block_models
        fill!(bm.x, zero(T))
    end

    predictions   = zeros(T, L, test_time + warmup)
    block_outputs = Vector{Vector{T}}(undef, num_blocks)
    uhat          = zeros(T, L)

    X = [Matrix{T}(undef, N, warmup + test_time) for _ in 1:num_blocks]

    # ---------- warmup with true data ----------
    for t in (t0+1):(t0+warmup)
        col = t - t0  # 1..warmup
        for (i, bm) in enumerate(block_models)
            X[i][:, col] .= bm.x

            # diagnostic output
            z = copy(bm.x)
            for k in 2:2:length(z)
                z[k] = z[k]^2
            end
            
            predictions[bm.rows_rec, col] .= bm.W_out * z

            u_rec   = @views data[bm.rows_rec, t]
            u_neigh = @views data[bm.rows_neigh, t]
            u_layer = @views data_layer[bm.rows_layer, train_time - warmup + col]  # aligned

            mul!(W_x, res.W, bm.x)
            mul!(Win_rec_u,   res.W_in_rec,   u_rec)
            mul!(Win_neigh_u, res.W_in_neigh, u_neigh)
            mul!(Win_layer_u, res.W_in_layer, u_layer)

            @inbounds for k in eachindex(bm.x)
                bm.x[k] = tanh(W_x[k] + Win_rec_u[k] + Win_neigh_u[k] + Win_layer_u[k])
            end
        end
    end

    # ---------- autonomous prediction ----------
    for t in 1:test_time

        # (1) compute each block output from current state, necessary for neighbors mixing
        for (i, bm) in enumerate(block_models)
            X[i][:, warmup + t] .= bm.x

            z = copy(bm.x)
            for k in 2:2:length(z)
                z[k] = z[k]^2
            end
            y = bm.W_out * z
            block_outputs[i] = y
            predictions[bm.rows_rec, warmup + t] .= y
        end

        # (2) advance each block state using closed-loop rec + mixed neighbors
        for (i, bm) in enumerate(block_models)
            u_rec   = @views predictions[bm.rows_rec, warmup + t]
            u_neigh = @views predictions[bm.rows_neigh, warmup + t]
            u_layer = @views data_layer[bm.rows_layer, train_time + t]  # aligned

            mul!(W_x,                res.W,      bm.x)
            mul!(Win_rec_u,   res.W_in_rec,     u_rec)
            mul!(Win_neigh_u, res.W_in_neigh, u_neigh)
            mul!(Win_layer_u, res.W_in_layer, u_layer)

            @inbounds for k in eachindex(bm.x)
                bm.x[k] = tanh(W_x[k] + Win_rec_u[k] + Win_neigh_u[k] + Win_layer_u[k])
            end
        end
    end

    return predictions, X
end

function run_single_layer(
    params::Tuple{Int, T, Int, T, T, T},
    data::Matrix{T},
    data_layer::Matrix{T},
    train_time::Int,
    test_time::Int,
    blocks::Vector{BlockType};
    washout::Int,
    warmup::Int,
    ridge_parameter::T,
    show_progress::Bool = false,
    input_mode::Symbol
) where T<:Real

    L, Ttot     = size(data)
    @assert train_time + test_time <= Ttot

    # block input dimension implied by your make_blocks topology
    dimensions = input_dimensions(blocks)

    res = generate_reservoir(params, dimensions; input_mode=input_mode)

    block_models, training_prediction, training_data =
        train_parallel_reservoir(
            res, data, data_layer, train_time, blocks;
            washout=washout,
            ridge_parameter=ridge_parameter,
            show_progress=show_progress
        )

    # start test at train_time (with warmup window ending at train_time)
    preds, X = test_parallel_reservoir(
        res, block_models, data, data_layer, train_time, test_time;
        warmup=warmup
    )

    return preds, training_prediction, training_data, X, block_models
end