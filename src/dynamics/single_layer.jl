"""
    build_W_in(N, rec_dim, neigh_dim, layer_dim, g_rec, g_neigh, g_layer; mode=:structured)

Build input weight matrices (W_in_rec, W_in_neigh, W_in_layer) for the reservoir.
- `mode=:structured`: one contiguous block of neurons per input dimension, scaled by gains.
- `mode=:random`: intended to build random full matrices with the given dimensions and gains;
  the current implementation has a bug (undefined variables in that branch).
Returns `(W_in_rec, W_in_neigh, W_in_layer)`.
"""
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

    if mode == :structured

        # global matrix
        W = zeros(N, D)

            gains = vcat(
            fill(g_rec,   rec_dim),
            fill(g_neigh, neigh_dim),
            fill(g_layer, layer_dim)
        )

        # contiguous neuron blocks
        q = div(N, D)
        for j in 1:D
            rows = (j-1)*q+1 : j*q
            W[rows, j] .= gains[j] .* (2 .* rand(q) .- 1)
        end

        # split by columns (cheap views)
        i1 = rec_dim
        i2 = i1 + neigh_dim

        W_rec   = W[:, 1:i1]
        W_neigh = W[:, i1+1:i2]
        W_layer = W[:, i2+1:end]

    elseif mode == :random

        W_rec   = g_rec .* (2 .* rand(N, rec_dim) .- 1)
        W_neigh = g_neigh .* (2 .* rand(N, neigh_dim) .- 1)
        W_layer = g_layer .* (2 .* rand(N, layer_dim) .- 1)
    end


    return W_rec, W_neigh, W_layer
end

"""
    generate_reservoir(params, input_dimensions; input_mode=:structured)

Build a `Reservoir`: sparse recurrent matrix (spectral radius from params), input weights
via `build_W_in`, and per-neuron dt/τ.
`params` = (N, g, degree, g_in_rec, g_in_neigh, g_in_layer, τ, dt), where `τ` is either
- a scalar `Real` → all N neurons share τ (classical single-timescale reservoir)
- a 2-tuple `(τ_min, τ_max)` → log-uniform distribution of τ across neurons
  (mixed-timescale reservoir: the i-th neuron gets τ = τ_min (τ_max/τ_min)^((i-1)/(N-1)))
- an `AbstractVector` of length N → explicit per-neuron τ.
"""
function generate_reservoir(params::Tuple, input_dimensions::Tuple{Int, Int, Int}; input_mode::Symbol)
    N, g, degree, g_in_rec, g_in_neigh, g_in_layer, τ, dt = params
    rec_dimensions, neigh_dimensions, layer_dimensions = input_dimensions
    sparsity = degree / N
    T = typeof(float(g))

    W = sprandn(T, N, N, sparsity)
    ρ = maximum(abs.(eigvals(Matrix(W))))
    W .*= T(g) / ρ

    W_in_rec, W_in_neigh, W_in_layer = build_W_in(
        N,
        rec_dimensions,
        neigh_dimensions,
        layer_dimensions,
        T(g_in_rec),
        T(g_in_neigh),
        T(g_in_layer);
        mode = input_mode
    )

    # Per-neuron leak rate.
    dt_τ = Vector{T}(undef, N)
    if isa(τ, Real)
        fill!(dt_τ, T(dt / τ))
    elseif isa(τ, Tuple) && length(τ) == 2
        τ_min, τ_max = T(τ[1]), T(τ[2])
        @assert τ_min > 0 && τ_max > 0 "τ_min and τ_max must be positive"
        # log-uniform from τ_min to τ_max across the N neurons
        log_τs = range(log(τ_min), log(τ_max); length=N)
        @inbounds for k in 1:N
            dt_τ[k] = T(dt) / exp(log_τs[k])
        end
    elseif isa(τ, AbstractVector)
        length(τ) == N || error("τ vector length ($(length(τ))) must equal N ($N)")
        @inbounds for k in 1:N
            dt_τ[k] = T(dt / τ[k])
        end
    else
        error("Unsupported τ type: $(typeof(τ)). Use Real, Tuple{Real,Real}, or AbstractVector.")
    end

    return Reservoir{T}(W, dt_τ, W_in_rec, W_in_neigh, W_in_layer)
end

"""
    fit_ridge_regression(X, Y, ridge, washout; mode=:linear)

Ridge regression readout: use columns from `washout+1` to end of X and Y; optionally apply
`square_even_rows` to the trimmed X when `mode=:quadratic`. Solves (X X' + ridge*I) W_out' = X Y'.
Returns `W_out` with size (size(Y,1) × size(X,1)), i.e. output dimension × reservoir size.
"""
function fit_ridge_regression(X::Matrix{T}, Y::Matrix{T}, ridge::T, washout::Int; mode::Symbol = :linear) where T<:AbstractFloat
    X_cut = X[:, washout+1:end]
    Y = Y[:, washout+1:end]
    N = size(X, 1)

    if mode == :quadratic
        X = square_even_rows(X_cut)

    else mode == :linear
        X = X_cut
    end

    I_N = Matrix{T}(I, N, N)
    XX = Symmetric(X * X' + ridge * I_N)
    W_out = (Y * X') / cholesky(XX)

    return W_out
end

"""
    train_parallel_reservoir(reservoir, data, data_layer, train_time, blocks; washout, ridge_parameter, ...)

Train one readout per block in parallel (teacher forcing). Each block uses local + neighbor + layer
inputs at time t; target at column t is local (recurrent) state at time t; X[:, t] is the reservoir
state before the update at t. Returns `(models, prediction, data[:, 1:train_time])`.
Optional: `show_progress`, `regression_mode` (`:linear` or `:quadratic`).
"""
function train_parallel_reservoir(
    reservoir::Reservoir{T},
    data::Matrix{T},
    data_layer::Matrix{T},
    train_time::Int,
    blocks::Vector{<:NamedTuple};
    washout::Int,
    ridge_parameter::T,
    show_progress::Bool = false,
    regression_mode::Symbol = :quadratic
) where T<:AbstractFloat

    L, _        = size(data)
    num_blocks  = length(blocks)

    N           = size(reservoir.W, 1)

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

            mul!(W_x,          reservoir.W,                x)
            mul!(Win_rec_u,    reservoir.W_in_rec,     u_rec)
            mul!(Win_neigh_u,  reservoir.W_in_neigh, u_neigh)
            mul!(Win_layer_u,  reservoir.W_in_layer, u_layer)

            @inbounds for k in eachindex(x)
                α = reservoir.dt_τ[k]
                x[k] = (1 - α) * x[k] + α * tanh(W_x[k] + Win_rec_u[k] + Win_neigh_u[k] + Win_layer_u[k])
            end

            # target is u_rec at time t+1
            Y[:, t] .= @views data[rows_rec, t]
        end

        W_out = fit_ridge_regression(X, Y, ridge_parameter, washout; mode = regression_mode)

        models[i] = BlockModel{T}(W_out, x, rows_rec, rows_neigh, rows_layer)
        prediction[rows_rec, :] .= W_out * square_even_rows(X)

        if show_progress
            next!(prog)
        end
    end

    return models, prediction, data[:, 1:train_time]
end

"""
    test_parallel_reservoir(reservoir, block_models, data, data_layer, train_time, test_time; warmup, ...)

Run closed-loop prediction: warmup with true data from (train_time - warmup) to train_time,
then autonomous prediction for `test_time` steps. Layer input uses true coarse data.
Returns `(predictions, X)` where X is the list of state matrices per block.
"""
function test_parallel_reservoir(
    reservoir::Reservoir{T},
    block_models::Vector{BlockModel{T}},
    data::Matrix{T},
    data_layer::Matrix{T},
    train_time::Int,
    test_time::Int;
    warmup::Int,
    regression_mode::Symbol = :quadratic
) where T<:Real

    L, Ttot = size(data)
    num_blocks = length(block_models)
    N = size(reservoir.W, 1)

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

            mul!(W_x, reservoir.W, bm.x)
            mul!(Win_rec_u,   reservoir.W_in_rec,   u_rec)
            mul!(Win_neigh_u, reservoir.W_in_neigh, u_neigh)
            mul!(Win_layer_u, reservoir.W_in_layer, u_layer)

            @inbounds for k in eachindex(bm.x)
                α = reservoir.dt_τ[k]
                bm.x[k] = (1 - α) * bm.x[k] + α * tanh(W_x[k] + Win_rec_u[k] + Win_neigh_u[k] + Win_layer_u[k])
            end
        end
    end

    # ---------- autonomous prediction ----------
    for t in 1:test_time

        # (1) compute each block output from current state, necessary for neighbors mixing
        for (i, bm) in enumerate(block_models)
            X[i][:, warmup + t] .= bm.x

            z = copy(bm.x)

            if regression_mode == :quadratic
                for k in 2:2:length(z)
                    z[k] = z[k]^2
                end
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

            mul!(W_x,                reservoir.W,      bm.x)
            mul!(Win_rec_u,   reservoir.W_in_rec,     u_rec)
            mul!(Win_neigh_u, reservoir.W_in_neigh, u_neigh)
            mul!(Win_layer_u, reservoir.W_in_layer, u_layer)

            @inbounds for k in eachindex(bm.x)
                α = reservoir.dt_τ[k]
                bm.x[k] = (1 - α) * bm.x[k] + α * tanh(W_x[k] + Win_rec_u[k] + Win_neigh_u[k] + Win_layer_u[k])
            end
        end
    end

    return predictions, X
end

"""
    run_single_layer(params, data, data_layer, train_time, test_time, blocks; washout, warmup, ridge_parameter, ...)

Full single-layer pipeline: build reservoir, train block readouts, run test. Returns
`(preds, training_prediction, training_data, X, block_models)`.
"""
function run_single_layer(
    params::Tuple,
    data::Matrix{T},
    data_layer::Matrix{T},
    train_time::Int,
    test_time::Int,
    blocks::Vector{<:NamedTuple};
    washout::Int,
    warmup::Int,
    ridge_parameter::T,
    show_progress::Bool = false,
    input_mode::Symbol = :structured,
    regression_mode::Symbol = :quadratic
) where T<:Real

    L, Ttot     = size(data)
    @assert train_time + test_time <= Ttot

    # block input dimension implied by your make_blocks topology
    dimensions = input_dimensions(blocks)

    reservoir = generate_reservoir(params, dimensions; input_mode=input_mode)

    block_models, training_prediction, training_data =
        train_parallel_reservoir(
            reservoir, data, data_layer, train_time, blocks;
            washout=washout,
            ridge_parameter=ridge_parameter,
            show_progress=show_progress,
            regression_mode=regression_mode
        )

    # start test at train_time (with warmup window ending at train_time)
    preds, X = test_parallel_reservoir(
        reservoir, block_models, data, data_layer, train_time, test_time;
        warmup=warmup, regression_mode=regression_mode
    )

    return preds, training_prediction, training_data, X, block_models
end