# Activate environment
using Pkg, Revise
Pkg.activate(".")
Pkg.instantiate()

using HierarchicalRC
using LinearAlgebra
using Plots

BLAS.set_num_threads(6)

# -------------------------------------------------------------------
# Data
# -------------------------------------------------------------------
Q = 64
L = 22
μ = 0.01
resolution_divisor = 4
Qeffective = div(Q, resolution_divisor)

data, τ = load_data(Q, L, μ; show_data=false, interpolate_data=false)
data = regrid_average(data, resolution_divisor)

# Experiment configuration
washout     = 1_000
train_len   = 50_000
predict_len = 1_000

M, Ttot = size(data)
washout + train_len + predict_len + 2 ≤ Ttot || error("Not enough data")

# -------------------------------------------------------------------
# NextGen-1 (NG-RC) implementation
#   - past = k (number of past values, i.e. delay taps, skip s=1)
#   - degree = p (polynomial degree)
# -------------------------------------------------------------------

# Generate all index-multisets of length k from 1:n (combinations with repetition).
# Returned as a Vector{Vector{Int}} where each vector is nondecreasing.
function combs_with_repetition(n::Int, k::Int)
    out = Vector{Vector{Int}}()
    buf = Vector{Int}(undef, k)

    function rec(pos::Int, start::Int)
        if pos > k
            push!(out, copy(buf))
            return
        end
        @inbounds for i in start:n
            buf[pos] = i
            rec(pos + 1, i)
        end
    end

    rec(1, 1)
    return out
end

function feature_length(nlin::Int, degree::Int)
    # 1 constant + nlin linear + sum_{p=2..degree} binomial(nlin+p-1, p)
    f = 1 + nlin
    for p in 2:degree
        f += binomial(nlin + p - 1, p)
    end
    return f
end

function fill_features!(
    ϕ::Vector{Float64},
    xhist::Matrix{Float64},          # size (M, past), col 1 is current, col 2 previous, ...
    poly_combos::Vector{Vector{Vector{Int}}}
)
    M, past = size(xhist)
    nlin = M * past

    # constant
    ϕ[1] = 1.0

    # linear terms (stack columns: [x(t); x(t-1); ...])
    pos = 2
    @inbounds for j in 1:past
        for i in 1:M
            ϕ[pos] = xhist[i, j]
            pos += 1
        end
    end

    # polynomial monomials on the linear stack (unique monomials)
    # Indices refer to the stacked linear segment ϕ[2 : 1+nlin]
    @inbounds for combs_k in poly_combos
        for idxs in combs_k
            prod = 1.0
            for idx in idxs
                prod *= ϕ[1 + idx]   # shift by 1 because ϕ[1] is constant
            end
            ϕ[pos] = prod
            pos += 1
        end
    end

    return ϕ
end

function nextgen_closedloop(
    data::AbstractMatrix{<:Real},
    train_len::Int,
    predict_len::Int;
    washout::Int = 0,
    past::Int = 2,
    degree::Int = 2,
    ridge::Float64 = 1e-6
)
    X = Matrix{Float64}(data)
    M, T = size(X)

    # Indices: features at time t, target is X[:, t+1] - X[:, t]
    t_start = washout + past
    t_end   = t_start + train_len - 1
    t_end + 1 ≤ T || error("Not enough samples for train_len given washout/past.")

    # Precompute polynomial index sets for each order 2..degree
    nlin = M * past
    poly_combos = Vector{Vector{Vector{Int}}}()
    for p in 2:degree
        push!(poly_combos, combs_with_repetition(nlin, p))
    end

    F = feature_length(nlin, degree)
    N = train_len

    Φ  = Matrix{Float64}(undef, F, N)      # feature matrix
    dY = Matrix{Float64}(undef, M, N)      # targets: ΔX

    ϕ = Vector{Float64}(undef, F)
    xhist = Matrix{Float64}(undef, M, past)

    # --- Build training matrices (teacher forcing for features) ---
    @inbounds for (col, t) in enumerate(t_start:t_end)
        # history: [t, t-1, ..., t-past+1]
        for j in 1:past
            @views xhist[:, j] .= X[:, t - (j - 1)]
        end
        fill_features!(ϕ, xhist, poly_combos)
        @views Φ[:, col] .= ϕ
        @views dY[:, col] .= X[:, t + 1] .- X[:, t]
    end

    # --- Ridge regression: Wout = dY * Φ' * (Φ*Φ' + ridge*I)^(-1) ---
    K = Symmetric(Φ * Φ' .+ ridge * I(F))
    Wout = (dY * Φ') / K

    # --- One-step-ahead predictions over training segment (still using true histories) ---
    preds_train = Matrix{Float64}(undef, M, train_len)
    @inbounds for (k, t) in enumerate(t_start:t_end)
        # reuse Φ[:, k] already computed as feature vector at time t
        Δ = Wout * @view(Φ[:, k])
        @views preds_train[:, k] .= X[:, t] .+ Δ
    end

    # --- Closed-loop prediction ---
    # Initialize from the last available true point (t_end+1) and its past history.
    t0 = t_end + 1
    for j in 1:past
        @views xhist[:, j] .= X[:, t0 - (j - 1)]
    end

    preds_test = Matrix{Float64}(undef, M, predict_len)
    xcur = @view(xhist[:, 1])

    @inbounds for k in 1:predict_len
        fill_features!(ϕ, xhist, poly_combos)
        Δ = Wout * ϕ
        xnext = xcur .+ Δ

        @views preds_test[:, k] .= xnext

        # shift history: newest becomes col 1
        for j in past:-1:2
            @views xhist[:, j] .= xhist[:, j - 1]
        end
        @views xhist[:, 1] .= xnext
    end

    return preds_test, preds_train, Wout, Φ
end

# -------------------------------------------------------------------
# Run NextGen
# -------------------------------------------------------------------
past   = 3      # number of past values (k)
degree = 2      # polynomial degree (p)
ridge  = 1e-4

preds_test, preds_train, Wout, Φ = nextgen_closedloop(
    data,
    train_len,
    predict_len;
    washout = washout,
    past = past,
    degree = degree,
    ridge = ridge
)

# Optional quick plot for one component (e.g. component 1)
t_start = washout + past
train_data = data[:, (t_start + 1):(t_start + train_len)];
test_data  = data[:, (t_start + train_len + 1):(t_start + train_len + predict_len)];

## Plotting
h1 = heatmap(test_data, clim=(-3, 3), cmap=:RdBu);
h2 = heatmap(preds_test, clim=(-3, 3), cmap=:RdBu);
h3 = heatmap(test_data - preds_test, clim=(-3, 3), cmap=:RdBu);
vline!(h3, [1], lw=2, color=:red, legend=false);

error_curve = collect(
    rmse_upto(test_data, preds_test; T=t) 
    for t in 1:size(test_data, 2)
);

p1 = plot(error_curve, grid=false, lw=2, color=:black, 
    ylabel="rmse_upto", xlabel="timestep", legend=false, title="Total err.")

display(plot(h1, h2, h3, p1, size=(800, 500)))