# Activate environment
using Pkg, Revise
Pkg.activate(".")
Pkg.instantiate()

using HierarchicalRC
using ReservoirComputing
using LinearAlgebra
using Plots, Measures, LaTeXStrings
using Random, Statistics
using SparseArrays

# Set threading
BLAS.set_num_threads(6)

# ============================================================================
# UTILITY FUNCTIONS: Feature Augmentation (quadratic style)
# ============================================================================

function augment_features(x::AbstractVector)
    """
    Augment feature vector with quadratic terms (square_even_rows pattern).
    Input:  [x₁, x₂, x₃, x₄, ...]
    Output: [x₁, x₂², x₃, x₄², ...]
    
    Replicates the behavior of square_even_rows() from single_layer.jl
    """
    x_aug = copy(x)
    for k in 2:2:length(x_aug)
        x_aug[k] = x_aug[k]^2
    end
    return x_aug
end

function augment_features!(x_aug::AbstractVector, x::AbstractVector)
    """
    In-place augmentation: x_aug[1] = x[1], x_aug[2] = x[2]², etc.
    Assumes x_aug is pre-allocated and same length as x.
    More efficient than allocating new vector each time.
    """
    copyto!(x_aug, x)
    for k in 2:2:length(x_aug)
        x_aug[k] = x_aug[k]^2
    end
end

function augment_features_matrix(X::AbstractMatrix)
    """
    Augment feature matrix column-by-column (square_even_rows pattern).
    
    Input:  X with shape (N, T) where each column is [x₁; x₂; ...; xₙ]
    Output: X_aug with shape (N, T) where column t: [x₁; x₂²; x₃; x₄²; ...]
    
    Matches fit_ridge_regression() behavior in single_layer.jl which calls
    X = square_even_rows(X_cut) before ridge regression.
    """
    X_aug = copy(X)
    N, T = size(X_aug)
    for t in 1:T
        for k in 2:2:N
            X_aug[k, t] = X_aug[k, t]^2
        end
    end
    return X_aug
end

# ============================================================================
# DeepESN with Quadratic Feature Augmentation
# ============================================================================

mutable struct DeepESN
    nu::Int # input dim.
    ny::Int # out dim.
    nl::Int # n. layers
    nr::Int # n. units per layer
    Win::Vector{Matrix{Float64}}      # Win[1]: (nr×nu), Win[l>1]: (nr×nr) inter-layer
    Wrec::Vector{Matrix{Float64}}     # recurrent (nr×nr) per layer
    leak::Vector{Float64}             # leaking rate a^(l)
    x::Matrix{Float64}                # states (nr×nl), column l is x^(l)(t)
    tmp::Matrix{Float64}              # preactivation buffer (nr×nl)
    Wout::Matrix{Float64}             # readout (ny×nr*nl) - applied to AUGMENTED features
    x_aug::Vector{Float64}            # buffer for augmented state vector
end

function DeepESN(nu::Integer, ny::Integer;
        nl::Integer = 3,
        nr::Integer = 100,
        rho = 0.9,
        leak = 1.0,
        input_scale = 1.0,
        inter_scale = 1.0,
        sparsity = 0.01,
        rng = Random.default_rng())
    
    nu = Int(nu); ny = Int(ny); nl = Int(nl); nr = Int(nr)
    
    rho_v = rho isa Number ? fill(Float64(rho), nl) : Float64.(collect(rho))
    leak_v = leak isa Number ? fill(Float64(leak), nl) : Float64.(collect(leak))
    @assert length(rho_v) == nl
    @assert length(leak_v) == nl
    
    Win = Vector{Matrix{Float64}}(undef, nl)
    Wrec = Vector{Matrix{Float64}}(undef, nl)
    
    Win[1], _, _ = build_W_in(
        nr,
        nu,
        0,
        0,
        input_scale,
        0.0,
        0.0;
        mode = :structured
    )
    for l in 2:nl
        Win[l] = (2rand(rng, nr, nr) .- 1) .* inter_scale
    end
    
    for l in 1:nl
        W = sprandn(rng, nr, nr, sparsity)
        sr = maximum(abs, eigvals(Matrix(W)))
        Wrec[l] = (rho_v[l] / sr) .* W
    end
    
    x = zeros(nr, nl)
    tmp = zeros(nr, nl)
    
    # Note: Wout dimension is (ny × nr*nl) because it operates on augmented features
    # But the augmentation happens externally in train!() and test_closed_loop()
    nfeat = nr * nl
    Wout = zeros(ny, nfeat)
    x_aug = zeros(nfeat)
    
    return DeepESN(nu, ny, nl, nr, Win, Wrec, leak_v, x, tmp, Wout, x_aug)
end

# Training with feature augmentation
function train!(m::DeepESN, U::AbstractMatrix, Y::AbstractMatrix;
        washout::Integer = 0,
        ridge::Real = 1e-6,
        reset_state::Bool = true,
        return_output::Bool = false)
    
    nu, T = size(U)
    ny, Ty = size(Y)
    @assert nu == m.nu
    @assert ny == m.ny
    @assert T == Ty
    @assert 0 <= washout < T
    
    reset_state && fill!(m.x, 0.0)
    
    nfeat = m.nr * m.nl
    Teff = T - washout
    X = Matrix{Float64}(undef, nfeat, Teff)
    Yeff = Matrix{Float64}(undef, m.ny, Teff)
    
    Xall = return_output ? Matrix{Float64}(undef, nfeat, T) : nothing
    
    k = 0
    @views for t in 1:T
        # State update: layer-by-layer pipeline at same time step
        for l in 1:m.nl
            x_l = view(m.x, :, l)
            pre = view(m.tmp, :, l)
            
            if l == 1
                mul!(pre, m.Win[l], view(U, :, t))
            else
                mul!(pre, m.Win[l], view(m.x, :, l - 1))
            end
            mul!(pre, m.Wrec[l], x_l, 1.0, 1.0)
            
            a = m.leak[l]
            @inbounds for i in 1:m.nr
                x_l[i] = (1 - a) * x_l[i] + a * tanh(pre[i])
            end
        end
        
        if return_output
            # Store raw state vector
            copyto!(view(Xall, :, t), vec(m.x))
        end
        
        if t > washout
            k += 1
            # Store raw state vector in feature matrix
            copyto!(view(X, :, k), vec(m.x))
            copyto!(view(Yeff, :, k), view(Y, :, t))
        end
    end
    
    X_aug = augment_features_matrix(X)  # Transform: [x₁ x₂ x₃ ...] → [x₁ x₂² x₃ x₄² ...]
    
    # Ridge readout on augmented features: Wout = (Y Φ') (Φ Φ' + ridge*I)^(-1)
    A = X_aug * X_aug'
    @inbounds for i in 1:nfeat
        A[i, i] += ridge
    end
    C = Yeff * X_aug'
    m.Wout .= C / A
    
    if return_output
        Yhat = similar(Y, Float64)
        @views for t in 1:T
            # Apply augmentation to stored raw states
            augment_features!(m.x_aug, view(Xall, :, t))
            mul!(view(Yhat, :, t), m.Wout, m.x_aug)
        end
        return Yhat
    end
    
    return nothing
end

# Testing with feature augmentation
function test_closed_loop(m::DeepESN;
        steps::Integer,
        warmup::AbstractMatrix = zeros(m.nu, 0),
        y_init = nothing,
        reset_state::Bool = true)
    
    reset_state && fill!(m.x, 0.0)
    
    Tw = size(warmup, 2)
    @assert size(warmup, 1) == m.nu
    @assert steps >= 1
    @assert m.nu == m.ny
    
    nfeat = m.nr * m.nl
    y_prev = y_init === nothing ? zeros(Float64, m.nu) : Float64.(collect(y_init))
    @assert length(y_prev) == m.nu
    
    T = Tw + Int(steps)
    Yhat = Matrix{Float64}(undef, m.ny, T)
    
    @views for t in 1:T
        u = (t <= Tw) ? view(warmup, :, t) : y_prev
        
        # State update
        for l in 1:m.nl
            x_l = view(m.x, :, l)
            pre = view(m.tmp, :, l)
            
            if l == 1
                mul!(pre, m.Win[l], u)
            else
                mul!(pre, m.Win[l], view(m.x, :, l - 1))
            end
            mul!(pre, m.Wrec[l], x_l, 1.0, 1.0)
            
            a = m.leak[l]
            @inbounds for i in 1:m.nr
                x_l[i] = (1 - a) * x_l[i] + a * tanh(pre[i])
            end
        end
        
        augment_features!(m.x_aug, vec(m.x))  # In-place: [x₁ x₂ x₃ ...] → [x₁ x₂² x₃ x₄² ...]
        
        # Prediction using augmented features
        mul!(view(Yhat, :, t), m.Wout, m.x_aug)
        
        y_prev = view(Yhat, :, t)  # feedback
    end
    
    return Yhat
end


# ============================================================================
# Main Script
# ============================================================================

##
# Data parameters
println("Loading KS data...")
Q = 64
L = 22
μ = 0.01
resolution_divisor = 4
Qeffective = div(Q, resolution_divisor)

data, τ = load_data(Q, L, μ; show_data=false, interpolate_data=false)
# regrid to lower the resolution
data = regrid_average(data, resolution_divisor)

# Experiment configuration
washout = 1_000
train_len = 50_000
warmup = 1_000
predict_len = 1_000

M_data, T_tot = size(data)
train_len + predict_len ≤ T_tot || error("Not enough data")

# Deep RC hyperparameters
nu = Qeffective
ny = Qeffective
nl = 2
nr = 500

reservoir_params = Dict(
    :radius => 0.1,
    :sparsity => 10/nr,
    :input_scale => 2.5/√(Qeffective),
    :leaky_coeff => 1.0,
    :ridge_param => 1e-4
)

# ============================================================================
# TRAINING PHASE
# ============================================================================
input_data = data[:, 1:(washout + train_len - 1)]
target_data = data[:, 2:(washout + train_len)]

println("\\n" * "="^70)
println("DEEP RESERVOIR COMPUTING ON KURAMOTO–SIVASHINSKY EQUATION")
println("="^70)
println("Data shape: $(size(data))")
println("Training samples: $train_len")
println("Prediction horizon: $predict_len")
println("Number of layers: $nl")
println("Layer sizes: $nr")
println("="^70 * "\\n")

# Build and train DeepESN
rng = MersenneTwister(42)


deep_rc = DeepESN(nu, ny;
    nl = nl,
    nr = nr,
    rho = reservoir_params[:radius],
    leak = reservoir_params[:leaky_coeff],
    input_scale = reservoir_params[:input_scale],
    inter_scale = 1.0,
    sparsity = reservoir_params[:sparsity],
    rng = rng
)

println("\\nTraining readout...")
preds_train = train!(deep_rc, input_data, target_data;
    washout = washout,
    ridge = reservoir_params[:ridge_param],
    reset_state = true,
    return_output = true
)

println("\\nDeep RC architecture built successfully!")

# ============================================================================
# TESTING PHASE: Use DIFFERENT data (after training_len)
# ============================================================================
##
println("Generating test predictions...")
test_start_idx = train_len - warmup + 1    # = 49_001
warmup_u = data[:, test_start_idx:(test_start_idx + warmup - 1)]

# Now predict
preds_test = test_closed_loop(deep_rc;
    steps = predict_len,
    warmup = warmup_u,
    reset_state = true
)

# Extract the corresponding true test data
test_data = data[:, (test_start_idx + warmup):(test_start_idx + warmup + predict_len - 1)]


println("\\nTest data range: indices $(test_start_idx + warmup) to $(test_start_idx + warmup + predict_len - 1)")
println("Training data range: indices 1 to $(washout + train_len)")
println("No overlap: $(test_start_idx + warmup) > $(washout + train_len) is $(test_start_idx + warmup > washout + train_len)")

## Plotting
h1 = heatmap(test_data, clim=(-3, 3), cmap=:RdBu)
h2 = heatmap(preds_test[:, warmup+1:end], clim=(-3, 3), cmap=:RdBu)
h3 = heatmap(test_data - preds_test[:, warmup+1:end], clim=(-3, 3), cmap=:RdBu)
vline!(h3, [1], lw=2, color=:red, legend=false)

error_curve = collect(
    rmse_upto(test_data[:, 1:end], preds_test[:, warmup+1:end]; T=t) 
    for t in 1:size(test_data, 2)
)

p1 = plot(error_curve, grid=false, lw=2, color=:black, 
    ylabel="rmse_upto", xlabel="timestep", legend=false, title="Total err.")

display(plot(h1, h2, h3, p1, size=(800, 500)))