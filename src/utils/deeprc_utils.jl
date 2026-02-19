"""
    augment_features(x)

Augment feature vector with quadratic terms (square_even_rows pattern).
Input: [x₁, x₂, x₃, x₄, ...] → Output: [x₁, x₂², x₃, x₄², ...].
"""
function augment_features(x::AbstractVector)
    x_aug = copy(x)
    for k in 2:2:length(x_aug)
        x_aug[k] = x_aug[k]^2
    end
    return x_aug
end

"""
    augment_features!(x_aug, x)

In-place augmentation: x_aug[1] = x[1], x_aug[2] = x[2]², etc. Assumes `x_aug` is
pre-allocated and same length as `x`. More efficient than allocating a new vector.
"""
function augment_features!(x_aug::AbstractVector, x::AbstractVector)
    copyto!(x_aug, x)
    for k in 2:2:length(x_aug)
        x_aug[k] = x_aug[k]^2
    end
end

"""
    augment_features_matrix(X)

Augment feature matrix column-by-column (square_even_rows pattern). X is (N×T);
each column becomes [x₁; x₂²; x₃; x₄²; ...]. Returns (N×T) matrix.
"""
function augment_features_matrix(X::AbstractMatrix)
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

"""
    DeepESN

Deep Echo State Network: multiple recurrent layers, quadratic feature augmentation
for readout. Fields: `nu`, `ny`, `nl`, `nr`, `Win`, `Wrec`, `leak`, `x`, `tmp`, `Wout`, `x_aug`.
"""
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

"""
    DeepESN(nu, ny; nl=3, nr=100, rho=0.9, leak=1.0, input_scale=1.0, inter_scale=1.0, sparsity=0.01, rng=...)

Construct a DeepESN with `nu` inputs, `ny` outputs, `nl` layers, `nr` units per layer.
Recurrent matrices scaled to `rho`; optional `rng` for reproducibility.
"""
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
    
    nfeat = nr * nl
    Wout = zeros(ny, nfeat)
    x_aug = zeros(nfeat)
    
    return DeepESN(nu, ny, nl, nr, Win, Wrec, leak_v, x, tmp, Wout, x_aug)
end

"""
    DeepESN_train!(m, U, Y; washout=0, ridge=1e-6, reset_state=true, return_output=false)

Train DeepESN readout on inputs `U` and targets `Y` using ridge regression on
augmented states. If `return_output=true`, returns predicted Y over the full sequence.
"""
function DeepESN_train!(m::DeepESN, U::AbstractMatrix, Y::AbstractMatrix;
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
    
    X_aug = augment_features_matrix(X)  # Transform: [x₁ x₂ x₃ ...] → [x₁ x₂² x₃ x₄² ...] , so dimensionality is still ny
    
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

"""
    test_closed_loop(m; steps, warmup=zeros(nu,0), y_init=nothing, reset_state=true)

Run closed-loop prediction for `steps` steps. Optional `warmup` matrix (nu×Tw) and
`y_init` for first feedback. Returns (ny × (Tw+steps)) prediction matrix.
"""
function DeepESN_test_closed_loop(m::DeepESN;
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