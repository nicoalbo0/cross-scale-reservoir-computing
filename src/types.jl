"""
    ResParams{T<:Real}

Reservoir hyperparameters.

# Fields
- `N`: Number of reservoir units.
- `g`: Spectral radius of the recurrent weight matrix.
- `degree`: Average degree (connectivity) of the recurrent matrix.
- `g_in`: Input scaling (gain).
"""
struct ResParams{T<:Real}
    N::Int
    g::T
    degree::Int
    g_in::T
end

"""
    Reservoir{T<:Real}

Echo state network reservoir: recurrent weights, leak rate, and input projections.

# Fields
- `W`: Sparse recurrent weight matrix (N×N).
- `dt_τ`: Leaking rate factor (dt/τ) for the reservoir state update.
- `W_in_rec`: Input weights for recurrent (local) inputs.
- `W_in_neigh`: Input weights for neighbor inputs.
- `W_in_layer`: Input weights for cross-layer (coarse) inputs.
"""
struct Reservoir{T<:Real}
    W::SparseMatrixCSC{T,Int}
    dt_τ::T
    W_in_rec::Matrix{T}
    W_in_neigh::Matrix{T}
    W_in_layer::Matrix{T}
end

"""
    BlockModel{T<:Real}

Trained readout and state for one spatial block.

# Fields
- `W_out`: Readout matrix (output dim × reservoir size).
- `x`: Current reservoir state vector.
- `rows_rec`: Row indices for recurrent (local) observables.
- `rows_neigh`: Row indices for neighbor observables.
- `rows_layer`: Row indices for cross-layer observables.
"""
struct BlockModel{T<:Real}
    W_out::Matrix{T}
    x::Vector{T}
    rows_rec::Vector{Int}
    rows_neigh::Vector{Int}
    rows_layer::Vector{Int}
end