struct ResParams{T<:Real}
    N::Int
    g::T
    degree::Int
    g_in::T
end

struct Reservoir{T<:Real}
    W::SparseMatrixCSC{T,Int}
    dt_τ::T
    W_in_rec::Matrix{T}             
    W_in_neigh::Matrix{T}
    W_in_layer::Matrix{T}
end

struct BlockModel{T<:Real}
    W_out::Matrix{T}  
    x::Vector{T}
    rows_rec::Vector{Int}    
    rows_neigh::Vector{Int}
    rows_layer::Vector{Int}
end

const BlockType = @NamedTuple{rows_rec::UnitRange{Int64}, rows_neigh::Vector{Int64}, rows_layer::Vector{Int64}}