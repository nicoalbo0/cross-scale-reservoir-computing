@testset "types" begin
    using SparseArrays
    using LinearAlgebra

    # ResParams
    rp = CrossScaleRC.ResParams{Float64}(10, 0.9, 4, 0.5)
    @test rp.N == 10
    @test rp.g == 0.9
    @test rp.degree == 4
    @test rp.g_in == 0.5

    # Reservoir (minimal valid construction)
    N = 4
    W = sparse(I, N, N) .* 0.5
    dt_τ = 1.0
    W_in_rec = zeros(N, 2)
    W_in_neigh = zeros(N, 0)
    W_in_layer = zeros(N, 0)
    res = CrossScaleRC.Reservoir{Float64}(W, dt_τ, W_in_rec, W_in_neigh, W_in_layer)
    @test size(res.W) == (N, N)
    @test res.dt_τ == 1.0
    @test size(res.W_in_rec) == (N, 2)

    # BlockModel (minimal valid construction)
    W_out = zeros(2, N)
    x = zeros(N)
    rows_rec = [1, 2]
    rows_neigh = Int[]
    rows_layer = Int[]
    bm = CrossScaleRC.BlockModel{Float64}(W_out, x, rows_rec, rows_neigh, rows_layer)
    @test size(bm.W_out) == (2, N)
    @test length(bm.x) == N
    @test bm.rows_rec == [1, 2]
end
