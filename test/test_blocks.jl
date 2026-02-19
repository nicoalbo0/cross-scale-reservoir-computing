@testset "blocks" begin
    # 1D single-layer
    L = 12
    nblocks = 3
    mixing = 1
    blocks = make_blocks(L, nblocks, mixing)
    @test length(blocks) == nblocks
    for (i, b) in enumerate(blocks)
        @test :rows_rec in keys(b)
        @test :rows_neigh in keys(b)
        @test :rows_layer in keys(b)
        @test length(b.rows_rec) == div(L, nblocks)
        @test length(b.rows_neigh) == 2 * mixing
        @test isempty(b.rows_layer)
    end
    rec_dim, neigh_dim, layer_dim = input_dimensions(blocks)
    @test rec_dim == 4
    @test neigh_dim == 2
    @test layer_dim == 0

    @test_throws Exception make_blocks(10, 3, 1)  # 10 not divisible by 3

    # 1D multi-layer
    L_f = 12
    divisor = 2
    n_fine = 4
    n_coarse = 2
    blocks_ml = make_blocks(L_f, divisor, n_fine, mixing, n_coarse; overlap_mode = :exclude)
    @test length(blocks_ml) == n_fine
    @test !isempty(blocks_ml[1].rows_layer)  # has layer input

    blocks_ml_inc = make_blocks(L_f, divisor, n_fine, mixing, n_coarse; overlap_mode = :include)
    @test length(blocks_ml_inc) == n_fine

    # linear_index (2D, internal)
    nlat = 5
    @test CrossScaleRC.linear_index(1, 1, nlat) == 1
    @test CrossScaleRC.linear_index(1, 2, nlat) == 2
    @test CrossScaleRC.linear_index(2, 1, nlat) == 6

    # 2D single-layer blocks
    nlon, nlat, nt = 6, 6, 10
    data_2d = randn(nlon, nlat, nt)
    grid = (2, 2)  # 2x2 blocks
    blocks_2d = CrossScaleRC.make_blocks_single_layer_2d(data_2d, grid; mixing = 0)
    @test length(blocks_2d) == 4
    for b in blocks_2d
        @test :rows_rec in keys(b)
        @test :rows_neigh in keys(b)
        @test :rows_layer in keys(b)
        @test isempty(b.rows_layer)
    end
    blocks_2d_mix = CrossScaleRC.make_blocks_single_layer_2d(data_2d, grid; mixing = 1)
    @test length(blocks_2d_mix) == 4

    # 2D multi-layer: add_cross_layer!
    fine_blocks = CrossScaleRC.make_blocks_single_layer_2d(randn(4, 4, 5), (2, 2); mixing = 0)
    coarse_blocks = CrossScaleRC.make_blocks_single_layer_2d(randn(2, 2, 5), (1, 1); mixing = 0)
    CrossScaleRC.add_cross_layer!(fine_blocks, coarse_blocks, 4, 4, 2, 2)
    @test !isempty(fine_blocks[1].rows_layer)

    # make_blocks multi-layer 2d
    data_vec_2d = [randn(2, 2, 8), randn(4, 4, 8)]
    grids_2d = [(1, 1), (2, 2)]
    layers_2d = CrossScaleRC.make_blocks_multi_layer_2d(data_vec_2d, grids_2d, 0)
    @test length(layers_2d) == 2
    @test length(layers_2d[1]) == 1
    @test length(layers_2d[2]) == 4
    @test !isempty(layers_2d[2][1].rows_layer)

    # make_blocks(::Vector{Array}, ::Vector{Tuple}, ::Int) dispatch
    out = make_blocks(data_vec_2d, grids_2d, 0)
    @test length(out) == 2
end
