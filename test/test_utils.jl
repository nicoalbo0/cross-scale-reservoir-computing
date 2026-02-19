@testset "utils" begin
    # input_dimensions
    blocks = [
        (rows_rec = 1:2, rows_neigh = [3, 4], rows_layer = Int[]),
        (rows_rec = 1:2, rows_neigh = [3, 4], rows_layer = Int[]),
    ]
    rec_dim, neigh_dim, layer_dim = input_dimensions(blocks)
    @test rec_dim == 2
    @test neigh_dim == 2
    @test layer_dim == 0

    @test_throws Exception input_dimensions([])
    bad_blocks = [(rows_rec = 1:2, rows_neigh = [3], rows_layer = Int[])]
    @test_throws Exception input_dimensions(vcat(blocks, bad_blocks))

    # regrid_average
    L, T = 8, 5
    data = rand(L, T)
    divisor = 2
    coarse = regrid_average(data, divisor)
    @test size(coarse) == (div(L, divisor), T)
    @test all(isfinite, coarse)
    # first coarse row should be mean of first two fine rows
    @test coarse[1, :] ≈ (data[1, :] .+ data[2, :]) ./ 2

    @test_throws Exception regrid_average(data, 3)  # 8 % 3 != 0

    # layer_params (internal, not exported)
    params = ([1, 2], [0.1, 0.2], [5, 5])
    p2 = CrossScaleRC.layer_params(params, 2)
    @test p2 == (2, 0.2, 5)

    # square_even_rows (internal)
    X = Float64[1 2; 3 4; 5 6; 7 8]
    Y = CrossScaleRC.square_even_rows(X)
    @test Y[1, :] == X[1, :]
    @test Y[2, :] == X[2, :] .^ 2
    @test Y[3, :] == X[3, :]
    @test Y[4, :] == X[4, :] .^ 2

    # rmse_upto
    a = rand(4, 10)
    b = copy(a)
    @test rmse_upto(a, b) ≈ 0.0 atol = 1e-12
    b[1, 1] += 1.0
    r = rmse_upto(a, b; T = 1, coords = [1])
    @test r ≈ 1.0
    r2 = rmse_upto(a, b; T = 5, coords = 1:2)
    @test r2 >= 0 && isfinite(r2)

    # cubic_time_interpolate (internal)
    data_small = rand(2, 5)
    interp = CrossScaleRC.cubic_time_interpolate(data_small, 1.0, 2)
    @test size(interp, 1) == 2
    @test size(interp, 2) > 5  # more time points
end
