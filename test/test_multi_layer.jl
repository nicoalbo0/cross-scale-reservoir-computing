@testset "multi_layer" begin
    using Random
    Random.seed!(43)

    L_fine = 8
    divisor = 2
    L_coarse = div(L_fine, divisor)
    T_tot = 200
    train_time = 150
    test_time = 40
    washout = 10
    warmup = 10
    ridge = 1e-4
    τ = 0.25
    dt = 0.25

    data_fine = randn(L_fine, T_tot) .* 0.1
    data_coarse = randn(L_coarse, T_tot) .* 0.1

    num_coarse = 2
    num_fine = 4
    mixing = 1
    blocks_coarse = make_blocks(L_coarse, num_coarse, mixing)
    blocks_fine = make_blocks(L_fine, divisor, num_fine, mixing, num_coarse; overlap_mode = :exclude)
    blocks = [blocks_coarse, blocks_fine]

    params = (
        [50, 50],           # N
        [0.8, 0.8],         # g
        [4, 4],             # degree
        [0.5, 0.5],         # g_in_rec
        [0.5, 0.5],         # g_in_neigh
        [0.0, 0.3],         # g_in_layer
        [τ, τ],
        [dt, dt],
    )
    ridge_vec = [ridge, ridge]
    regression_mode = [:linear, :linear]

    preds_fine, preds_coarse, train_pred_fine, train_pred_coarse,
    train_data_coarse, train_data_fine, data_c_out, Xc, Xf = run_multi_layer(
        params,
        data_fine,
        data_coarse,
        train_time,
        test_time,
        blocks;
        washout = washout,
        warmup = warmup,
        ridge_parameter = ridge_vec,
        show_progress = false,
        input_mode = :structured,
        regression_mode = regression_mode,
    )

    @test size(preds_fine, 1) == L_fine
    @test size(preds_fine, 2) == test_time + warmup
    @test size(preds_coarse, 1) == L_coarse
    @test size(preds_coarse, 2) == test_time + warmup
    @test size(train_pred_fine, 2) == train_time
    @test size(train_pred_coarse, 2) == train_time
    @test length(Xc) == num_coarse
    @test length(Xf) == num_fine
end
