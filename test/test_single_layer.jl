@testset "single_layer" begin
    using Random
    Random.seed!(42)

    # build_W_in (structured mode)
    N = 20
    rec_dim, neigh_dim, layer_dim = 2, 2, 0
    W_rec, W_neigh, W_layer = build_W_in(N, rec_dim, neigh_dim, layer_dim, 0.5, 0.3, 0.0; mode = :structured)
    @test size(W_rec) == (N, rec_dim)
    @test size(W_neigh) == (N, neigh_dim)
    @test size(W_layer) == (N, 0)
    @test all(isfinite, W_rec)
    @test all(isfinite, W_neigh)

    # Zero total input dim
    Wr, Wn, Wl = build_W_in(10, 0, 0, 0, 0.0, 0.0, 0.0)
    @test size(Wr) == (10, 0)
    @test size(Wn) == (10, 0)
    @test size(Wl) == (10, 0)

    # generate_reservoir
    τ, dt = 0.25, 0.25
    params = (N, 0.9, 5, 0.5, 0.3, 0.0, τ, dt)
    dims = (rec_dim, neigh_dim, layer_dim)
    res = CrossScaleRC.generate_reservoir(params, dims; input_mode = :structured)
    @test size(res.W, 1) == N
    @test size(res.W, 2) == N
    @test res.dt_τ ≈ dt / τ
    @test size(res.W_in_rec) == (N, rec_dim)

    # Small pipeline: make data, blocks, run_single_layer
    L = 8
    T_tot = 200
    train_time = 150
    test_time = 40
    washout = 10
    warmup = 10
    data = randn(L, T_tot) .* 0.1  # small amplitude
    data_layer = zeros(L, T_tot)
    blocks = make_blocks(L, 2, 1)
    ridge = 1e-4
    res_params = (30, 0.8, 4, 0.5, 0.5, 0.0, τ, dt)

    preds, train_pred, train_data, X, models = run_single_layer(
        res_params,
        data,
        data_layer,
        train_time,
        test_time,
        blocks;
        washout = washout,
        warmup = warmup,
        ridge_parameter = ridge,
        show_progress = false,
        input_mode = :structured,
        regression_mode = :linear,
    )
    @test size(preds, 1) == L
    @test size(preds, 2) == test_time + warmup
    @test length(models) == length(blocks)
    @test length(X) == length(blocks)
    @test size(train_pred, 2) == train_time

    # regression_mode = :quadratic (default for train)
    preds_q, train_pred_q, _, X_q, _ = run_single_layer(
        res_params,
        data,
        data_layer,
        train_time,
        test_time,
        blocks;
        washout = washout,
        warmup = warmup,
        ridge_parameter = ridge,
        show_progress = false,
        input_mode = :structured,
        regression_mode = :quadratic,
    )
    @test size(preds_q, 2) == test_time + warmup
    @test length(X_q) == length(blocks)

    # fit_ridge_regression directly (:linear and :quadratic)
    N_res = 30
    T_train = 100
    wash = 5
    rows_rec = blocks[1].rows_rec
    X_fit = randn(N_res, T_train)
    Y_fit = randn(length(rows_rec), T_train)
    W_lin = CrossScaleRC.fit_ridge_regression(X_fit, Y_fit, Float64(ridge), wash; mode = :linear)
    @test size(W_lin, 1) == size(Y_fit, 1)
    @test size(W_lin, 2) == N_res
    W_quad = CrossScaleRC.fit_ridge_regression(X_fit, Y_fit, Float64(ridge), wash; mode = :quadratic)
    @test size(W_quad) == size(W_lin)

    # show_progress = true (smoke test)
    preds_p, _, _, _, _ = run_single_layer(
        res_params,
        data,
        data_layer,
        train_time,
        test_time,
        blocks;
        washout = washout,
        warmup = warmup,
        ridge_parameter = ridge,
        show_progress = true,
        input_mode = :structured,
        regression_mode = :linear,
    )
    @test size(preds_p, 1) == L
end
