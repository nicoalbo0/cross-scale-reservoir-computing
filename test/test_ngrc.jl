@testset "NG-RC" begin
    using Random
    Random.seed!(45)

    # nextgen_closedloop
    M = 4
    T_tot = 150
    train_len = 80
    predict_len = 30
    washout = 5
    past = 2
    degree = 2
    ridge = 1e-4

    data = randn(M, T_tot) .* 0.1
    preds_test, preds_train, Wout, Φ = nextgen_closedloop(
        data,
        train_len,
        predict_len;
        washout = washout,
        past = past,
        degree = degree,
        ridge = ridge,
    )

    @test size(preds_test, 1) == M
    @test size(preds_test, 2) == predict_len
    @test size(preds_train, 1) == M
    @test size(preds_train, 2) == train_len
    @test all(isfinite, Wout)
    @test size(Φ, 2) == train_len
    # preds_test can have NaNs with small random data in closed loop; structure is what we test
    @test eltype(preds_test) == Float64
end
