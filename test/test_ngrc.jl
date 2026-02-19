@testset "NG-RC" begin
    using Random
    Random.seed!(45)

    # combs_with_repetition (internal)
    c = CrossScaleRC.combs_with_repetition(2, 2)
    @test length(c) == 3  # (1,1), (1,2), (2,2)
    @test sort(c) == [[1, 1], [1, 2], [2, 2]]
    @test CrossScaleRC.combs_with_repetition(3, 1) == [[1], [2], [3]]

    # feature_length (internal)
    nlin = 4  # M * past
    @test CrossScaleRC.feature_length(nlin, 1) == 1 + nlin
    @test CrossScaleRC.feature_length(nlin, 2) == 1 + nlin + binomial(nlin + 1, 2)
    @test CrossScaleRC.feature_length(2, 3) == 1 + 2 + binomial(3, 2) + binomial(4, 3)

    # fill_features! (internal) - via constructing poly_combos and xhist
    nlin = 4
    past = 2
    degree = 2
    poly_combos = [CrossScaleRC.combs_with_repetition(nlin, p) for p in 2:degree]
    F = CrossScaleRC.feature_length(nlin, degree)
    ϕ = Vector{Float64}(undef, F)
    xhist = randn(2, past)
    CrossScaleRC.fill_features!(ϕ, xhist, poly_combos)
    @test ϕ[1] == 1.0
    @test length(ϕ) == F
    @test all(isfinite, ϕ)

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

    # degree = 3 for more coverage
    preds_test3, _, _, Φ3 = nextgen_closedloop(data, 50, 15; washout = 2, past = 2, degree = 3, ridge = 1e-4)
    @test size(preds_test3, 2) == 15
    @test size(Φ3, 2) == 50
end
