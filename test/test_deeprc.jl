@testset "DeepESN" begin
    using Random
    Random.seed!(44)

    nu = 4
    ny = 4
    nl = 2
    nr = 20

    m = DeepESN(nu, ny; nl = nl, nr = nr, rho = 0.9, leak = 1.0, input_scale = 0.5, sparsity = 0.1)
    @test m.nu == nu
    @test m.ny == ny
    @test m.nl == nl
    @test m.nr == nr
    @test length(m.Win) == nl
    @test length(m.Wrec) == nl
    @test size(m.x) == (nr, nl)
    @test size(m.Wout) == (ny, nr * nl)

    # DeepESN with vector rho and leak
    m2 = DeepESN(nu, ny; nl = 2, nr = 10, rho = [0.8, 0.9], leak = [0.9, 1.0], input_scale = 0.5, sparsity = 0.2)
    @test m2.nl == 2
    @test length(m2.leak) == 2

    # augment_features (internal)
    x = [1.0, 2.0, 3.0, 4.0]
    aug = CrossScaleRC.augment_features(x)
    @test aug[1] == 1.0
    @test aug[2] == 4.0
    @test aug[3] == 3.0
    @test aug[4] == 16.0
    # augment_features!
    buf = zeros(4)
    CrossScaleRC.augment_features!(buf, x)
    @test buf ≈ aug
    # augment_features_matrix
    X = [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0]
    Xa = CrossScaleRC.augment_features_matrix(X)
    @test Xa[1, :] == X[1, :]
    @test Xa[2, :] == X[2, :] .^ 2
    @test Xa[3, :] == X[3, :]
    @test Xa[4, :] == X[4, :] .^ 2

    # Train and test (aliases)
    T = 100
    U = randn(nu, T) .* 0.1
    Y = randn(ny, T) .* 0.1
    washout = 5
    ridge = 1e-5

    DeepESN_train!(m, U, Y; washout = washout, ridge = ridge, reset_state = true)
    @test all(isfinite, m.Wout)

    # DeepESN_train! with return_output = true
    Yhat_train = DeepESN_train!(m, U, Y; washout = washout, ridge = ridge, reset_state = true, return_output = true)
    @test size(Yhat_train) == size(Y)
    @test all(isfinite, Yhat_train)

    steps = 20
    warmup = zeros(nu, 10)
    Yhat = DeepESN_test_closed_loop(m; steps = steps, warmup = warmup, reset_state = true)
    @test size(Yhat, 1) == ny
    @test size(Yhat, 2) == size(warmup, 2) + steps
    @test all(isfinite, Yhat)

    # DeepESN_test_closed_loop with y_init and no warmup
    Yhat2 = DeepESN_test_closed_loop(m; steps = 5, warmup = zeros(nu, 0), y_init = zeros(nu), reset_state = true)
    @test size(Yhat2) == (ny, 5)
end
