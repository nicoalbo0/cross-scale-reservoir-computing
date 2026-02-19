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

    # Train and test (aliases)
    T = 100
    U = randn(nu, T) .* 0.1
    Y = randn(ny, T) .* 0.1
    washout = 5
    ridge = 1e-5

    DeepESN_train!(m, U, Y; washout = washout, ridge = ridge, reset_state = true)
    @test all(isfinite, m.Wout)

    steps = 20
    warmup = zeros(nu, 10)
    Yhat = DeepESN_test_closed_loop(m; steps = steps, warmup = warmup, reset_state = true)
    @test size(Yhat, 1) == ny
    @test size(Yhat, 2) == size(warmup, 2) + steps
    @test all(isfinite, Yhat)
end
