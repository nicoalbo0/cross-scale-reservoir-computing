@testset "plots" begin
    # Small data for plotting (avoid opening windows; just test no error and return type)
    # plot_train_test_heatmaps expects warmup >= n_lyap_warm * steps_per_lyap and enough columns
    τ = 0.25
    λ_max = 0.1
    steps_per_lyap = Int(round(1 / (τ * λ_max)))  # 40
    Q, train_len = 4, 100
    n_lyap_train = 2
    n_lyap_test = 2
    n_lyap_warm = 1
    warmup = 50  # >= n_lyap_warm * steps_per_lyap (40)
    total_test_cols = (n_lyap_warm + n_lyap_test) * steps_per_lyap
    plot_start = warmup - n_lyap_warm * steps_per_lyap + 1
    plot_end = plot_start + total_test_cols - 1
    min_forecast_cols = plot_end
    min_data_cols = train_len - warmup + plot_end

    training_data = randn(Q, train_len) .* 0.5
    training_forecast = randn(Q, train_len) .* 0.5
    data = randn(Q, min_data_cols) .* 0.5
    forecast = randn(Q, min_forecast_cols) .* 0.5

    p = plot_train_test_heatmaps(
        training_data,
        training_forecast,
        data,
        forecast;
        τ = τ,
        λ_max = λ_max,
        warmup = warmup,
        train_len = train_len,
        Q = Q,
        L = 22,
        n_lyap_train = n_lyap_train,
        n_lyap_test = n_lyap_test,
        n_lyap_warm = n_lyap_warm,
        clims = (-2, 2),
        title_prefix = "Test",
    )
    @test p !== nothing

    # plot_units_activity (n_units must be <= size(X[1], 1) to avoid step zero in 1:n_skip:end)
    X_units = [randn(20, 50)]
    p2 = plot_units_activity(X_units; n_units = 10)
    @test p2 !== nothing
    p3 = plot_units_activity(X_units; n_units = 20)
    @test p3 !== nothing
end
