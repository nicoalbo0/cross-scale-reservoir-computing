function plot_train_test_heatmaps(
    training_data::AbstractMatrix,
    training_forecast::AbstractMatrix,
    data::AbstractMatrix,
    forecast::AbstractMatrix;
    τ::Real,
    λ_max::Real,
    warmup::Int,
    train_len::Int,
    Q::Int,
    L::Int,
    n_lyap_train::Int = 10,
    n_lyap_test::Int  = 10,
    n_lyap_warm::Int  = 1,
    clims = (-3, 3),
    title_prefix::AbstractString = "",
)

    tp = isempty(title_prefix) ? "" : title_prefix * " – "

    # ---------------------------
    # Basic quantities
    # ---------------------------
    s = (1:Q) .* (L / Q)

    steps_per_lyap = Int(round(1 / (τ * λ_max)))

    # ===========================
    # TEST PLOT (same as before)
    # ===========================
    total_steps = (n_lyap_warm + n_lyap_test) * steps_per_lyap

    plot_start = warmup - n_lyap_warm * steps_per_lyap + 1
    plot_end   = plot_start + total_steps - 1

    forecast_plot = forecast[:, plot_start:plot_end]
    actual_plot   = data[:,
        train_len - warmup + plot_start :
        train_len - warmup + plot_end
    ]

    err_plot = forecast_plot .- actual_plot

    t_plot = (-n_lyap_warm * steps_per_lyap :
               n_lyap_test * steps_per_lyap - 1) .* τ .* λ_max

    t_test_start = 0.0

    p_act = heatmap(
        t_plot, s, actual_plot;
        title=tp * "Test Data",
        c=:jet, clims=clims,
        framestyle=:box,
        yticks=false, xticks=false
    )

    p_pred = heatmap(
        t_plot, s, forecast_plot;
        title=tp * "Test Forecast",
        c=:jet, clims=clims,
        framestyle=:box,
        yticks=false, xticks=false,
        legend=:bottomright
    )

    p_err = heatmap(
        t_plot, s, err_plot;
        title=tp * "Test Error",
        xlabel=L"Λt"*" (Lyapunov time)",
        c=:jet, clims=clims,
        framestyle=:box,
        yticks=false,
        legend=:bottomright
    )

    vline!(p_pred, [t_test_start]; color=:red, linewidth=2, label="Test Start")
    vline!(p_err,  [t_test_start]; color=:red, linewidth=2, label="Test Start")

    p_test = plot(p_act, p_pred, p_err;
                  layout=(3,1), size=(1100,900))

    # ===========================
    # TRAINING PLOT (same as before)
    # ===========================
    train_plot_len = n_lyap_train * steps_per_lyap

    Ttrain = size(training_forecast, 2)
    train_plot_start = max(1, Ttrain - train_plot_len + 1)

    train_err = training_forecast .- training_data

    t_train = (0:(size(training_forecast[:, train_plot_start:end], 2)-1)) .* τ .* λ_max

    p_train_act = heatmap(
        t_train, s,
        training_data[:, train_plot_start:end];
        title=tp * "Training Data",
        c=:jet, clims=clims,
        colorbar=false, framestyle=:box,
        ylabel=L"x", xticks=false
    )

    p_train_pred = heatmap(
        t_train, s,
        training_forecast[:, train_plot_start:end];
        title=tp * "Training Forecast",
        c=:jet, clims=clims,
        colorbar=false, framestyle=:box,
        ylabel=L"x", xticks=false
    )

    p_train_err = heatmap(
        t_train, s,
        train_err[:, train_plot_start:end];
        title=tp * "Training Error",
        xlabel=L"Λt"*" (Lyapunov time)",
        c=:jet, clims=clims,
        colorbar=false, framestyle=:box,
        ylabel=L"x"
    )

    p_train = plot(
        p_train_act, p_train_pred, p_train_err;
        layout=(3,1),
        size=(1100,900),
        left_margin=5mm
    )

    # ===========================
    # COMPOSE FINAL FIGURE
    # ===========================
    l = @layout [a{0.45w} b{0.55w}]
    p = plot(p_train, p_test; layout=l)

    return p
end

function plot_units_activity(X::Vector{Matrix{T}}) where T<:Real

    p = plot(X[1][1:50,:]', label="", framestyle=:box, xlabel="Time", ylabel="Unit Activity", ylims=(-1,1))

    return p
end