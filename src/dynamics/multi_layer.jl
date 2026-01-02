function run_multi_layer(
    params::Tuple{Vector,Vector,Vector,Vector,Vector,Vector},
    data::Matrix{T},
    data_coarse::Matrix{T},
    train_time::Int,
    test_time::Int;
    washout::Int,
    warmup::Int,
    num_networks::Vector{Int},
    mixing::Vector{Int},
    ridge_parameter::Vector{T},
    div::Int,
    show_progress::Bool = false
) where T<:Real

    # --------------------------------------------------
    # 1. Build COARSE data + blocks
    # --------------------------------------------------
    #data_coarse = regrid_average(data, div)
    blocks_coarse = make_blocks(size(data_coarse, 1), num_networks[1], mixing[1])

    # --------------------------------------------------
    # 2. Run COARSE layer
    # --------------------------------------------------
    preds_coarse, train_pred_coarse, train_data_coarse, Xc, _ =
        run_single_layer(
            layer_params(params, 1),
            data_coarse,
            zeros(T, size(data_coarse)),   # no layer input
            train_time,
            test_time,
            blocks_coarse;
            washout = washout,
            warmup  = warmup,
            ridge_parameter = ridge_parameter[1],
            show_progress = show_progress,
            input_mode = :structured
        )

    # --------------------------------------------------
    # 3. Lift coarse outputs to fine grid
    # --------------------------------------------------
    data_layer = hcat(train_pred_coarse, preds_coarse[:, warmup+1:end])

    @assert size(train_pred_coarse, 2) == train_time
    @assert size(data_layer, 2) == train_time + test_time

    # --------------------------------------------------
    # 4. Build FINE blocks (hard hierarchy)
    # --------------------------------------------------
    blocks_fine = make_blocks(size(data, 1), div, num_networks[2], mixing[2], num_networks[1]; overlap_mode = :exclude)

    # --------------------------------------------------
    # 5. Run FINE layer
    # --------------------------------------------------
    preds_fine, train_pred_fine, train_data_fine, Xf, _ =
        run_single_layer(
            layer_params(params, 2),
            data,
            data_layer,
            train_time,
            test_time,
            blocks_fine;
            washout = washout,
            warmup  = warmup,
            ridge_parameter = ridge_parameter[2],
            show_progress = show_progress,
            input_mode = :structured
        )

    return (
        preds_fine,
        preds_coarse,
        train_pred_fine,
        train_pred_coarse,
        train_data_coarse,
        train_data_fine,
        data_coarse,
        Xc,
        Xf,
    )
end