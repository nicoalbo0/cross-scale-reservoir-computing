"""
    run_multi_layer(params, data, data_coarse, train_time, test_time, blocks; washout, warmup, ridge_parameter, ...)

Two-layer pipeline: (1) run coarse layer, (2) concatenate coarse training prediction and test
predictions as `data_layer` for the fine layer, (3) run fine layer with that layer input.
`blocks` is `[blocks_coarse, blocks_fine]`; `params` is an 8-tuple of vectors (per-layer);
`ridge_parameter` and `regression_mode` are per-layer. Returns a tuple:
`(preds_fine, preds_coarse, train_pred_fine, train_pred_coarse, train_data_coarse, train_data_fine, data_coarse, Xc, Xf)`.
"""
function run_multi_layer(
    params::Tuple{Vector,Vector,Vector,Vector,Vector,Vector,Vector,Vector},
    data::Matrix{T},
    data_coarse::Matrix{T},
    train_time::Int,
    test_time::Int,
    blocks::Vector{<:Vector{<:NamedTuple}};
    washout::Int,
    warmup::Int,
    ridge_parameter::Vector{T},
    show_progress::Bool = false,
    input_mode::Symbol = :structured,
    regression_mode = Vector{Symbol}
) where T<:Real

    blocks_coarse = blocks[1]
    blocks_fine   = blocks[2]

    # --------------------------------------------------
    # 1. Run COARSE layer
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
            input_mode = input_mode,
            regression_mode = regression_mode[1]
        )

    # --------------------------------------------------
    # 2. Lift coarse outputs to fine grid
    # --------------------------------------------------
    data_layer = hcat(train_pred_coarse, preds_coarse[:, warmup+1:end])

    @assert size(train_pred_coarse, 2) == train_time
    @assert size(data_layer, 2) == train_time + test_time

    # --------------------------------------------------
    # 3. Run FINE layer
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
            input_mode = input_mode,
            regression_mode = regression_mode[2]
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