# Keep vector-/tuple-valued parameters in a *single* CSV cell.
using DelimitedFiles, LinearAlgebra
_cell(x) = (x isa AbstractArray || x isa Tuple || x isa NamedTuple) ? repr(x) : x

"""
    grid_search(run_once, grids; nrep=20, outfile="grid.csv", param_order=..., error_names=nothing, progress=true)

Cartesian grid search over `grids` (Dict mapping parameter name => vector of values).
`run_once(params)` receives a `NamedTuple` and must return a vector/tuple of numeric errors
(fixed length across reps). Writes CSV with columns: param columns..., mean_<err_i>, std_<err_i>.
`param_order` controls column order; `error_names` names the error metrics.
"""
function grid_search(run_once, grids::AbstractDict{Symbol,<:AbstractVector};
    nrep::Integer = 20,
    outfile::AbstractString = "grid.csv",
    param_order::Vector{Symbol} = collect(keys(grids)),
    error_names::Union{Nothing,Vector{Symbol}} = nothing,
    progress::Bool = true,
)
    iters = (grids[s] for s in param_order)
    total_jobs = prod(length(grids[s]) for s in param_order)
    job_id = 0

    rows = Vector{Vector{Any}}()

    for vals in Iterators.product(iters...)  # Cartesian product
        job_id += 1
        params = NamedTuple{Tuple(param_order)}(vals)
        progress && @info "Grid point" job_id total_jobs params

        sum_err = Float64[]
        sumsq_err = Float64[]

        for rep in 1:nrep
            progress && @info "  Rep" job_id total_jobs rep nrep

            err_raw = run_once(params)
            err = Float64.(collect(err_raw))

            if rep == 1
                sum_err = zeros(length(err))
                sumsq_err = zeros(length(err))
            elseif length(err) != length(sum_err)
                throw(ArgumentError("run_once returned length $(length(err)) but expected $(length(sum_err))"))
            end

            @inbounds for i in eachindex(err)
                sum_err[i] += err[i]
                sumsq_err[i] += err[i]^2
            end
        end

        mean = sum_err ./ nrep
        var = nrep > 1 ? (sumsq_err .- (sum_err .^ 2) ./ nrep) ./ (nrep - 1) : zeros(length(mean))
        std = sqrt.(max.(var, 0.0))

        if isempty(rows)
            names = error_names === nothing ? [Symbol("err", i) for i in 1:length(mean)] : error_names
            length(names) == length(mean) || throw(ArgumentError("error_names length must match error length"))

            header = Any[param_order...]
            for nm in names
                push!(header, Symbol("mean_", nm))
                push!(header, Symbol("std_", nm))
            end
            push!(rows, header)
        end

        row = Any[_cell(v) for v in vals]
        for i in eachindex(mean)
            push!(row, mean[i])
            push!(row, std[i])
        end
        push!(rows, row)
    end

    # writedlm supports "iterable collection of iterable rows".
    writedlm(outfile, rows, ',')

    return nothing
end
