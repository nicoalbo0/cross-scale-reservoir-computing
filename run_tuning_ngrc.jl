# Activate environment
using Pkg, Revise
Pkg.activate(".")

using CrossScaleRC
using LinearAlgebra, Measures
using Plots
using Random
using DelimitedFiles

BLAS.set_num_threads(Threads.nthreads())

# Experiment setup
Q0 = 128
L = 44
μ = 0.01

resolution_divisor = 1
Q = div(Q0, resolution_divisor)
data, τ = load_data(Q0, L, μ; show_data=false, refinement=1)
data = regrid_average(data, resolution_divisor)

# Experiment configuration
washout     = 1_000
train_len   = 50_000
predict_len = 1_000

M, Ttot = size(data)
washout + train_len + predict_len + 2 ≤ Ttot || error("Not enough data")

Perrors = 25

using MultivariateStats
pca_model = fit(PCA, data; pratio=0.999, method=:auto);
Ψ = eigvecs(pca_model);
data_red= ψ' * data;

function run_once(params)
    past, degree, delay, ridge, noise_std = params

    preds_test_red, _, _ = nextgen_closedloop(
        data_red,
        train_len,
        predict_len;
        washout = washout,
        past=past,
        delay = delay,
        degree = degree,
        ridge = ridge,
        nfeat = 0,
        noise_std = noise_std
    )
    preds_test = ψ * preds_test_red    

    t_start = washout + past
    test_data = data[:, (t_start + train_len + 1):(t_start + train_len + predict_len)]

    nT    = size(test_data, 2)
    Tgrid = round.(Int, range(1, nT; length=Perrors))  # P evenly spaced indices, 1 and nT always included

    rmses = Vector{Float64}(undef, Perrors)
    @inbounds for (i, T) in enumerate(Tgrid)
        rmses[i] = rmse_upto(test_data, preds_test; T=T)
    end
    return rmses
end

# ---------------------------
# Choose grids + wrapper
# ---------------------------
grids = Dict(
    :past   => [1,2,3,4,5,6],
    :degree => [2],
    :delay  => [1,4,8],
    :ridge  => [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    :noise_std => [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
)

run_once_grid(p) = run_once((p.past, p.degree, p.delay, p.ridge, p.noise_std))

grid_search(
    run_once_grid,
    grids;
    nrep = 5,
    outfile = "gridsearch_ngrc_Q$(Q).csv",
    param_order = [:past, :degree, :delay, :ridge, :noise_std],
    error_names = Symbol.(:rmse, 1:Perrors),
    progress = true,
)