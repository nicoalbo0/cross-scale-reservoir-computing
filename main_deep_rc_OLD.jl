# Activate environment
using Pkg, Revise
Pkg.activate(".")
Pkg.instantiate()

using HierarchicalRC
using Statistics, LinearAlgebra
using Plots, Measures
using ReservoirComputing

Q = 64
L = 22
μ = 0.0

data, τ = load_data(Q, L, μ; show_data=false, interpolate_data=false);

# Experiment configuration
washout      = 1_000
train_len    = 50_000
predict_len  = 1_000
warmup       = 1_000

U = data[:, 1:train_len-1]      # inputs
Y = data[:, 2:train_len]        # targets

N_res      = 300        # neurons per layer
spectral_radius = 0.9
input_scaling   = 0.1
leak_rate       = 0.2

model = DeepESN(
    input_size      = Q,
    reservoir_size  = N_res,
    output_size     = Q,
    nlayers         = L,
    spectral_radius = spectral_radius,
    input_scaling   = input_scaling,
    leaking_rate    = leak_rate,
    bias            = true,
    activation      = tanh
)

ridge = 1e-6

esn = train(
    model,
    U,
    Y;
    washout = washout,
    ridge   = ridge
)

# Initial condition for prediction
u0 = data[:, train_len]

Ŷ = predict(
    esn,
    u0,
    predict_len;
    continuation = true
)

t = 1:predict_len

p = plot(
    t,
    data[1, train_len+1:train_len+predict_len],
    label = "True",
    lw = 2
)
plot!(
    p, 
    t,
    Ŷ[1, :],
    label = "DeepESN",
    lw = 2,
    ls = :dash
)