# Technical Report: Cross-Scale Reservoir Computing

## A guide for new contributors to the CrossScaleRC codebase

---

## Table of Contents

1. [What This Project Is About](#1-what-this-project-is-about)
2. [The Science: Reservoir Computing in a Nutshell](#2-the-science-reservoir-computing-in-a-nutshell)
3. [The Innovation: Cross-Scale Spatial Blocking](#3-the-innovation-cross-scale-spatial-blocking)
4. [The Four Architectures](#4-the-four-architectures)
5. [The Data: What Gets Forecasted](#5-the-data-what-gets-forecasted)
6. [Results and What They Mean](#6-results-and-what-they-mean)
7. [Code Architecture: The Full Picture](#7-code-architecture-the-full-picture)
8. [How the Code Runs Step by Step](#8-how-the-code-runs-step-by-step)
9. [The Key Algorithms in Detail](#9-the-key-algorithms-in-detail)
10. [How to Modify Things](#10-how-to-modify-things)
11. [Glossary](#11-glossary)

---

## 1. What This Project Is About

This project tackles a hard problem in physics and climate science: **forecasting the future of chaotic spatiotemporal systems**. Think of a field of values (like ocean temperature across the globe) that evolves in time according to complex, chaotic dynamics. Traditional numerical simulation is expensive and requires knowing the exact equations. Machine learning can learn these dynamics from data alone.

The approach used here is called **Reservoir Computing (RC)**, a type of recurrent neural network that is fast to train (no backpropagation) because only the output layer is learned. The key innovation of this project is a **cross-scale** method: instead of one giant model, the spatial domain is split into blocks, and multiple resolution scales (coarse and fine) are coupled together so that large-scale patterns inform small-scale predictions.

**Paper**: *"Cross-Scale Reservoir Computing"* by Nicola Albore, Gabriele Di Antonio, Fabrizio Coccetti, Andrea Gabrielli (arXiv: 2510.11209).

**Key result**: On sea surface temperature data, the cross-scale method outperforms standard parallel reservoir approaches in long-term prediction. The authors also discovered that optimal network dynamics become progressively more linear at coarser scales, with slower modes transferred to deeper layers.

---

## 2. The Science: Reservoir Computing in a Nutshell

### 2.1 The Problem

You have a spatiotemporal system: a matrix of values **u(x, t)** where x is space and t is time. You observe it for a while (training), then want to predict its future evolution (testing) without knowing the underlying equations.

### 2.2 Echo State Networks (ESNs)

An ESN is a recurrent neural network with three parts:

```
Input u(t) --> [Fixed random reservoir] --> Trained linear readout --> Output y(t)
```

1. **Input layer**: Projects the input data into a high-dimensional reservoir state. The weight matrices are random and fixed (never trained).
2. **Reservoir**: A large sparse recurrent network with N neurons. Its state evolves as:

   ```
   x(t+1) = (1 - dt/tau) * x(t) + (dt/tau) * tanh(W*x(t) + W_in*u(t))
   ```

   where:
   - `x(t)` is the N-dimensional reservoir state
   - `W` is the N x N sparse recurrent weight matrix (fixed, random)
   - `W_in` is the input weight matrix (fixed, random)
   - `dt/tau` is the leak rate (controls memory timescale)
   - `tanh` is the nonlinear activation

3. **Readout**: A linear map from reservoir states to outputs, trained via ridge regression:

   ```
   W_out = argmin || W_out * X - Y ||^2 + ridge * ||W_out||^2
   ```

   This is the **only thing that gets trained**. It is a simple linear algebra problem (no gradient descent, no epochs).

### 2.3 Why This Works

The reservoir acts as a nonlinear kernel that maps the input into a rich feature space. If the reservoir is well-designed (right spectral radius, right sparsity), these features capture the relevant dynamics and a simple linear readout can extract the prediction.

### 2.4 Key Hyperparameters

| Parameter | Symbol | Role |
|-----------|--------|------|
| Reservoir size | N | Number of neurons. Bigger = more expressive but slower |
| Spectral radius | g | Largest eigenvalue of W. Controls echo memory length. < 1 for stability |
| Connectivity | degree | Average connections per neuron. Controls sparsity of W |
| Input scaling | g_in | How strongly input drives the reservoir |
| Leak rate | dt/tau | Blending of old state and new update. Small = long memory |
| Ridge parameter | ridge | Regularization. Prevents overfitting in the readout |

---

## 3. The Innovation: Cross-Scale Spatial Blocking

### 3.1 The Scalability Problem

A naive ESN for a system with L=128 spatial points needs a reservoir that sees all 128 inputs. This doesn't scale to large domains (like global SST with thousands of grid points).

### 3.2 Spatial Blocking (Parallelized RC)

The domain is divided into **blocks**. Each block gets its own independent reservoir and readout. Block b is responsible for predicting the values at its spatial locations.

For a 1D domain of L=128 with 16 blocks:
- Each block "owns" 8 spatial points (rows_rec = local input)
- Each block also sees its neighbors (rows_neigh), e.g. 8 points on each side (with periodic wrapping)
- This neighbor mixing prevents blocks from being completely isolated

```
Block structure for L=128, 16 blocks, mixing=8:

Block 1:  owns [1:8],   sees neighbors from [121:128] and [9:16]
Block 2:  owns [9:16],  sees neighbors from [1:8] and [17:24]
...
Block 16: owns [121:128], sees neighbors from [113:120] and [1:8]
```

### 3.3 Multi-Scale Coupling (The Cross-Scale Part)

This is the paper's main contribution. Two resolution levels are coupled:

1. **Coarse layer**: The data is spatially downsampled (e.g., averaging every 4 fine cells into 1 coarse cell). A set of block-wise ESNs forecasts at this coarse resolution.

2. **Fine layer**: The full-resolution data is forecasted by a second set of ESNs. But each fine block also receives the coarse layer's predictions as an additional input (`rows_layer`).

```
Coarse data (32 points) --> Coarse ESNs (4 blocks) --> Coarse predictions
                                                           |
                                                           v  (cross-layer input)
Fine data (128 points)  --> Fine ESNs (16 blocks)   --> Fine predictions
```

The coarse layer captures large-scale, slow-moving patterns. The fine layer uses this information to improve its local, fast-scale predictions. Each fine block's reservoir has three input channels:

- **W_in_rec**: local data (8 fine points)
- **W_in_neigh**: neighbor data (16 fine points from adjacent blocks)
- **W_in_layer**: coarse data from the parent coarse block

---

## 4. The Four Architectures

This codebase implements four distinct RC architectures:

### 4.1 Single-Layer Block ESN (`main_single_layer.jl`)

The simplest architecture. One resolution, domain split into blocks, each block has local + neighbor inputs.

- **When to use**: Baseline experiments, understanding the blocking mechanism
- **Strengths**: Fast, simple, good for benchmarking
- **Limitations**: No cross-scale information

### 4.2 Multi-Layer Cross-Scale ESN (`main_multi_layer.jl`)

Two resolution layers (coarse and fine), coupled via cross-layer input. This is the paper's main contribution.

- **When to use**: When you want to exploit multi-scale structure
- **Strengths**: Captures both large-scale and small-scale dynamics
- **How it works**: Coarse layer runs first (fully), then its predictions feed into the fine layer

### 4.3 Deep ESN (`main_deep_rc.jl`)

Multiple reservoir layers stacked vertically (not spatially). No spatial blocking. The output of layer l feeds into layer l+1.

- **When to use**: When you want depth rather than spatial decomposition
- **Strengths**: Can learn hierarchical temporal representations
- **Key difference**: Uses quadratic feature augmentation in the readout (even-indexed state components are squared)

### 4.4 Next-Generation RC (`main_ngrc.jl`)

No reservoir at all. Instead, uses polynomial features on time-delay embeddings.

- **When to use**: Lightweight, fast alternative; no random initialization
- **How it works**:
  1. Build a delay embedding: stack the state at times t, t-delay, t-2*delay, ..., t-(past-1)*delay
  2. Compute all polynomial monomials up to a given degree
  3. Fit a linear map from polynomial features to the next state
  4. Run closed-loop prediction

---

## 5. The Data: What Gets Forecasted

### 5.1 Kuramoto-Sivashinsky (KS) Equation

A 1D chaotic PDE used as a standard benchmark for spatiotemporal forecasting:

```
du/dt = -u * du/dx - d2u/dx2 - d4u/dx4 + mu * u * d2u/dx2
```

- **Parameters**: Q = number of spatial grid points (128), L = domain length (44), mu = asymmetry (0.01)
- **Properties**: Chaotic, spatially extended, positive Lyapunov exponent (~0.08)
- **Data format**: CSV file, matrix of size (Q x T), stored in `data/kuramoto/`
- **Preprocessing**: Normalize each spatial coordinate to zero mean, unit variance

The KS equation is an ideal testbed because:
- It has well-understood chaotic dynamics
- You can generate unlimited data (via MATLAB scripts)
- Lyapunov time provides a natural prediction horizon metric

### 5.2 Sea Surface Temperature (SST)

Real-world climate data from the Copernicus Climate Data Store.

- **Source**: Satellite SST ensemble product (1982-2016)
- **Format**: Daily gridded fields, stored as JLD2 (Julia's binary format)
- **Resolutions**: Multiple (e.g., 18 degrees coarse, 6 degrees fine)
- **Preprocessing pipeline**:
  1. `scripts/download_sst_data.py` - Downloads raw data from CDS API
  2. `scripts/main_regrid.jl` - Regrids to desired resolution
  3. `scripts/main_produce_sst_data.jl` - Concatenates into final 3D arrays

SST data has real-world complications: missing values (NaN), land masks, seasonal cycles, and multi-scale spatial structure.

---

## 6. Results and What They Mean

### 6.1 How Results Are Measured

- **RMSE (Root Mean Square Error)**: Averaged over all spatial points, computed cumulatively up to time T via `rmse_upto()`
- **Lyapunov time**: Time is rescaled by the maximum Lyapunov exponent (lambda_max ~ 0.08 for KS). One Lyapunov time is the timescale over which small errors double. Predicting beyond ~5-10 Lyapunov times is very challenging.
- **Heatmaps**: Space (y-axis) vs time (x-axis) colored by value. Comparing data vs forecast vs error visually shows when and where the prediction diverges.

### 6.2 What the Code Produces

Each main script generates:
1. **Train/test heatmap panels**: 3 rows (data, forecast, error) for both training and test phases
2. **RMSE curves**: Error as a function of prediction horizon
3. **Unit activity plots**: How the reservoir neurons behave over time (useful for diagnosing reservoir dynamics)

### 6.3 Key Findings from the Paper

1. **Cross-scale coupling improves fine-scale prediction**: The multi-layer architecture outperforms the single-layer baseline on both KS and SST data
2. **Progressive linearization**: Optimal dynamics at coarser scales are more linear (lower spectral radius, higher regularization), meaning the coarse layer acts more like a linear smoother
3. **Slower modes migrate to coarse layers**: The coarse layer captures the slow, large-scale dynamics, freeing the fine layer to focus on fast local fluctuations

---

## 7. Code Architecture: The Full Picture

### 7.1 Module Structure

```
CrossScaleRC/
  src/
    CrossScaleRC.jl        # Module definition: imports, includes, exports
    types.jl               # ResParams, Reservoir, BlockModel structs
    data/
      loading.jl           # load_data() for KS and SST
      grids.jl             # make_blocks() for 1D and 2D spatial decomposition
    dynamics/
      single_layer.jl      # build_W_in, generate_reservoir, train, test, run
      multi_layer.jl       # run_multi_layer (coarse-to-fine pipeline)
    utils/
      generic.jl           # regrid_average, ridge helpers, rmse_upto, etc.
      deeprc_utils.jl      # DeepESN struct and methods
      ngrc_utils.jl        # Next-Gen RC polynomial features and prediction
      gridsearch_utils.jl  # Hyperparameter grid search
    plots.jl               # Visualization functions
  test/
    runtests.jl            # Test runner (11 test files)
  main_*.jl                # Entry-point scripts
  run_tuning_*.jl          # Hyperparameter search scripts
  scripts/                 # Data preparation (SST download, regrid, etc.)
```

### 7.2 The Three Core Structs

**ResParams{T}** - A simple parameter container (used less in practice; scripts pass tuples directly):
```julia
struct ResParams{T<:Real}
    N::Int       # reservoir units
    g::T         # spectral radius
    degree::Int  # connectivity
    g_in::T      # input scaling
end
```

**Reservoir{T}** - The actual ESN (created by `generate_reservoir`):
```julia
struct Reservoir{T<:Real}
    W::SparseMatrixCSC{T,Int}  # recurrent weights (N x N, sparse)
    dt_tau::T                   # leak rate
    W_in_rec::Matrix{T}        # input weights for local data
    W_in_neigh::Matrix{T}      # input weights for neighbor data
    W_in_layer::Matrix{T}      # input weights for cross-layer data
end
```

**BlockModel{T}** - One trained block (created during training):
```julia
struct BlockModel{T<:Real}
    W_out::Matrix{T}       # trained readout (the ONLY learned part)
    x::Vector{T}           # current reservoir state
    rows_rec::Vector{Int}  # which data rows are "mine"
    rows_neigh::Vector{Int}# which data rows are my neighbors
    rows_layer::Vector{Int}# which coarse data rows inform me
end
```

### 7.3 Data Flow Diagram

```
                        +-----------+
                        |  Raw Data |
                        |  (CSV/JLD2)|
                        +-----+-----+
                              |
                        load_data()
                              |
                     normalize (zero mean, unit var)
                              |
                    +----+----+----+
                    |              |
              [if multi-layer]  [single layer]
                    |              |
          regrid_average()    make_blocks()
              |       |            |
         data_coarse  data    blocks (with rows_rec,
              |       |             rows_neigh, rows_layer)
              |       |            |
         make_blocks() x2     generate_reservoir()
              |                    |
    [blocks_coarse,           Reservoir struct
     blocks_fine]                  |
              |              train_parallel_reservoir()
              |                    |
         run_multi_layer()    BlockModel per block
              |                    |
         [coarse first,       test_parallel_reservoir()
          then fine with           |
          coarse predictions   predictions matrix
          as layer input]          |
              |              plot / rmse_upto()
              v
         predictions
```

---

## 8. How the Code Runs Step by Step

Let's trace a complete execution of `main_single_layer.jl`:

### Step 1: Load and Prepare Data
```julia
data, tau = load_data(128, 44, 0.01)  # Load KS data from CSV
data = regrid_average(data, 1)        # No regridding (divisor=1)
# data is now a 128 x T matrix (128 spatial points, T timesteps)
```

### Step 2: Define Spatial Blocks
```julia
blocks = make_blocks(128, 16, 8)
# Creates 16 blocks, each with:
#   rows_rec:   8 contiguous spatial indices (owned by this block)
#   rows_neigh: 16 spatial indices (8 left + 8 right neighbors, periodic)
#   rows_layer: empty (no cross-layer input in single-layer mode)
```

### Step 3: Set Hyperparameters
```julia
res_params = (400, 0.3, 10, 0.25, 0.35, 0.0, 0.25, 0.25)
#             N    g   deg  g_rec g_neigh g_lay  tau   dt
```

Input scalings are set as `1/sqrt(input_dimension)` to normalize the drive to the reservoir. This is a common heuristic.

### Step 4: Build the Reservoir (`generate_reservoir`)
1. Create a sparse random matrix W of size 400x400 with sparsity = 10/400
2. Compute its spectral radius (largest eigenvalue magnitude)
3. Scale W so spectral radius = 0.3
4. Build input weight matrices via `build_W_in`:
   - W_in_rec: 400 x 8 (local input)
   - W_in_neigh: 400 x 16 (neighbor input)
   - W_in_layer: 400 x 0 (no cross-layer)
   - In `:structured` mode: each input dimension drives a contiguous block of ~16 neurons

### Step 5: Train (`train_parallel_reservoir`)
For each of the 16 blocks (in parallel via threads):
1. Initialize state x = zeros(400)
2. For t = 1 to 50,000:
   - Record X[:, t] = x (state before update)
   - Record Y[:, t] = data[rows_rec, t] (target = local data)
   - Gather inputs: u_rec, u_neigh, u_layer from data
   - Update: x = (1-dt/tau)*x + (dt/tau)*tanh(W*x + W_rec*u_rec + W_neigh*u_neigh + W_layer*u_layer)
3. Apply quadratic features: square even rows of X
4. Solve ridge regression: W_out = (Y * X') / (X * X' + ridge * I)

This is **teacher forcing**: the true data drives the reservoir, and the readout learns to predict the local data from the reservoir state.

### Step 6: Test (`test_parallel_reservoir`)

**Warmup phase** (1,000 steps): Drive with true data to synchronize the reservoir state.

**Autonomous phase** (1,000 steps): Closed-loop - the model feeds its own predictions back as input.
1. Each block computes its output from current state: y = W_out * square_even_rows(x)
2. All block outputs are collected (needed for neighbor mixing)
3. Each block's state is updated using its own prediction (u_rec) and neighbors' predictions (u_neigh) as input

This is where the model must fly on its own. Errors accumulate because each prediction becomes the next input.

### Step 7: Visualize
- Heatmaps of data vs prediction vs error
- RMSE curve over the prediction horizon

---

## 9. The Key Algorithms in Detail

### 9.1 Structured Input Weights (`build_W_in`)

With D total input dimensions and N reservoir neurons, each input dimension j drives a contiguous block of q = floor(N/D) neurons:

```
Neurons  1..q      -->  driven by input dimension 1 (scaled by g_rec)
Neurons  q+1..2q   -->  driven by input dimension 2 (scaled by g_rec)
...
Neurons  (D-1)*q+1..D*q  -->  driven by last dimension (scaled by g_layer)
```

This "structured" mode means each neuron responds to only one input dimension, creating specialized subgroups in the reservoir. The alternative `:random` mode makes every neuron respond to every input.

### 9.2 Quadratic Feature Augmentation (`square_even_rows`)

Before computing the readout, the state vector is transformed: every even-indexed component is squared.

```
[x1, x2, x3, x4, x5, x6] --> [x1, x2^2, x3, x4^2, x5, x6^2]
```

This gives the linear readout access to second-order nonlinear features, which often improves forecasting of chaotic systems without adding much computational cost.

### 9.3 Ridge Regression

The readout is trained by solving:

```
W_out = Y * X' * (X * X' + ridge * I)^(-1)
```

where:
- X is the (N x T_train) matrix of reservoir states
- Y is the (output_dim x T_train) matrix of targets
- ridge is the regularization parameter

In practice, this is solved via Cholesky decomposition for numerical stability:
```julia
XX = Symmetric(X * X' + ridge * I)
W_out = (Y * X') / cholesky(XX)
```

### 9.4 Multi-Layer Pipeline (`run_multi_layer`)

1. Run the coarse layer as a single-layer ESN (with `data_layer = zeros`)
2. Concatenate the coarse training predictions and test predictions into one timeline: `data_layer = [train_pred_coarse | test_pred_coarse]`
3. Run the fine layer as a single-layer ESN, but now each block receives the coarse predictions via `W_in_layer * data_layer[rows_layer, t]`

Note: during fine-layer testing, the coarse predictions used are the coarse model's own predictions (not the true coarse data). This is because in a real deployment, you wouldn't have access to future true data at any resolution.

### 9.5 Closed-Loop Neighbor Coupling

During autonomous testing, block predictions must be coordinated. At each timestep:

1. **First pass**: All blocks compute their outputs from current states (no state update yet)
2. **Second pass**: All blocks update their states using the freshly computed outputs of themselves and their neighbors

This two-pass scheme ensures that neighbor information is up-to-date when computing the state update, avoiding stale data.

---

## 10. How to Modify Things

### 10.1 Changing Hyperparameters

Open any `main_*.jl` script and modify the parameter block. Key parameters to experiment with:

| What to change | Where | Effect |
|----------------|-------|--------|
| Reservoir size | `res_size = 400` | More neurons = more capacity, slower training |
| Spectral radius | `res_radius = 0.3` | Higher = longer memory, risk of instability |
| Number of blocks | `num_networks = 16` | More blocks = finer spatial decomposition |
| Neighbor width | `mixing = 8` | More mixing = more inter-block communication |
| Ridge parameter | `ridge_param = 1e-3` | Higher = more regularization, less overfitting |
| Input scalings | `g_in_rec`, `g_in_neigh`, `g_in_layer` | Balance between input channels |
| Leak rate | `tau`, `dt` | dt/tau controls memory timescale |
| Readout mode | `regression_mode = :quadratic` | `:quadratic` adds nonlinear features |

### 10.2 Running Hyperparameter Searches

The `run_tuning_*.jl` scripts perform grid searches. Example structure:

```julia
grid = Dict(
    :res_radius  => [0.1, 0.2, 0.3, 0.5],
    :g_in_rec    => [0.01, 0.1, 0.5],
    :ridge_param => [1e-6, 1e-4, 1e-2],
)
grid_search(run_once_function, grid; nrep=20)
```

This evaluates every combination, repeating each 20 times (due to random reservoir initialization), and writes results to a CSV file.

**To add a new parameter to the search**: Add it to the `grid` dictionary and make sure the `run_once` function receives and uses it.

### 10.3 Using Different Data

**For a new 1D dataset:**
1. Format your data as a (space x time) CSV matrix
2. Add a loading method in `src/data/loading.jl` or just load it manually in a new main script
3. Normalize to zero mean and unit variance (the reservoir expects this)
4. Set `blocks = make_blocks(L, nblocks, mixing)` with appropriate values

**For a new 2D dataset:**
1. Store as a 3D array (lon x lat x time) in JLD2 format
2. Use `make_blocks_single_layer_2d(data, (div_lon, div_lat); mixing=m)` for spatial blocking
3. Flatten the 3D array to 2D for the reservoir: `reshape(data, nlon*nlat, nt)`

### 10.4 Adding a New Architecture

If you want to add a new dynamics engine (e.g., a different reservoir variant):

1. Create a new file in `src/dynamics/` (e.g., `my_architecture.jl`)
2. Implement `train_*` and `test_*` functions following the pattern in `single_layer.jl`
3. Include the file in `src/CrossScaleRC.jl`
4. Export your new functions
5. Create a `main_my_architecture.jl` entry script

### 10.5 Modifying the Reservoir State Update

The core state update equation is in `train_parallel_reservoir` and `test_parallel_reservoir` (both in `src/dynamics/single_layer.jl`). Look for the inner loop:

```julia
@inbounds for k in eachindex(x)
    x[k] = (1 - reservoir.dt_tau) * x[k] +
            reservoir.dt_tau * tanh(W_x[k] + Win_rec_u[k] + Win_neigh_u[k] + Win_layer_u[k])
end
```

To change the activation function, replace `tanh`. To add a bias, add a term. To change the leaking integration, modify the `(1-dt_tau)*x + dt_tau*...` structure. **Important**: any change here must be applied consistently in both `train_parallel_reservoir` and `test_parallel_reservoir`.

### 10.6 Modifying the Readout

The readout is computed in `fit_ridge_regression` (`src/dynamics/single_layer.jl`). To change from ridge regression to another method (e.g., LASSO, neural readout):
1. Replace the body of `fit_ridge_regression`
2. Keep the same signature: takes X (states), Y (targets), returns W_out matrix

### 10.7 Changing the Block Topology

Block creation is in `src/data/grids.jl`. Each block is a NamedTuple with three fields:
- `rows_rec`: which data rows this block predicts
- `rows_neigh`: which neighboring rows it sees
- `rows_layer`: which coarse-layer rows it receives

To create a custom topology (e.g., non-uniform blocks, graph-based neighborhoods), write a function that returns a `Vector{NamedTuple}` with these three fields. The only constraint is that all blocks must have the same `length(rows_rec)`, `length(rows_neigh)`, and `length(rows_layer)`.

### 10.8 Common Julia Workflow Tips

```bash
# Run a script
julia --project=. main_single_layer.jl

# Interactive development (recommended)
julia --project=.
julia> using Revise          # auto-reloads changed source files
julia> using CrossScaleRC
julia> include("main_single_layer.jl")

# Run tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Run a single test file
julia --project=. -e 'include("test/test_single_layer.jl")'
```

**Tip**: Use `Revise` during development. After `using Revise; using CrossScaleRC`, any changes you make to files in `src/` are automatically picked up without restarting Julia.

**Tip**: `BLAS.set_num_threads(1)` is called in the main scripts. This is because each block is parallelized via Julia threads (`Threads.@threads`), and nested BLAS parallelism causes thread contention. Start Julia with multiple threads: `julia --project=. -t 8`.

---

## 11. Glossary

| Term | Meaning |
|------|---------|
| **ESN** | Echo State Network - a recurrent neural network where only the output layer is trained |
| **Reservoir** | The fixed random recurrent network that transforms inputs into rich feature representations |
| **Readout** | The trained linear map from reservoir states to predictions (W_out) |
| **Teacher forcing** | Training mode where the true data drives the reservoir (not its own predictions) |
| **Closed-loop** | Test mode where the model feeds its own predictions back as input |
| **Spectral radius** | Largest absolute eigenvalue of the recurrent weight matrix W |
| **Leak rate (dt/tau)** | Controls how fast the reservoir forgets its past state. Small = long memory |
| **Ridge regression** | Linear regression with L2 regularization |
| **Spatial blocking** | Dividing the spatial domain into sub-regions, each with its own model |
| **Mixing** | Number of neighbor cells shared between adjacent blocks |
| **Cross-layer input** | Coarse-resolution predictions fed as input to the fine-resolution layer |
| **Lyapunov time** | Characteristic timescale of chaos: time for nearby trajectories to diverge by factor e |
| **Warmup** | Initial test phase where true data is fed to synchronize the reservoir state |
| **Washout** | Initial training steps discarded so the reservoir forgets its initial condition |
| **KS equation** | Kuramoto-Sivashinsky: a 1D chaotic PDE used as a benchmark |
| **SST** | Sea Surface Temperature: real-world 2D climate data |
| **JLD2** | Julia's binary data format (like HDF5) |
| **Quadratic features** | Augmenting the state vector by squaring every other component |
| **NG-RC** | Next-Generation Reservoir Computing: uses polynomial features instead of a reservoir |
| **DeepESN** | Deep Echo State Network: multiple reservoir layers stacked vertically |

---

*Generated for the CrossScaleRC repository. For the full scientific details, see the paper: https://doi.org/10.48550/arXiv.2510.11209*
