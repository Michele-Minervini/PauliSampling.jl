# PauliSampling.jl

Sampling from quantum states via **Pauli propagation**, and training **Quantum
Boltzmann Machines (QBMs)** as generative models. Code for the paper
*"Sampling from Quantum States via Pauli Propagation."*

## Repository layout

| Path | What it is |
|------|------------|
| `src/` | The `PauliSampling` package. `Sampling/` = Algorithm-1 sampler & probabilities; `QuantumBoltzmannMachine/` = Hamiltonians, imaginary-time evolution (ITE), optimization; `Training/` = QBM generative-modeling pipeline (data, models, loss, gradients, `train_qbm`). |
| `test/` | Package tests. |
| `Project.toml`, `Manifest.toml` | The exact Julia environment — instantiate to reproduce. |

## Setup

```julia
julia --project=.
julia> using Pkg; Pkg.instantiate()      # builds the exact environment
```

## Train a QBM (MNIST, single digit)

The training pipeline is part of the package (`src/Training/`):

```julia
using PauliSampling

res = train_qbm(5, 5; digit=1, nsteps=150)    # 5×5 MNIST, exact AD gradient (default)
# res.theta  res.kls  res.rho  res.supp  res.probs
```

**Key hyperparameters** (`train_qbm` keywords):

| Knob | Meaning |
|------|---------|
| `beta` | inverse temperature of the thermal state |
| `min_abs_coeff` | Pauli truncation threshold |
| `max_weight` | max Pauli weight kept |
| `num_layers` | ITE layers |
| `nsteps`, `lr0`, `lr1` | Adam steps + cosine learning-rate schedule |
| `gradient_fn` | `ad_gradient` (exact, default) · `spsa_gradient` · `forward_fd_gradient_spawn` |
| `Htemplate=build_h_general(r, c; max_distance, max_order)` | tunable connectivity: **`max_distance`** = neighbor_distance (1 = nearest-neighbor → diameter = all-to-all), **`max_order`** = interaction order (2-body, 3-body, …) |

Threading: launch with `--threads=N` (the AD gradient parallelizes over chunks);
keep `LinearAlgebra.BLAS.set_num_threads(1)`.

## Cluster runs

Batch entry point in `cluster/main.jl` (`main(ARGS)` format). It trains one digit
and serializes per-iteration loss / params / Pauli-count / time to `cluster/results/`.

```bash
julia --project=. --threads=N cluster/main.jl <min_abs_coeff> <max_weight> <neighbor_distance> [nsteps]
# e.g.
julia --project=. --threads=8 cluster/main.jl 1e-2 25 4
```

The swept hyperparameters are the CLI args; the fixed config (system size, digit,
β, `max_order`, gradient, …) lives in `cluster/setup.jl`. Load results with
`using Serialization; r = deserialize("cluster/results/<name>")`.
