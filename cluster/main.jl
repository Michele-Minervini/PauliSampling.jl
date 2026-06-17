# ════════════════════════════════════════════════════════════════════
# cluster/main.jl — batch QBM training on reduced MNIST (cluster entry point).
#
# Trains a Quantum Boltzmann Machine ρ_θ ∝ exp(-βH(θ)) (truncated Pauli-ITE)
# to fit a single MNIST digit. Records loss / params / Pauli-count / step-time /
# gradient-norm EVERY iteration and serializes incrementally (robust to job kills).
#
# SETUP (once): instantiate the exact environment
#     julia --project=<repo> -e 'using Pkg; Pkg.instantiate()'
#
# RUN:
#     julia --project=<repo> --threads=N cluster/main.jl <min_abs_coeff> <max_weight> <neighbor_distance> [nsteps]
#   examples:
#     julia --project=. --threads=8 cluster/main.jl 1e-2 25 4
#     julia --project=. --threads=8 cluster/main.jl 1e-3 25 2 500
#
#   ARGS = the swept hyperparameters:
#     1  min_abs_coeff      Pauli truncation threshold   (Float64, e.g. 1e-2)
#     2  max_weight         max Pauli weight kept        (Int, e.g. 25 = no weight cut on 5×5)
#     3  neighbor_distance  connectivity range           (Int: 1 = nearest-neighbor … diameter = all-to-all)
#     4  nsteps (optional)  training iterations          (Int; default from setup.jl)
#
# The FIXED config (system size, digit, β, max_order, gradient, …) is in cluster/setup.jl.
#
# RESULTS: serialized to cluster/results/<descriptive name>. Load with:
#     using Serialization; r = deserialize("cluster/results/<name>")
#     # r.losses  r.params  r.paulis  r.times  r.grad_norms  r.best_loss  r.best_theta  r.config  …
# ════════════════════════════════════════════════════════════════════
include(joinpath(@__DIR__, "setup.jl"))
using PauliSampling
using Serialization, Random, Printf, LinearAlgebra
using Base.Threads

const CURRENT_DIR = @__DIR__

function main(ARGS)
    LinearAlgebra.BLAS.set_num_threads(1)   # avoid BLAS × Julia-thread oversubscription

    length(ARGS) >= 3 || error("usage: cluster/main.jl <min_abs_coeff::Float64> <max_weight::Int> <neighbor_distance::Int> [nsteps::Int]")
    min_abs_coeff     = parse(Float64, ARGS[1])
    max_weight        = parse(Int,     ARGS[2])
    neighbor_distance = parse(Int,     ARGS[3])

    s = getsetup()
    nsteps = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : s.nsteps
    nq = s.rows * s.cols

    savedir = joinpath(CURRENT_DIR, "results"); mkpath(savedir)
    savename = joinpath(savedir, @sprintf("qbm_%dx%d_digit%d_d%d_o%d_mac%.0e_mw%d_beta%.1f_%s",
        s.rows, s.cols, s.digit, neighbor_distance, s.max_order, min_abs_coeff, max_weight, s.beta, s.gradient))

    println("="^72)
    @printf "QBM cluster run | %d×%d (n=%d) digit=%d\n" s.rows s.cols nq s.digit
    @printf "  swept:  min_abs_coeff=%.0e  max_weight=%d  neighbor_distance=%d\n" min_abs_coeff max_weight neighbor_distance
    @printf "  fixed:  beta=%.2f  num_layers=%d  max_order=%d  nsteps=%d  gradient=%s  seed=%d\n" s.beta s.num_layers s.max_order nsteps s.gradient s.seed
    @printf "  threads=%d  BLAS=%d\n  save -> %s\n" Threads.nthreads() LinearAlgebra.BLAS.get_num_threads() savename
    println("="^72); flush(stdout)

    # ── data + model ──────────────────────────────────────────────
    ds = generate_mnist_dataset(s.rows, s.cols; digit_classes=[s.digit],
                                n_per_class=s.n_per_class, binarize_method=:adaptive, seed=1)
    supp, probs = extract_support(ds)
    H = build_h_general(s.rows, s.cols; max_distance=neighbor_distance, max_order=s.max_order, seed=s.seed)
    K = length(H)
    @printf "data support = %d patterns | model K = %d parameters\n" length(supp) K; flush(stdout)

    mkrho(θ) = prepare_thermal_state(h_from_flat(θ, H), nq; beta=s.beta, num_layers=s.num_layers,
                                     min_abs_coeff=min_abs_coeff, max_weight=max_weight)
    loss(θ)  = kl_support(mkrho(θ), supp, probs)

    # gradient estimator (persistent RNG for SPSA so each step uses fresh directions)
    rng = MersenneTwister(s.seed)
    gradfn = s.gradient === :ad   ? (l, θ) -> ad_gradient(l, θ) :
             s.gradient === :spsa ? (l, θ) -> spsa_gradient(l, θ; rng=rng) :
             s.gradient === :fd   ? (l, θ) -> forward_fd_gradient_spawn(l, θ) :
             error("unknown gradient $(s.gradient) — use :ad, :spsa, or :fd")

    # ── training loop: record EVERYTHING per iteration, serialize incrementally ──
    theta = [h.coeff for h in H]
    adam  = AdamState(K)
    cos_lr(t) = s.lr1 + 0.5 * (s.lr0 - s.lr1) * (1 + cos(pi * (t - 1) / max(nsteps - 1, 1)))

    losses     = Float64[]
    params     = Vector{Float64}[]
    paulis     = Int[]
    times      = Float64[]
    grad_norms = Float64[]
    best_loss  = Inf
    best_theta = copy(theta)

    gradfn(loss, theta)   # warm / compile the chosen gradient
    for t in 1:nsteps
        dt = @timed begin
            grad, _ = gradfn(loss, theta)
            adam_step!(theta, grad, cos_lr(t), adam)
            grad
        end
        g  = dt.value
        ρ  = mkrho(theta)                       # state at updated θ → loss + Pauli count from one build
        l  = kl_support(ρ, supp, probs)
        np = length(ρ)
        gn = sqrt(sum(abs2, g))

        push!(losses, l); push!(params, copy(theta)); push!(paulis, np)
        push!(times, dt.time); push!(grad_norms, gn)
        if l < best_loss; best_loss = l; copyto!(best_theta, theta); end

        @printf "step %4d/%d | loss=%.5f | Paulis=%6d | |g|=%.3f | %.2fs\n" t nsteps l np gn dt.time
        flush(stdout)

        serialize(savename, (; losses, params, paulis, times, grad_norms,
                               best_loss, best_theta,
                               config = s, min_abs_coeff, max_weight, neighbor_distance,
                               nsteps, K, support = length(supp), nq))
    end

    @printf "DONE | best loss = %.5f | results -> %s\n" best_loss savename
    flush(stdout)
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
