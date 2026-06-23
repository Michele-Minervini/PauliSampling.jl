###############################################################
#   TRAINING — entry point                                    #
#   train_qbm: fit a QBM to reduced MNIST (default = exact AD) #
###############################################################

"""
    train_qbm(rows, cols; kwargs...) -> (; theta, H, kls, rho, supp, probs)

Train a fully-visible QBM on reduced MNIST (single `digit`) by minimizing the
support-KL of the truncated Pauli-ITE thermal state with Adam + a cosine LR.

The gradient DEFAULTS to exact threaded automatic differentiation
(`gradient_fn = ad_gradient`): at these sizes it is the most accurate gradient
*and* the cheapest per step. Override with `gradient_fn = spsa_gradient` (noisy
8-eval estimate — wins for large K / all-to-all) or `forward_fd_gradient_spawn`.
Any function with the `(loss_fn, theta) -> (gradient, loss)` signature works.

Pass `Htemplate = build_h_general(rows, cols; max_distance, max_order)` to train a
generalized (range+order) connectivity model instead of the physical `model`.

Returns a NamedTuple: best-KL coefficients `theta`, template `H`, per-step KL
history `kls`, final thermal state `rho`, and the data `supp`/`probs`.
"""
function train_qbm(rows::Int, cols::Int;
        digit::Int = 1, n_per_class::Int = 1000, min_count::Int = 2, model::Symbol = :tfim_tilted,
        beta::Float64 = 2.0, min_abs_coeff::Float64 = 1e-2, num_layers::Int = 1,
        nsteps::Int = 150, lr::Float64 = 0.05,
        seed::Int = 7, init::Symbol = :randn, init_clamp::Float64 = 0.9, init_alpha::Float64 = 1.0,
        Htemplate = nothing, gradient_fn = ad_gradient, verbose::Bool = true)
    nq = rows * cols
    ds = generate_mnist_dataset(rows, cols; digit_classes=[digit], n_per_class=n_per_class,
                                binarize_method=:adaptive, seed=1)
    supp, probs = extract_support(ds; min_count=min_count)
    # Default: physical model via build_h_template. Pass `Htemplate=build_h_general(...)`
    # to train a generalized (range+order) connectivity model instead.
    H = Htemplate === nothing ? build_h_template(rows, cols; model=model, init=:randn,
                         field_scale=0.1, coupling_scale=0.1, seed=seed) : Htemplate
    K = length(H)
    lf = theta -> kl_support(prepare_thermal_state(h_from_flat(theta, H), nq;
                             beta=beta, num_layers=num_layers,
                             min_abs_coeff=min_abs_coeff, max_weight=nq), supp, probs)
    # constant learning rate (cosine schedule removed — with exact AD a fixed LR works fine)
    lr_at(s) = lr

    # init options (all reproducible via MersenneTwister(seed)):
    #   :randn         → template's small random coefficients (default; warm, safe at all sizes,
    #                    but high-variance at ≥4×4 — forward-KL gradient can explode if the random
    #                    model misses a support point).
    #   :data          → full data mean-field (fields←marginals, ZZ←correlations): ~10× lower start
    #                    + best fit at SMALL sizes, but COLD → operator blow-up at ≥5×5.
    #   :data_warm     → warm-directed: fields clamped (init_clamp) & α-scaled (init_alpha) + data
    #                    couplings. Keeps the data signal but tamed → cluster-safer (lower clamp/α = warmer).
    #   :data_coupling → random fields + data couplings (warm & stable, but fields carry the fidelity,
    #                    so the gain over :randn is small).
    theta = init === :randn         ? [h.coeff for h in H] :
            init === :data          ? data_init(H, supp, probs, nq; beta=beta, rng=MersenneTwister(seed)) :
            init === :data_warm     ? data_init(H, supp, probs, nq; beta=beta, clampval=init_clamp, alpha=init_alpha, rng=MersenneTwister(seed)) :
            init === :data_coupling ? data_init(H, supp, probs, nq; beta=beta, field_mode=:random, rng=MersenneTwister(seed)) :
            error("unknown init=$(init) (use :randn, :data, :data_warm, or :data_coupling)")
    adam = AdamState(K)
    kls = Float64[]; best = Inf; btheta = copy(theta)
    gradient_fn(lf, theta)   # warm / compile the chosen gradient
    verbose && (println("train_qbm $(rows)x$(cols): n=$nq K=$K |supp|=$(length(supp)), $nsteps steps, grad=$(nameof(gradient_fn))"); flush(stdout))
    for s in 1:nsteps
        g, l = gradient_fn(lf, theta)
        adam_step!(theta, g, lr_at(s), adam)
        push!(kls, l); (l < best) && (best = l; copyto!(btheta, theta))
        verbose && (s % 25 == 0 || s == 1) && (@printf "  step %3d  KL=%.4f\n" s l; flush(stdout))
    end
    rho = prepare_thermal_state(h_from_flat(btheta, H), nq; beta=beta, num_layers=num_layers,
                                min_abs_coeff=min_abs_coeff, max_weight=nq)
    verbose && (@printf "  done: best KL=%.4f  (final state: %d Pauli terms)\n" best length(rho); flush(stdout))
    return (theta=btheta, H=H, kls=kls, rho=rho, supp=supp, probs=probs)
end
