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
        digit::Int = 1, n_per_class::Int = 1000, model::Symbol = :tfim_tilted,
        beta::Float64 = 2.0, min_abs_coeff::Float64 = 1e-2, num_layers::Int = 1,
        nsteps::Int = 150, lr0::Float64 = 0.05, lr1::Float64 = 0.015,
        seed::Int = 7, init::Symbol = :randn, Htemplate = nothing,
        gradient_fn = ad_gradient, verbose::Bool = true)
    nq = rows * cols
    ds = generate_mnist_dataset(rows, cols; digit_classes=[digit], n_per_class=n_per_class,
                                binarize_method=:adaptive, seed=1)
    supp, probs = extract_support(ds)
    # Default: physical model via build_h_template. Pass `Htemplate=build_h_general(...)`
    # to train a generalized (range+order) connectivity model instead.
    H = Htemplate === nothing ? build_h_template(rows, cols; model=model, init=:randn,
                         field_scale=0.1, coupling_scale=0.1, seed=seed) : Htemplate
    K = length(H)
    lf = theta -> kl_support(prepare_thermal_state(h_from_flat(theta, H), nq;
                             beta=beta, num_layers=num_layers,
                             min_abs_coeff=min_abs_coeff, max_weight=nq), supp, probs)
    # cosine LR from lr0 -> lr1 over nsteps
    cos_lr(s) = lr1 + 0.5 * (lr0 - lr1) * (1 + cos(pi * (s - 1) / max(nsteps - 1, 1)))

    # init = :randn → the template's random coefficients (default; safe at all sizes).
    #        :data  → data-driven (mean-field) start: ~10× lower starting KL and a better
    #                 fit at SMALL sizes (3×3/4×4), but the cold start can blow up the
    #                 truncated operator at ≥5×5 — use only at small sizes (see data_init).
    theta = init === :data  ? data_init(H, supp, probs, nq; beta=beta, rng=MersenneTwister(seed)) :
            init === :randn ? [h.coeff for h in H] :
            error("unknown init=$(init) (use :data or :randn)")
    adam = AdamState(K)
    kls = Float64[]; best = Inf; btheta = copy(theta)
    gradient_fn(lf, theta)   # warm / compile the chosen gradient
    verbose && (println("train_qbm $(rows)x$(cols): n=$nq K=$K |supp|=$(length(supp)), $nsteps steps, grad=$(nameof(gradient_fn))"); flush(stdout))
    for s in 1:nsteps
        g, l = gradient_fn(lf, theta)
        adam_step!(theta, g, cos_lr(s), adam)
        push!(kls, l); (l < best) && (best = l; copyto!(btheta, theta))
        verbose && (s % 25 == 0 || s == 1) && (@printf "  step %3d  KL=%.4f\n" s l; flush(stdout))
    end
    rho = prepare_thermal_state(h_from_flat(btheta, H), nq; beta=beta, num_layers=num_layers,
                                min_abs_coeff=min_abs_coeff, max_weight=nq)
    verbose && (@printf "  done: best KL=%.4f  (final state: %d Pauli terms)\n" best length(rho); flush(stdout))
    return (theta=btheta, H=H, kls=kls, rho=rho, supp=supp, probs=probs)
end
