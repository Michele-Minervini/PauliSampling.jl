# Batch QBM training on reduced MNIST.
#   julia --project=. --threads=N cluster/main.jl <min_abs_coeff> <max_weight> <neighbor_distance> [nsteps]
# Fixed settings live in setup.jl; results are serialized to cluster/results/.

include(joinpath(@__DIR__, "setup.jl"))
using PauliSampling
using Serialization, Random, Printf, LinearAlgebra
using Base.Threads 

function main(ARGS)
    LinearAlgebra.BLAS.set_num_threads(1)
    length(ARGS) >= 3 || error("usage: cluster/main.jl <min_abs_coeff> <max_weight> <neighbor_distance> [nsteps]")
    min_abs_coeff     = parse(Float64, ARGS[1])
    max_weight        = parse(Int, ARGS[2])
    neighbor_distance = parse(Int, ARGS[3])

    println("Starting run with min_abs_coeff=$(min_abs_coeff), max_weight=$(max_weight), neighbor_distance=$(neighbor_distance) ", nthreads()," thread(s)")
    
    s  = getsetup()
    nsteps = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : s.nsteps
    nq = s.rows * s.cols

    savedir  = joinpath(@__DIR__, "results"); mkpath(savedir)
    savename = joinpath(savedir, @sprintf("qbm_%dx%d_digit%d_d%d_o%d_mac%.0e_mw%d_beta%.1f_%s",
        s.rows, s.cols, s.digit, neighbor_distance, s.max_order, min_abs_coeff, max_weight, s.beta, s.gradient))

    ds = generate_mnist_dataset(s.rows, s.cols; digit_classes=[s.digit], n_per_class=s.n_per_class,
                                binarize_method=:adaptive, seed=1)
    supp, probs = extract_support(ds; min_count=s.min_count)
    H  = build_h_general(s.rows, s.cols; max_distance=neighbor_distance, max_order=s.max_order, seed=s.seed)
    K  = length(H)

    mkrho(θ) = prepare_thermal_state(h_from_flat(θ, H), nq; beta=s.beta, num_layers=s.num_layers,
                                     min_abs_coeff=min_abs_coeff, max_weight=max_weight)
    loss(θ)  = kl_support(mkrho(θ), supp, probs)

    rng = MersenneTwister(s.seed)
    gradfn = s.gradient === :ad   ? (l, θ) -> ad_gradient(l, θ) :
             s.gradient === :spsa ? (l, θ) -> spsa_gradient(l, θ; rng=rng) :
             s.gradient === :fd   ? (l, θ) -> forward_fd_gradient_spawn(l, θ) :
             error("unknown gradient $(s.gradient)")

    theta = s.init === :randn         ? [h.coeff for h in H] :
            s.init === :data          ? data_init(H, supp, probs, nq; beta=s.beta, rng=MersenneTwister(s.seed)) :
            s.init === :data_warm     ? data_init(H, supp, probs, nq; beta=s.beta, clampval=s.init_clamp, alpha=s.init_alpha, rng=MersenneTwister(s.seed)) :
            s.init === :data_coupling ? data_init(H, supp, probs, nq; beta=s.beta, field_mode=:random, rng=MersenneTwister(s.seed)) :
            error("unknown init $(s.init)")
    adam = AdamState(K)

    losses=Float64[]; params=Vector{Float64}[]; paulis=Int[]; times=Float64[]; grad_norms=Float64[]
    best_loss=Inf; best_theta=copy(theta)

    @printf("%dx%d digit=%d | min_abs_coeff=%.0e max_weight=%d neighbor_distance=%d | init=%s gradient=%s nsteps=%d\n",
            s.rows, s.cols, s.digit, min_abs_coeff, max_weight, neighbor_distance, s.init, s.gradient, nsteps)
    @printf("support=%d (min_count=%d) | K=%d | threads=%d\n",
            length(supp), s.min_count, K, Base.Threads.nthreads())
    flush(stdout)

    gradfn(loss, theta)
    for t in 1:nsteps
        dt = @timed begin
            g, _ = gradfn(loss, theta)
            adam_step!(theta, g, s.lr, adam)
            g
        end
        g   = dt.value
        rho = mkrho(theta)
        l   = kl_support(rho, supp, probs)
        np  = length(rho)
        gn  = sqrt(sum(abs2, g))
        push!(losses, l); push!(params, copy(theta)); push!(paulis, np); push!(times, dt.time); push!(grad_norms, gn)
        l < best_loss && (best_loss = l; copyto!(best_theta, theta))
        @printf("step %4d/%d | loss=%.5f | paulis=%d | |g|=%.3f | %.2fs\n", t, nsteps, l, np, gn, dt.time)
        flush(stdout)
        serialize(savename, (; losses, params, paulis, times, grad_norms, best_loss, best_theta,
                               config=s, min_abs_coeff, max_weight, neighbor_distance, nsteps, K,
                               support=length(supp), nq))
    end
    @printf("done | best loss=%.5f | %s\n", best_loss, savename)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end

