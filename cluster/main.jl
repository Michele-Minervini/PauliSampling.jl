# Batch QBM training on reduced MNIST.
#   julia --project=. --threads=N cluster/main.jl <min_abs_coeff> <max_weight> <neighbor_distance> [nsteps]
# Fixed settings live in setup.jl; results are serialized to cluster/results/.
using Pkg
Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "setup.jl"))
using PauliSampling
using Serialization, Random, Printf, LinearAlgebra
using Base.Threads
using Plots

const IMAGES_PER_EPOCH = 100

function main(ARGS)
    LinearAlgebra.BLAS.set_num_threads(1)
    length(ARGS) ==5 || error("usage: cluster/main.jl <nx> <ny> <min_abs_coeff> <max_weight> <neighbor_distance> [nsteps]")
    nx                = parse(Int, ARGS[1])
    ny                = parse(Int, ARGS[2])
    min_abs_coeff     = parse(Float64, ARGS[3])
    max_weight        = parse(Float64, ARGS[4])
    neighbor_distance = parse(Int, ARGS[5])

    s  = getsetup()
    nsteps =  s.nsteps
    nq = nx * ny

    max_weight = isinf(max_weight) ? nq : Int(max_weight)

    println("Starting run with min_abs_coeff=$(min_abs_coeff), max_weight=$(max_weight), neighbor_distance=$(neighbor_distance) ", nthreads()," thread(s)")

    savedir  = joinpath(@__DIR__, "results"); mkpath(savedir)
    savename = joinpath(savedir, @sprintf("qbm_%dx%d_digit%d_d%d_o%d_mac%.0e_mw%d_beta%.1f_%s",
        nx, ny, s.digit, neighbor_distance, s.max_order, min_abs_coeff, max_weight, s.beta, s.gradient))

    imgdir = joinpath(@__DIR__, "images", join(ARGS, "_"))
    rm(imgdir; force=true, recursive=true); mkpath(imgdir)

    ds = generate_mnist_dataset(nx, ny; digit_classes=[s.digit], n_per_class=s.n_per_class,
                                binarize_method=:adaptive, seed=1)
    supp, probs = extract_support(ds; min_count=s.min_count)
    probs = [1.0 / length(probs) for _ in probs] # make it uniform

    H  = build_h_general(nx, ny; max_distance=neighbor_distance, max_order=s.max_order, seed=s.seed)
    K  = length(H)

    mkrho(θ) = prepare_thermal_state(h_from_flat(θ, H), nq; beta=s.beta, num_layers=s.num_layers,
                                     min_abs_coeff=min_abs_coeff, max_weight=max_weight)
    loss(θ)  = kl_support(mkrho(θ), supp, probs)

    rng = MersenneTwister(s.seed)
    gradfn = s.gradient === :ad   ? (l, θ) -> ad_gradient(l, θ; max_parallel=s.max_parallel) :
             s.gradient === :ad_serial   ? (l, θ) -> ad_gradient_serial(l, θ) :
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
            nx, ny, s.digit, min_abs_coeff, max_weight, neighbor_distance, s.init, s.gradient, nsteps)
    @printf("support=%d (min_count=%d) | K=%d | threads=%d\n",
            length(supp), s.min_count, K, Base.Threads.nthreads())
    flush(stdout)

    @time gradfn(loss, theta)
    println("first gradient passed")
    flush(stdout)
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

        epochdir = joinpath(imgdir, "epoch$t")
        mkpath(epochdir)
        samples  = sample_bitstrings(rho, IMAGES_PER_EPOCH)
        for (i, bv) in enumerate(samples)
            img = bitvector_to_image(reverse(bv), nx, ny)
            savefig(heatmap(img; yflip=true, color=:grays, aspect_ratio=:equal, axis=false, colorbar=false),
                    joinpath(epochdir, "$i.pdf"))
        end
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

