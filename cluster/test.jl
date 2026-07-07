# Timing bench for the two expensive pieces of the training loop, isolated
# from gradients/Adam: thermal state construction and the KL loss.
#   julia --project=. cluster/test.jl [min_abs_coeff] [max_weight] [neighbor_distance]
using Pkg
Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "setup.jl"))
using PauliSampling
using Random, Printf
using PauliPropagation

s = getsetup()
min_abs_coeff     = 5e-3
max_weight        = Inf
neighbor_distance  = 2

nq = s.rows * s.cols
max_weight = isinf(max_weight) ? nq : Int(max_weight)

ds = generate_mnist_dataset(s.rows, s.cols; digit_classes=[s.digit], n_per_class=s.n_per_class,
                             binarize_method=:adaptive, seed=1)
supp, probs = extract_support(ds; min_count=s.min_count)
# sort(probs, rev=true)
probs = [1.0 / length(probs) for _ in probs]
H = build_h_general(s.rows, s.cols; max_distance=neighbor_distance, max_order=s.max_order, seed=s.seed)

theta = s.init === :randn         ? [h.coeff for h in H] :
        s.init === :data          ? data_init(H, supp, probs, nq; beta=s.beta, rng=MersenneTwister(s.seed)) :
        s.init === :data_warm     ? data_init(H, supp, probs, nq; beta=s.beta, clampval=0.5, alpha=0.3, rng=MersenneTwister(s.seed)) :
        s.init === :data_coupling ? data_init(H, supp, probs, nq; beta=s.beta, field_mode=:random, rng=MersenneTwister(s.seed)) :
        error("unknown init $(s.init)")

mkrho(θ) = prepare_thermal_state(h_from_flat(θ, H), nq; beta=s.beta, num_layers=s.num_layers,
                                  min_abs_coeff=min_abs_coeff, max_weight=max_weight)

@printf("%dx%d | min_abs_coeff=%.0e max_weight=%d neighbor_distance=%d | K=%d support=%d\n",
        s.rows, s.cols, min_abs_coeff, max_weight, neighbor_distance, length(H), length(supp))

# warm up once so @time below measures runtime, not JIT compilation
rho = mkrho(theta)

l   = kl_support(rho, supp, probs)
println("paulis=", length(rho), "  loss=", l)

println("--- thermal state creation ---")
@time rho = mkrho(theta);

println("--- KL loss ---")
@time l = kl_support(rho, supp, probs);

VSCodeServer.@profview kl_support(rho, supp, probs)




# This is to get a sense of how the data looks
using Plots
nx = 6 #length(ARGS) >= 1 ? parse(Int, ARGS[1]) : s.rows
ny = 5 #length(ARGS) >= 2 ? parse(Int, ARGS[2]) : s.cols
n_show = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 8

ds = generate_mnist_dataset(nx, ny; digit_classes=[0], n_per_class=s.n_per_class,
                             binarize_method=:adaptive, seed=1)

supp, probs = extract_support(ds; min_count=s.min_count)
inds = sortperm(probs; rev=true)
supp = supp[inds]
probs = probs[inds]
probs

n_show = min(n_show, length(ds))
plots = [heatmap(bitvector_to_image(supp[i], nx, ny); yflip=true, color=:grays,
                  aspect_ratio=:equal, axis=false, colorbar=false, title="digit $i")
         for i in 1:n_show]

fig = plot(plots...; layout=(1, n_show), size=(200 * n_show, 220))
# outfile = joinpath(@__DIR__, "mnist_preview.png")
# savefig(fig, outfile)
# println("saved preview of $n_show images ($(nx)x$(ny)) to $outfile")