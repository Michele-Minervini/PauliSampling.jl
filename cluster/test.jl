# Quick look at the reduced-MNIST images main.jl trains on.
#   julia --project=. cluster/test.jl [nx] [ny] [n_show]
using Pkg
Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "setup.jl"))
using PauliSampling
using Plots

s = getsetup()
nx = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : s.rows
ny = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : s.cols
n_show = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 8

ds = generate_mnist_dataset(nx, ny; digit_classes=[s.digit], n_per_class=s.n_per_class,
                             binarize_method=:adaptive, seed=1)

n_show = min(n_show, length(ds))
plots = [heatmap(bitvector_to_image(ds[i], nx, ny); yflip=true, color=:grays,
                  aspect_ratio=:equal, axis=false, colorbar=false, title="digit $i")
         for i in 1:n_show]

fig = plot(plots...; layout=(1, n_show), size=(200 * n_show, 220))
# outfile = joinpath(@__DIR__, "mnist_preview.png")
# savefig(fig, outfile)
println("saved preview of $n_show images ($(nx)x$(ny)) to $outfile")
