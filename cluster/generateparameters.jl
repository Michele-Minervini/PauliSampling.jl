CURRENT_DIR = @__DIR__

## Sweep over the hyperparameters main.jl expects:
##   min_abs_coeff  max_weight  neighbor_distance  [nsteps]
min_abs_coeff_vals = [1e-2, 1e-3, 1e-4]

max_weight_vals = [Inf]

neighbor_distance_vals = [1, 2, 3, 4]

f = open("$CURRENT_DIR/parameters.txt"; write=true)
for (min_abs_coeff, max_weight, neighbor_distance) in Iterators.product(min_abs_coeff_vals, max_weight_vals, neighbor_distance_vals)
    str = ""
    str *= string(min_abs_coeff) * " "
    str *= string(max_weight) * " "
    str *= string(neighbor_distance)

    @show str
    write(f, str * "\n")
end

close(f)