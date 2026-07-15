CURRENT_DIR = @__DIR__

## Sweep over the hyperparameters main.jl expects:
sizes = [(7, 6)]
##   min_abs_coeff  max_weight  neighbor_distance  [nsteps]
min_abs_coeff_vals = [3e-2, 1e-2, 7e-3] # [1e-2, 1e-3, 1e-4]

max_weight_vals = [Inf]

neighbor_distance_vals = [2]

continue_run = false

f = open("$CURRENT_DIR/parameters.txt"; write=true)
for ((nx, ny), min_abs_coeff, max_weight, neighbor_distance) in Iterators.product(sizes, min_abs_coeff_vals, max_weight_vals, neighbor_distance_vals)
    str = ""
    str *= string(nx) * " "
    str *= string(ny) * " "
    str *= string(min_abs_coeff) * " "
    str *= string(max_weight) * " "
    str *= string(neighbor_distance) * " "
    str *= string(continue_run)

    @show str
    write(f, str * "\n")
end

close(f)