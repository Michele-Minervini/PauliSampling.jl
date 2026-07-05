# Fixed configuration for a cluster run. The swept hyperparameters
# (min_abs_coeff, max_weight, neighbor_distance) are passed on the command line.
function getsetup()
    return (
        rows        = 6,
        cols        = 6,
        digit       = 1,
        n_per_class = 1000,
        beta        = 1.0,
        num_layers  = 1,
        max_order   = 2,
        nsteps      = 300,
        lr          = 0.05,
        seed        = 7,
        gradient    = :ad,         # :ad | :spsa | :fd
        init        = :data_warm,      # :randn | :data | :data_warm | :data_coupling
        init_clamp  = 0.9,         # data_warm: marginal clamp (lower = warmer start)
        init_alpha  = 1.0,         # data_warm: scale on data params (lower = warmer start)
        min_count   = 2,           # drop target patterns seen fewer than this many times
    )
end
