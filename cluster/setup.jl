# ────────────────────────────────────────────────────────────────────
# Cluster run configuration (FIXED across a hyperparameter sweep).
#
# The SWEPT hyperparameters — min_abs_coeff, max_weight, neighbor_distance —
# are passed on the command line (see cluster/main.jl). Edit the values below
# to change the fixed part of the experiment (system size, digit, β, …).
# ────────────────────────────────────────────────────────────────────
function getsetup()
    return (
        rows        = 5,      # lattice rows   (system = rows × cols qubits)
        cols        = 5,      # lattice cols   (5×5 = 25 qubits)
        digit       = 1,      # MNIST digit to learn
        n_per_class = 1000,   # number of training images
        beta        = 2.0,    # inverse temperature of the thermal state
        num_layers  = 1,      # imaginary-time-evolution layers
        max_order   = 2,      # interaction order: 2 = 2-body; >2 requires neighbor_distance ≥ 2
        nsteps      = 300,    # training iterations (override with the optional 4th CLI arg)
        lr0         = 0.05,   # cosine learning-rate schedule: start
        lr1         = 0.015,  # cosine learning-rate schedule: end
        seed        = 7,      # Hamiltonian-initialization seed
        gradient    = :ad,    # :ad (exact, default) | :spsa (cheap/noisy) | :fd
    )
end
