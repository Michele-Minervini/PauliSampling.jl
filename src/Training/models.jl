###############################################################
#   TRAINING — model templates & thermal state                #
#   Build H(θ) and the truncated Pauli-ITE thermal state ρ_θ  #
###############################################################

"""
    build_h_template(rows, cols; model, connectivity, init, field_scale, coupling_scale, seed)

Return a `Vector{PauliString}` defining the trainable Pauli terms of a physical
H(θ) (e.g. `:tfim_tilted`) on a `rows`×`cols` square lattice, with initial
coefficients. The terms are fixed; only the coefficients (`theta`) change during training.
"""
function build_h_template(rows::Int, cols::Int;
        model::Symbol = :tfim_tilted,
        connectivity::Symbol = :nearest,
        init::Symbol = :randn,
        field_scale::Real = 0.1,
        coupling_scale::Real = 0.1,
        seed::Int = 0)
    seed > 0 && Random.seed!(seed)
    lat = SquareLattice(rows, cols)
    params = parameters(model, lat.n; init = init,
        field_scale = field_scale, coupling_scale = coupling_scale)
    return makehamiltonian(params, lat;
        connectivity = connectivity, periodic = false, model = model)
end

"""
    build_h_general(rows, cols; max_distance=1, max_order=2, field_axes=(:X,:Z),
                    interaction_axis=:Z, ...) -> Vector{PauliString}

Generalized-connectivity template (RANGE + ORDER knobs) for training: 2-body couplings
out to graph-distance `max_distance`, PLUS k-body terms (k ≤ `max_order`) on every
distance-clique. `max_distance=1, max_order=2` reproduces the *structure* of the physical
`:tfim_tilted` model (X,Z fields + nearest-neighbor ZZ). Feed the result to
`train_qbm(rows, cols; Htemplate = build_h_general(...))` to train that model with AD.
"""
function build_h_general(rows::Int, cols::Int;
        max_distance::Int = 1, max_order::Int = 2,
        field_axes = (:X, :Z), interaction_axis::Symbol = :Z,
        field_scale::Real = 0.1, coupling_scale::Real = 0.1,
        order_decay::Real = 1.0, seed::Int = 0)
    seed > 0 && Random.seed!(seed)
    lat = SquareLattice(rows, cols)
    return makehamiltonian_generalized(lat; max_distance=max_distance, max_order=max_order,
        field_axes=field_axes, interaction_axis=interaction_axis, init=:randn,
        field_scale=field_scale, coupling_scale=coupling_scale, order_decay=order_decay)
end

"""
    prepare_thermal_state(H, nq; beta, num_layers, min_abs_coeff, max_weight)

Truncated Pauli-ITE thermal state ρ_θ ≈ exp(-βH(θ)) / Z, with `optimize_for_z_basis=true`
(drops X/Y terms in the final ITE layer since we sample in computational basis).
"""
function prepare_thermal_state(H::Vector{<:PauliString}, nq::Int;
        beta::Float64 = 1.0, num_layers::Int = 1,
        min_abs_coeff::Float64 = 1e-4, max_weight::Int = nq)
    circuit, thetas = paulistringtocircuit(H)
    return makethermalstate(nq, circuit, thetas, num_layers;
        beta = beta, max_weight = max_weight, min_abs_coeff = min_abs_coeff,
        optimize_for_z_basis = true)
end

"""
    h_from_flat(theta, H_template) -> Vector{PauliString}

Rebuild H with new coefficients while keeping the template's Pauli structure.
`theta` is typed `AbstractVector` (not `Vector{Float64}`) so ForwardDiff `Dual`
vectors flow through unchanged — this is what makes exact AD (`ad_gradient`) possible.
"""
h_from_flat(theta::AbstractVector, H_template::Vector{<:PauliString}) =
    [PauliString(h.nqubits, h.term, theta[i]) for (i, h) in enumerate(H_template)]
