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

"""
    data_init(H, supp, probs, nq; beta=2.0, coupling_scale=1.0,
              field_mode=:data, clampval=0.999, alpha=1.0, rng=Random.default_rng()) -> theta

Data-driven initialization of the coefficients for template `H`, from the empirical
distribution (`supp`, `probs`) — a classical mean-field starting point:
  • 1-body Zᵢ field        →  θ_i  = α·atanh(-clamp(⟨z_i⟩,±clampval))/β   (marginal match)
  • 2-body ZᵢZⱼ coupling   →  θ_ij = -α·coupling_scale·cov(z_i,z_j)/β       (observed correlation)
  • transverse (X/Y) or higher-order terms → small random                  (no classical analog)
where z = +1 for bit 0, -1 for bit 1.

Knobs that TAME the cold start. (Empirically the FIELDS are both the explosion source —
βθ_i = atanh(-⟨z_i⟩) → ±∞ for always-on/off pixels — *and* the fidelity source; the
couplings are bounded |θ_ij|≤1/β and mainly add training stability.)
  • `field_mode=:random` → warm random fields instead of marginals (couplings stay
                           data-driven). Warm/stable but low fidelity (fields carry the signal).
  • `clampval<0.999`     → cap the divergent extreme-pixel fields (warmer; the explosion source).
  • `alpha<1`            → scale the whole data-derived θ down (warmer).

Defaults reproduce the FULL mean-field init: ~10× lower starting support-KL than `randn`
and the best fit at SMALL sizes (3×3/4×4), but COLD → blows up the truncated operator at
≥5×5. For the cluster use a *warm-directed* start
(`train_qbm(...; init=:data_warm, init_clamp=…, init_alpha=…)`), lowering clamp/α until the
init is feasible. See [`train_qbm`].
"""
function data_init(H, supp::Vector{BitVector}, probs::Vector{Float64}, nq::Int;
        beta::Real = 2.0, coupling_scale::Real = 1.0,
        field_mode::Symbol = :data, clampval::Real = 0.999, alpha::Real = 1.0,
        rng = Random.default_rng())
    zmean = zeros(nq); zz = zeros(nq, nq)              # z = +1 if bit 0, -1 if bit 1
    for (x, p) in zip(supp, probs)
        z = [x[i] ? -1.0 : 1.0 for i in 1:nq]
        for i in 1:nq
            zmean[i] += p * z[i]
            for j in 1:nq; zz[i, j] += p * z[i] * z[j]; end
        end
    end
    theta = zeros(length(H))
    @inbounds for (k, h) in enumerate(H)
        zq = Int[]; nonZ = false
        for q in 1:nq
            op = Int(getpauli(h.term, q))
            op == 3 && push!(zq, q)
            (op == 1 || op == 2) && (nonZ = true)
        end
        if nonZ
            theta[k] = 0.05 * randn(rng)                                  # transverse: no classical analog
        elseif length(zq) == 1
            if field_mode === :random
                theta[k] = 0.1 * randn(rng)                               # warm random field (coupling-only init)
            else
                zi = clamp(zmean[zq[1]], -clampval, clampval)
                theta[k] = alpha * atanh(-zi) / beta                      # field ← marginal (clamped & α-scaled)
            end
        elseif length(zq) == 2
            i, j = zq
            theta[k] = alpha * (-coupling_scale * (zz[i, j] - zmean[i] * zmean[j]) / beta)   # coupling ← correlation
        else
            theta[k] = 0.05 * randn(rng)                                  # higher-order: no simple analog
        end
    end
    return theta
end
