struct ImaginaryPauliRotation <: ParametrizedGate
    symbols::Vector{Symbol}
    qinds::Vector{Int}
end

function PauliPropagation._tomaskedpaulirotation(imag_pauli_gate::ImaginaryPauliRotation, nqubits)
    pstr_term = symboltoint(nqubits, imag_pauli_gate.symbols, imag_pauli_gate.qinds)
    return MaskedPauliRotation(imag_pauli_gate.symbols, imag_pauli_gate.qinds, pstr_term)
end

function PauliPropagation.applytoall!(gate::ImaginaryPauliRotation, theta::Real, psum, aux_psum; kwargs...)

    # turn the PauliRotation gate into a MaskedPauliRotation gate
    # this allows for faster operations
    gate = PauliPropagation._tomaskedpaulirotation(gate, paulitype(psum))

    # pre-compute the sinh and cosh values because they are used for every Pauli string that commutes with the gate
    cosh_val = cosh(theta)
    sinh_val = sinh(theta)

    # loop over all Pauli strings and their coefficients in the Pauli sum
    for (pstr, coeff) in psum

        if !commutes(gate, pstr)
            # if the gate does not commute with the pauli string, do nothing
            continue
        end
        # else the gate splits the Pauli string into two
        coeff1 = coeff * cosh_val
        new_pstr, phase = pauliprod(gate.generator_mask, pstr, gate.qinds)
        coeff2 = -1 * phase * coeff * sinh_val

        # set the coefficient of the original Pauli string
        set!(psum, pstr, coeff1)

        # set the coefficient of the new Pauli string in the aux_psum
        # we can set the coefficient because PauliRotations create non-overlapping new Pauli strings
        set!(aux_psum, new_pstr, coeff2)
    end

    return
end

function applymergetruncate_ite!(gate, psum, aux_psum, thetas, param_idx;max_weight=Inf, min_abs_coeff=1e-10, max_freq=Inf, max_sins=Inf, customtruncfunc=nothing, normalization=false, kwargs...)

    # Pick out the next theta if gate is a ParametrizedGate.
    # Else set the paramter to nothing for clarity that theta is not used.
    if gate isa ParametrizedGate
        theta = thetas[param_idx]
        # If the gate is parametrized, decrement theta index by one.
        param_idx -= 1
    else
        theta = nothing
    end
    # Apply the gate to all Pauli strings in psum, potentially writing into auxillary aux_psum in the process.
    # The pauli sums will be changed in-place
    applytoall!(gate, theta, psum, aux_psum; kwargs...)

    # Any contents of psum and aux_psum are merged into the larger of the two, which is returned as psum.
    # The other is emptied and returned as aux_psum.
    psum, aux_psum = mergeandempty!(psum, aux_psum)

    if normalization
        coeff_I = getcoeff(psum, :I, 1)
        if coeff_I isa PauliFreqTracker
            coeff_I = coeff_I.coeff  # extract numeric value
        end
        min_abs_coeff = min_abs_coeff * coeff_I
    end

    # Check truncation conditions on all Pauli strings in psum and remove them if they are truncated.
    PauliPropagation.checktruncationonall!(psum; max_weight, min_abs_coeff, max_freq, max_sins, customtruncfunc)

    return psum, aux_psum, param_idx
end

function propagate_ite!(
    circ, psum, thetas=nothing;
    max_weight=Inf, min_abs_coeff=1e-10, max_freq=Inf, max_sins=Inf,
    customtruncfunc=nothing, normalization=false, kwargs...
)
    # Check that max_freq and max_sins are only used if psum tracks them
    PauliPropagation._checkfreqandsinfields(psum, max_freq, max_sins)

    # Promote circuit and thetas if needed
    circ, thetas = PauliPropagation._promotecircandthetas(circ, thetas)

    # Check consistency between circ and thetas
    PauliPropagation._checkcircandthetas(circ, thetas)

    # Start from the last parameter if thetas is not nothing
    param_idx = thetas === nothing ? nothing : length(thetas)

    # Create an auxiliary Pauli sum for intermediate terms
    aux_psum = similar(psum)

    # Loop through gates in reverse order
    for gate in reverse(circ)
        psum, aux_psum, param_idx = applymergetruncate_ite!(
            gate, psum, aux_psum, thetas, param_idx;
            max_weight=max_weight, min_abs_coeff=min_abs_coeff,
            max_freq=max_freq, max_sins=max_sins,
            customtruncfunc=customtruncfunc, normalization=normalization,
            kwargs...
        )
    end

    return psum
end

function makethermalstate(nq::Integer, circuit::Vector{Gate}, thetas::AbstractVector{CT}, num_layers::Integer; 
    beta::Real = 1.0, max_weight=Inf, max_sins=Inf, min_abs_coeff=1e-10) where {CT}

    psum = PauliSum(CT, nq)
    add!(psum, PauliString(nq, :I, 1, 1))
    wrapped_psum = wrapcoefficients(psum, PauliFreqTracker)

    for i in 1:num_layers
        wrapped_psum = propagate_ite!(circuit, wrapped_psum, (beta / num_layers) * thetas; max_weight=max_weight, max_sins=max_sins, min_abs_coeff=min_abs_coeff, normalization=true)
    end

    unwrapped_psum = unwrapcoefficients(wrapped_psum)
    mult!(unwrapped_psum, 1 / getcoeff(unwrapped_psum, :I, 1))  # Normalize c_I = 1
    mult!(unwrapped_psum, 1 / 2.0^nq)  # Normalize trace = 1
    return unwrapped_psum
end