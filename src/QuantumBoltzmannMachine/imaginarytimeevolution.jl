# struct ImaginaryPauliRotation <: ParametrizedGate
#     symbols::Vector{Symbol}
#     qinds::Vector{Int}
# end

# function PauliPropagation._tomaskedpaulirotation(imag_pauli_gate::ImaginaryPauliRotation, nqubits)
#     pstr_term = symboltoint(nqubits, imag_pauli_gate.symbols, imag_pauli_gate.qinds)
#     return MaskedPauliRotation(imag_pauli_gate.symbols, imag_pauli_gate.qinds, pstr_term)
# end

# # Helper to check if a PauliString is purely diagonal (only I or Z)
# function _is_diagonal(pstr, nq)
#     for i in 1:nq
#         p = getpauli(pstr, i)
#         # Pauli encoding: 0=I, 1=X, 2=Y, 3=Z. 
#         # We fail if we see 1 (X) or 2 (Y).
#         if ispauli(p, 1) || ispauli(p, 2)
#             return false
#         end
#     end
#     return true
# end

# function PauliPropagation.applytoall!(gate::ImaginaryPauliRotation, theta::Real, psum, aux_psum; prune_non_diagonal::Bool=false, kwargs...)

#     # nq = psum.nqubits

#     # turn the PauliRotation gate into a MaskedPauliRotation gate
#     # this allows for faster operations with bitwise speed
#     gate = PauliPropagation._tomaskedpaulirotation(gate, paulitype(psum))

#     # pre-compute the sinh and cosh values because they are used for every Pauli string that commutes with the gate
#     cosh_val = cosh(theta)
#     sinh_val = sinh(theta)

#     # loop over all Pauli strings and their coefficients in the Pauli sum
#     for (pstr, coeff) in psum
        
#         # If they anti-commute, the term is unchanged. Skip it.
#         # (WARNING: This means an existing X/Y term might survive here! 
#         #  That is why the final filter in makethermalstate is still required)
#         if !commutes(gate, pstr)
#             # if the gate does not commute with the pauli string, do nothing
#             continue
#         end

#         # else the gate splits the Pauli string into two
#         # Evolution Rule: P -> cosh(θ)P - sinh(θ)PG
#         new_pstr, phase = pauliprod(gate.generator_mask, pstr, gate.qinds)

#         # --- OPTIMIZATION for Sampling START ---
#         # If we are in the final layer (prune_non_diagonal=true), check before we write!        
#         if prune_non_diagonal
#             # 1. Optimize Child: If the NEW term has X or Y, don't even calculate it.
#             #    We skip writing to aux_psum entirely.
#             if PauliPropagation.containsXorY(new_pstr)
#                 # We strictly discard this branch. 
#                 # But we MUST still process the parent branch below.
#                 # So we just don't write to aux_psum.
#             else
#                 coeff2 = -1 * real(phase) * coeff * sinh_val
#                 set!(aux_psum, new_pstr, coeff2)
#             end

#             # 2. Optimize Parent: If the OLD term has X or Y, kill it now.
#             if PauliPropagation.containsXorY(pstr)
#                 # Effectively delete it from psum by setting coeff to 0
#                 set!(psum, pstr, 0.0)
#             else
#                 # Valid parent: Update its coefficient
#                 coeff1 = coeff * cosh_val
#                 set!(psum, pstr, coeff1)
#             end
            
#             # We are done with this term. Continue to next.
#             continue
#         end

#         # --- OPTIMIZATION for Sampling END ---

#         # Standard behavior (if not pruning)
#         coeff1 = coeff * cosh_val
#         coeff2 = -1 * real(phase) * coeff * sinh_val

#         set!(psum, pstr, coeff1)
#         set!(aux_psum, new_pstr, coeff2)
#     end

#     return
# end

# function applymergetruncate_ite!(gate, psum, aux_psum, thetas, param_idx; prune_non_diagonal=false, max_weight=Inf, min_abs_coeff=1e-10, max_freq=Inf, max_sins=Inf, customtruncfunc=nothing, normalization=false, kwargs...)

#     # Pick out the next theta if gate is a ParametrizedGate.
#     # Else set the paramter to nothing for clarity that theta is not used.
#     if gate isa ParametrizedGate
#         theta = thetas[param_idx]
#         # If the gate is parametrized, decrement theta index by one.
#         param_idx -= 1
#     else
#         theta = nothing
#     end
#     # Apply the gate to all Pauli strings in psum, potentially writing into auxillary aux_psum in the process.
#     # The pauli sums will be changed in-place
#     applytoall!(gate, theta, psum, aux_psum; prune_non_diagonal=prune_non_diagonal, kwargs...)

#     # Any contents of psum and aux_psum are merged into the larger of the two, which is returned as psum.
#     # The other is emptied and returned as aux_psum.
#     psum, aux_psum = mergeandempty!(psum, aux_psum)

#     if normalization
#         coeff_I = getcoeff(psum, :I, 1)
#         if coeff_I isa PauliFreqTracker
#             coeff_I = coeff_I.coeff  # extract numeric value
#         end
#         min_abs_coeff = min_abs_coeff * coeff_I
#     end

#     # Check truncation conditions on all Pauli strings in psum and remove them if they are truncated.
#     PauliPropagation.checktruncationonall!(psum; max_weight, min_abs_coeff, max_freq, max_sins, customtruncfunc)

#     return psum, aux_psum, param_idx
# end

# function propagate_ite!(
#     circ, psum, thetas=nothing; 
#     prune_at_final_step::Bool=false, 
#     max_weight=Inf, min_abs_coeff=1e-10, max_freq=Inf, max_sins=Inf,
#     customtruncfunc=nothing, normalization=false, kwargs...
# )
#     # Check that max_freq and max_sins are only used if psum tracks them
#     PauliPropagation._checkfreqandsinfields(psum, max_freq, max_sins)

#     # Promote circuit and thetas if needed
#     circ, thetas = PauliPropagation._promotecircandthetas(circ, thetas)

#     # Check consistency between circ and thetas
#     PauliPropagation._checkcircandthetas(circ, thetas)

#     # Start from the last parameter if thetas is not nothing
#     param_idx = thetas === nothing ? nothing : length(thetas)

#     # Create an auxiliary Pauli sum for intermediate terms
#     aux_psum = similar(psum)

#     # We iterate through the circuit (reversed). 
#     # The "Last Step" of the simulation corresponds to the LAST gate in this loop.
#     reversed_circ = reverse(circ)
#     len_circ = length(reversed_circ)

#     for (i, gate) in enumerate(reversed_circ)
        
#         # Check if we are at the very last gate of the sequence AND pruning is requested
#         is_last_gate = (i == len_circ)
#         do_prune = (prune_at_final_step && is_last_gate)

#         psum, aux_psum, param_idx = applymergetruncate_ite!(
#             gate, psum, aux_psum, thetas, param_idx;
#             prune_non_diagonal=do_prune, # Only true at the very end
#             max_weight=max_weight, min_abs_coeff=min_abs_coeff,
#             max_freq=max_freq, max_sins=max_sins,
#             customtruncfunc=customtruncfunc, normalization=normalization,
#             kwargs...
#         )
#     end

#     return psum
# end

function makethermalstate(nq::Integer, circuit::Vector{Gate}, thetas::AbstractVector{CT}, num_layers::Integer; 
    beta::Real = 1.0, 
    max_weight=Inf, 
    max_sins=Inf, 
    min_abs_coeff=1e-10,
    optimize_for_z_basis::Bool = false # <--- New Argument: Master Switch
) where {CT}

    # psum = PauliSum(CT, nq)
    psum = VectorPauliSum(CT, nq) # THIS USES MULTITHREADING!
    add!(psum, PauliString(nq, :I, 1, 1))

    # Only wrap in PauliFreqTracker if we are actually using sine-based truncation.
    # The tracker is not compatible with ForwardDiff.Dual, so we skip it during AD/Optimization.
    # When training, max_sins is typically Inf anyway.
    use_tracker = (max_sins < Inf)

    if use_tracker
        wrapped_psum = wrapcoefficients(psum, PauliFreqTracker)
    else
        wrapped_psum = psum
    end

    for i in 1:num_layers
        # 1. OPTIMIZATION: Tell the propagator to avoid creating NEW junk in the final layer
        # Logic: We prune ONLY if the user requested the optimization 
        #        AND we are currently in the final layer.

        wrapped_psum = propagate!(
            circuit, wrapped_psum, (beta / num_layers) * thetas; 
            # prune_at_final_step=should_prune, # Pass the dynamic condition
            max_weight=max_weight, max_sins=max_sins, 
            min_abs_coeff=min_abs_coeff, normalization=true, heisenberg=false
        )
    end

    if use_tracker
        unwrapped_psum = unwrapcoefficients(wrapped_psum)
    else
        unwrapped_psum = wrapped_psum
    end    # Normalize c_I

    # mult!(unwrapped_psum, 1 / getcoeff(unwrapped_psum, :I, 1))

    # 2. SAFETY NET: The Final Sweep
    # This catches:
    #   a) Terms that didn't commute with the final gate (skipped by applytoall)
    #   b) Numerical noise (abs < 1e-15)
    if optimize_for_z_basis
        zerofilter!(psum)
    end

    ## THIS IS NOT NECESSARY BECAUSE PROBS NORMALIZE AGAIN
    # # Final Trace Normalization
    # mult!(unwrapped_psum, 1 / 2.0^nq)

    return unwrapped_psum
end

# function makethermalstate(nq::Integer, circuit::Vector{Gate}, thetas::AbstractVector{CT}, num_layers::Integer; 
#     beta::Real = 1.0, max_weight=Inf, max_sins=Inf, min_abs_coeff=1e-10) where {CT}

#     psum = PauliSum(CT, nq)
#     add!(psum, PauliString(nq, :I, 1, 1))
#     wrapped_psum = wrapcoefficients(psum, PauliFreqTracker)

#     for i in 1:num_layers
#         wrapped_psum = propagate_ite!(circuit, wrapped_psum, (beta / num_layers) * thetas; max_weight=max_weight, max_sins=max_sins, min_abs_coeff=min_abs_coeff, normalization=true)
#     end

#     unwrapped_psum = unwrapcoefficients(wrapped_psum)
#     mult!(unwrapped_psum, 1 / getcoeff(unwrapped_psum, :I, 1))  # Normalize c_I = 1
#     mult!(unwrapped_psum, 1 / 2.0^nq)  # Normalize trace = 1
#     return unwrapped_psum
# end


"""
    makethermalstate_and_probs(H::AbstractMatrix{<:Number}, β::Real) -> (ρ::Matrix, probs::Vector)

Calculates the thermal state density matrix ρ = exp(-βH) / Z and its eigenvalues (thermal probabilities).
"""
function makethermalstate_matrix(H::AbstractMatrix{<:Number}, β::Real)
    
    # 1. Convert to Dense and Diagonalize H
    H_dense = Matrix(H) 

    if !ishermitian(H_dense)
        # Handle non-Hermitian case if necessary (less efficient)
        @warn "Hamiltonian is not Hermitian. Falling back to matrix exponentiation."
        A = -β .* H_dense
        E = exp(A)
        ρ = E ./ tr(E)
        
        # Calculate eigenvalues of ρ for the probabilities
        # NOTE: This still requires a second diagonalization for non-Hermitian case
        probs = eigen(ρ).values 
        return ρ, real.(probs) # Return real part of eigenvalues
    end

    # 2. Eigendecomposition (Only ONCE)
    vals, vecs = eigen(Hermitian(H_dense)) # vals are the energy eigenvalues E_n

    # 3. Calculate Thermal Probabilities (Eigenvalues of ρ)
    # The eigenvalues of ρ are p_n = exp(-βE_n) / Z
    
    min_E = minimum(vals)
    # Numerically stable calculation of unnormalized probabilities
    unnorm_probs = exp.(-β .* (vals .- min_E))
    
    Z = sum(unnorm_probs)
    
    # Final thermal probabilities: p_n
    probs = unnorm_probs ./ Z
    
    # 4. Reconstruct Density Matrix (ρ)
    # ρ = V * Diagonal(p_n) * V'
    ρ = vecs * Diagonal(probs) * vecs'

    return ρ, probs
end

#########

function propagate_ite_with_stats!(
    circ, psum, thetas=nothing; 
    prune_at_final_step::Bool=false, 
    max_weight=Inf, min_abs_coeff=1e-10, max_freq=Inf, max_sins=Inf,
    customtruncfunc=nothing, normalization=false, kwargs...
)
    PauliPropagation._checkfreqandsinfields(psum, max_freq, max_sins)
    circ, thetas = PauliPropagation._promotecircandthetas(circ, thetas)
    PauliPropagation._checkcircandthetas(circ, thetas)

    param_idx = thetas === nothing ? nothing : length(thetas)
    aux_psum = similar(psum)
    
    pre_prune_count = 0
    reversed_circ = reverse(circ)
    len_circ = length(reversed_circ)

    for (i, gate) in enumerate(reversed_circ)
        is_last_gate = (i == len_circ)
        do_prune = (prune_at_final_step && is_last_gate)

        # CAPTURE POINT: Right before the last gate prunes XY terms
        # This is the "Total Paulis before final pruning" your colleague wants.
        if do_prune
            pre_prune_count = length(psum)
        end

        psum, aux_psum, param_idx = applymergetruncate_ite!(
            gate, psum, aux_psum, thetas, param_idx;
            prune_non_diagonal=do_prune, 
            max_weight=max_weight, min_abs_coeff=min_abs_coeff,
            max_freq=max_freq, max_sins=max_sins,
            customtruncfunc=customtruncfunc, normalization=normalization,
            kwargs...
        )
    end

    # Returns the modified psum and the count recorded at the start of the final gate
    return psum, pre_prune_count
end

function makethermalstate_with_stats(nq::Integer, circuit::Vector{Gate}, thetas::AbstractVector{CT}, num_layers::Integer; 
    beta::Real = 1.0, 
    max_weight=Inf, 
    max_sins=Inf, 
    min_abs_coeff=1e-10,
    optimize_for_z_basis::Bool = true
) where {CT}

    psum = PauliSum(CT, nq)
    add!(psum, PauliString(nq, :I, 1, 1))

    use_tracker = (max_sins < Inf)
    wrapped_psum = use_tracker ? wrapcoefficients(psum, PauliFreqTracker) : psum

    total_paulis_before_prune = 0

    for i in 1:num_layers
        # Optimization only triggers in the final Trotter layer
        should_prune = optimize_for_z_basis && (i == num_layers)

        wrapped_psum, last_count = propagate_ite_with_stats!(
            circuit, wrapped_psum, (beta / num_layers) * thetas; 
            prune_at_final_step=should_prune,
            max_weight=max_weight, max_sins=max_sins, 
            min_abs_coeff=min_abs_coeff, normalization=true
        )
        
        if should_prune
            total_paulis_before_prune = last_count
        end
    end

    unwrapped_psum = use_tracker ? unwrapcoefficients(wrapped_psum) : wrapped_psum
    mult!(unwrapped_psum, 1 / getcoeff(unwrapped_psum, :I, 1))

    # Final filter to ensure purity and handle small numerical noise
    filter!(unwrapped_psum) do p_int, coeff
        if abs(coeff) <= 1e-15
            return false
        end
        if optimize_for_z_basis
            if PauliPropagation.containsXorY(p_int)
                return false
            end
        end
        return true
    end

    # Final Normalization to get density matrix traces
    final_diagonal_count = length(unwrapped_psum)
    mult!(unwrapped_psum, 1 / 2.0^nq)

    # Return the state and the stats as requested for the x-axis analysis
    return unwrapped_psum, (before=total_paulis_before_prune, after=final_diagonal_count)
end