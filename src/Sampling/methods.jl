using Random

# :exact - Uses FWHT for global normalization (slow for large $N$, exponentially precise).
# :approx - Uses the new Mask-Based Ancestral Sampling (orders of magnitude faster)
function sample_bitstring(psum; prob_method::Symbol = :approx)
    nq = psum.nqubits
    
    # --- Method :exact (Global L1-normalized sampling) ---
    if prob_method == :exact
        # 1. Compute the full diagonal vector <x|rho|x> for all x
        #    This uses FWHT to get all 2^N amplitudes simultaneously.
        #    Note: raw_probs[k] corresponds to integer index k-1 (0-based)
        #    where Qubit 1 is the Least Significant Bit (LSB).
        raw_probs = compute_grouped_traces(psum, 1:nq)
        
        # 2. Apply the L1-normalization logic: P(x) = |raw(x)| / sum(|raw|)
        total_norm = sum(abs, raw_probs)
        if total_norm == 0
            error("State has zero norm; cannot sample.")
        end
        
        # 3. Sample an integer index from this distribution
        target = rand() * total_norm
        cumulative = 0.0
        sample_idx = 0 # This will hold the 0-based index of our sample
        
        for (i, val) in enumerate(raw_probs)
            cumulative += abs(val)
            if cumulative >= target
                sample_idx = i - 1 # Convert 1-based Julia index to 0-based integer
                break
            end
        end
        
        # 4. Convert integer `sample_idx` to BitVector matching :approx format
        #    In :approx, Qubit 1 is stored at bitstring[nq], Qubit n at bitstring[1].
        #    In FWHT `sample_idx`, Qubit 1 is the LSB (bit 0).
        bitstring = BitVector(undef, nq)
        for q in 1:nq
            # Extract bit (q-1) from sample_idx
            bit_val = (sample_idx >> (q - 1)) & 1 == 1
            # Store at nq+1-q to match the :approx "Little Endian" visual order
            bitstring[nq + 1 - q] = bit_val
        end
        
        return bitstring

    # --- Method :approx (Optimized Diagonal Ancestral Sampling) ---
    elseif prob_method == :approx
        # This replaces the old slow 'apply_projector' loop with the 
        # fast bitwise mask logic.
        
        # --- Pre-processing: Integer Masks ---
        term_coeffs = Float64[]
        term_masks = Vector{UInt64}() # Supports up to 64 qubits
        
        for (pstr, coeff) in psum
            mask = UInt64(0)
            is_diagonal = true
            
            for i in 1:nq
                p = getpauli(pstr, i)
                # Check for Z (3). If X(1) or Y(2), it's not diagonal.
                if ispauli(p, 3) 
                    mask |= (UInt64(1) << (i - 1))
                elseif ispauli(p, 1) || ispauli(p, 2)
                    is_diagonal = false
                    break
                end
            end
            
            # Only keep significant real diagonal terms
            if is_diagonal && abs(real(coeff)) > 1e-15
                push!(term_coeffs, real(coeff))
                push!(term_masks, mask)
            end
        end
        
        # --- Ancestral Sampling Loop ---
        bitstring = BitVector(undef, nq)
        history_mask = UInt64(0) 
        
        for q in 1:nq
            q_bit = UInt64(1) << (q - 1)
            
            # Mask of all future bits (q+1 to nq)
            future_mask = ~((UInt64(1) << q) - 1) 
            if q == nq
                future_mask = UInt64(0)
            end

            prob_unnorm_0 = 0.0
            prob_unnorm_1 = 0.0
            
            for k in 1:length(term_coeffs)
                mask = term_masks[k]
                coeff = term_coeffs[k]
                
                # 1. FUTURE CHECK: Skip if support on future qubits (traces to 0)
                if (mask & future_mask) != 0
                    continue
                end
                
                # 2. HISTORY CHECK: Parity from past samples
                past_overlap = mask & history_mask
                parity_sign = (count_ones(past_overlap) % 2 == 0) ? 1.0 : -1.0
                
                # 3. CURRENT QUBIT CHECK
                val = coeff * parity_sign
                
                if (mask & q_bit) == 0
                    # Identity on q
                    prob_unnorm_0 += val
                    prob_unnorm_1 += val
                else
                    # Z on q
                    prob_unnorm_0 += val
                    prob_unnorm_1 -= val
                end
            end
            
            # Normalize
            denom = prob_unnorm_0 + prob_unnorm_1
            p0 = 0.5 
            if abs(denom) > 1e-15
                p0 = prob_unnorm_0 / denom
            end
            p0 = clamp(p0, 0.0, 1.0)
            
            # Sample
            if rand() <= p0
                bitstring[nq+1-q] = false
            else
                bitstring[nq+1-q] = true
                history_mask |= q_bit
            end
        end
        
        return bitstring

    else
        error("Unknown prob_method: $prob_method. Use :exact or :approx.")
    end
end

function unnormalized_prob(psum, qind, x_i)
    a = getcoeff(psum, :I, qind)
    b = getcoeff(psum, :Z, qind)

    # Unwrap coefficients if they are PauliFreqTracker
    a_val = a isa PauliFreqTracker ? a.coeff : a
    b_val = b isa PauliFreqTracker ? b.coeff : b

    return abs(a_val + (-1.0)^x_i * b_val)
end

function normalized_prob(psum, qind, x_i)
    # Calculates the probability of observing a 0 for qubit qind under the distribution paramaterized by psum. 
    p0 = unnormalized_prob(psum, qind, 0)
    p1 = unnormalized_prob(psum, qind, 1)

    return (x_i == 0 ? p0 : p1) / (p0 + p1)
end

function sample_bitstring(psum, ::Type{UT}; marginal_func=bayes_marginal, proj_func=trace_project) where {UT <: Unsigned}
    nq = psum.nqubits
    running_psum = psum
    bitstring = zero(UT)
    for i in 1:nq
        p0 = marginal_func(running_psum, i, false)
        b_i = rand() ≥ p0
        bitstring = set_bit(bitstring, i, b_i)
        running_psum = proj_func(running_psum, i, b_i)
    end
    return bitstring
end


# AUXILIARIES FOR EXACT Method

"""
Calculates the diagonal elements <x|rho|x> for all bitstrings defined by `qinds`.
Returns a vector of length 2^length(qinds).
"""
function compute_grouped_traces(psum, qinds)
    num_qinds = length(qinds)
    n_states = 1 << num_qinds # 2^num_qinds
    coeffs = Vector{ComplexF64}(undef, n_states)
    
    # Pre-allocate buffer for constructing Z-strings
    # This replaces the need for `get_one_inds!`
    Z_str = fill(:Z, num_qinds)
    
    for i in 0:(n_states - 1)
        # 1. Identify which qubits in `qinds` are active (1) for index `i`
        #    to construct the corresponding Pauli Z string.
        #    (We only need Z terms because <x|X|x> = 0, so only I and Z survive diagonal)
        
        # We perform a "partial" retrieval: we only care about the coefficient 
        # of the Z string formed by the bits of `i`.
        
        # Construct the specific Z-term for this index `i`
        # e.g., if i=1 (binary 01), we want coeff of I Z
        current_one_inds = Int[]
        for bit_pos in 0:(num_qinds - 1)
            if (i >> bit_pos) & 1 == 1
                push!(current_one_inds, bit_pos + 1)
            end
        end
        
        # Get coefficient from PauliSum
        if isempty(current_one_inds)
            # i=0 corresponds to the Identity term (all I's)
            # We ask for the coefficient of Identity on the requested qinds
            coeffs[i + 1] = getcoeff(psum, :I, qinds[1]) # Simplified: Coeff of Identity is global
        else
            @views Z_view = Z_str[1:length(current_one_inds)]
            @views active_qinds = qinds[current_one_inds]
            coeffs[i + 1] = getcoeff(psum, Z_view, active_qinds)
        end
    end

    # 2. Perform Inverse Fast Walsh-Hadamard Transform (IFWHT)
    #    This converts Pauli-Z coefficients into Computational Basis amplitudes.
    return naive_ifwht!(coeffs)    
end

"""
In-place Inverse Fast Walsh-Hadamard Transform.
Converts Pauli coefficients -> State probabilities (unnormalized).
"""
function naive_ifwht!(x::Vector{ComplexF64})
    n = length(x)
    logn = trailing_zeros(n)
    @assert 2^logn == n "Length must be a power of 2"
    
    for i in 0:logn-1
        step = 1 << (i + 1)
        half = 1 << i
        for j in 1:step:n
            for k in 0:half-1
                a = x[j + k]
                b = x[j + k + half]
                x[j + k] = a + b
                x[j + k + half] = a - b
            end
        end
    end
    # Note: Standard IFWHT scaling is usually handled here or in the coefficients.
    # For relative probabilities |P|/sum|P|, constant scaling factors cancel out, 
    # so strictly speaking x ./= n isn't required for the distribution shape,
    # but we keep it for numerical consistency with density matrix values.
    x ./= n 
    return x
end