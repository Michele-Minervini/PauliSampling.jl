function sample_bitstring(psum; prob_method::Symbol = :approx)
    nq = psum.nqubits
    running_psum = psum
    bitstring = BitVector(undef, nq)

    if prob_method == :coset
        f = coset_normalizing_factor(running_psum)
    end

    for qind in 1:nq
        if prob_method == :coset
            proj0 = apply_projector(running_psum, qind, false)
            bitstring[qind] = false
            f0 = coset_normalizing_factor(proj0; num_set_bits=qind, set_bits=@view bitstring[1:qind])
            p0 = f0 / f

            if rand() <= p0
                bitstring[qind] = false
                f = f0
                running_psum = proj0
            else
                bitstring[qind] = true
                f = f - f0
                running_psum = running_psum - proj0
            end

        elseif prob_method == :approx

            p0 = normalized_prob(running_psum, qind, 0)        
            
            if rand() <= p0
                bitstring[qind] = false
            else
                bitstring[qind] = true
            end
            running_psum = apply_projector(running_psum, qind, bitstring[qind])

        else
            error("Unknown prob_method: $prob_method. Use :coset or :approx.")
        end
    end

    return bitstring
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