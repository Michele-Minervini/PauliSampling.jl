function normalizing_factor(psum)
    nq = psum.nqubits
    acc = 0
    for one_inds in powerset(1:nq)
        acc += abs(overlapwithcomputational(psum, one_inds))
    end
    return acc
end

function faster_normalizing_factor(psum)
    nq = psum.nqubits
    f_psum = filter_no_XY(psum)
    return normalizing_factor(f_psum)
end

function coset_normalizing_factor(psum; to_filter=true, num_set_bits=0, set_bits=BitVector())
    nq = psum.nqubits
    if to_filter
        f_psum = filter_no_XY(psum)
    end
    num_pstrs = length(f_psum)

    Z_matrix = BitMatrix(undef, num_pstrs, nq)
    coeffs = Vector{ComplexF64}(undef, num_pstrs)
    for (i, (term, coeff)) in enumerate(f_psum)
        for j in 1:nq
            pauli = getpauli(term, j)
            Z_matrix[i, j] = ispauli(pauli, 3)
        end
        coeffs[i] = coeff
    end

    @views Z_matrix_a = Z_matrix[:, 1:num_set_bits]
    Z_matrix_b = Z_matrix[:, num_set_bits+1:nq]

    y = f2_mul(Z_matrix_a, set_bits)

    RREF, rank, pivots = F2Algebra.rref(Z_matrix_b)
    @views basis = Z_matrix_b[:, pivots]

    acc = 0.0
    v = zero(Z_matrix[:, 1])
    for i in 0:(2^rank - 1)
        fill!(v, false)
        for j in 0:(rank - 1)
            if ((i >> j) & 1) == 1
                v .⊻= @view basis[:, j + 1]
            end
        end

        tmp = 0.0 + 0.0im
        for k in 1:num_pstrs
            sgn = ifelse(v[k] ⊻ y[k], -1.0, 1.0)
            tmp += coeffs[k] * sgn
        end
        acc += abs(tmp)
    end
    acc *= 2^(nq - rank - num_set_bits)
    return acc
end

function compute_grouped_traces(psum, qinds)
    # TODO: I don't think I'm normalizing correctly, but it's still giving correct results. Why?
    num_qinds = length(qinds)
    coeffs = Vector{ComplexF64}(undef, 2^num_qinds)
    one_inds = Int[]
    sizehint!(one_inds, num_qinds)
    Z_str = fill(:Z, num_qinds)

    for i in 0:(2^(num_qinds) - 1)
        empty!(one_inds)
        get_one_inds!(one_inds, num_qinds, i)
        @views Z_view = Z_str[1:length(one_inds)]
        @views qinds_view = qinds[one_inds]
        coeffs[i + 1] = getcoeff(psum, Z_view, qinds_view)
    end
    # eturn ifwht(coeffs)
    return naive_ifwht!(coeffs)    
end

function naive_ifwht!(x::Vector{ComplexF64})
    n = length(x)
    logn = trailing_zeros(n)
    @assert 2^logn == n "length must be a power of 2"
    
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
    x ./= n
    return x
end

function approximate_normalizing_factor(psum, qinds)
    vals = compute_grouped_traces(psum, qinds)
    return sum(abs, vals)
end

function projection_prob(psum, one_bit_inds::Vector{<:Integer})
    numerator = abs(overlapwithcomputational(psum, one_bit_inds))
    denominator = coset_normalizing_factor(psum)

    return numerator / denominator
end

function projection_prob(psum, x::BitVector)
    return projection_prob(psum, findall(x))
end

function projection_prob(psum, x::Integer)
    nq = psum.nqubits
    return projection_prob(psum, BitVector(digits(x, base=2, pad=nq)))
end

function exact_marginal(psum, i, b_i; to_filter=true, num_set_bits=0, set_bits=BitVector())
    b = coset_normalizing_factor(nq, psum; to_filter=to_filter, num_set_bits=num_set_bits, set_bits=set_bits)
    proj_psum = apply_projector(psum, i, b_i)
    a = coset_normalizing_factor(nq, proj_psum; to_filter=to_filter, num_set_bits=num_set_bits, set_bits=set_bits)

    return a / b
end

function approximate_marginal(psum, qinds, i, b_i)
    # i must be the first element in qinds. Could get rid of this assumption, but probably not worth it. 
    vals = compute_grouped_traces(psum, qinds)
    p0_unnormalized = sum(abs, @view vals[1:end÷2])
    p1_unnormalized = sum(abs, @view vals[end÷2 + 1:end])
    a = p0_unnormalized
    if b_i == 1
        a = p1_unnormalized
    end
    return a / (p0_unnormalized + p1_unnormalized)
end

function approximate_prob(psum, x::BitVector; qinds_func=qinds_base_func)
    running_psum = psum
    nq = running_psum.nqubits
    p = 1.0
    for i in 1:nq
        qinds = qinds_base_func(i, nq)
        p *= approximate_marginal(running_psum, qinds, i, x[i])
        running_psum = apply_projector(running_psum, i, x[i])
    end
    return p
end

function approximate_prob(psum, x::Integer)
    nq = psum.nqubits
    return approximate_prob(psum, BitVector(digits(x, base=2, pad=nq)))
end