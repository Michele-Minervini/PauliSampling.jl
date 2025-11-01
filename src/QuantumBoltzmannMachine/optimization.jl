"""
    pauliweight(p::PauliString)

Return the number of non-identity Pauli operators in a PauliString.
"""
function pauliweight(p::PauliString)
    w = 0
    for i in 1:p.nqubits
        pauli = getpauli(p.term, i)
        # getpauli returns 0 for :I, 1 for :X, 2 for :Y, 3 for :Z
        if pauli != 0
            w += 1
        end
    end
    return w
end

# Gradient component ∂_{θ_i} S(η||ρ_θ)
"""
    computepartialderivative(H_i::PauliString, eta::PauliSum, rho_theta::PauliSum; spin_scaling=true)

Compute the partial derivative of the quantum relative entropy
term corresponding to `H_i`, optionally applying the spin-½ scaling.

- If `spin_scaling=true` (default):
    - divide by 2 if `H_i` has Pauli weight 1
    - divide by 4 if `H_i` has Pauli weight 2
- Otherwise, return the unscaled difference.

The result is (Tr[η H_i] - Tr[ρ_θ H_i]) with optional scaling.
"""
function computepartialderivative(H_i::PauliString, eta::PauliSum, rho_theta::PauliSum; spin_scaling::Bool=false)
    n = H_i.nqubits
    val_eta  = getcoeff(eta, H_i)  * 2^n
    val_rho  = getcoeff(rho_theta, H_i) * 2^n

    if spin_scaling
        # compute Pauli weight (number of non-identity symbols)
        w = pauliweight(H_i)
        scale = w == 1 ? 2 : w == 2 ? 4 : 1
        return (val_eta - val_rho) / scale
    else
        return val_eta - val_rho
    end
end

function computegradients(params::HamiltonianParameters, eta::PauliSum, rho::PauliSum)
    n = length(params.wi_xyz[1])
    grad_wi = ntuple(_ -> zeros(n), 3)
    grad_wij = ntuple(_ -> zeros(n, n), 9)

    H = makehamiltonian(params)
    for h in H
        val = computepartialderivative(h, eta, rho)
        pauli_str = PauliPropagation.inttostring(h.term, h.nqubits)
        indices = findall(c -> c != 'I', pauli_str)
        w = length(indices)

        if w == 1
            i = indices[1]
            sym = pauli_str[i]
            k = sym == 'X' ? 1 : sym == 'Y' ? 2 : 3
            grad_wi[k][i] = val

        elseif w == 2
            i, j = indices
            s1, s2 = pauli_str[i], pauli_str[j]
            ai = s1 == 'X' ? 1 : s1 == 'Y' ? 2 : 3
            aj = s2 == 'X' ? 1 : s2 == 'Y' ? 2 : 3
            pidx = 3*(ai-1) + aj  # map to 1:9
            grad_wij[pidx][i, j] = val
            grad_wij[pidx][j, i] = val
        else
            @warn "Unexpected Pauli term with weight=$w : $pauli_str"
        end
    end

    return HamiltonianParameters(grad_wi, grad_wij)
end

function updateparameters!(params::HamiltonianParameters, grads::HamiltonianParameters, γ::Float64)
    for k in 1:3
        params.wi_xyz[k]  .-= γ .* grads.wi_xyz[k]
    end
    for k in 1:9
        params.wij_xyz[k] .-= γ .* grads.wij_xyz[k]
    end
end