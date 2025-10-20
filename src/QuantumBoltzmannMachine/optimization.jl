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
function computepartialderivative(H_i::PauliString, eta::PauliSum, rho_theta::PauliSum; spin_scaling::Bool=true)
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

# computegradient[i] = ⟨h_i⟩_η - ⟨h_i⟩_ρ
function computegradient(h::Vector{<:PauliString}, eta::PauliSum, rho::PauliSum)
    [computepartialderivative(hi, eta, rho) for hi in h]
end

# gradient descent!
gradientdescentstep(params::Vector{<:Real}, grads::Vector{<:Real}, gamma::Real) = params .- gamma .* grads 