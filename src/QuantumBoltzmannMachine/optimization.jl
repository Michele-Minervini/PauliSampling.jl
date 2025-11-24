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

function computepartialderivative(H_i::PauliString, eta::AbstractMatrix, rho_theta::PauliSum; spin_scaling::Bool=false)
    n = H_i.nqubits
    
    # 1. Compute Expectation Value <P_i>_η = Tr(η * P_i)

    """
    tr_eta_pauli(eta::AbstractMatrix, P::PauliString)

    Computes Tr(eta * P) without allocating the matrix for P.
    """
    function tr_eta_pauli(eta::AbstractMatrix, P::PauliString)
        # P corresponds to a permutation and a phase factor for each index
        # But generally, P maps basis |k> to phase_k |k'>
        # Tr(eta * P) = sum_k <k| eta P |k>
        
        dim = size(eta, 1)
        n = P.nqubits
        total = 0.0 + 0.0im
        
        # Iterate over diagonal elements if P is diagonal (Z only)
        # OR iterate over relevant elements if P is off-diagonal
        # This logic can be complex to implement from scratch without a library.
        
        # FASTEST ALTERNATIVE using your existing dense matrix approach:
        # Use the dot product property: Tr(A' * B) = dot(A, B)
        # But since P is sparse, we shouldn't build it.
        
        # Fallback to the corrected matrix approach if you want to keep code simple:
        P_mat = paulistringtomatrix([PauliString(P.nqubits, P.term, 1.0)])
        return real(tr(eta * P_mat))
    end

    val_eta = tr_eta_pauli(eta, H_i)

    # 2. Get Expectation Value <P_i>_ρ
    # getcoeff usually returns the projection, we multiply by 2^n to get the trace overlap
    val_rho  = getcoeff(rho_theta, H_i) * 2^n

    if spin_scaling
        w = pauliweight(H_i)
        scale = w == 1 ? 2 : w == 2 ? 4 : 1
        return (val_eta - val_rho) / scale
    else
        return val_eta - val_rho
    end
end


"""
    computegradients(H::Vector{PauliString}, eta::PauliSum, rho::PauliSum)

Compute gradients ∂L/∂cᵢ for each coefficient cᵢ of the Hamiltonian terms in `H`.
Returns a vector of Float64 with same length as H.
"""
function computegradients(H::Vector{<:PauliString}, eta::PauliSum, rho::PauliSum)
    grads = Vector{Float64}(undef, length(H))
    @inbounds for (i, h) in enumerate(H)
        grads[i] = computepartialderivative(h, eta, rho)
    end
    return grads
end

function computegradients(H::Vector{<:PauliString}, eta::AbstractMatrix, rho::PauliSum)
    grads = Vector{Float64}(undef, length(H))
    @inbounds for (i, h) in enumerate(H)
        grads[i] = computepartialderivative(h, eta, rho)
    end
    return grads
end

"""
    updatehamiltonian!(H::Vector{PauliString}, grads::Vector{Float64}, γ::Real)

In-place update of Hamiltonian coefficients:  cᵢ ← cᵢ - γ * grads[i].
"""
function updatehamiltonian!(H::Vector{<:PauliString}, grads::Vector{Float64}, γ::Real)
    @assert length(H) == length(grads)
    @inbounds for i in eachindex(H)
        H[i] = PauliString(H[i].nqubits, H[i].term, H[i].coeff - γ * grads[i])
    end
    return H
end