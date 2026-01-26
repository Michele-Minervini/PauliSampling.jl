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

function computepartialderivative(H_i::PauliString, state_matrix::AbstractMatrix, rho_theta::PauliSum; spin_scaling::Bool=false)
    n = H_i.nqubits
    dim = size(state_matrix, 1)
    
    val_eta = 0.0

    # ---------------------------------------------------------
    # BRANCH 1: BdG Covariance Matrix (Dimension 2 * N)
    # ---------------------------------------------------------
    if dim == 2 * n
        # It is a Covariance Matrix -> Use Wick's Theorem logic
        
        # Convert Pauli to String (e.g., "ZIIZ") for the BdG solver
        p_str = PauliPropagation.inttostring(H_i.term, n)
        val_eta = bdg_expectation(state_matrix, p_str)

    # ---------------------------------------------------------
    # BRANCH 2: Exact Density Matrix (Dimension 2^N)
    # ---------------------------------------------------------
    elseif dim == 2^n
        # It is a Full Density Matrix -> Use Matrix Trace
        
        # Helper inner function for trace
        function tr_eta_pauli(eta::AbstractMatrix, P::PauliString)
             # Fallback to sparse matrix construction if needed
             # Or use your existing optimized trace logic
             P_mat = paulistringtomatrix([PauliString(P.nqubits, P.term, 1.0)])
             return real(tr(eta * P_mat))
        end

        val_eta = tr_eta_pauli(state_matrix, H_i)

    # ---------------------------------------------------------
    # ERROR HANDLING
    # ---------------------------------------------------------
    else
        error("Matrix dimension $dim does not match N=$n. Expected size 2N=$(2*n) (BdG) or 2^N=$(2^n) (Exact).")
    end

    # ---------------------------------------------------------
    # COMMON LOGIC: Calculate Rho part and Gradient
    # ---------------------------------------------------------
    val_rho = getcoeff(rho_theta, H_i) * 2^n

    if spin_scaling
        w = pauliweight(H_i)
        scale = w == 1 ? 2.0 : (w == 2 ? 4.0 : 1.0)
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

"""
Computes the gradient vector for a list of Pauli strings H.
Automatic Detection:
- If `state_matrix` has dimension 2*N, it uses efficient BdG/Wick's theorem (O(N^3)).
- If `state_matrix` has dimension 2^N, it uses Exact Matrix Trace (Exponential cost).
"""
function computegradients(H::Vector{<:PauliString}, state_matrix::AbstractMatrix, rho::PauliSum)
    # Pre-allocate the gradient vector
    grads = Vector{Float64}(undef, length(H))
    
    # Loop over all Hamiltonian terms
    # Using @inbounds for slight performance gain since indices are safe
    @inbounds for (i, h) in enumerate(H)
        # The logic for BdG vs Exact is handled inside this function call:
        grads[i] = computepartialderivative(h, state_matrix, rho)
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