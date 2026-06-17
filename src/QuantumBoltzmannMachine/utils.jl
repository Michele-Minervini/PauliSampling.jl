# Compute expectation Tr(ρ H)
computeexpectationvalue(rho::AbstractMatrix, H::AbstractMatrix) = real(tr(rho * H))

# function to compute the quantum relative entropy between the target density and the QBM
function computequantumrelativeentropy(eta::AbstractMatrix, evals_eta::AbstractVector, h_qbm::AbstractMatrix)
    # --- Step 1. Regularize small negative eigenvalues ---
    # Replace tiny negatives by eps() and renormalize so that ∑p = 1
    safe_evals = max.(evals_eta, eps(Float64))
    safe_evals ./= sum(safe_evals)   # keep probabilities normalized

    # --- Step 2. Compute -S(η) = ∑ p log p ---
    h = sum(safe_evals .* log.(safe_evals))

    # --- Step 3. Model partition function ---
    evalsH = computesymmetriceigenvalues(h_qbm)
    Z = sum(exp.(-evalsH))

    # --- Step 4. Mean energy of η under H_qbm ---
    eta_stat = computeexpectationvalue(eta, h_qbm)

    # --- Step 5. Return quantum relative entropy ---
    return h + eta_stat + log(Z)
end

"""
Computes Quantum Relative Entropy efficiently.
Inputs:
    - Gamma_eta: 2N x 2N Covariance Matrix of the target state
    - H_pauli: Vector of PauliStrings representing the model Hamiltonian (beta * H)
"""
function computequantumrelativeentropy(Gamma_eta::AbstractMatrix, H_pauli::Vector{<:PauliString})
    
    n_sites = size(Gamma_eta, 1) ÷ 2

    # ====================================================
    # STEP 1: Entropy of Target State (-S(eta))
    # ====================================================
    # Helper to compute entropy from Covariance Matrix
    function compute_entropy_from_gamma(Gamma::AbstractMatrix)
        n_sites = size(Gamma, 1) ÷ 2
        evals = eigen(Gamma).values
        
        # Due to p vs (1-p) symmetry, we can just sum the entropy of ALL 2N eigenvalues
        # and divide by 2. This avoids sorting/filtering issues entirely.
        total_entropy = 0.0
        for p in evals
            p_safe = clamp(real(p), eps(Float64), 1.0 - eps(Float64))
            total_entropy += -p_safe * log(p_safe) - (1.0 - p_safe) * log(1.0 - p_safe)
        end
        
        return total_entropy / 2.0
    end
    # We use the helper function defined previously
    S_eta = compute_entropy_from_gamma(Gamma_eta)
    term1 = -S_eta 

    # ====================================================
    # STEP 2: Log Partition Function (log Z)
    # ====================================================
    # 1. Build the BdG matrix for the current model H
    H_bdg_model = build_bdg_from_pauli(H_pauli)
    
    # 2. Diagonalize to get single-particle energies E_k
    # We take the positive half of the eigenvalues
    energies_model = eigen(H_bdg_model).values[n_sites+1:end]
    
    # 3. Calculate log Z
    # Note: 'beta' is already absorbed into H_pauli coefficients, so we use beta=1.0 here.
    log_Z = sum(log.(2.0 .* cosh.(energies_model ./ 2.0)))

    # ====================================================
    # STEP 3: Expectation Value <H_model>_eta
    # ====================================================
    # We calculate Tr(eta * H_model) using Wick's theorem (bdg_expectation)
    eta_stat = 0.0
    
    for ps in H_pauli
        # Convert to string for the expectation function
        p_str = PauliPropagation.inttostring(ps.term, n_sites)
        
        # Calculate <P_k> using the covariance matrix
        val = bdg_expectation(Gamma_eta, p_str)
        
        # Add c_k * <P_k>
        eta_stat += ps.coeff * val
    end

    # ====================================================
    # STEP 4: Combine
    # ====================================================
    # Relative Entropy = -S(eta) + <H>_eta + log Z
    return term1 + eta_stat + log_Z
end

# Symmetric eigenvalues 
computesymmetriceigenvalues(M::AbstractMatrix) = eigen(Hermitian(Matrix(M))).values

"""
Constructs the BdG Hamiltonian matrix directly from a vector of PauliStrings.
Logic strictly matches the trusted 'jordan_wigner_tfim' function:
- Field Diagonal: 2 * coeff (recovers 'h')
- Hopping/Pairing: -1 * coeff (recovers '-J/4')
"""
function build_bdg_from_pauli(H_pauli::Vector{<:PauliString})
    n_sites = H_pauli[1].nqubits

    A = zeros(Float64, n_sites, n_sites)
    B = zeros(Float64, n_sites, n_sites)

    for ps in H_pauli
        coeff = ps.coeff
        
        # Parse the string to identify the term type
        p_str = PauliPropagation.inttostring(ps.term, n_sites)
        
        inds_X = findall(c -> c == 'X', p_str)
        inds_Z = findall(c -> c == 'Z', p_str)

        # --- FIELD TERM (X_i) ---
        if length(inds_X) == 1 && isempty(inds_Z)
            i = inds_X[1]
            
            # Trusted: A[i,i] = h.  Pauli: coeff = h/2.
            # Map: A[i,i] = 2 * coeff
            A[i, i] += 2.0 * coeff

        # --- INTERACTION TERM (Z_i Z_j) ---
        elseif length(inds_Z) == 2 && isempty(inds_X)
            i, j = inds_Z[1], inds_Z[2]
            
            # Trusted: A[i,j] = -J/4.  Pauli: coeff = J/4.
            # Map: Val = -coeff
            val = -1.0 * coeff
            
            # Symmetric Hopping (A)
            A[i, j] += val
            A[j, i] += val
            
            # Antisymmetric Pairing (B)
            # Matches 'jordan_wigner_tfim': B[i, j] = -J/4 (val), B[j, i] = +J/4 (-val)
            if i < j
                B[i, j] += val
                B[j, i] -= val
            else
                B[i, j] -= val
                B[j, i] += val
            end
        end
    end

    # Construct the full 2N x 2N matrix
    return [A B; -B -A]
end



function getpaulisymbols(pstr)
    nq = pstr.nqubits
    symbols = Vector{Symbol}()
    idxs = Vector{Integer}()
    for i in 1:nq
        pauli = getpauli(pstr.term, i)
        if pauli != 0
            push!(symbols, inttosymbol(pauli))
            push!(idxs, i)
        end
    end
    return symbols, idxs
end

function paulistringtocircuit(hamiltonian::Vector{<:PauliString})
    circuit = Gate[]
    thetas = typeof(hamiltonian[1].coeff)[]
    for pstr in hamiltonian
        symbols, idxs = getpaulisymbols(pstr)
        push!(circuit, ImaginaryPauliRotation(symbols, idxs))
        push!(thetas, pstr.coeff)
    end
    return circuit, thetas
end


# 1. Helper: Sparse Single Pauli Matrices
# We force them to be ComplexF64 so X, Y, Z all have the same type.
const I_sp = sparse([1.0+0.0im 0.0; 0.0 1.0])
const X_sp = sparse([0.0+0.0im 1.0; 1.0 0.0])
const Y_sp = sparse([0.0 -1.0im; 1.0im 0.0])
const Z_sp = sparse([1.0+0.0im 0.0; 0.0 -1.0])

const MATS_SP = Dict('I' => I_sp, 'X' => X_sp, 'Y' => Y_sp, 'Z' => Z_sp)

# 2. Builder: Interpret a Pauli word into a SPARSE matrix 
function paulitomatrix(p::AbstractString)
    # Start with the first operator
    mat = MATS_SP[p[1]]
    
    # Kronecker product preserves sparsity efficiently in Julia
    for i in 2:length(p)
        mat = kron(mat, MATS_SP[p[i]])
    end
    return mat
end

# 3. Builder: Vector of PauliStrings -> Sparse Matrix
function paulistringtomatrix(pstring::Vector{<:PauliString})
    nq = pstring[1].nqubits
    dim = 2^nq
    
    # Initialize empty sparse matrix
    Hmat = spzeros(ComplexF64, dim, dim)
    
    for term in pstring
        coeff = term.coeff
        ps = PauliPropagation.inttostring(term.term, term.nqubits)
        
        # Accumulate: Sparse addition handles structural changes automatically
        # Note: 'Hmat += ...' is slightly better than 'Hmat = Hmat + ...' in newer Julia
        Hmat = Hmat + coeff * paulitomatrix(ps)
    end
    return Hmat
end

# 4. Builder: PauliSum -> Sparse Matrix
function paulistringtomatrix(psum::PauliSum)
    nq = psum.nqubits
    dim = 2^nq
    Hmat = spzeros(ComplexF64, dim, dim)
    
    for (term, coeff) in psum.terms
        ps = PauliPropagation.inttostring(term, nq)
        Hmat = Hmat + coeff * paulitomatrix(ps)
    end
    return Hmat
end



###################################################
### Bogoliubov-de Gennes matrix for Ising model ###
###################################################

"""
Construct the Bogoliubov-de Gennes matrix for Ising model.
"""
function jordan_wigner_tfim(n_sites; J=1.0, h=1.0)
# --- SCALING CORRECTION ---
    # Exact Method does: Field / 2.  BdG needs: Field / 1.  -> Ratio: 1.0
    # Exact Method does: Coupl / 4.  BdG needs: Coupl / 2.  -> Ratio: 0.5
    
    h_internal = h * 1.0
    J_internal = J * 0.5  
    
    # We assume standard TFIM where J comes from X-coupling (Jx) and Jy=0
    # If you were doing XY model, you would scale both Jx and Jy by 0.5

    A = zeros(Float64, n_sites, n_sites)
    B = zeros(Float64, n_sites, n_sites)

    # 1. Fill Diagonal
    for i in 1:n_sites
        A[i, i] = h_internal
    end

    # 2. Fill Off-Diagonals
    for i in 1:(n_sites-1)
        # Hopping
        A[i, i+1] = -J_internal / 2
        A[i+1, i] = -J_internal / 2

        # Pairing
        B[i, i+1] = -J_internal / 2
        B[i+1, i] =  J_internal / 2
    end

    M = [A B; -B -A]
    return M
end

"""
Constructs the 2N x 2N Covariance Matrix Gamma.
Gamma[m, n] = < Psi_m * Psi_n' >
where Psi = (c_1...c_N, c_1'...c_N')
"""
function get_covariance_matrix(H_bdg, beta)
    
    # 2. Diagonalize: H = U * E * U'
    evals, U = eigen(H_bdg)
    
    # 3. Compute Occupation numbers (Fermi-Dirac)
    # For a mode with energy E, occupation is 1 / (1 + exp(beta*E))
    # We apply this to ALL 2N eigenvalues (positive and negative)
    fermi_weights = 1.0 ./ (1.0 .+ exp.(beta .* evals))
    
    # 4. Rotate back to get Correlation Matrix
    # Gamma = U * Diagonal(fermi_weights) * U'
    Gamma = U * Diagonal(fermi_weights) * U'
    
    return Gamma
end

"""
Gets statistics from 2N x 2N matrix and reconstructs the full 2^N spectrum.
"""
function get_bdg_reconstructed_eigenvalues(H_bdg, beta)

    n_sites = Int(dim(H_bdg)/2)

    # Diagonalize
    # LinearAlgebra.eigen returns an object. .values gives the list of eigenvalues.
    # They are typically sorted ascendingly by default for symmetric matrices.
    all_eigvals = eigen(H_bdg).values
    
    # Get positive eigenvalues (single particle energies)
    # Python: eigvals[n_sites:] (The second half)
    # Julia: all_eigvals[n_sites+1:end] (1-based indexing)
    energies = all_eigvals[n_sites+1:end]
    
    # Occupation probabilities (using dot . for element-wise operations)
    occupations = 1.0 ./ (1.0 .+ exp.(beta .* energies))
    
    # Reconstruct full list of 2^N probabilities using product rule
    full_probs = Float64[]
    
    # Iterators.product is the Julia equivalent to itertools.product
    # We create a product of (0,1) repeated n_sites times
    for pattern in Iterators.product(fill(0:1, n_sites)...)
        prob = 1.0
        # pattern is a tuple, e.g., (0, 1, 0, ...)
        for (k, bit) in enumerate(pattern)
            if bit == 1
                prob *= occupations[k]
            else
                prob *= (1.0 - occupations[k])
            end
        end
        push!(full_probs, prob)
    end
    
    return full_probs
end

# =========================================================
# 2. EXPECTATION VALUE CALCULATOR
# =========================================================

"""
Helper: Computes the contraction < B_u A_v >
where B_u = (c_u' - c_u) and A_v = (c_v' + c_v)
"""
function get_BA_contraction(Gamma::AbstractMatrix, u::Int, v::Int)
    N = size(Gamma, 1) ÷ 2
    
    # We want < (c_u' - c_u)(c_v' + c_v) >
    # Expand: <c_u' c_v'> + <c_u' c_v> - <c_u c_v'> - <c_u c_v>
    
    # Map to Gamma indices (Gamma[a,b] = <Psi_a Psi_b'>)
    # Psi = (c_1...c_N, c'_1...c'_N)
    
    # 1. <c_u' c_v'> -> Gamma[u+N, v]
    val = Gamma[u+N, v]
    
    # 2. <c_u' c_v>  -> Gamma[u+N, v+N]
    val += Gamma[u+N, v+N]
    
    # 3. -<c_u c_v'> -> -Gamma[u, v]
    val -= Gamma[u, v]
    
    # 4. -<c_u c_v>  -> -Gamma[u, v+N]
    val -= Gamma[u, v+N]
    
    return val
end

"""
Computes expectation values using the BdG Covariance Matrix and a Pauli String.
Input `pauli_str` should be in the basis of the 'Exact' Hamiltonian 
(where X is the Field and Z is the Interaction axis).

Supported Patterns:
1. Single 'X' (e.g., "IIXIII"): Computes Field Expectation (Magnetization).
2. Double 'Z' (e.g., "ZIIZI"): Computes Interaction Expectation (Correlation).
"""
function bdg_expectation(Gamma::AbstractMatrix, pauli_str::String)
    N = size(Gamma, 1) ÷ 2
    
    # 1. Parse the string to find active sites
    # 'collect' converts string to char array for easy indexing
    chars = collect(pauli_str)
    
    if length(chars) != N
        error("String length ($(length(chars))) does not match system size ($N)")
    end
    
    # Find indices of X and Z operators
    inds_X = findall(c -> c == 'X', chars)
    inds_Z = findall(c -> c == 'Z', chars)
    
    # Check that we don't have mixed types or unsupported operators (like Y)
    has_Y = any(c -> c == 'Y', chars)
    if has_Y
        error("Operator 'Y' is not currently supported in this BdG mapping.")
    end
    
    # --- CASE 1: FIELD TERM (Single X) ---
    # Corresponds to <Z> in the BdG basis
    if length(inds_X) == 1 && isempty(inds_Z)
        i = inds_X[1]
        
        # Calculation: 1 - 2<c'_i c_i>
        n_occ = real(Gamma[i+N, i+N])
        return 1.0 - 2.0 * n_occ

    # --- CASE 2: INTERACTION TERM (Two Zs) ---
    # Corresponds to <XX> in the BdG basis
    elseif length(inds_Z) == 2 && isempty(inds_X)
        i, j = inds_Z[1], inds_Z[2]
        
        # Ensure i < j for the determinant formula
        if i > j
            i, j = j, i
        end
        
        # Distance
        r = j - i
        
        # If r=0 (same site), <Z_i Z_i> = <I> = 1.0
        if r == 0
            return 1.0
        end

        # Construct the Toeplitz-like matrix T of size r x r
        # The expectation value <X_i ... X_j> is the determinant of this matrix.
        T = zeros(ComplexF64, r, r)
        
        for row in 1:r
            for col in 1:r
                u_site = i + row - 1
                v_site = i + col
                T[row, col] = get_BA_contraction(Gamma, u_site, v_site)
            end
        end
        
        return real(det(T))

    else
        # Fallback for empty strings (Identity) or unsupported combinations
        if isempty(inds_X) && isempty(inds_Z)
            return 1.0 # Identity
        else
            error("Unsupported operator pattern: $pauli_str. Only single 'X' or double 'Z' supported.")
        end
    end
end


# ------------------------------------------------------------------
# RBM / PARTIAL TRACE UTILITIES
# ------------------------------------------------------------------

"""
    marginalize(psum::PauliSum, n_visible::Int)

Performs the partial trace over the hidden qubits (indices n_visible+1 to n_total).
Returns a new PauliSum defined only on the visible qubits (1 to n_visible).
"""
function marginalize(psum::PauliSum, n_visible::Int)
    n_total = psum.nqubits
    n_hidden = n_total - n_visible
    
    # We will accumulate terms in a dictionary to merge duplicates automatically
    # (e.g. Z_1 I_2 and Z_1 Z_2 might both map to Z_1 if we trace out qubit 2)
    # However, for partial trace, we ONLY keep terms that are Identity on hidden units.
    # So we don't need to merge; we just filter.
    
    visible_pstrs = Vector{PauliString}()
    # The trace of I on m qubits is 2^m.
    # In Pauli representation, Tr_B(P_A x I_B) = P_A * 2^m.
    # We must scale coefficients by this factor to maintain normalization.
    trace_factor = 2.0^n_hidden
    
    # Helper to reconstruct symbols from codes
    # Assuming standard mapping: 1=>X, 2=>Y, 3=>Z, else I
    function code_to_symbol(p_code)
        if ispauli(p_code, 1); return :X
        elseif ispauli(p_code, 2); return :Y
        elseif ispauli(p_code, 3); return :Z
        else; return :I
        end
    end

    for (pstr, coeff) in psum
        # 1. Check if Hidden Part is Identity
        is_hidden_identity = true
        for k in n_visible+1:n_total
            p_code = getpauli(pstr, k)
            # If not Identity (assuming ispauli checks 1,2,3 for X,Y,Z)
            if ispauli(p_code, 1) || ispauli(p_code, 2) || ispauli(p_code, 3)
                is_hidden_identity = false
                break
            end
        end

        # 2. If valid, extract visible part and rescale
        if is_hidden_identity
            # Construct new PauliString for visible qubits
            syms = Vector{Symbol}(undef, n_visible)
            for k in 1:n_visible
                p_code = getpauli(pstr, k)
                syms[k] = code_to_symbol(p_code)
            end
            
            new_coeff = coeff * trace_factor
            push!(visible_pstrs, PauliString(n_visible, syms, 1:n_visible, new_coeff))
        end
    end

    return PauliSum(visible_pstrs)
end