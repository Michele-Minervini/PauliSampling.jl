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

# Symmetric eigenvalues 
computesymmetriceigenvalues(M::AbstractMatrix) = eigen(Hermitian(Matrix(M))).values





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