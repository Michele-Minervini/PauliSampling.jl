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
computesymmetriceigenvalues(M::AbstractMatrix) = eigen(Hermitian(M)).values






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


# Interpret a Pauli word into a matrix 
function paulitomatrix(p::AbstractString)
    I = [1 0; 0 1]
    X = [0 1; 1 0]
    Y = [0 -1im; 1im 0]
    Z = [1 0; 0 -1]
    mats = Dict('I' => I, 'X' => X, 'Y' => Y, 'Z' => Z)
    mat = mats[p[1]]
    for i in 2:length(p)
        mat = kron(mat, mats[p[i]])
    end
    return mat
end

# Convert a vector of PauliString terms into its full Hamiltonian matrix
## Vector-of-PauliStrings as input
function paulistringtomatrix(pstring::Vector{<:PauliString})
    nq = pstring[1].nqubits
    dim = 2^nq
    Hmat = zeros(ComplexF64, dim, dim)
    for term in pstring
        coeff = term.coeff
        ps = PauliPropagation.inttostring(term.term, term.nqubits)
        Hmat .+= coeff .* paulitomatrix(ps)
    end
    return Hmat
end

## PauliSum as input
function paulistringtomatrix(psum::PauliSum)
    nq = psum.nqubits
    dim = 2^nq
    Hmat = zeros(ComplexF64, dim, dim)
    for (term, coeff) in psum.terms
        ps = PauliPropagation.inttostring(term, nq)
        Hmat .+= coeff .* paulitomatrix(ps)
    end
    return Hmat
end
