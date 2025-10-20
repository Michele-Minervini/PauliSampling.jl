function maketfim(nq::Integer;
              jz=1.0, bx=1.0,
              coeffs::Union{Nothing, AbstractVector}=nothing,
              periodic::Bool=false, alltoall::Bool=false,
              random::Bool=false, dist=Uniform(-1,1))

    """
    Construct a transverse-field Ising-model Hamiltonian as a vector of PauliStrings.
    The Hamiltonian follows the spin-½ convention:
    H = (jz/4) * Σ ZᵢZⱼ  -  (bx/2) * Σ Xᵢ

    Keyword arguments:
    - `jz`: coefficient for all two-local ZZ terms
    - `bx`: coefficient for all one-local X terms
    - `coeffs`: if not nothing, explicit coefficients for each term
    - `periodic`: whether to include periodic boundary (ZₙZ₁)
    - `alltoall`: if true, use every pair (i<j) instead of only neighbors  
    - `random`: if true, sample coefficients for each term independently from `dist`
    - `dist`: distribution for random coefficients (default `Uniform(-1,1)`)
    """
    hamiltonian = PauliString[]

    # --- 1-local transverse-field terms ---
    for i in 1:nq
        coeff = random ? rand(dist) : (-bx / 2)
        push!(hamiltonian, PauliString(nq, :X, i, coeff))
    end

    # --- 2-local ZZ interactions ---
    if alltoall
        for i in 1:(nq-1), j in (i+1):nq
            coeff = random ? rand(dist) : (jz / 4)
            push!(hamiltonian, PauliString(nq, [:Z, :Z], [i, j], coeff))
        end
    else
        for i in 1:(nq-1)
            coeff = random ? rand(dist) : (jz / 4)
            push!(hamiltonian, PauliString(nq, [:Z, :Z], [i, i+1], coeff))
        end
        if periodic && nq > 2
            coeff = random ? rand(dist) : (jz / 4)
            push!(hamiltonian, PauliString(nq, [:Z, :Z], [nq, 1], coeff))
        end
    end

    ## Apply custom coefficients if provided 
    if coeffs !== nothing
        n_terms = length(hamiltonian)
        if length(coeffs) != n_terms
            error("Length of coeffs ($(length(coeffs))) must equal number of terms ($n_terms).")
        end
        for (i, c) in enumerate(coeffs)
            hamiltonian[i] = PauliString(hamiltonian[i].nqubits,
                                         hamiltonian[i].term,
                                         c)
        end
    end

    return hamiltonian
end