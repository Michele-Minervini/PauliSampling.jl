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


"""
    make_spin_hamiltonian(nq; wi_xyz=(0,0,0), wij_xyz=(0,0,0);
                          periodic=false, alltoall=false)

Build a generic spin-½ Hamiltonian

    H = sum_i sum_{k∈{x,y,z}} (wi^k / 2) * σ_i^k
      + sum_{i<j} sum_{k∈{x,y,z}} (wij^k / 4) * σ_i^k σ_j^k

Arguments
---------
- `nq::Integer`: number of qubits/spins.

Keyword arguments
-----------------
- `wi_xyz`: tuple `(wi_x, wi_y, wi_z)`. Each entry can be a scalar (applied to all sites)
           or a vector of length `nq` (per-site coefficients).
- `wij_xyz`: tuple `(wij_x, wij_y, wij_z)`. Each entry can be a scalar (applied to all pairs)
            or a vector of length equal to the number of pairs (see connectivity).
- `periodic::Bool=false`: if `true` and `alltoall=false`, include the (n,1) neighbour.
- `alltoall::Bool=false`: if `true`, use all pairs i<j; otherwise nearest neighbours only.

Vector ordering for pairwise coefficients
-----------------------------------------
If a vector is supplied for a given axis in `wij_xyz`, entries are assigned in this order:
- `alltoall=true`: pairs in lexicographic order `(i=1,j=2..n), (i=2,j=3..n), ...`.
- `alltoall=false`: nearest neighbours `(1,2), (2,3), ..., (n-1,n)`, and if `periodic=true`
  also `(n,1)` appended at the end.

Returns
-------
A `Vector{PauliString}` with spin-½ normalization:
- single-site terms carry coefficient `wi^k / 2`,
- two-site terms carry coefficient `wij^k / 4`.

Notes
-----
- Pass zeros for axes you don't want (e.g., TFIM: `wi_xyz=(bx,0,0)`, `wij_xyz=(0,0,jz)`).
- Signs are up to you. For TFIM with the common convention `-(bx/2) Σ X + (jz/4) Σ ZZ`,
  pass `wi_xyz=(-bx, 0, 0)` and `wij_xyz=(0, 0, jz)`.

"""
function makehamiltonian(nq::Integer;
                               wi_xyz::Tuple=(-1,0,0),
                               wij_xyz::Tuple=(0,0,1),
                               periodic::Bool=false,
                               alltoall::Bool=false)

    # --- helpers ---
    _tofloat(x) = x isa Number ? float(x) : x
    function _expand_coeff(c, count, label)
        c = _tofloat(c)
        if c isa Number
            return fill(c, count)
        elseif c isa AbstractVector
            length(c) == count || error("$label: expected length $count, got $(length(c))")
            return collect(float.(c))
        else
            error("$label must be a Number or a Vector")
        end
    end

    axes_syms = (:X, :Y, :Z)

    # --- build pair list per connectivity ---
    pairs = Tuple{Int,Int}[]
    if alltoall
        for i in 1:(nq-1), j in (i+1):nq
            push!(pairs, (i, j))
        end
    else
        for i in 1:(nq-1)
            push!(pairs, (i, i+1))
        end
        if periodic && nq > 2
            push!(pairs, (nq, 1))
        end
    end
    npairs = length(pairs)

    # --- expand coefficients (per-axis) ---
    wi_x, wi_y, wi_z   = wi_xyz
    wij_x, wij_y, wij_z = wij_xyz

    wi_vecs = (
        _expand_coeff(wi_x,  nq,    "wi_x"),
        _expand_coeff(wi_y,  nq,    "wi_y"),
        _expand_coeff(wi_z,  nq,    "wi_z")
    )
    wij_vecs = (
        _expand_coeff(wij_x, npairs, "wij_x"),
        _expand_coeff(wij_y, npairs, "wij_y"),
        _expand_coeff(wij_z, npairs, "wij_z")
    )

    # --- assemble Hamiltonian as Pauli strings ---
    H = PauliString[]

    # single-site: (wi^k / 2) * σ_i^k
    for (k, sym) in enumerate(axes_syms)
        coeffs_k = wi_vecs[k]
        for i in 1:nq
            c = coeffs_k[i]
            if c != 0
                push!(H, PauliString(nq, sym, i, c/2))
            end
        end
    end

    # two-site: (wij^k / 4) * σ_i^k σ_j^k
    for (k, sym) in enumerate(axes_syms)
        coeffs_k = wij_vecs[k]
        for (idx, (i, j)) in enumerate(pairs)
            c = coeffs_k[idx]
            if c != 0
                push!(H, PauliString(nq, [sym, sym], [i, j], c/4))
            end
        end
    end

    return H
end
