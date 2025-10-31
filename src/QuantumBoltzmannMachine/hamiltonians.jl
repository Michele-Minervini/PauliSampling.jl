import Base: *, /  # to extend these operators
struct HamiltonianParams
    wi_xyz::NTuple{3, Vector{Float64}}   # e.g. [Bx_i], [By_i], [Bz_i] per site
    wij_xyz::NTuple{3, Matrix{Float64}}  # e.g. [Jx_ij], [Jy_ij], [Jz_ij] per pair
end


function HamiltonianParams(n::Int;
        wi_xyz=(0.0, 0.0, 0.0),
        wij_xyz=(1.0, 1.0, 1.0))
    
    wi = ntuple(k -> fill(wi_xyz[k], n), 3)
    wij = ntuple(k -> fill(wij_xyz[k], n, n), 3)
    return HamiltonianParams(wi, wij)
end

# scalar * HamiltonianParams
*(β::Real, p::HamiltonianParams) = HamiltonianParams(
    (β .* p.wi_xyz[1], β .* p.wi_xyz[2], β .* p.wi_xyz[3]),
    (β .* p.wij_xyz[1], β .* p.wij_xyz[2], β .* p.wij_xyz[3])
)

# HamiltonianParams * scalar (optional, for symmetry)
*(p::HamiltonianParams, β::Real) = β * p

# division by scalar
/(p::HamiltonianParams, β::Real) = (1/β) * p

"""
    makehamiltonian(params::HamiltonianParams; connectivity=:nearest, periodic=false)

Build a generic spin-½ Hamiltonian:
H = Σ_i Σ_k (wi_xyz[k][i]/2) σᵢᵏ  +  Σ_{i<j} Σ_k (wij_xyz[k][i,j]/4) σᵢᵏ σⱼᵏ

Keyword arguments:
- `connectivity`: :nearest (default), :nextnearest, or :alltoall
- `periodic`: whether to include periodic boundary conditions
"""
function makehamiltonian(params::HamiltonianParams; 
                         connectivity::Symbol = :nearest, 
                         periodic::Bool = false)

    n = length(params.wi_xyz[1])
    H = PauliString[]

    # ---- 1. Add single-site (field) terms ----
    for i in 1:n, (k, sym) in enumerate((:X, :Y, :Z))
        coeff = params.wi_xyz[k][i] / 2
        if coeff != 0
            push!(H, PauliString(n, sym, i, coeff))
        end
    end

    # ---- 2. Define neighbor list depending on connectivity type ----
    pairs = Tuple{Int,Int}[]
    if connectivity == :nearest
        for i in 1:n-1
            push!(pairs, (i, i+1))
        end
        if periodic
            push!(pairs, (n, 1))
        end
    elseif connectivity == :nextnearest
        for i in 1:n-2
            push!(pairs, (i, i+2))
        end
        if periodic
            push!(pairs, (n-1, 1))
            push!(pairs, (n, 2))
        end
    elseif connectivity == :alltoall
        for i in 1:n-1, j in i+1:n
            push!(pairs, (i, j))
        end
    else
        error("Unknown connectivity type: $connectivity")
    end

    # ---- 3. Build the two-body terms ----
    for (i, j) in pairs, (k, sym) in enumerate((:X, :Y, :Z))
        coeff = params.wij_xyz[k][i, j] / 4
        if coeff != 0
            qinds = sort([i, j])  # ensure consistent internal order
            push!(H, PauliString(n, [sym, sym], qinds, coeff))
        end
    end

    return H
end