###############################################################
#               GENERAL HAMILTONIAN GENERATOR                 #
#    Supports 1-local and full 9-combination 2-local terms    #
#    Supports 1D Chain and 2D Square Lattice Geometries      #
###############################################################

using SparseArrays, LinearAlgebra, Random

# Each site can have 3 field components (X,Y,Z)
# Each pair can have 9 coupling components (XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ)
struct HamiltonianParameters
    wi_xyz::NTuple{3, Vector{Float64}}        # per-site fields 
    wij_xyz::NTuple{9, Matrix{Float64}}       # per-pair couplings
end

# --- Geometry Abstraction ---
abstract type Lattice end

struct ChainLattice <: Lattice
    n::Int
end

struct SquareLattice <: Lattice
    rows::Int
    cols::Int
    n::Int
    SquareLattice(r, c) = new(r, c, r * c)
end

# Low-level constructor
function HamiltonianParameters(n::Int;
        wi_xyz::NTuple{3, Real} = (0.0, 0.0, 0.0),
        wij_xyz::NTuple{9, Real} = ntuple(_ -> 0.0, 9))

    wi  = ntuple(k -> fill(Float64(wi_xyz[k]), n), 3)
    wij = ntuple(k -> fill(Float64(wij_xyz[k]), n, n), 9)
    return HamiltonianParameters(wi, wij)
end

# ------------------------------------------------------------------
# 1. MODEL CATALOGUE 
# ------------------------------------------------------------------
const MODEL_AXES = Dict{Symbol, NamedTuple}(
    :ising                   => (p1 = (:Z,),              p2 = (:ZZ,)),
    :tfim                    => (p1 = (:X,),              p2 = (:ZZ,)),
    :tfim_tilted             => (p1 = (:X, :Z),          p2 = (:ZZ,)),
    :xx                      => (p1 = (),                p2 = (:XX,)),
    :xy                      => (p1 = (),                p2 = (:XX, :YY)),
    :xyz                     => (p1 = (),                p2 = (:XX, :YY, :ZZ)),
    :heisenberg              => (p1 = (),                p2 = (:XX, :YY, :ZZ)),
    :heisenberg_fields       => (p1 = (:X, :Y, :Z),      p2 = (:XX, :YY, :ZZ)),
    :heisenberg_fields_real  => (p1 = (:X, :Z),          p2 = (:XX, :YY, :ZZ)),
    :generic                 => (p1 = (:X, :Y, :Z),      p2 = (:XX, :XY, :XZ, :YX, :YY, :YZ, :ZX, :ZY, :ZZ)),
    :generic_real            => (p1 = (:X, :Z),          p2 = (:XX, :XZ, :YY, :ZX, :ZZ))
)

@inline function axis_index(ax::Symbol)
    ax === :X && return 1
    ax === :Y && return 2
    ax === :Z && return 3
    error("Unknown axis $ax")
end

@inline function pair_index(term::Symbol)
    s = String(term)
    length(s) == 2 || error("Invalid 2-local term symbol $term")
    i = axis_index(Symbol(s[1]))
    j = axis_index(Symbol(s[2]))
    return 3*(i-1) + j 
end

# ------------------------------------------------------------------
# 2. MODEL-AWARE INITIALIZATION
# ------------------------------------------------------------------

"""
    parameters(model::Symbol, n::Int; ...)
Restored high-level constructor that populates fields/couplings based on model spec.
"""
function parameters(model::Symbol, n::Int;
        init::Symbol = :zeros,
        field_scale::Real = 0.1,
        coupling_scale::Real = 0.1,
        field_vals::Union{Nothing, Dict{Symbol,<:Real}} = nothing,
        coupling_vals::Union{Nothing, Dict{Symbol,<:Real}} = nothing,
        rng = Random.default_rng())

    haskey(MODEL_AXES, model) || error("Unknown model: $model")
    spec = MODEL_AXES[model]

    wi  = ntuple(_ -> zeros(n), 3)
    wij = ntuple(_ -> zeros(n, n), 9)

    # ---- 1-local terms ----
    for ax in spec.p1
        k = axis_index(ax)
        val = field_vals !== nothing && haskey(field_vals, ax) ? field_vals[ax] : nothing
        if val !== nothing
            wi[k] .= val
        elseif init === :randn
            wi[k] .= field_scale .* randn(rng, n)
        end
    end

    # ---- 2-local terms ----
    for term in spec.p2
        pidx = pair_index(term)
        val = coupling_vals !== nothing && haskey(coupling_vals, term) ? coupling_vals[term] : nothing
        if val !== nothing
            wij[pidx] .= val
        elseif init === :randn
            wij[pidx] .= coupling_scale .* randn(rng, n, n)
        end
    end

    return HamiltonianParameters(wi, wij)
end

# ------------------------------------------------------------------
# 3. NEIGHBORHOOD LOGIC (Multiple Dispatch)
# ------------------------------------------------------------------

# 1D Chain Neighbors 
function get_pairs(lat::ChainLattice, connectivity::Symbol, periodic::Bool)
    pairs = Tuple{Int,Int}[]
    n = lat.n
    if connectivity == :nearest
        for i in 1:n-1; push!(pairs, (i, i+1)); end
        periodic && push!(pairs, (n, 1))
    elseif connectivity == :nextnearest
        for i in 1:n-2; push!(pairs, (i, i+2)); end
        if periodic
            push!(pairs, (n-1, 1)); push!(pairs, (n, 2))
        end
    elseif connectivity == :alltoall
        for i in 1:n-1, j in i+1:n; push!(pairs, (i, j)); end
    else
        error("Unknown connectivity: $connectivity")
    end
    return pairs
end

# 2D Square Lattice Neighbors 
function get_pairs(lat::SquareLattice, connectivity::Symbol, periodic::Bool)
    pairs = Tuple{Int,Int}[]
    r, c = lat.rows, lat.cols
    idx(i, j) = (i - 1) * c + j # Helper for 2D -> 1D index mapping

    if connectivity == :nearest
        for i in 1:r, j in 1:c
            # Horizontal neighbor
            if j < c
                push!(pairs, (idx(i, j), idx(i, j+1)))
            elseif periodic
                push!(pairs, (idx(i, j), idx(i, 1)))
            end
            # Vertical neighbor
            if i < r
                push!(pairs, (idx(i, j), idx(i+1, j)))
            elseif periodic
                push!(pairs, (idx(i, j), idx(1, j)))
            end
        end
    elseif connectivity == :alltoall
        for i in 1:lat.n-1, j in i+1:lat.n; push!(pairs, (i, j)); end
    else
        error("2D lattice currently only supports :nearest or :alltoall")
    end
    return pairs
end

# ------------------------------------------------------------------
# 3. BUILDER FUNCTIONS
# ------------------------------------------------------------------

function makehamiltonian(params::HamiltonianParameters, lat::Lattice;
        connectivity::Symbol = :nearest,
        periodic::Bool = false,
        model::Union{Nothing,Symbol} = nothing)

    n = lat.n
    H = PauliString[]
    spec = model !== nothing ? MODEL_AXES[model] : nothing

    # 1-local field terms 
    for i in 1:n, (k, sym) in enumerate((:X, :Y, :Z))
        if spec !== nothing && !(sym in spec.p1); continue; end
        coeff = params.wi_xyz[k][i] / 2
        if coeff != 0.0; push!(H, PauliString(n, sym, i, coeff)); end
    end

    # 2-local interaction terms 
    pairs = get_pairs(lat, connectivity, periodic)
    AXES = (:X, :Y, :Z)
    for (i, j) in pairs
        for ai in 1:3, aj in 1:3
            sym1, sym2 = AXES[ai], AXES[aj]
            term = Symbol(string(sym1, sym2))
            if spec !== nothing && !(term in spec.p2); continue; end
            pidx = 3*(ai-1) + aj
            coeff = params.wij_xyz[pidx][i, j] / 4
            if coeff != 0.0; push!(H, PauliString(n, [sym1, sym2], [i, j], coeff)); end
        end
    end
    return H
end

function makehamiltonian_matrix(params::HamiltonianParameters,   
        lat::Lattice;
        connectivity::Symbol = :nearest,
        periodic::Bool = false,
        model::Union{Nothing,Symbol} = nothing)

    n = lat.n
    dim = 2^n
    H = spzeros(ComplexF64, dim, dim) 
    spec = model !== nothing ? MODEL_AXES[model] : nothing

    # 1-local fields 
    for i in 1:n, k in 1:3
        sym = (:X, :Y, :Z)[k]
        if spec !== nothing && !(sym in spec.p1); continue; end
        coeff = params.wi_xyz[k][i] / 2
        if coeff != 0.0
            op = embed_op_sparse(n, Dict(i => PAULIS_SP[k]))
            H += coeff * op
        end
    end

    # 2-local interactions
    pairs = get_pairs(lat, connectivity, periodic)
    for (i, j) in pairs, ai in 1:3, aj in 1:3
        term_sym = Symbol(string((:X, :Y, :Z)[ai], (:X, :Y, :Z)[aj]))
        if spec !== nothing && !(term_sym in spec.p2); continue; end
        pidx = 3*(ai-1) + aj
        coeff = params.wij_xyz[pidx][i, j] / 4
        if coeff != 0.0
            op = embed_op_sparse(n, Dict(i => PAULIS_SP[ai], j => PAULIS_SP[aj]))
            H += coeff * op
        end
    end
    return H
end

# ------------------------------------------------------------------
# 4. UTILS & SPARSE OPS
# ------------------------------------------------------------------
const I2_sp = sparse([1.0 0.0; 0.0 1.0] .+ 0.0im)
const σx_sp = sparse([0.0 1.0; 1.0 0.0] .+ 0.0im)
const σy_sp = sparse([0.0 -1.0im; 1.0im 0.0])
const σz_sp = sparse([1.0 0.0; 0.0 -1.0] .+ 0.0im)
const PAULIS_SP = (σx_sp, σy_sp, σz_sp)

function embed_op_sparse(n::Int, ops::Dict{Int, <:AbstractSparseMatrix})
    mats = Vector{SparseMatrixCSC{ComplexF64, Int}}(undef, n)
    for i in 1:n; mats[i] = get(ops, i, I2_sp); end
    return reduce(kron, mats)
end

# ------------------------------------------------------------------
# 5. EXAMPLE: 2D HEISENBERG MODEL (4x3 LATTICE)
# ------------------------------------------------------------------

# # 1. Define the 2D Geometry (4x3 = 12 spins)
# lat_2d = SquareLattice(4, 3)

# # 2. Initialize parameters for the Heisenberg model (XX + YY + ZZ)
# # We set all interaction strengths to 1.0
# h_params = parameters(:heisenberg, lat_2d.n; 
#                      init=:zeros, 
#                      coupling_vals=Dict(:XX => 1.0, :YY => 1.0, :ZZ => 1.0))

# # 3. Build the Hamiltonian as a list of PauliStrings
# H_pauli = makehamiltonian(h_params, lat_2d; 
#                          connectivity=:nearest, 
#                          periodic=false, 
#                          model=:heisenberg)

# # 4. Build the Hamiltonian as a Sparse Matrix
# H_matrix = makehamiltonian_matrix(h_params, lat_2d; 
#                                  connectivity=:nearest, 
#                                  periodic=false, 
#                                  model=:heisenberg)

# println("2D Heisenberg (4x3) initialized.")
# println("Number of Pauli terms: ", length(H_pauli))