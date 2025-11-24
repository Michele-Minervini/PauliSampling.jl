###############################################################
#               GENERAL HAMILTONIAN GENERATOR                 #
#    Supports 1-local and full 9-combination 2-local terms    #
#          Fully compatible with gradients optimizer          #
###############################################################

# Each site can have 3 field components (X,Y,Z)
# Each pair can have 9 coupling components (XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ)
struct HamiltonianParameters
    wi_xyz::NTuple{3, Vector{Float64}}        # per-site fields
    wij_xyz::NTuple{9, Matrix{Float64}}       # per-pair couplings
end


"""
    HamiltonianParameters(n; wi_xyz=(0,0,0), wij_xyz=fill(0,9))

Low-level constructor: creates dense per-site and per-pair arrays.
"""
function HamiltonianParameters(n::Int;
        wi_xyz::NTuple{3, Real} = (0.0, 0.0, 0.0),
        wij_xyz::NTuple{9, Real} = ntuple(_ -> 0.0, 9))

    wi  = ntuple(k -> fill(Float64(wi_xyz[k]), n), 3)
    wij = ntuple(k -> fill(Float64(wij_xyz[k]), n, n), 9)
    return HamiltonianParameters(wi, wij)
end


# ------------------------------------------------------------------
# 1. MODEL CATALOGUE (Allowed local and pairwise terms per model)
# ------------------------------------------------------------------

const MODEL_AXES = Dict{Symbol, NamedTuple}(
    # Ising family
    :ising                   => (p1 = (:Z,),              p2 = (:ZZ,)),
    :tfim                    => (p1 = (:X,),              p2 = (:ZZ,)),
    :tfim_tilted             => (p1 = (:X, :Z),          p2 = (:ZZ,)),

    # 2-local lattice models
    :xx                      => (p1 = (),                p2 = (:XX,)),
    :xy                      => (p1 = (),                p2 = (:XX, :YY)),
    :xyz                     => (p1 = (),                p2 = (:XX, :YY, :ZZ)),

    # Heisenberg family
    :heisenberg              => (p1 = (),                p2 = (:XX, :YY, :ZZ)),
    :heisenberg_fields       => (p1 = (:X, :Y, :Z),      p2 = (:XX, :YY, :ZZ)),
    :heisenberg_fields_real  => (p1 = (:X, :Z),          p2 = (:XX, :YY, :ZZ)),

    # Fully general
    :generic                 => (p1 = (:X, :Y, :Z),      p2 = (:XX, :XY, :XZ, :YX, :YY, :YZ, :ZX, :ZY, :ZZ)),
    :generic_real            => (p1 = (:X, :Z),          p2 = (:XX, :XZ, :YY, :ZX, :ZZ))
)


# helper: single-axis symbol → index
@inline function axis_index(ax::Symbol)
    ax === :X && return 1
    ax === :Y && return 2
    ax === :Z && return 3
    error("Unknown axis $ax")
end

# helper: pair axis symbol (e.g. :XY) → index in 1:9
@inline function pair_index(term::Symbol)
    s = String(term)
    length(s) == 2 || error("Invalid 2-local term symbol $term")
    ax1, ax2 = Symbol(s[1]), Symbol(s[2])
    i = axis_index(ax1)
    j = axis_index(ax2)
    return 3*(i-1) + j   # maps (1,1)->1, (1,2)->2, ..., (3,3)->9
end


# ------------------------------------------------------------------
# 2. MODEL-AWARE INITIALIZATION
# ------------------------------------------------------------------

"""
    parameters(model::Symbol, n::Int;
                          init=:zeros, field_scale=0.1, coupling_scale=0.1,
                          rng=Random.default_rng())

Create a `HamiltonianParameters` for a specific model.

Each allowed axis (from MODEL_AXES[model]) is populated;
everything else is left zero.

Arguments
---------
- `model`: one of keys(MODEL_AXES)
- `n`: number of qubits/spins
- `init`: one of `:zeros`, `:randn`, or `:custom`
- `field_scale`, `coupling_scale`: scaling for random init
- `field_vals`: Dict of specific field coefficients, e.g. `Dict(:X => -0.8, :Z => 0.1)`
- `coupling_vals`: Dict of specific couplings, e.g. `Dict(:ZZ => 0.3, :XX => 0.2)`
- `rng`: random number generator

If `field_vals` or `coupling_vals` is provided, those values override
the default initialization for the corresponding axes.
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
        elseif init === :zeros
            # already zero
        else
            error("Unknown init mode $init")
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
        elseif init === :zeros
            # already zero
        else
            error("Unknown init mode $init")
        end
    end

    return HamiltonianParameters(wi, wij)
end


# ------------------------------------------------------------------
# 3. HAMILTONIAN BUILDER
# ------------------------------------------------------------------

"""
    makehamiltonian(params::HamiltonianParameters;
                    connectivity=:nearest, periodic=false, model=nothing)

Build the full Hamiltonian as a vector of `PauliString`s.
Supports all 9 cross-axis 2-local terms.
"""
function makehamiltonian(params::HamiltonianParameters;
        connectivity::Symbol = :nearest,
        periodic::Bool = false,
        model::Union{Nothing,Symbol} = nothing)

    n = length(params.wi_xyz[1])
    H = PauliString[]

    allowed_p1 = nothing
    allowed_p2 = nothing
    if model !== nothing
        haskey(MODEL_AXES, model) || error("Unknown model $model")
        allowed_p1 = MODEL_AXES[model].p1
        allowed_p2 = MODEL_AXES[model].p2
    end

    # ---- 1-local field terms ----
    for i in 1:n, (k, sym) in enumerate((:X, :Y, :Z))
        if allowed_p1 !== nothing && !(sym in allowed_p1)
            continue
        end
        coeff = params.wi_xyz[k][i] / 2
        if coeff != 0.0
            push!(H, PauliString(n, sym, i, coeff))
        end
    end

    # ---- define neighbor list ----
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

    # ---- 2-local interaction terms ----
    AXES = (:X, :Y, :Z)
    for (i, j) in pairs
        for ai in 1:3, aj in 1:3
            sym1, sym2 = AXES[ai], AXES[aj]
            term = Symbol(string(sym1, sym2))
            if allowed_p2 !== nothing && !(term in allowed_p2)
                continue
            end
            pidx = 3*(ai-1) + aj
            coeff = params.wij_xyz[pidx][i, j] / 4
            if coeff != 0.0
                push!(H, PauliString(n, [sym1, sym2], [i, j], coeff))
            end
        end
    end

    return H
end


# ------------------------------------------------------------------
# 4. MATRIX REPRESENTATIONS (SPARSE)
# ------------------------------------------------------------------
# Functions to convert model parameters into explicit Sparse Matrices 
# Distinct from the PauliString approach

# Pre-allocate Sparse Pauli Matrices
# We add 0.0im to X and Z to ensure they are all the same ComplexF64 type.
const I2_sp = sparse([1.0 0.0; 0.0 1.0] .+ 0.0im)
const σx_sp = sparse([0.0 1.0; 1.0 0.0] .+ 0.0im)
const σy_sp = sparse([0.0 -1.0im; 1.0im 0.0])
const σz_sp = sparse([1.0 0.0; 0.0 -1.0] .+ 0.0im)

# Tuple for indexing (1=>X, 2=>Y, 3=>Z)
const PAULIS_SP = (σx_sp, σy_sp, σz_sp)

"""
    embed_op_sparse(n::Int, ops::Dict{Int, SparseMatrixCSC})

Constructs the sparse tensor product operator for an n-qubit system.
"""
function embed_op_sparse(n::Int, ops::Dict{Int, <:AbstractSparseMatrix})
    # We build the list of matrices for the Kronecker product
    mats = Vector{SparseMatrixCSC{ComplexF64, Int}}(undef, n)
    for i in 1:n
        mats[i] = get(ops, i, I2_sp)
    end
    # reduce(kron, ...) is highly optimized for SparseArrays in Julia
    return reduce(kron, mats)
end

"""
    makehamiltonian_matrix(params::HamiltonianParameters;
                            connectivity=:nearest, periodic=false, model=nothing)

Build the full Hamiltonian as a SparseMatrixCSC.
"""
function makehamiltonian_matrix(params::HamiltonianParameters;
        connectivity::Symbol = :nearest,
        periodic::Bool = false,
        model::Union{Nothing,Symbol} = nothing)

    n = length(params.wi_xyz[1])
    dim = 2^n
    
    # Initialize an empty sparse matrix of size 2^n x 2^n
    # spzeros does not allocate memory for elements, so this is cheap.
    H = spzeros(ComplexF64, dim, dim)

    # ---- Filter Allowed Terms based on Model ----
    allowed_p1 = nothing
    allowed_p2 = nothing
    if model !== nothing
        haskey(MODEL_AXES, model) || error("Unknown model $model")
        allowed_p1 = MODEL_AXES[model].p1
        allowed_p2 = MODEL_AXES[model].p2
    end

    # ---- 1-local field terms ----
    for i in 1:n, k in 1:3
        sym = (:X, :Y, :Z)[k]
        
        if allowed_p1 !== nothing && !(sym in allowed_p1)
            continue
        end

        coeff = params.wi_xyz[k][i] / 2
        if coeff != 0.0
            # Create sparse op
            op = embed_op_sparse(n, Dict(i => PAULIS_SP[k]))
            H = H + coeff * op  # Sparse addition
        end
    end

    # ---- Define Neighbor List ----
    pairs = Tuple{Int,Int}[]
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
        error("Unknown connectivity type: $connectivity")
    end

    # ---- 2-local interaction terms ----
    AXES_SYMBOLS = (:X, :Y, :Z)
    
    for (i, j) in pairs
        for ai in 1:3, aj in 1:3
            sym1, sym2 = AXES_SYMBOLS[ai], AXES_SYMBOLS[aj]
            term_sym = Symbol(string(sym1, sym2))

            if allowed_p2 !== nothing && !(term_sym in allowed_p2)
                continue
            end

            pidx = 3*(ai-1) + aj
            coeff = params.wij_xyz[pidx][i, j] / 4
            
            if coeff != 0.0
                # Create sparse term σ_i ⊗ σ_j
                op = embed_op_sparse(n, Dict(i => PAULIS_SP[ai], j => PAULIS_SP[aj]))
                H = H + coeff * op
            end
        end
    end

    return H
end