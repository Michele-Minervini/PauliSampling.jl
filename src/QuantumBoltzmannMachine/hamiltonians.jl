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
# 3b. GENERALIZED CONNECTIVITY  (tunable interaction RANGE + ORDER)
#     Lets us go beyond physical models: pick a neighbor distance `d`
#     and a max interaction order `K`, and place k-body terms on every
#     group of qubits that are mutually within distance `d`. This is the
#     complexity (#coefficients) <-> expressivity knob for model search.
# ------------------------------------------------------------------

"""
    lattice_distance_matrix(lat::Lattice; periodic=false) -> Matrix{Int}

All-pairs graph (hop) distance on the nearest-neighbor lattice, via BFS.
For a `SquareLattice` this equals the Manhattan distance |Δrow|+|Δcol|.
`D[i,j]` is the number of nearest-neighbor hops between qubits i and j.
(No external graph library needed — BFS on the lattice's own NN edges.)
"""
function lattice_distance_matrix(lat::Lattice; periodic::Bool=false)
    n = lat.n
    adj = [Int[] for _ in 1:n]
    for (i, j) in get_pairs(lat, :nearest, periodic)
        push!(adj[i], j); push!(adj[j], i)
    end
    D = fill(typemax(Int), n, n)
    for s in 1:n
        D[s, s] = 0
        queue = [s]; head = 1
        while head <= length(queue)
            u = queue[head]; head += 1
            for v in adj[u]
                if D[s, v] == typemax(Int)
                    D[s, v] = D[s, u] + 1
                    push!(queue, v)
                end
            end
        end
    end
    return D
end

"""
    interaction_cliques(D, n; max_distance=1, max_order=2) -> Vector{Vector{Int}}

Every qubit group (sorted, size 2..max_order) that forms a *clique* in the graph
"connect i,j iff D[i,j] ≤ max_distance" — i.e. every member is within
`max_distance` hops of every other member. With `max_distance=1, max_order=2`
these are exactly the nearest-neighbor edges. NOTE: a square lattice's NN graph is
triangle-free, so 3+ body groups require `max_distance ≥ 2`.
"""
function interaction_cliques(D::AbstractMatrix{<:Integer}, n::Int;
        max_distance::Int = 1, max_order::Int = 2)
    @assert max_order >= 2 "max_order must be ≥ 2 (got $max_order)"
    within(i, j) = i != j && D[i, j] <= max_distance
    groups = Vector{Vector{Int}}()
    # order 2 (edges within distance)
    cur = [[i, j] for i in 1:n for j in (i+1):n if within(i, j)]
    append!(groups, cur)
    # orders 3..max_order: extend each clique by a higher-index vertex adjacent to ALL members
    for _ in 3:max_order
        nxt = Vector{Int}[]
        for cl in cur, v in (cl[end] + 1):n
            if all(within(u, v) for u in cl)
                push!(nxt, vcat(cl, v))
            end
        end
        isempty(nxt) && break
        append!(groups, nxt)
        cur = nxt
    end
    return groups
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

"""
    makehamiltonian_generalized(lat; max_distance=1, max_order=2, field_axes=(:Z,),
        interaction_axis=:Z, init=:randn, field_scale=0.1, coupling_scale=0.1,
        order_decay=1.0, periodic=false, rng=Random.default_rng()) -> Vector{PauliString}

Build a (possibly NON-physical) Hamiltonian with a tunable connectivity, generalizing
the physical models along two independent axes:

  * RANGE  — `max_distance` d: terms live on groups of qubits that are mutually within
             graph-distance ≤ d (d=1 = nearest neighbor, d = lattice diameter = all-to-all).
  * ORDER  — `max_order` K: include 2-body, 3-body, …, up to K-body terms, each an
             `interaction_axis`^k string (e.g. Z⊗Z⊗Z), on every such group.

`field_axes` (e.g. `(:Z,)` or `(:X,:Z)`) adds 1-local fields on every site. Coefficients
are random (`init=:randn`); `order_decay` optionally damps higher orders
(`scale = coupling_scale * order_decay^(k-2)`). With `max_distance=1, max_order=2,
field_axes=(:X,:Z), interaction_axis=:Z` this reproduces the *structure* of `:tfim_tilted`.

This is the model-complexity (#coefficients) vs expressivity knob for non-physical model search.
Returns a `Vector{PauliString}` ready for `prepare_thermal_state` / training.
"""
function makehamiltonian_generalized(lat::Lattice;
        max_distance::Int = 1, max_order::Int = 2,
        field_axes = (:Z,), interaction_axis::Symbol = :Z,
        init::Symbol = :randn, field_scale::Real = 0.1, coupling_scale::Real = 0.1,
        order_decay::Real = 1.0, periodic::Bool = false,
        rng = Random.default_rng())
    n = lat.n
    H = PauliString[]

    # --- 1-local fields on every site ---
    for i in 1:n, ax in field_axes
        coeff = init === :randn ? field_scale * randn(rng) : 0.0
        coeff != 0.0 && push!(H, PauliString(n, ax, i, coeff))
    end

    # --- k-body interaction terms on distance-cliques (k = 2..max_order) ---
    D = lattice_distance_matrix(lat; periodic = periodic)
    groups = interaction_cliques(D, n; max_distance = max_distance, max_order = max_order)
    for g in groups
        k = length(g)
        scale = coupling_scale * order_decay^(k - 2)
        coeff = init === :randn ? scale * randn(rng) : 0.0
        coeff != 0.0 && push!(H, PauliString(n, fill(interaction_axis, k), collect(g), coeff))
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



# ------------------------------------------------------------------
# 5. RESTRICTED BOLTZMANN MACHINE (RBM)
#    Bipartite Graph: n visible nodes, m hidden nodes
# ------------------------------------------------------------------

struct RBMParameters
    n_visible::Int
    n_hidden::Int
    # Fields for visible units (size n_visible)
    visible_fields::NTuple{3, Vector{Float64}}
    # Fields for hidden units (size n_hidden)
    hidden_fields::NTuple{3, Vector{Float64}}
    # Couplings between visible and hidden (matrix size n_visible x n_hidden)
    # index 1=>XX, 2=>XY ... 9=>ZZ
    couplings::NTuple{9, Matrix{Float64}} 
    # Couplings between visible and visible (matrix size n_visible x n_visible)
    vis_vis_couplings::NTuple{9, Matrix{Float64}}
    # Couplings between hidden and hidden (matrix size n_hidden x n_hidden)
    hid_hid_couplings::NTuple{9, Matrix{Float64}}
end

"""
    RBMParameters(n, m; ...)

Low-level constructor for BM parameters.
"""
function RBMParameters(n::Int, m::Int;
        visible_xyz::NTuple{3, Real} = (0.0, 0.0, 0.0),
        hidden_xyz::NTuple{3, Real}  = (0.0, 0.0, 0.0),
        wij_xyz::NTuple{9, Real}     = ntuple(_ -> 0.0, 9),
        w_vv_xyz::NTuple{9, Real}    = ntuple(_ -> 0.0, 9),
        w_hh_xyz::NTuple{9, Real}    = ntuple(_ -> 0.0, 9))

    vis = ntuple(k -> fill(Float64(visible_xyz[k]), n), 3)
    hid = ntuple(k -> fill(Float64(hidden_xyz[k]), m), 3)
    wij = ntuple(k -> fill(Float64(wij_xyz[k]), n, m), 9)
    
    # Square matrices for intra-layer connections
    w_vv = ntuple(k -> fill(Float64(w_vv_xyz[k]), n, n), 9)
    w_hh = ntuple(k -> fill(Float64(w_hh_xyz[k]), m, m), 9)
    
    return RBMParameters(n, m, vis, hid, wij, w_vv, w_hh)
end

# --- RBM Model Catalogue ---
# keys: vis (visible 1-local), hid (hidden 1-local), int (interaction 2-local),
#       vis_vis (intra-visible 2-local), hid_hid (intra-hidden 2-local)
const RBM_MODEL_AXES = Dict{Symbol, NamedTuple}(
    # Standard RBM: Only Z fields and ZZ interactions between layers (Strictly Bipartite)
    :ising => (vis = (:Z,), hid = (:Z,), int = (:ZZ,), vis_vis = (), hid_hid = ()),
    
    # NEW: General Boltzmann Machine (ZZ everywhere)
    :gbm_ising => (vis = (:Z,), hid = (:Z,), int = (:ZZ,), vis_vis = (:ZZ,), hid_hid = (:ZZ,)),
    
    # Example of generalization: TFIM RBM (Transverse Field on Visible)
    :tfim_visible => (vis = (:X,), hid = (:Z,), int = (:ZZ,), vis_vis = (), hid_hid = ()),
    
    # Fully general (for future use)
    :generic => (vis = (:X,:Y,:Z), hid = (:X,:Y,:Z), 
                 int = (:XX,:XY,:XZ,:YX,:YY,:YZ,:ZX,:ZY,:ZZ),
                 vis_vis = (:XX,:XY,:XZ,:YX,:YY,:YZ,:ZX,:ZY,:ZZ),
                 hid_hid = (:XX,:XY,:XZ,:YX,:YY,:YZ,:ZX,:ZY,:ZZ))
)

"""
    rbm_parameters(model::Symbol, n::Int, m::Int; ...)

High-level constructor for RBMs and GBMs. 
"""
function rbm_parameters(model::Symbol, n::Int, m::Int;
        init::Symbol = :zeros,
        field_scale::Real = 0.1,
        coupling_scale::Real = 0.1,
        visible_vals::Union{Nothing, Dict{Symbol,<:Real}} = nothing,
        hidden_vals::Union{Nothing, Dict{Symbol,<:Real}} = nothing,
        coupling_vals::Union{Nothing, Dict{Symbol,<:Real}} = nothing,
        vis_vis_vals::Union{Nothing, Dict{Symbol,<:Real}} = nothing,
        hid_hid_vals::Union{Nothing, Dict{Symbol,<:Real}} = nothing,
        rng = Random.default_rng())

    haskey(RBM_MODEL_AXES, model) || error("Unknown RBM model: $model")
    spec = RBM_MODEL_AXES[model]

    vis = ntuple(_ -> zeros(n), 3)
    hid = ntuple(_ -> zeros(m), 3)
    wij = ntuple(_ -> zeros(n, m), 9)
    w_vv = ntuple(_ -> zeros(n, n), 9)
    w_hh = ntuple(_ -> zeros(m, m), 9)

    # ---- Visible Fields ----
    for ax in spec.vis
        k = axis_index(ax)
        val = visible_vals !== nothing && haskey(visible_vals, ax) ? visible_vals[ax] : nothing
        if val !== nothing
            vis[k] .= val
        elseif init === :randn
            vis[k] .= field_scale .* randn(rng, n)
        end
    end

    # ---- Hidden Fields ----
    for ax in spec.hid
        k = axis_index(ax)
        val = hidden_vals !== nothing && haskey(hidden_vals, ax) ? hidden_vals[ax] : nothing
        if val !== nothing
            hid[k] .= val
        elseif init === :randn
            hid[k] .= field_scale .* randn(rng, m)
        end
    end

    # ---- Interactions (Visible-Hidden) ----
    for term in spec.int
        pidx = pair_index(term)
        val = coupling_vals !== nothing && haskey(coupling_vals, term) ? coupling_vals[term] : nothing
        if val !== nothing
            wij[pidx] .= val
        elseif init === :randn
            wij[pidx] .= coupling_scale .* randn(rng, n, m)
        end
    end

    # ---- Interactions (Visible-Visible) ----
    for term in get(spec, :vis_vis, ())
        pidx = pair_index(term)
        val = vis_vis_vals !== nothing && haskey(vis_vis_vals, term) ? vis_vis_vals[term] : nothing
        if val !== nothing
            w_vv[pidx] .= val
        elseif init === :randn
            # Only fill the upper triangle to avoid double counting undirected edges
            for i in 1:n, j in (i+1):n
                w_vv[pidx][i, j] = coupling_scale * randn(rng)
            end
        end
    end

    # ---- Interactions (Hidden-Hidden) ----
    for term in get(spec, :hid_hid, ())
        pidx = pair_index(term)
        val = hid_hid_vals !== nothing && haskey(hid_hid_vals, term) ? hid_hid_vals[term] : nothing
        if val !== nothing
            w_hh[pidx] .= val
        elseif init === :randn
            # Only fill the upper triangle
            for i in 1:m, j in (i+1):m
                w_hh[pidx][i, j] = coupling_scale * randn(rng)
            end
        end
    end

    return RBMParameters(n, m, vis, hid, wij, w_vv, w_hh)
end

"""
    makehamiltonian(params::RBMParameters)

Build the RBM Hamiltonian as a vector of PauliStrings.
"""
function makehamiltonian(params::RBMParameters; model::Union{Nothing,Symbol}=nothing)
    n = params.n_visible
    m = params.n_hidden
    total_qubits = n + m
    H = PauliString[]

    spec = model !== nothing ? RBM_MODEL_AXES[model] : nothing
    AXES = (:X, :Y, :Z)

    # ---- Visible Fields (Indices 1..n) ----
    for i in 1:n, (k, sym) in enumerate(AXES)
        if spec !== nothing && !(sym in spec.vis); continue; end
        coeff = params.visible_fields[k][i] / 2
        if coeff != 0.0
            push!(H, PauliString(total_qubits, sym, i, coeff))
        end
    end

    # ---- Hidden Fields (Indices n+1..n+m) ----
    for j in 1:m, (k, sym) in enumerate(AXES)
        if spec !== nothing && !(sym in spec.hid); continue; end
        coeff = params.hidden_fields[k][j] / 2
        if coeff != 0.0
            push!(H, PauliString(total_qubits, sym, n + j, coeff))
        end
    end

    # ---- Interactions (Visible i -- Hidden n+j) ----
    for i in 1:n, j in 1:m
        for ai in 1:3, aj in 1:3
            sym1, sym2 = AXES[ai], AXES[aj]
            term = Symbol(string(sym1, sym2))
            
            if spec !== nothing && !(term in spec.int); continue; end
            
            pidx = 3*(ai-1) + aj
            coeff = params.couplings[pidx][i, j] / 4
            
            if coeff != 0.0
                push!(H, PauliString(total_qubits, [sym1, sym2], [i, n + j], coeff))
            end
        end
    end

    # ---- NEW: Interactions (Visible i -- Visible j) ----
    for i in 1:n, j in (i+1):n # i < j to avoid double counting
        for ai in 1:3, aj in 1:3
            sym1, sym2 = AXES[ai], AXES[aj]
            term = Symbol(string(sym1, sym2))
            
            if spec !== nothing && haskey(spec, :vis_vis) && !(term in spec.vis_vis); continue; end
            
            pidx = 3*(ai-1) + aj
            coeff = params.vis_vis_couplings[pidx][i, j] / 4
            
            if coeff != 0.0
                push!(H, PauliString(total_qubits, [sym1, sym2], [i, j], coeff))
            end
        end
    end

    # ---- NEW: Interactions (Hidden n+i -- Hidden n+j) ----
    for i in 1:m, j in (i+1):m # i < j to avoid double counting
        for ai in 1:3, aj in 1:3
            sym1, sym2 = AXES[ai], AXES[aj]
            term = Symbol(string(sym1, sym2))
            
            if spec !== nothing && haskey(spec, :hid_hid) && !(term in spec.hid_hid); continue; end
            
            pidx = 3*(ai-1) + aj
            coeff = params.hid_hid_couplings[pidx][i, j] / 4
            
            if coeff != 0.0
                push!(H, PauliString(total_qubits, [sym1, sym2], [n + i, n + j], coeff))
            end
        end
    end

    return H
end

"""
    makehamiltonian_matrix(params::RBMParameters)

Build the RBM Hamiltonian as a SparseMatrixCSC.
"""
function makehamiltonian_matrix(params::RBMParameters; model::Union{Nothing,Symbol}=nothing)
    n = params.n_visible
    m = params.n_hidden
    total_qubits = n + m
    dim = 2^total_qubits
    
    H = spzeros(ComplexF64, dim, dim)
    spec = model !== nothing ? RBM_MODEL_AXES[model] : nothing

    # ---- Visible Fields (Indices 1..n) ----
    for i in 1:n, k in 1:3
        sym = (:X, :Y, :Z)[k]
        if spec !== nothing && !(sym in spec.vis); continue; end
        
        coeff = params.visible_fields[k][i] / 2
        if coeff != 0.0
            op = embed_op_sparse(total_qubits, Dict(i => PAULIS_SP[k]))
            H += coeff * op
        end
    end

    # ---- Hidden Fields (Indices n+1..n+m) ----
    for j in 1:m, k in 1:3
        sym = (:X, :Y, :Z)[k]
        if spec !== nothing && !(sym in spec.hid); continue; end
        
        coeff = params.hidden_fields[k][j] / 2
        if coeff != 0.0
            op = embed_op_sparse(total_qubits, Dict(n + j => PAULIS_SP[k]))
            H += coeff * op
        end
    end

    # ---- Interactions (Visible i -- Hidden n+j) ----
    for i in 1:n, j in 1:m, ai in 1:3, aj in 1:3
        term_sym = Symbol(string((:X, :Y, :Z)[ai], (:X, :Y, :Z)[aj]))
        if spec !== nothing && !(term_sym in spec.int); continue; end

        pidx = 3*(ai-1) + aj
        coeff = params.couplings[pidx][i, j] / 4
        
        if coeff != 0.0
            op = embed_op_sparse(total_qubits, Dict(i => PAULIS_SP[ai], n + j => PAULIS_SP[aj]))
            H += coeff * op
        end
    end

    # ---- NEW: Interactions (Visible i -- Visible j) ----
    for i in 1:n, j in (i+1):n, ai in 1:3, aj in 1:3
        term_sym = Symbol(string((:X, :Y, :Z)[ai], (:X, :Y, :Z)[aj]))
        if spec !== nothing && haskey(spec, :vis_vis) && !(term_sym in spec.vis_vis); continue; end

        pidx = 3*(ai-1) + aj
        coeff = params.vis_vis_couplings[pidx][i, j] / 4
        
        if coeff != 0.0
            op = embed_op_sparse(total_qubits, Dict(i => PAULIS_SP[ai], j => PAULIS_SP[aj]))
            H += coeff * op
        end
    end

    # ---- NEW: Interactions (Hidden n+i -- Hidden n+j) ----
    for i in 1:m, j in (i+1):m, ai in 1:3, aj in 1:3
        term_sym = Symbol(string((:X, :Y, :Z)[ai], (:X, :Y, :Z)[aj]))
        if spec !== nothing && haskey(spec, :hid_hid) && !(term_sym in spec.hid_hid); continue; end

        pidx = 3*(ai-1) + aj
        coeff = params.hid_hid_couplings[pidx][i, j] / 4
        
        if coeff != 0.0
            op = embed_op_sparse(total_qubits, Dict(n + i => PAULIS_SP[ai], n + j => PAULIS_SP[aj]))
            H += coeff * op
        end
    end

    return H
end