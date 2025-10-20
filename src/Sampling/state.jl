function zero_state(nq, ::Type{T}) where T
    return prod(
        qind -> PauliSum(nq, Dict(
            PauliString(nq, :I, qind).term => Complex{T}(1//2),
            PauliString(nq, :Z, qind).term => Complex{T}(1//2)
        )),
        1:nq
    )
end

function zero_state(nq::Int)
    return zero_state(nq, Float64)
end

function random_psum(nq::Integer)
    labels = [:I, :X, :Y, :Z]
    total = 4^nq
    pstrs = Vector{PauliString}(undef, total)

    for idx in 0:total-1
        paulis = Vector{Symbol}(undef, nq)
        tmp = idx
        for i in 1:nq
            paulis[i] = labels[mod1(tmp, 4)]
            tmp ÷= 4
        end
        coeff = randn() + randn()*im
        pstrs[idx + 1] = PauliString(nq, paulis, 1:nq, coeff)
    end
    return PauliSum(pstrs)
end

function build_circuit(nq; nl=6, topology_type=:bricklayer, circuit_type=:hardwareefficient, periodic=false)
    # Select topology
    topology = 
        topology_type == :bricklayer  ? bricklayertopology(nq; periodic=periodic) :
        topology_type == :rectangle   ? rectangletopology(nq, nq; periodic=periodic) :
        topology_type == :staircase   ? staircasetopology(nq; periodic=periodic) :
        error("Unknown topology type: $topology_type")

    # Select circuit ansatz
    circuit =
        circuit_type == :hardwareefficient   ? hardwareefficientcircuit(nq, nl; topology=topology) :
        circuit_type == :tfitrotter          ? tfitrottercircuit(nq, nl; topology=topology, start_with_ZZ=true) :
        circuit_type == :tiltedtfitrotter    ? tiltedtfitrottercircuit(nq, nl; topology=topology) :
        circuit_type == :heisenbergtrotter   ? heisenbergtrottercircuit(nq, nl; topology=topology) :
        error("Unknown circuit type: $circuit_type")

    return circuit
end