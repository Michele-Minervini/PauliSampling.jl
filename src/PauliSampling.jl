module PauliSampling

using Base.Threads

using PauliPropagation
using F2Algebra
using Hadamard
using Combinatorics
using BenchmarkTools
using Random
using LinearAlgebra
using LinearMaps
using Arpack
using Plots
using ProgressMeter
using Distributions
using Optimisers
using ReverseDiff
using MAT
using IterTools
using StatsBase
using LaTeXStrings
using Measures

# Extend Base methods for PauliFreqTracker
import Base: real, imag, abs, complex, convert, float

real(p::PauliFreqTracker) = real(p.coeff)
imag(p::PauliFreqTracker) = imag(p.coeff)
abs(p::PauliFreqTracker) = abs(p.coeff)
float(p::PauliFreqTracker) = float(p.coeff)
complex(p::PauliFreqTracker) = complex(p.coeff)
convert(::Type{ComplexF64}, p::PauliFreqTracker) = convert(ComplexF64, p.coeff)

include("./Sampling/Sampling.jl")
export 
    get_dist,
    get_bit,
    paulis_to_matrix,
    hamiltonian_to_circuit,
    approximate_prob,
    projection_prob,
    sample_bitstring,
    zero_state,
    build_circuit,
    tvd,
    kl_div,
    avg_metrics_for_weight,
    bit_marginals,
    shannon_entropy

include("QuantumBoltzmannMachine/QuantumBoltzmannMachine.jl")
export
    makethermalstate,
    maketfim,
    parameters,
    HamiltonianParameters,
    makehamiltonian,
    computequantumrelativeentropy,
    computesymmetriceigenvalues,
    paulistringtocircuit,
    paulistringtomatrix,
    computegradients,
    updatehamiltonian!

end
