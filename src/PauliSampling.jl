module PauliSampling

using Base.Threads

using PauliPropagation
using F2Algebra
using Hadamard
using Combinatorics
using BenchmarkTools
using Random
using LinearAlgebra
using SparseArrays
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
using Statistics      # mean (dataset binarization/downsampling)
using Printf          # @printf in train_qbm verbose output
using MLDatasets      # reduced-MNIST loader (Training)
using ForwardDiff     # exact threaded AD gradient (Training)

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
    sample_bitstrings,
    zero_state,
    build_circuit,
    tvd,
    kl_div,
    avg_metrics_for_weight,
    bit_marginals,
    get_exact_prob,
    get_approx_prob,
    get_rbm_prob,
    shannon_entropy

include("QuantumBoltzmannMachine/QuantumBoltzmannMachine.jl")
export
    makethermalstate,
    makethermalstate_matrix,
    maketfim,
    parameters,
    HamiltonianParameters,
    Lattice,        
    ChainLattice,   
    SquareLattice,
    makehamiltonian,
    makehamiltonian_matrix,
    makehamiltonian_generalized,
    lattice_distance_matrix,
    interaction_cliques,
    rbm_parameters,
    RBMParameters,
    computequantumrelativeentropy,
    build_bdg_from_pauli,
    computesymmetriceigenvalues,
    paulistringtocircuit,
    paulistringtomatrix,
    computegradients,
    updatehamiltonian!,
    jordan_wigner_tfim,
    get_covariance_matrix,
    get_bdg_reconstructed_eigenvalues,
    bdg_expectation,
    marginalize

# ── Training: QBM generative-modeling pipeline (MNIST) ──────────────
include("Training/Training.jl")
export
    # data
    generate_mnist_dataset, extract_support,
    downsample_image, binarize_image, image_to_bitvector, bitvector_to_image,
    # model templates + thermal state
    build_h_template, build_h_general, prepare_thermal_state, h_from_flat,
    # loss
    model_prob, kl_support,
    # optimizer + gradient estimators (interchangeable: (loss_fn, theta) -> (grad, loss))
    AdamState, adam_step!,
    spsa_gradient, spsa_gradient_serial, spsa_gradient_threads, spsa_gradient_spawn,
    forward_fd_gradient_spawn, ad_gradient,
    # training entry point
    train_qbm

end
