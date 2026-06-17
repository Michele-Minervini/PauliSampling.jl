###############################################################
#   TRAINING SUBMODULE                                        #
#   QBM generative-modeling pipeline: data -> model -> loss   #
#   -> gradients -> train_qbm. (Included into module          #
#   PauliSampling, after Sampling/ and QuantumBoltzmannMachine/.)
#
#   Depends on: get_approx_prob / extract_diagonal_terms /     #
#   approx_prob_from_terms (Sampling), SquareLattice /         #
#   parameters / makehamiltonian[_generalized] /               #
#   paulistringtocircuit / makethermalstate (QBM).             #
###############################################################
include("datasets.jl")     # reduced-MNIST loaders
include("models.jl")       # H(θ) templates + thermal state
include("loss.jl")         # Algorithm-1 model_prob + support-KL
include("gradients.jl")    # Adam + SPSA / forward-FD / exact AD
include("train.jl")        # train_qbm entry point
