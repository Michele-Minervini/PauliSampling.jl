###############################################################
#   TRAINING — loss                                           #
#   Algorithm-1 model probability + support-KL                #
###############################################################

"""
    model_prob(rho, x::BitVector)

Algorithm-1 (ancestral) probability of bitstring `x` under the thermal state `rho`.
(`reverse(x)` matches the qubit ordering of `get_approx_prob`.)
"""
model_prob(rho, x::BitVector) = get_approx_prob(rho, reverse(x))

"""
    extract_support(dataset) -> (supp, probs)

Empirical support of a `Vector{BitVector}`: the distinct patterns and their frequencies.
"""
function extract_support(dataset::Vector{BitVector})
    counts = Dict{BitVector, Int}()
    for x in dataset; counts[x] = get(counts, x, 0) + 1; end
    supp = collect(keys(counts)); probs = Float64[counts[x] / length(dataset) for x in supp]
    return supp, probs
end

"""
    kl_support(rho, supp, probs) -> KL

Forward support-KL between the empirical data distribution (`supp`,`probs`) and the
model. Extracts the diagonal terms ONCE (independent of the support point), then scores
every support bitstring against them — ~|supp|× less extraction work than calling
`get_approx_prob` per point, and type-stable under ForwardDiff (`kl` starts at the
coefficient type so it stays a `Dual` under AD).
"""
function kl_support(rho, supp::Vector{BitVector}, probs::Vector{Float64})
    term_coeffs, term_masks = extract_diagonal_terms(rho)
    nq = rho.nqubits
    kl = zero(eltype(term_coeffs))
    @inbounds for i in eachindex(supp)
        pd = probs[i]
        if pd > 1e-12
            pm = max(approx_prob_from_terms(term_coeffs, term_masks, nq, reverse(supp[i])), 1e-20)
            kl += pd * log(pd / pm)
        end
    end
    return kl
end
