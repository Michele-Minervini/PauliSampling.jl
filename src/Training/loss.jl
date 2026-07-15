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
    extract_support(dataset; min_count=1) -> (supp, probs)

Distinct patterns and their renormalized empirical frequencies. Patterns seen fewer than
`min_count` times are dropped before renormalizing (`min_count=1` keeps everything).
"""
function extract_support(dataset::Vector{BitVector}; min_count::Int = 1)
    counts = Dict{BitVector, Int}()
    for x in dataset; counts[x] = get(counts, x, 0) + 1; end
    supp = BitVector[]; kept = Int[]
    for (x, c) in counts
        c >= min_count && (push!(supp, x); push!(kept, c))
    end
    total = sum(kept)
    probs = Float64[c / total for c in kept]
    return supp, probs
end

"""
    kl_support(rho, supp, probs; thread=true) -> KL

Forward support-KL between the empirical data distribution (`supp`,`probs`) and the
model. Extracts the diagonal terms ONCE (independent of the support point), then scores
every support bitstring against them — ~|supp|× less extraction work than calling
`get_approx_prob` per point, and type-stable under ForwardDiff (`kl` starts at the
coefficient type so it stays a `Dual` under AD).
`thread=false` runs the reduction over support points single-threaded, e.g. when the
caller already runs several evaluations concurrently.
"""
function kl_support(rho, supp::Vector{BitVector}, probs::Vector{Float64}; thread::Bool = true)
    term_coeffs, term_masks = extract_diagonal_terms(rho)
    nq = rho.nqubits

    
    ## Multi-threading the computation
    return AcceleratedKernels.mapreduce(
        (i) -> begin
            pd = probs[i]
            if pd > 1e-12
                pm = max(approx_prob_from_terms(term_coeffs, term_masks, nq, reverse(supp[i])), 1e-20)
                return pd * log(pd / pm)
            else
                return zero(eltype(term_coeffs))
            end
        end,
        +,
        collect(eachindex(supp));
        init=zero(eltype(term_coeffs)),
        max_tasks=thread ? Threads.nthreads() : 1
    )

end
