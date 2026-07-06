###############################################################
#   TRAINING — optimizer & gradient estimators                #
#   Adam · SPSA (q-sample) · forward-FD · exact threaded AD    #
#   All gradient fns share the interface (loss_fn, theta) ->   #
#   (gradient, loss_value), so they are interchangeable.       #
###############################################################

# ===== optimizer (ADAM) =====
mutable struct AdamState
    m::Vector{Float64}; v::Vector{Float64}; b1::Float64; b2::Float64; eps::Float64; t::Int
end
AdamState(K::Int; b1=0.9, b2=0.999, eps=1e-8) = AdamState(zeros(K), zeros(K), b1, b2, eps, 0)
function adam_step!(theta, grad, lr, s::AdamState)
    s.t += 1
    @inbounds for i in eachindex(theta)
        s.m[i]=s.b1*s.m[i]+(1-s.b1)*grad[i]; s.v[i]=s.b2*s.v[i]+(1-s.b2)*grad[i]^2
        mh=s.m[i]/(1-s.b1^s.t); vh=s.v[i]/(1-s.b2^s.t); theta[i]-=lr*mh/(sqrt(vh)+s.eps)
    end
end

# ===== SPSA gradient (serial / @threads / @spawn share identical math) =====
function _spsa_points(theta, c, rng, q)
    K = length(theta)
    Ds = [2.0 .* rand(rng, Bool, K) .- 1.0 for _ in 1:q]      # RNG used only here (serial)
    pts = Vector{Vector{Float64}}(undef, 2q)
    @inbounds for j in 1:q
        pts[2j-1] = theta .+ c .* Ds[j]; pts[2j] = theta .- c .* Ds[j]
    end
    return Ds, pts
end
function _spsa_reduce(L, Ds, c, q)
    g = zeros(length(Ds[1])); La = 0.0
    @inbounds for j in 1:q
        Lp = L[2j-1]; Lm = L[2j]; g .+= (Lp .- Lm) ./ (2 .* c .* Ds[j]); La += 0.5*(Lp+Lm)
    end
    return g ./ q, La / q
end
function spsa_gradient_serial(loss_fn, theta::Vector{Float64}; c=0.05, rng=Random.default_rng(), q::Int=4)
    Ds, pts = _spsa_points(theta, c, rng, q); L = Float64[loss_fn(pts[i]) for i in 1:2q]; _spsa_reduce(L, Ds, c, q)
end
function spsa_gradient_threads(loss_fn, theta::Vector{Float64}; c=0.05, rng=Random.default_rng(), q::Int=4)
    Ds, pts = _spsa_points(theta, c, rng, q); L = Vector{Float64}(undef, 2q)
    Threads.@threads for i in 1:2q; L[i] = loss_fn(pts[i]); end
    _spsa_reduce(L, Ds, c, q)
end
function spsa_gradient_spawn(loss_fn, theta::Vector{Float64}; c=0.05, rng=Random.default_rng(), q::Int=4)
    Ds, pts = _spsa_points(theta, c, rng, q); tasks = Vector{Task}(undef, 2q)
    @inbounds for i in 1:2q; p = pts[i]; tasks[i] = Threads.@spawn loss_fn(p); end
    L = Float64[fetch(t) for t in tasks]; _spsa_reduce(L, Ds, c, q)
end
const spsa_gradient = spsa_gradient_spawn

# ===== forward finite-difference gradient (threaded via @spawn): base + K perturbations =====
function forward_fd_gradient_spawn(loss_fn, theta::Vector{Float64}; delta=5e-3)
    K=length(theta); base=loss_fn(theta); tasks=Vector{Task}(undef,K)
    @inbounds for k in 1:K
        tp=copy(theta); tp[k]+=delta; tasks[k]=Threads.@spawn loss_fn(tp)
    end
    g=Float64[(fetch(tasks[k])-base)/delta for k in 1:K]
    return g, base
end

# ===== EXACT gradient via threaded forward-mode AD (ForwardDiff) =====
# The exact gradient: no finite-difference δ, no stochastic noise. Splits the K
# partials into chunks of width `chunk` and runs the chunks concurrently with
# @spawn. (PolyesterForwardDiff would thread this for us but is broken on Apple
# Silicon: "cfunction: closures not supported on this platform".) Drop-in for
# spsa_gradient / forward_fd_gradient_spawn — returns (gradient, loss_value).
# chunk=8 was measured optimal on an M3 Pro (8 threads); C=1 is ~5x slower.
struct _ADGradTag end
function _ad_chunk(loss_fn, theta, lo::Int, hi::Int, ::Val{C}) where {C}
    K = length(theta)
    td = Vector{ForwardDiff.Dual{_ADGradTag,Float64,C}}(undef, K)
    @inbounds for k in 1:K
        part = ntuple(s -> (lo <= k <= hi && s == k - lo + 1) ? 1.0 : 0.0, C)
        td[k] = ForwardDiff.Dual{_ADGradTag,Float64,C}(theta[k], ForwardDiff.Partials(part))
    end
    r = loss_fn(td)
    return Float64[ForwardDiff.partials(r, s) for s in 1:(hi - lo + 1)], ForwardDiff.value(r)
end
function _ad_gradient(loss_fn, theta::Vector{Float64}, ::Val{C}, max_parallel::Int) where {C}
    K = length(theta); g = zeros(K); nch = cld(K, C); base = 0.0
    ci = 1
    @inbounds while ci <= nch
        bhi = min(ci + max_parallel - 1, nch)
        tasks = Vector{Task}(undef, bhi - ci + 1)
        for cj in ci:bhi
            lo = (cj - 1) * C + 1; hi = min(cj * C, K)
            tasks[cj - ci + 1] = Threads.@spawn _ad_chunk(loss_fn, theta, lo, hi, Val(C))
        end
        for cj in ci:bhi
            lo = (cj - 1) * C + 1
            pr, val = fetch(tasks[cj - ci + 1]); cj == 1 && (base = val)
            for s in eachindex(pr); g[lo + s - 1] = pr[s]; end
        end
        ci = bhi + 1
    end
    return g, base
end
# max_parallel bounds how many chunks are ever in flight at once (memory guard);
# default keeps the old behavior of spawning every chunk up front.
ad_gradient(loss_fn, theta::Vector{Float64}; chunk::Int=8, max_parallel::Int=typemax(Int)) =
    _ad_gradient(loss_fn, theta, Val(chunk), max_parallel)

function _ad_gradient_serial(loss_fn, theta::Vector{Float64}, ::Val{C}) where {C}
    K = length(theta); g = zeros(K); nch = cld(K, C); base = 0.0
    @inbounds for ci in 1:nch
        lo = (ci - 1) * C + 1; hi = min(ci * C, K)
        pr, val = _ad_chunk(loss_fn, theta, lo, hi, Val(C)); ci == 1 && (base = val)
        for s in eachindex(pr); g[lo + s - 1] = pr[s]; end
    end
    return g, base
end
ad_gradient_serial(loss_fn, theta::Vector{Float64}; chunk::Int=8) = _ad_gradient_serial(loss_fn, theta, Val(chunk))
