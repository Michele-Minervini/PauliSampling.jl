function mul(nq::Integer, pstr::S, i::T) where {S, T}
    res = i
    phase = 1.0 + 0.0im
    @inbounds for j in 0:nq-1
        op = (pstr >> (2j)) & 0x3
        b  = (i >> j) & 0x1
        if op == 0x0
            continue
        elseif op == 0x1
            res ⊻= (1 << j)
        elseif op == 0x2
            res ⊻= (1 << j)
            phase *= if b == 0
                1im
            else
                -1im
            end
        elseif op == 0x3
            phase *= if b == 0
                1
            else
                -1
            end
        end
    end
    return phase, res
end

function apply!(y::AbstractVector, pstr::S, coeff::C, nq::Integer, x::AbstractVector) where {S, C}
    @inbounds for idx in 1:(1<<nq)
        i = idx - 1
        phase, flipped = mul(nq, pstr, i)
        y[flipped + 1] += coeff * phase * x[idx]
    end
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, H::PauliSum{S, C},
                            x::AbstractVector, α::Number, β::Number) where {S, C}
    nq = H.nqubits
    fill!(y, 0)
    for (term, coeff) in H
        apply!(y, term, coeff, nq, x)
    end
    @. y = α*y + β*y
    return y
end

function (H::PauliSum{S, C})(y::AbstractVector, x::AbstractVector) where {S, C}
    mul!(y, H, x, 1.0, 0.0)
    return y
end

function build_lin_map(psum)
    nq = psum.nqubits
    dim = 1<<nq
    return LinearMap{ComplexF64}(psum, dim, dim; issymmetric=true)
end

function compute_most_negative_eigenvalue(lin_map)
    λ, = eigs(lin_map; nev=1, which=:SR, ritzvec=false)
    @assert sum(abs, imag.(λ)) < 1e-9
    return real(λ[1])
end

function compute_neg_eigs(lin_map)
    dim = size(lin_map, 1)
    if dim == 2
        λ = eigen(Hermitian(Matrix(lin_map))).values
    else
        nev = dim - 2 # I still don't get why I have to do - 2.
        λ, = eigs(lin_map; nev=nev, which=:SR, ritzvec=false)
    end
    @assert sum(abs, imag.(λ)) < 1e-9
    λ = real(λ)
    neg = λ[λ .< 0]
    return neg
end

function compute_negativity(lin_map; maxnev=256)
    neg_vals = compute_neg_eigs(lin_map)
    neg_sum = sum(abs, neg_vals)
    return neg_sum
end

"""
# Experimental stuff (not working right now). See: Ubaru, Shashanka, Jie Chen, and Yousef Saad. "Fast estimation of tr(f(A)) via stochastic Lanczos quadrature." SIAM Journal on Matrix Analysis and Applications 38.4 (2017): 1075-1099

function lanczos_T!(T::SymTridiagonal, H, v; m::Int)
    n = length(v)
    α = Vector{Float64}(undef, m)
    β = Vector{Float64}(undef, m-1)
    vprev = zeros(ComplexF64, n)
    w = similar(v)
    βprev = 0.0

    for j in 1:m
        mul!(w, H, v, 1, 0)
        if j > 1
            @. w -= βprev * vprev
        end
        α[j] = real(dot(v, w))
        @. w -= α[j] * v
        if j < m
            β[j] = LinearAlgebra.norm(w)
            vprev .= v
            @. v = w / β[j]
            βprev = β[j]
        end
    end

    for i in 1:m
        T.dv[i] = α[i]
        if i < m
            T.ev[i] = β[i]
        end
    end
    return T
end

function estimate_negativity(Hmap; nprobe=40, m=120, rng=Random.default_rng())
    n = size(Hmap, 1)
    acc = 0.0
    T = SymTridiagonal(zeros(m), zeros(m-1))

    for _ in 1:nprobe
        v = randn(rng, ComplexF64, n)
        v ./= LinearAlgebra.norm(v)

        lanczos_T!(T, Hmap, copy(v); m=m)

        T_dense = Matrix(T)
        θ, Q = eigen(T_dense)

        w = view(Q, 1, :)
        acc += sum(abs2.(w) .* max.(-θ, 0.0))
    end

    return acc * n / nprobe
end
"""