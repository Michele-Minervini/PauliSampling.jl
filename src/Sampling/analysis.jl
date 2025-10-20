function tvd(p, q)
    return 0.5 * sum(abs.(p .- q))
end

function kl_div(p, q)
    res = 0.0
    for (i, pi) in enumerate(p)
        qi = q[i]
        res += pi * log(pi/qi)
    end
    return res
end

function shannon_entropy(prob)
    entropy = 0.0
    for p in prob
        if p > 0  # avoid log(0)
            entropy -= p * log2(p)
        end
    end
    return entropy
end



function avg_metrics_for_weight(nq, max_weight, M)
    tvd_coset = 0.0
    tvd_bayes = 0.0
    kl_coset  = 0.0
    kl_bayes  = 0.0

    circuit_nq  = build_circuit(nq)
    nparams  = countparameters(circuit_nq)
    init_psum = zero_state(nq)

    for _ in 1:M
        thetas = randn(nparams) * 0.5
        exact_psum = propagate(circuit_nq, init_psum, thetas)
        trunc_psum = propagate(circuit_nq, init_psum, thetas; max_weight=max_weight)

        exact_dist = PauliSampling.get_dist(exact_psum,  PauliSampling.approximate_prob)
        coset_dist = PauliSampling.get_dist(trunc_psum, PauliSampling.projection_prob)
        bayes_dist = PauliSampling.get_dist(trunc_psum, PauliSampling.approximate_prob)

        tvd_coset += PauliSampling.tvd(exact_dist, coset_dist)
        kl_coset += PauliSampling.kl_div(exact_dist, coset_dist)
        tvd_bayes += PauliSampling.tvd(exact_dist, bayes_dist)
        kl_bayes += PauliSampling.kl_div(exact_dist, bayes_dist)
    end
    return tvd_coset/M, kl_coset/M, tvd_bayes/M, kl_bayes/M
end

function boolean_fourier_dist_diff(nq, p, q)
    dist_diff = p - q
    fourier_coeffs = naive_fwht(dist_diff)
    bins = zeros(nq + 1)
    for i in 1:2^nq
        bins[count_ones(i-1) + 1] += abs(fourier_coeffs[i])
    end

    return bins
end

function naive_fwht!(x::Vector{Float64})
    n = length(x)
    @assert ispow2(n) "Length must be a power of 2"
    h = 1
    while h < n
        for i in 1:2h:n
            for j in i:i+h-1
                u = x[j]
                v = x[j + h]
                x[j] = u + v
                x[j + h] = u - v
            end
        end
        h *= 2
    end
    return x
end

function naive_fwht(x::Vector{Float64})
    return naive_fwht!(copy(x))
end

function plot_distribution_fourier_analysis(nq; max_weight=4, plot_difference=true)
    init_psum = zero_state(nq)
    circuit = build_circuit(nq)
    nparams = countparameters(circuit)
    thetas = randn(nparams) * 0.5

    # Propagate
    exact_psum = propagate(circuit, init_psum, thetas)
    trunc_psum = propagate(circuit, init_psum, thetas; max_weight=max_weight)

    # Get distributions
    exact_dist = get_dist(exact_psum, approximate_prob)
    coset_dist = get_dist(trunc_psum, projection_prob)
    bayes_dist = get_dist(trunc_psum, approximate_prob)

    x_bins = 0:nq

    # Plot styling
    plot_kwargs = (
        xlabel = "Hamming weight",
        ylabel = "|Fourier coeff| sum",
        size = (700, 450),
        top_margin = 10Plots.mm,
        right_margin = 8Plots.mm,
        bottom_margin = 5Plots.mm,
        left_margin = 6Plots.mm,
        legend = :outertopright,
        tickfont = font(10),
        guidefont = font(12),
        legendfont = font(10)
    )

    if plot_difference
        # Plot difference in Fourier spectrum
        coset_diff = boolean_fourier_dist_diff(nq, exact_dist, coset_dist)
        bayes_diff = boolean_fourier_dist_diff(nq, exact_dist, bayes_dist)

        plt = plot(x_bins, coset_diff; label="Coset vs Exact", marker=:circle, lw=2, plot_kwargs...)
        plot!(plt, x_bins, bayes_diff; label="Bayes vs Exact", marker=:square, lw=2)
        plot_title = "Fourier difference: Coset/Bayes vs Exact (max_weight = $max_weight)"

    else
        # Plot individual Fourier spectra
        zero = zeros(2^nq)
        coset_bins = boolean_fourier_dist_diff(nq, zero, coset_dist)
        bayes_bins = boolean_fourier_dist_diff(nq, zero, bayes_dist)
        exact_bins = boolean_fourier_dist_diff(nq, zero, exact_dist)

        plt = plot(x_bins, coset_bins; label="Coset", marker=:circle, lw=2, plot_kwargs...)
        plot!(plt, x_bins, bayes_bins; label="Bayes", marker=:square, lw=2)
        plot!(plt, x_bins, exact_bins; label="Exact", marker=:diamond, lw=2)
        plot_title = "Fourier spectrum of individual distributions (max_weight = $max_weight)"
    end

    plot!(plt; title=plot_title, titlefont=font(13))
    display(plt)
end

function plot_all_weight_fourier(nq::Int)
    plots = []

    for max_weight in 1:nq
        # Compute distributions
        init_psum = zero_state(nq)
        circuit = build_circuit(nq)
        nparams = countparameters(circuit)
        thetas = randn(nparams) * 0.5

        exact_psum = propagate(circuit, init_psum, thetas)
        trunc_psum = propagate(circuit, init_psum, thetas; max_weight=max_weight)

        exact_dist = get_dist(exact_psum, approximate_prob)
        coset_dist = get_dist(trunc_psum, projection_prob)
        bayes_dist = get_dist(trunc_psum, approximate_prob)

        exact_coset_bins = PauliSampling.boolean_fourier_dist_diff(nq, exact_dist, coset_dist)
        exact_bayes_bins = PauliSampling.boolean_fourier_dist_diff(nq, exact_dist, bayes_dist)

        x_bins = 0:nq
        plt = plot(x_bins, exact_coset_bins;
                   label="Coset vs Exact",
                   lw=2, marker=:circle,
                   xlabel="Weight", ylabel="|Fourier Coeff| sum",
                   title="max_weight = $max_weight",
                   legend=:topright,
                   margin=5Plots.mm,
                   size=(400, 300),
                   tickfont=font(8),
                   titlefont=font(10))

        plot!(plt, x_bins, exact_bayes_bins; label="Bayes vs Exact", lw=2, marker=:square)
        push!(plots, plt)
    end

    # Compute layout
    ncols = ceil(Int, sqrt(nq))
    nrows = ceil(Int, nq / ncols)

    # Display all plots in a grid
    gridplot = plot(plots...; layout=(nrows, ncols), size=(400 * ncols, 300 * nrows))
    display(gridplot)
end


function bit_marginals(nq, dist)
    marg = zeros(nq)
    for i in 1:length(dist)
        bits = digits(i - 1, base=2, pad=nq)
        for j in 1:nq
            marg[j] += bits[j] * dist[i]
        end
    end
    return marg
end