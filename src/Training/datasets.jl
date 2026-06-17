###############################################################
#   TRAINING — datasets                                       #
#   Reduced-MNIST loaders (downsample + binarize + flatten)   #
###############################################################

"""
    downsample_image(img, target_rows, target_cols) -> Matrix{Float32}

Block-average downsampling. Block boundaries use integer floor division, so
non-divisor target sizes (e.g. 28→5) work but blocks are not all the same size.
"""
function downsample_image(img::AbstractMatrix{<:Real}, target_rows::Int, target_cols::Int)
    src_rows, src_cols = size(img)
    out = zeros(Float32, target_rows, target_cols)
    @inbounds for i in 1:target_rows
        r_start = ((i - 1) * src_rows) ÷ target_rows + 1
        r_end   = (i * src_rows) ÷ target_rows
        for j in 1:target_cols
            c_start = ((j - 1) * src_cols) ÷ target_cols + 1
            c_end   = (j * src_cols) ÷ target_cols
            out[i, j] = mean(view(img, r_start:r_end, c_start:c_end))
        end
    end
    return out
end

"""
    binarize_image(img; method, threshold) -> BitMatrix

Four methods: `:fixed`, `:mean`, `:adaptive`, `:otsu`.
"""
function binarize_image(img::AbstractMatrix{<:Real}; method::Symbol = :adaptive, threshold::Float64 = 0.5)
    if method === :fixed
        return img .> threshold
    elseif method === :mean
        return img .> mean(img)
    elseif method === :adaptive
        nz = filter(>(0.05), vec(img))
        thr = isempty(nz) ? threshold : mean(nz) * 0.5
        return img .> thr
    elseif method === :otsu
        vals = vec(img); best_thr = 0.5; best_var = -Inf
        for t in 0.0:0.01:1.0
            fg = vals[vals .> t]; bg = vals[vals .<= t]
            (isempty(fg) || isempty(bg)) && continue
            wf, wb = length(fg)/length(vals), length(bg)/length(vals)
            v = wf * wb * (mean(fg) - mean(bg))^2
            if v > best_var; best_var = v; best_thr = t; end
        end
        return img .> best_thr
    else
        error("Unknown binarization method: $method")
    end
end

"""
    image_to_bitvector(bin_img) -> BitVector

Row-major flattening. `bv[(r-1)*cols + c] = bin_img[r,c]`, so the top-left pixel
lands at position 1 = qubit 1.
"""
function image_to_bitvector(bin_img::AbstractMatrix{Bool})
    rows, cols = size(bin_img)
    bv = falses(rows * cols)
    @inbounds for r in 1:rows, c in 1:cols
        bv[(r-1)*cols + c] = bin_img[r, c]
    end
    return BitVector(bv)
end

function bitvector_to_image(bv::BitVector, rows::Int, cols::Int)
    img = falses(rows, cols)
    @inbounds for r in 1:rows, c in 1:cols
        img[r, c] = bv[(r-1)*cols + c]
    end
    return img
end

"""
    generate_mnist_dataset(rows, cols; digit_classes, n_per_class, binarize_method, seed)

Build a `Vector{BitVector}` of binarized + downsampled MNIST images. Each BitVector has
length `rows*cols`, with top-left pixel at position 1 (qubit 1).

Note on MNIST orientation: MLDatasets returns features as a (28, 28, N) array where
the first two axes are swapped vs. the visual convention — i.e., `X_all[i, j, k]`
corresponds to original_image[row=j, col=i]. We `permutedims` to recover (row, col)
so that the digit "1" appears as a vertical stroke.
"""
function generate_mnist_dataset(
        rows::Int, cols::Int;
        digit_classes::Vector{Int} = [1],
        n_per_class::Int = 500,
        binarize_method::Symbol = :adaptive,
        seed::Int = 42)

    @assert all(0 .<= digit_classes .<= 9) "digit_classes must be in 0..9"
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
    Random.seed!(seed)

    mnist = MLDatasets.MNIST(split = :train)
    X_all = mnist.features
    y_all = mnist.targets

    dataset = BitVector[]
    for d in digit_classes
        idx = findall(==(d), y_all)
        n_take = min(n_per_class, length(idx))
        chosen = idx[randperm(length(idx))[1:n_take]]
        for k in chosen
            img = permutedims(X_all[:, :, k])    # (28,28) with (row, col) order
            small = downsample_image(img, rows, cols)
            binary = binarize_image(small; method = binarize_method)
            push!(dataset, image_to_bitvector(binary))
        end
    end
    return shuffle(dataset)
end
