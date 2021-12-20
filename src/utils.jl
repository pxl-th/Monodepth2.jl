eye_like(::AbstractArray{T}, shape) where T = Array{T, 2}(I, shape)
eye_like(::CuArray{T}, shape) where T = CuArray{T, 2}(I, shape)

zeros_like(::AbstractArray{T}, shape) where T = zeros(T, shape)
zeros_like(::CuArray{T}, shape) where T = CUDA.zeros(T, shape)

ones_like(::AbstractArray{T}, shape) where T = ones(T, shape)
ones_like(::CuArray{T}, shape) where T = CUDA.ones(T, shape)

struct SSIM{P}
    pool::P
    c1::Float64
    c2::Float64
end
Flux.@functor SSIM
SSIM() = SSIM(MeanPool((3, 3); stride=1), 0.01^2, 0.03^2)

"""
The more similar `x` and `y` are, the lower output values will be.
The function is symmetric.
"""
function (ssim::SSIM)(x::AbstractArray{T}, y) where T
    x_ref, y_ref = pad_reflect(x, 1), pad_reflect(y, 1)
    μx, μy = ssim.pool(x_ref), ssim.pool(y_ref)

    two, c1, c2 = T(2.0), T(ssim.c1), T(ssim.c2)

    σx = ssim.pool(x_ref .* x_ref) .- μx .* μx
    σy = ssim.pool(y_ref .* y_ref) .- μy .* μy
    σxy = ssim.pool(x_ref .* y_ref) .- μx .* μy

    ssim_n = (two .* μx .* μy .+ c1) .* (two .* σxy .+ c2)
    ssim_d = (μx .* μx .+ μy .* μy .+ c1) .* (σx .+ σy .+ c2)
    clamp.((one(T) .- ssim_n ./ ssim_d) .* T(0.5), zero(T), one(T))
end

struct Backproject{C}
    coordinates::C
end
Flux.@functor Backproject
function Backproject(::Type{T} = Float64; width, height) where T <: Number
    coordinates = Array{T}(undef, 3, width, height)
    @inbounds for w ∈ 1:width, h ∈ 1:height
        coordinates[1, w, h] = w
        coordinates[2, w, h] = h
        coordinates[3, w, h] = 1.0
    end
    Backproject(reshape(coordinates, (3, width * height)))
end

"""
depth (1, W*H, N)
invK (3, 3)

# Returns

(3, W*H, N)
"""
function (b::Backproject)(depth, invK)
    depth .* reshape(invK * b.coordinates, (3, size(b.coordinates, 2), 1))
end

struct Project{N}
    normalizer::N
end
Flux.@functor Project
function Project(::Type{T} = Float64; width, height) where T <: Number
    Project(reshape(T[width - 1.0, height - 1.0], (2, 1, 1)))
end

"""
Assumes pixels coordinates start at `(1, 1)` and end at `(width, height)`.
Normalizes coordinates to be in `(-1, 1)` range.
"""
function normalize(p::Project, pixels::AbstractArray{T}) where T
    (((pixels .- one(T)) ./ p.normalizer) .- T(0.5)) .* T(2.0)
end

"""
points (3, W*H, N)
K (3, 3, 1)
R (3, 3, N)
t (3, 1, N)

# Returns

(2, W*H, N)

Normalized projected coordinates in `(-1, 1)` range.
"""
function (p::Project)(points::AbstractArray{V}, K, R, t) where V
    camera_points = K ⊠ ((R ⊠ points) .+ t)
    denom = one(V) ./ (camera_points[[3], :, :] .+ V(1e-7))
    normalize(p, camera_points[1:2, :, :] .* denom)
end

# rvec 3xN
function so3_exp_map(rvec)
    T, N = eltype(rvec), size(rvec, 2)

    skew = hat(rvec)
    skew² = skew ⊠ skew

    # NOTE: regular sqrt gives NaN gradient on 0.
    # Need to use subgradient (e.g. return 0 on 0).
    θ = sqrt.(sum(abs2, rvec; dims=1)) # 1xN
    θ_inv = one(T) ./ max.(θ, T(1e-4))

    f1 = reshape(θ_inv .* sin.(θ), (1, 1, N))
    f2 = reshape(θ_inv .* θ_inv .* (one(T) .- cos.(θ)), (1, 1, N))

    f1 .* skew .+ f2 .* skew² .+ eye_like(rvec, (3, 3))
end

function hat(rvec)
    S = zeros_like(rvec, (3, 3, size(rvec, 2)))
    S[2, 1, :] .=  rvec[3, :]
    S[1, 2, :] .= -rvec[3, :]
    S[3, 1, :] .= -rvec[2, :]
    S[1, 3, :] .=  rvec[2, :]
    S[3, 2, :] .=  rvec[1, :]
    S[2, 3, :] .= -rvec[1, :]
    S
end

function rrule(::typeof(hat), v)
    Y = hat(v)
    function hat_pullback(Δ)
        d = unthunk(Δ)
        ∇v = zeros_like(v, size(v))
        ∇v[1, :] .= d[3, 2, :] .- d[2, 3, :]
        ∇v[2, :] .= -d[3, 1, :] .+ d[1, 3, :]
        ∇v[3, :] .= d[2, 1, :] .- d[1, 2, :]
        NoTangent(), ∇v
    end
    return Y, hat_pullback
end

"""
Compute smoothness loss for a disparity image.
`image` is used for edge-aware smoothness.

The disparity smoothness loss penalizes the inverse depth spatial gradients.
The goal of this loss is to make nearby pixels have the similar depth -> spatial smoothness.

# Arguments

- `disparity`: `WHN` shape.
- `image`: `WHCN` shape.

# Returns

Single-value smoothness loss.
"""
function smooth_loss(disparity, image)
    ∇disparity_x = abs.(disparity[1:(end - 1), :, :] .- disparity[2:end, :, :])
    ∇disparity_y = abs.(disparity[:, 1:(end - 1), :] .- disparity[:, 2:end, :])

    ∇image_x = mean(abs.(image[1:(end - 1), :, :, :] .- image[2:end, :, :, :]); dims=3)[:, :, 1, :]
    ∇image_y = mean(abs.(image[:, 1:(end - 1), :, :] .- image[:, 2:end, :, :]); dims=3)[:, :, 1, :]

    mean(∇disparity_x .* exp.(-∇image_x)) + mean(∇disparity_y .* exp.(-∇image_y))
end

function disparity_to_depth(disparity::AbstractArray{T}, min_depth, max_depth) where T
    min_disp = T(1.0 / max_depth)
    max_disp = T(1.0 / min_depth)
    one(T) ./ (disparity .* (max_disp - min_disp) .+ min_disp)
end
