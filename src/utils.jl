zeros_like(::AbstractArray{T}, shape) where T = zeros(T, shape)
zeros_like(::CuArray{T}, shape) where T = CUDA.zeros(T, shape)

ones_like(::AbstractArray{T}, shape) where T = ones(T, shape)
ones_like(::CuArray{T}, shape) where T = CUDA.ones(T, shape)

to_homogeneous(x::AbstractArray{T}) where T = vcat(x, ones(T, 1, size(x)[2:end]...))
to_homogeneous(x::CuArray{T}) where T = vcat(x, CUDA.ones(T, 1, size(x)[2:end]...))

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

    σx = ssim.pool(x_ref .^ 2) .- μx .^ 2
    σy = ssim.pool(y_ref .^ 2) .- μy .^ 2
    σxy = ssim.pool(x_ref .* y_ref) .- μx .* μy

    c1, c2 = T(ssim.c1), T(ssim.c2)
    ssim_n = (T(2.0) .* μx .* μy .+ c1) .* (T(2.0) .* σxy .+ c2)
    ssim_d = (μx .^ 2 .+ μy .^ 2 .+ c1) .* (σx .+ σy .+ c2)
    clamp.((T(1.0) .- ssim_n ./ ssim_d) .* T(0.5), T(0.0), T(1.0))
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
invK (4, 4)

# Returns

(4, W*H, N)
"""
function (b::Backproject)(depth, invK)
    points = reshape(invK[1:3, 1:3] * b.coordinates, (3, size(b.coordinates, 2), 1))
    to_homogeneous(points .* depth)
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
    (((pixels .- T(1.0)) ./ p.normalizer) .- T(0.5)) .* T(2.0)
end

"""
points (4, W*H, N)
K (4, 4)
T (4, 4, N)

# Returns

Normalized projected coordinates in `(-1, 1)` range.
"""
function (p::Project)(points::AbstractArray{V}, K, T) where V
    camera_points = (K ⊠ T) ⊠ points
    denom = V(1.0) ./ (camera_points[[3], :, :] .+ V(1e-7))
    normalize(p, camera_points[1:2, :, :] .* denom)
end

"""
v (3, N)
t (3, N)
"""
function translation_matrix(t)
    N = size(t, 2)
    ze = zeros_like(t, (1, N))
    on = ones_like(t, (1, N))

    translation = cat([
        on, ze, ze, ze,
        ze, on, ze, ze,
        ze, ze, on, ze,
        t[[1], :], t[[2], :], t[[3], :], on,
    ]...; dims=1)
    reshape(translation, (4, 4, N))
end

function rotation_from_axis_angle(v::AbstractArray{T}) where T
    angle = sqrt.(sum(abs2, v; dims=1))
    axis = v ./ (angle .+ T(1e-7))

    ca, sa = cos.(angle), sin.(angle)
    C = T(1.0) .- ca

    x, y, z = axis[[1], :], axis[[2], :], axis[[3], :]
    xs, ys, zs = x .* sa, y .* sa, z .* sa
    xC, yC, zC = x .* C, y .* C, z .* C
    xyC, yzC, zxC = x .* yC, y .* zC, z .* xC

    N = size(v, 2)
    ze = zeros_like(v, (1, N))
    on = ones_like(v, (1, N))

    rotation = cat([
        x .* xC .+ ca, xyC .+ zs, zxC .- ys, ze,
        xyC .- zs, y .* yC .+ ca, yzC .+ xs, ze,
        zxC .+ ys, yzC .- xs, z .* zC .+ ca, ze,
        ze, ze, ze, on,
    ]...; dims=1)
    reshape(rotation, (4, 4, N))
end

get_transformation(v, t, invert::Val{true}) =
    permutedims(rotation_from_axis_angle(v), (2, 1, 3)) ⊠ translation_matrix(-t)

"""
v (3, N)
t (3, N)
"""
function get_transformation(v::AbstractArray{T}, t, invert::Val{false}) where T
    angle = sqrt.(sum(abs2, v; dims=1))
    axis = v ./ (angle .+ T(1e-7))

    ca, sa = cos.(angle), sin.(angle)
    C = T(1.0) .- ca

    x, y, z = axis[[1], :], axis[[2], :], axis[[3], :]
    xs, ys, zs = x .* sa, y .* sa, z .* sa
    xC, yC, zC = x .* C, y .* C, z .* C
    xyC, yzC, zxC = x .* yC, y .* zC, z .* xC

    N = size(v, 2)
    ze = zeros_like(v, (1, N))
    on = ones_like(v, (1, N))

    transformation = cat([
        x .* xC .+ ca, xyC .+ zs, zxC .- ys, ze,
        xyC .- zs, y .* yC .+ ca, yzC .+ xs, ze,
        zxC .+ ys, yzC .- xs, z .* zC .+ ca, ze,
        t[[1], :], t[[2], :], t[[3], :], on,
    ]...; dims=1)
    reshape(transformation, (4, 4, N))
end

"""
Compute smoothness loss for a disparity image.
`image` is used for edge-aware smoothness.

# Arguments

- `disparity`: `WHCN` shape.
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
    T(1.0) ./ (disparity .* (max_disp - min_disp) .+ min_disp)
end
