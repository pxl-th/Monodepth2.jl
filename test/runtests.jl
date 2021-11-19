using Test
using ChainRulesTestUtils

using LinearAlgebra
using Statistics
using Rotations
using CUDA
using Zygote
using Flux
using Monodepth
CUDA.allowscalar(false)

@testset "Test rotations" begin
    v = rand(Float64, (3, 1))
    p = rand(Float64, (3, 1))

    target = RotationVec(v...)
    source = Monodepth.compose_rotation(v)
    @test all(isapprox.(target, source[:, :, 1]; atol=1e-5))
    test_rrule(Monodepth.hat, v)

    vg = CuArray(v)
    source = Monodepth.compose_rotation(vg)
    @test source isa CuArray
    @test all(isapprox.(target, collect(source)[:, :, 1]; atol=1e-5))
end

@testset "Test transformation" begin
    rvec = rand(Float64, (3, 1))
    tvec = rand(Float64, (3, 1, 1))
    p = rand(Float64, (3, 1, 1))

    R, t = Monodepth._get_transformation(rvec, tvec, false)
    tp = RotationVec(rvec...) * p[:, 1, 1] .+ t[:, 1, 1]
    np = R ⊠ p .+ t
    @test all(isapprox.(np, tp; atol=1e-6))

    R, t = Monodepth._get_transformation(rvec, tvec, true)
    invR = transpose(RotationVec(rvec...))
    invt = -(invR * tvec[:, 1, 1])
    tp = invR * np[:, 1, 1] .+ invt
    op = R ⊠ np .+ t
    @test all(isapprox.(op, tp; atol=1e-6))
    @test all(isapprox.(op, p; atol=1e-6))
end

@testset "Test SSIM" begin
    ssim = Monodepth.SSIM()

    source = ones(Float64, 2, 2, 1, 1)
    target = ones(Float64, 2, 2, 1, 1)
    score = ssim(source, target)
    @test all(isapprox.(score, 0.0))

    target = zeros(Float64, 2, 2, 1, 1)
    score = ssim(source, target)
    @test all(isapprox.(score, 0.5; atol=1e-1))

    source = rand(Float64, 2, 2, 1, 2)
    target = rand(Float64, 2, 2, 1, 2)
    @test all(ssim(source, target) .≈ ssim(target, source))
end

@testset "Test smooth loss" begin
    disp = reshape(transpose(reshape(collect(0:0.1:0.3), (2, 2))), (2, 2, 1, 1))
    image = ones(Float64, (2, 2, 1, 1))

    sl = Monodepth.smooth_loss(disp, image)
    tl = mean(abs.(disp[1:(end - 1), :, :, :] .- disp[2:end, :, :, :])) +
        mean(abs.(disp[:, 1:(end - 1), :, :] .- disp[:, 2:end, :, :]))
    @test sl ≈ tl

    image = reshape(transpose(reshape(collect(0.1:0.1:0.4), (2, 2))), (2, 2, 1, 1))
    sl = Monodepth.smooth_loss(disp, image)
    @test isapprox(sl, 0.2542; atol=1e-4)
end

@testset "Test disparity → depth conversion" begin
    disp = rand(Float64, 32, 32, 2)
    depth = Monodepth.disparity_to_depth(disp, 0.1, 100.0)
    @test minimum(depth) ≥ 0.1
    @test maximum(depth) ≤ 100.0
end

@testset "Test identity image warping" begin
    res = 16
    N = 2
    image = rand(Float64, res, res, 1, N)
    depth = rand(Float64, (1, res * res, N))
    K = reshape(Float64[
        910, 0, 0,
        0, 910, 0,
        res / 2, res / 2, 1], (3, 3))
    invK = inv(K)
    K = reshape(K, (3, 3, 1))

    v = zeros(Float64, (3, N))
    t = zeros(Float64, (3, 1, N))
    R = Monodepth.compose_rotation(v)

    projection = Monodepth.Project(;width=res, height=res)
    backprojection = Monodepth.Backproject(;width=res, height=res)

    camera_points = backprojection(depth, invK)
    warped_uv = projection(camera_points, K, R, t)
    grid = reshape(warped_uv, (2, res, res, N))
    sampled = Monodepth.Flux.grid_sample(image, grid)
    @test all(isapprox.(image, sampled; atol=1e-3))
end
