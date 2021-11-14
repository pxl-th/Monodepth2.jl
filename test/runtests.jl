using Test
using LinearAlgebra
using Rotations
using Statistics
using Monodepth

@testset "Test transformations construction" begin
    v = rand(Float64, (3, 1))
    t = reshape([0.0, 0.0, 0.0], (3, 1))

    target = RotationVec(v...)
    source = Monodepth.transformation_from_axis_angle_translation(
        v, t, Val(false))
    @test all(isapprox.(target, source[1:3, 1:3, 1]; atol=1e-5))

    target = transpose(target)
    source = Monodepth.transformation_from_axis_angle_translation(
        v, t, Val(true))
    @test all(isapprox.(target, source[1:3, 1:3, 1]; atol=1e-5))
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
    res = 32
    image = rand(Float64, res, res, 1, 1)

    K = reshape(Float64[
        910, 0, 0, 0,
        0, 910, 0, 0,
        res / 2, res / 2, 1, 0,
        0, 0, 0, 1,
    ], (4, 4))
    invK = inv(K)

    v = zeros(Float64, (3, 1))
    t = zeros(Float64, (3, 1))
    T = Monodepth.transformation_from_axis_angle_translation(v, t, Val(false))
    depth = rand(Float64, (1, res * res, 1))

    projection = Monodepth.Project(;width=res, height=res)
    backprojection = Monodepth.Backproject(;width=res, height=res)

    grid = reshape(projection(backprojection(depth, invK), K, T), (2, res, res, 1))
    sampled = Monodepth.Flux.grid_sample(image, grid)
    @test all(isapprox.(image, sampled; atol=1e-3))
end
