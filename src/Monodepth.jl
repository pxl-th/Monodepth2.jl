module Monodepth

using Plots
gr()

using StaticArrays
using FileIO
using ImageCore
using ImageTransformations
using Printf
using MLDataPattern: shuffleobs
using DataLoaders
using LinearAlgebra

using Rotations
using Statistics
import ChainRulesCore: rrule
using ChainRulesCore
using CUDA
using Zygote
using Flux
using EfficientNet
CUDA.allowscalar(false)

import BSON
using BSON: @save, @load

include("kitty.jl")
include("dtk.jl")
include("utils.jl")
include("depth_decoder.jl")
include("pose_decoder.jl")
include("model.jl")

Zygote.@nograd CUDA.ones
Zygote.@nograd CUDA.zeros
Zygote.@nograd eye_like

Base.@kwdef struct Params
    min_depth::Float64 = 0.1
    max_depth::Float64 = 100.0
    disparity_smoothness::Float64 = 1e-3
    frame_ids::Vector{Int64} = [1, 2, 3]

    target_size::Tuple{Int64, Int64} = (128, 128) # width, height format TODO dataset
    batch_size::Int64 = 1
end

struct TrainCache{S, B, P, K, I}
    ssim::S
    backprojections::B
    projections::P
    Ks::K
    invKs::I

    scales::Vector{Float64}
    source_ids::Vector{Int64}
    target_pos_id::Int64
end

function photometric_loss(
    ssim, predicted::AbstractArray{T}, target::AbstractArray{T}; α = T(0.85),
) where T
    l1_loss = mean(abs.(target .- predicted); dims=3)
    ssim_loss = mean(ssim(predicted, target); dims=3)
    α .* ssim_loss .+ (T(1.0) - α) .* l1_loss
end

automasking_loss(ssim, inputs, target; source_ids) =
    minimum(cat(map(sid -> photometric_loss(ssim, inputs[sid], target), source_ids)...; dims=3); dims=3)
prediction_loss(ssim, predictions, target) =
    minimum(cat(map(p -> photometric_loss(ssim, p, target), predictions)...; dims=3); dims=3)

function warp(
    disparity, inputs, Ps, backproject, project, invK, K;
    min_depth, max_depth, source_ids,
)
    depth = disparity_to_depth(disparity, min_depth, max_depth)
    _, dw, dh, dn = size(depth)
    depth = reshape(depth, (1, dw * dh, dn))

    cam_coordinates = backproject(depth, invK)
    function _warp(i, sid)
        R, t = Ps[i]
        warped_uv = reshape(project(cam_coordinates, K, R, t), (2, dw, dh, dn))
        grid_sample(inputs[sid], warped_uv; padding_mode=:border)
    end
    map(si -> _warp(si[1], si[2]), enumerate(source_ids))
end

function _get_transformation(rvec, t, invert)
    R = compose_rotation(rvec)
    if invert
        R = permutedims(R, (2, 1, 3))
        t = R ⊠ -t
    end
    R, t
end

function train_loss(model, x::AbstractArray{T}, train_cache::TrainCache, parameters::Params) where T
    loss = T(0.0)
    xs = map(i -> x[:, :, :, i, :], 1:length(parameters.frame_ids))

    disparities, poses = model(
        x; source_ids=train_cache.source_ids, target_pos_id=train_cache.target_pos_id)
    rvecs, tvecs = poses # 3, 1, N
    println(cpu(rvecs)[end][:, 1, 1], " | ", cpu(tvecs)[end][:, 1, 1])

    Ps = map(si -> _get_transformation(
        rvecs[si[1]][:, 1, :],
        tvecs[si[1]],
        si[2] > train_cache.target_pos_id), enumerate(train_cache.source_ids))

    vis_warped = nothing
    for (i, scale) in enumerate(train_cache.scales)
        disparity = disparities[i]
        if i != length(train_cache.scales)
            disparity = upsample_bilinear(disparity; size=parameters.target_size)
        end

        dw, dh, _, db = size(disparity)
        disparity = reshape(disparity, (1, dw, dh, db))
        warped_images = warp(
            disparity, xs, Ps, train_cache.backprojections, train_cache.projections,
            train_cache.invKs, train_cache.Ks;
            min_depth=parameters.min_depth, max_depth=parameters.max_depth,
            source_ids=train_cache.source_ids)

        vis_warped = cpu(warped_images[end])

        auto_loss = automasking_loss(
            train_cache.ssim, xs, xs[train_cache.target_pos_id]; source_ids=train_cache.source_ids)
        pred_loss = prediction_loss(
            train_cache.ssim, warped_images, xs[train_cache.target_pos_id])
        warp_loss = minimum(cat(auto_loss, pred_loss; dims=3); dims=3)
        loss += mean(warp_loss)

        disparity = reshape(disparity, size(disparity)[2:end])
        disparity = disparity ./ (mean(disparity; dims=(1, 2)) .+ T(1e-7))
        disparity_loss = smooth_loss(disparity, xs[train_cache.target_pos_id]) .*
            T(parameters.disparity_smoothness) .* T(scale)
        loss += disparity_loss
    end

    loss / T(length(train_cache.scales)), cpu(disparities[end]), vis_warped
end

function save_disparity(disparity, epoch, i)
    disparity = disparity[:, :, 1, 1]
    disparity = permutedims(disparity, (2, 1))
    fig = heatmap(disparity; c=:thermal, aspect_ratio=:equal)
    png(fig, "/home/pxl-th/projects/disp-$epoch-$i.png")
end

function save_warped(warped, epoch, i)
    c = size(warped, 3)
    if c == 1
        warped = warped[:, :, 1, 1]
        warped = permutedims(warped, (2, 1))
    else
        warped = warped[:, :, :, 1]
        warped = colorview(RGB, permutedims(warped, (3, 2, 1)))
    end
    save("/home/pxl-th/projects/warped-$epoch-$i.png", warped)
end

function nn()
    device = cpu
    precision = f32

    dataset = Depth10k("/home/pxl-th/projects/depth10k/imgs")
    parameters = Params(;
        batch_size=1, disparity_smoothness=1e-3,
        target_size=dataset.resolution)
        # target_size=(224, 64))

    # original_resolution = (1241, 376)
    # dataset = KittyDataset(
    #     "/home/pxl-th/Downloads/kitty-dataset", "00";
    #     original_resolution, target_size=parameters.target_size,
    #     frame_ids=parameters.frame_ids, n_frames=4541)

    @show length(dataset)
    display(dataset.K); println()

    max_scale = 5
    scale_levels = collect(2:5)

    scales = Float64[]
    scale_sizes = Tuple{Int64, Int64}[]
    for scale_level in scale_levels
        scale = 1.0 / 2.0^(max_scale - scale_level)
        scale_size = ceil.(Int64, parameters.target_size .* scale)
        push!(scales, scale)
        push!(scale_sizes, scale_size)
    end
    @show scale_sizes

    # Transfer to the device.
    projections = device(precision(Project(; width=parameters.target_size[1], height=parameters.target_size[2])))
    backprojections = device(precision(Backproject(; width=parameters.target_size[1], height=parameters.target_size[2])))
    Ks = device(precision(Array(dataset.K)))
    invKs = device(precision(inv(Array(dataset.K))))
    ssim = SSIM() |> precision |> device

    train_cache = TrainCache(
        ssim, backprojections, projections, Ks, invKs,
        scales, dataset.source_ids, dataset.target_pos_id)

    encoder = EffNet("efficientnet-b0"; include_head=false, in_channels=3)
    encoder_channels = collect(encoder.stages_channels)
    depth_decoder = DepthDecoder(;encoder_channels, scale_levels)
    pose_decoder = PoseDecoder(encoder_channels[end], 2, 1)
    model = Model(encoder, depth_decoder, pose_decoder) |> precision |> device

    θ = model |> params
    optimizer = ADAM(1e-4) |> precision

    for epoch in 1:100
        i = 0
        loader = DataLoader(shuffleobs(dataset), parameters.batch_size)
        for images in loader
            x = images |> precision |> device

            model |> trainmode!
            loss_cpu = 0.0
            disparity = nothing
            warped = nothing

            @time ∇ = gradient(θ) do
                loss, disparity, warped = train_loss(model, x, train_cache, parameters)
                loss_cpu = loss |> cpu
                loss
            end
            Flux.Optimise.update!(optimizer, θ, ∇)

            if i % 13 == 0
                println("$epoch | $i | Loss: $loss_cpu")
                save_disparity(disparity, epoch, i)
                save_warped(warped, epoch, i)
            end
            if i % 101 == 0
                model_host = model |> cpu
                @save "./models/epoch-$epoch-loss-$loss_cpu.bson" model_host
                # model_host = BSON.load("./models/epoch-$epoch-loss-$loss_cpu.bson", @__MODULE__)
            end
            i += 1
        end
    end
end
# nn()

end
