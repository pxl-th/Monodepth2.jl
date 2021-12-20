module Monodepth

using LinearAlgebra
using Printf
using Statistics

using Augmentations
import BSON
using BSON: @save, @load
using DataLoaders
using FileIO
using ImageCore
using ImageTransformations
using MLDataPattern: shuffleobs
using ProgressMeter
using Plots
using StaticArrays
gr()

import ChainRulesCore: rrule
using ChainRulesCore
using Zygote
using CUDA
using Flux
using EfficientNet
using ResNet

CUDA.allowscalar(false)

Zygote.@nograd CUDA.ones
Zygote.@nograd CUDA.zeros

Base.@kwdef struct Params
    min_depth::Float64 = 0.1
    max_depth::Float64 = 100.0
    disparity_smoothness::Float64 = 1e-3
    frame_ids::Vector{Int64} = [1, 2, 3]

    automasking::Bool = true

    target_size::Tuple{Int64, Int64} # (width, height)
    batch_size::Int64
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

include("dtk.jl")
include("utils.jl")
include("depth_decoder.jl")
include("pose_decoder.jl")
include("model.jl")
include("simple_depth.jl")

Zygote.@nograd eye_like

function _get_transformation(rvec, t, invert)
    R = so3_exp_map(rvec)
    if invert
        R = permutedims(R, (2, 1, 3))
        t = R ⊠ -t
    end
    R, t
end

function warp(
    disparity, x, Ps, backproject, project, invK, K;
    min_depth, max_depth, source_ids,
)
    depth = disparity_to_depth(disparity, min_depth, max_depth)
    _, dw, dh, dn = size(depth)
    depth = reshape(depth, (1, dw * dh, dn))
    cam_coordinates = backproject(depth, invK)

    function _warp(wid)
        R, t = Ps[wid[1]]
        warped_uv = reshape(project(cam_coordinates, K, R, t), (2, dw, dh, dn))
        grid_sample(x[:, :, :, wid[2], :], warped_uv; padding_mode=:border)
    end
    map(_warp, enumerate(source_ids))
end

function photometric_loss(
    ssim, predicted::AbstractArray{T}, target::AbstractArray{T}; α = T(0.85),
) where T
    l1_loss = mean(abs.(target .- predicted); dims=3)
    ssim_loss = mean(ssim(predicted, target); dims=3)
    α .* ssim_loss .+ (one(T) - α) .* l1_loss
end

@inline automasking_loss(ssim, inputs, target; source_ids) =
    minimum(cat(map(i -> photometric_loss(ssim, inputs[:, :, :, i, :], target), source_ids)...; dims=3); dims=3)

@inline prediction_loss(ssim, predictions, target) =
    minimum(cat(map(p -> photometric_loss(ssim, p, target), predictions)...; dims=3); dims=3)

function train_loss(
    model, x::AbstractArray{T}, auto_loss, cache::TrainCache, parameters::Params,
    do_visualization,
) where T
    target_x = x[:, :, :, cache.target_pos_id, :]
    disparities, poses = model(x, cache.source_ids, cache.target_pos_id)

    # TODO pass as parameter to function
    inverse_transform = cache.source_ids .< cache.target_pos_id
    Ps = map(
        p -> _get_transformation(p[1].rvec, p[1].tvec, p[2]),
        zip(poses, inverse_transform))

    vis_warped, vis_loss, vis_disparity = nothing, nothing, nothing
    if do_visualization
        vis_disparity = cpu(disparities[end])
    end

    loss = zero(T)
    for (i, scale) in enumerate(cache.scales)
        disparity = disparities[i]
        if i != length(cache.scales)
            disparity = upsample_bilinear(disparity; size=parameters.target_size)
        end

        dw, dh, _, db = size(disparity)
        disparity = reshape(disparity, (1, dw, dh, db))
        warped_images = warp(
            disparity, x, Ps,
            cache.backprojections, cache.projections,
            cache.invKs, cache.Ks;
            min_depth=parameters.min_depth, max_depth=parameters.max_depth,
            source_ids=cache.source_ids)

        warp_loss = prediction_loss(cache.ssim, warped_images, target_x)
        if parameters.automasking
            warp_loss = minimum(cat(auto_loss, warp_loss; dims=3); dims=3)
        end
        loss += mean(warp_loss)

        normalized_disparity = reshape(disparity, size(disparity)[2:end])
        normalized_disparity = normalized_disparity ./
            (mean(normalized_disparity; dims=(1, 2)) .+ T(1e-7))
        disparity_loss =
            smooth_loss(normalized_disparity, target_x) .*
            T(parameters.disparity_smoothness) .* T(scale)
        loss += disparity_loss

        if do_visualization && i == length(cache.scales)
            vis_warped = cpu.(warped_images)
            vis_loss = cpu(warp_loss)
        end
    end

    loss / T(length(cache.scales)), vis_disparity, vis_warped, vis_loss
end

function save_disparity(disparity, path)
    disparity = permutedims(disparity, (2, 1))[end:-1:1, :]
    fig = heatmap(
        disparity; c=:thermal, aspect_ratio=:equal,
        colorbar=:none, legend=:none, grid=false, showaxis=false)
    png(fig, path)
end

function save_warped(warped, path)
    if ndims(warped) == 2
        warped = permutedims(warped, (2, 1))
    else
        warped = colorview(RGB, permutedims(warped, (3, 2, 1)))
    end
    save(path, warped)
end

function train()
    device = gpu
    precision = f32

    dtk_dir = "/home/pxl-th/projects/datasets/depth10k"
    log_dir = "/home/pxl-th/projects/Monodepth2.jl/logs"
    save_dir = "/home/pxl-th/projects/Monodepth2.jl/models"

    isdir(log_dir) || mkdir(log_dir)
    isdir(save_dir) || mkdir(save_dir)

    image_dir = joinpath(dtk_dir, "imgs")
    image_files = readlines(joinpath(dtk_dir, "trainable-nonstatic"))

    flip_augmentation = Sequential([FlipX(0.5), ToGray(0.2)])
    dataset = Depth10k(image_dir, image_files; flip_augmentation)
    parameters = Params(;
        batch_size=1, target_size=dataset.resolution,
        disparity_smoothness=1e-3, automasking=false)
    max_scale, scale_levels = 5, collect(2:5)
    scales = [1.0 / 2.0^(max_scale - slevel) for slevel in scale_levels]

    println(parameters)

    # Transfer to the device.
    transfer = device ∘ precision
    width, height = parameters.target_size
    projections = Project(; width, height) |> transfer
    backprojections = Backproject(; width, height) |> transfer
    Ks = Array(dataset.K) |> transfer
    invKs = inv(Array(dataset.K)) |> transfer
    ssim = SSIM() |> transfer

    train_cache = TrainCache(
        ssim, backprojections, projections, Ks, invKs,
        scales, dataset.source_ids, dataset.target_pos_id)

    encoder = ResidualNetwork(18; in_channels=3, classes=nothing)
    encoder_channels = collect(encoder.stages)
    depth_decoder = DepthDecoder(;encoder_channels, scale_levels)
    pose_decoder = PoseDecoder(encoder_channels[end])
    model = Model(encoder, depth_decoder, pose_decoder) |> transfer

    θ = params(model)
    optimizer = ADAM(3e-4)
    trainmode!(model) # TODO: switch to test mode once it is implemented

    n_epochs = 20
    log_iter, save_iter = 250, 500

    for epoch in 1:n_epochs
        loader = DataLoader(shuffleobs(dataset), parameters.batch_size)
        bar = get_pb(length(loader), "Epoch $epoch / $n_epochs: ")

        for (i, images) in enumerate(loader)
            x = precision(images)

            auto_loss = nothing
            if parameters.automasking
                auto_loss = automasking_loss(
                    train_cache.ssim, x, x[:, :, :, train_cache.target_pos_id, :];
                    source_ids=train_cache.source_ids) |> device
            end

            x = device(x)

            loss_cpu = 0.0
            disparity, warped, vis_loss = nothing, nothing, nothing
            do_visualization = i % log_iter == 0

            Flux.Optimise.update!(optimizer, θ, gradient(θ) do
                loss, disparity, warped, vis_loss = train_loss(
                    model, x, auto_loss, train_cache, parameters, do_visualization)
                loss_cpu = cpu(loss)
                loss
            end)

            if do_visualization
                save_disparity(disparity[:, :, 1, 1], joinpath(log_dir, "disp-$epoch-$i.png"))
                save(joinpath(log_dir, "loss-$epoch-$i.png"), permutedims(vis_loss[:, :, 1, 1], (2, 1)))

                # for l in 1:size(x, 4)
                #     xi = permutedims(cpu(x[:, :, :, l, 1]), (3, 2, 1))
                #     save(joinpath(log_dir, "x-$epoch-$i-$l.png"), colorview(RGB, xi))
                # end
                for sid in 1:length(warped)
                    save_warped(warped[sid][:, :, :, 1], joinpath(log_dir, "warp-$epoch-$i-$sid.png"))
                end
            end
            if i % save_iter == 0
                model_host = cpu(model)
                @save joinpath(save_dir, "$epoch-$i-$loss_cpu.bson") model_host
            end

            next!(bar; showvalues=[(:i, i), (:loss, loss_cpu)])
        end
    end
end

get_pb(n, desc::String) = Progress(
    n; desc, dt=1, barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:white)

function eval()
    dtk_dir = "/home/pxl-th/projects/datasets/depth10k"
    image_dir = joinpath(dtk_dir, "imgs")
    image_files = readlines(joinpath(dtk_dir, "trainable-nonstatic"))
    dataset = Depth10k(image_dir, image_files)

    model_path = "/home/pxl-th/projects/Monodepth2.jl/models/epoch-1-loss-0.10200599.bson"
    model = BSON.load(model_path, @__MODULE__)[:model_host]
    model = model |> testmode!

    x = dataset[1][:, :, :, [dataset.target_pos_id]]
    disparities = eval_disparity(model, x)
    save_disparity(disparities[end][:, :, 1, 1], "/home/pxl-th/d.png")
end

function refine_dtk()
    dtk_dir = "/home/pxl-th/projects/datasets/depth10k"
    image_dir = joinpath(dtk_dir, "imgs")
    image_files = readlines(joinpath(dtk_dir, "trainable"))
    dataset = Depth10k(image_dir, image_files)

    non_staic = find_static(dataset, 0.03)
    open(joinpath(dtk_dir, "trainable-nonstatic"), "w") do io
        for ns in non_staic
            write(io, ns, "\n")
        end
    end
end

train()
# eval()
# refine_dtk()
# simple_depth()

end
