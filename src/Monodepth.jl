module Monodepth

using LinearAlgebra
using Printf
using Statistics

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

using Rotations
import ChainRulesCore: rrule
using ChainRulesCore
using Zygote
using CUDA
using Flux
using EfficientNet
using ResNet

CUDA.allowscalar(false)

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

    automasking::Bool = false

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

function photometric_loss(
    ssim, predicted::AbstractArray{T}, target::AbstractArray{T}; α = T(0.85),
) where T
    l1_loss = mean(abs.(target .- predicted); dims=3)
    ssim_loss = mean(ssim(predicted, target); dims=3)
    α .* ssim_loss .+ (one(T) - α) .* l1_loss
end

@inline automasking_loss(ssim, inputs, target; source_ids) =
    minimum(cat(map(sid -> photometric_loss(ssim, inputs[sid], target), source_ids)...; dims=3); dims=3)

@inline prediction_loss(ssim, predictions, target) =
    minimum(cat(map(p -> photometric_loss(ssim, p, target), predictions)...; dims=3); dims=3)

function train_loss(model, x::AbstractArray{T}, train_cache::TrainCache, parameters::Params) where T
    # TODO get rid of xs? use only x
    xs = map(i -> x[:, :, :, i, :], 1:length(parameters.frame_ids))
    disparities, poses = model(
        x; source_ids=train_cache.source_ids,
        target_pos_id=train_cache.target_pos_id)
    rvecs, tvecs = poses # 3, 1, N

    Ps = map(si -> _get_transformation(
        rvecs[si[1]][:, 1, :], tvecs[si[1]],
        si[2] < train_cache.target_pos_id), enumerate(train_cache.source_ids))

    vis_warped, vis_loss = nothing, nothing

    loss = T(0.0)
    for (i, scale) in enumerate(train_cache.scales)
        disparity = disparities[i]
        if i != length(train_cache.scales)
            disparity = upsample_bilinear(disparity; size=parameters.target_size)
        end

        dw, dh, _, db = size(disparity)
        disparity = reshape(disparity, (1, dw, dh, db))
        warped_images = warp(
            disparity, xs, Ps,
            train_cache.backprojections, train_cache.projections,
            train_cache.invKs, train_cache.Ks;
            min_depth=parameters.min_depth,
            max_depth=parameters.max_depth,
            source_ids=train_cache.source_ids)

        warp_loss = prediction_loss(
            train_cache.ssim, warped_images, xs[train_cache.target_pos_id])
        if parameters.automasking
            auto_loss = automasking_loss(
                train_cache.ssim, xs, xs[train_cache.target_pos_id];
                source_ids=train_cache.source_ids)
            warp_loss = minimum(cat(auto_loss, warp_loss; dims=3); dims=3)
        end
        loss += mean(warp_loss)

        disparity = reshape(disparity, size(disparity)[2:end])
        disparity = disparity ./ (mean(disparity; dims=(1, 2)) .+ T(1e-7))
        disparity_loss =
            smooth_loss(disparity, xs[train_cache.target_pos_id]) .*
            T(parameters.disparity_smoothness) .* T(scale)
        loss += disparity_loss

        # Visualization.
        if i == length(train_cache.scales)
            vis_warped = cpu.(warped_images)
            vis_loss = cpu(warp_loss)
        end
    end

    loss / T(length(train_cache.scales)), cpu(disparities[end]), vis_warped, vis_loss
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
    device = cpu
    precision = f32

    dtk_dir = "/home/pxl-th/projects/depth10k"
    log_dir = "/home/pxl-th/projects/Monodepth.jl/logs"
    save_dir = "/home/pxl-th/projects/Monodepth.jl/models"

    isdir(log_dir) || mkdir(log_dir)
    isdir(save_dir) || mkdir(save_dir)

    image_dir = joinpath(dtk_dir, "imgs")
    image_files = readlines(joinpath(dtk_dir, "trainable-nonstatic"))

    dataset = Depth10k(image_dir, image_files)
    parameters = Params(;
        batch_size=3, target_size=dataset.resolution,
        disparity_smoothness=1e-3, automasking=true)
    max_scale, scale_levels = 5, collect(2:5)
    scales = [1.0 / 2.0^(max_scale - slevel) for slevel in scale_levels]

    display(parameters); println()

    # Transfer to the device.
    projections = device(precision(Project(;
        width=parameters.target_size[1], height=parameters.target_size[2])))
    backprojections = device(precision(Backproject(;
        width=parameters.target_size[1], height=parameters.target_size[2])))
    Ks = device(precision(Array(dataset.K)))
    invKs = device(precision(inv(Array(dataset.K))))
    ssim = device(precision(SSIM()))

    train_cache = TrainCache(
        ssim, backprojections, projections, Ks, invKs,
        scales, dataset.source_ids, dataset.target_pos_id)

    encoder = ResidualNetwork(18; in_channels=3, classes=nothing)
    encoder_channels = collect(encoder.stages)
    # encoder = EffNet("efficientnet-b0"; include_head=false, in_channels=3)
    # encoder_channels = collect(encoder.stages_channels)
    depth_decoder = DepthDecoder(;encoder_channels, scale_levels)
    pose_decoder = PoseDecoder(encoder_channels[end], 2, 1)
    model = device(precision(Model(encoder, depth_decoder, pose_decoder)))

    θ = params(model)
    optimizer = ADAM(1e-4)
    trainmode!(model) # TODO: switch to test mode once it is implemented

    n_epochs = 20
    log_iter, save_iter = 11, 31

    for epoch in 1:n_epochs
        loader = DataLoader(shuffleobs(dataset), parameters.batch_size)
        bar = get_pb(length(loader), "Epoch $epoch / $n_epochs: ")

        for (i, images) in enumerate(loader)
            x = device(precision(images))

            loss_cpu = 0.0
            disparity, warped, vis_loss = nothing, nothing, nothing

            ∇ = gradient(θ) do
                loss, disparity, warped, vis_loss = train_loss(model, x, train_cache, parameters)
                loss_cpu = cpu(loss)
                loss
            end
            Flux.Optimise.update!(optimizer, θ, ∇)

            if i % log_iter == 0
                save_disparity(disparity[:, :, 1, 1], joinpath(log_dir, "disp-$epoch-$i.png"))
                save(joinpath(log_dir, "loss-$epoch-$i.png"), permutedims(vis_loss[:, :, 1, 1], (2, 1)))

                for l in 1:size(x, 4)
                    xi = permutedims(cpu(x[:, :, :, l, 1]), (3, 2, 1))
                    save(joinpath(log_dir, "x-$epoch-$i-$l.png"), colorview(RGB, xi))
                end
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
    model_path = "/home/pxl-th/projects/Monodepth.jl/models/epoch-1-loss-0.10200599.bson"
    model = BSON.load(model_path, @__MODULE__)[:model_host]

    dataset = Depth10k("/home/pxl-th/projects/depth10k/imgs")
    x = dataset[1][:, :, :, [dataset.target_pos_id]]
    disparities = eval_disparity(model, x)
    save_disparity(disparities[end][:, :, 1, 1], "/home/pxl-th/d.png")
end

function refine_dtk()
    image_dir = "/home/pxl-th/projects/depth10k/imgs"
    image_files = readlines("/home/pxl-th/projects/depth10k/trainable")
    dataset = Depth10k(image_dir, image_files)

    non_staic = find_static(dataset)
    open("trainable-nonstatic", "w") do io
        for ns in non_staic
            write(io, ns, "\n")
        end
    end
end

train()
# eval()
# refine_dtk()

end
