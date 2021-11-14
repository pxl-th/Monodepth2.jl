module Monodepth

using StaticArrays
using FileIO
using ImageCore
using ImageTransformations
using Printf
using DataLoaders
using LinearAlgebra

using Statistics
using CUDA
using Zygote
using Flux
using EfficientNet
CUDA.allowscalar(false)

Zygote.@nograd CUDA.ones
Zygote.@nograd CUDA.zeros

include("kitty.jl")
include("utils.jl")
include("depth_decoder.jl")
include("pose_decoder.jl")
include("model.jl")

function photometric_loss(
    ssim, predicted::AbstractArray{T}, target::AbstractArray{T}; α = T(0.85),
) where T
    l1_loss = mean(abs.(target .- predicted); dims=3)
    ssim_loss = mean(ssim(predicted, target); dims=3)
    α .* ssim_loss .+ (T(1.0) - α) .* l1_loss
end

function automasking_loss(ssim, inputs; source_ids, target_pos_id)
    target = inputs[target_pos_id]
    minimum(cat(map(sid -> photometric_loss(ssim, inputs[sid], target), source_ids)...; dims=3); dims=3)
end

prediction_loss(ssim, target, predictions) =
    minimum(cat(map(p -> photometric_loss(ssim, p, target), predictions)...; dims=3); dims=3)

function generate_scale_image_predictions(
    backproject, project, inputs, disparity, K, invK, # should have the same scale
    Rs, ts; min_depth, max_depth, source_ids, target_pos_id,
)
    depth = disparity_to_depth(disparity, min_depth, max_depth)

    _, dw, dh, dn = size(depth)
    depth = reshape(depth, (1, dw * dh, dn))
    cam_coordinates = backproject(depth, invK)

    inv_depth = eltype(depth)(1.0) ./ depth
    mean_inv_depth = reshape(mean(inv_depth; dims=2), (1, dn))
    println("Mean inv depth $(mean_inv_depth)")

    function warp(i, sid)
        P = get_transformation(
            Rs[i][:, 1, :], ts[i][:, 1, :] .* mean_inv_depth, Val(sid < target_pos_id))
        projections = reshape(project(cam_coordinates, K, P), (2, dw, dh, dn))
        grid_sample(inputs[sid], projections; padding_mode=:zeros)
    end
    map(si -> warp(si[1], si[2]), enumerate(source_ids))
end

function generate_scale_image_predictions(
    backproject, project, inputs, disparity, K, invK, # should have the same scale
    P; min_depth, max_depth, source_ids,
)
    depth = disparity_to_depth(disparity, min_depth, max_depth)
    _, dw, dh, dn = size(depth)
    depth = reshape(depth, (1, dw * dh, dn))
    cam_coordinates = backproject(depth, invK)

    function warp(i, sid)
        projections = reshape(project(cam_coordinates, K, P[i]), (2, dw, dh, dn))
        grid_sample(inputs[sid], projections; padding_mode=:zeros)
    end
    map(si -> warp(si[1], si[2]), enumerate(source_ids))
end

function train_loss(
    model, x, projections, backprojections, ssim;
    Ks, invKs, source_ids, target_pos_id, seq_length,
    scales, scale_sizes, min_depth, max_depth, disparity_smoothness,
)
    T = eltype(x)
    loss = T(0.0)

    disparities, poses = model(x; source_ids, target_pos_id)
    Rs, ts = poses
    P = map(
        si -> get_transformation(Rs[si[1]][:, 1, :], ts[si[1]][:, 1, :], Val(si[2] < target_pos_id)),
        enumerate(source_ids))

    warped = nothing
    for (i, (scale, scale_size)) in enumerate(zip(scales, scale_sizes))
        scale_xs = map(s -> upsample_bilinear(x[:, :, :, s, :]; size=scale_size), 1:seq_length)

        disparity = disparities[i]
        dw, dh, _, db = size(disparity)
        disparity = reshape(disparity, (1, dw, dh, db))

        # scale_predictions = generate_scale_image_predictions(
        #     backprojections[i], projections[i], scale_xs, disparity,
        #     Ks[i], invKs[i], Rs, ts; min_depth, max_depth, source_ids, target_pos_id)
        scale_predictions = generate_scale_image_predictions(
            backprojections[i], projections[i], scale_xs, disparity,
            Ks[i], invKs[i], P; min_depth, max_depth, source_ids)
        warped = scale_predictions[end]

        auto_loss = automasking_loss(ssim, scale_xs; source_ids, target_pos_id)
        pred_loss = prediction_loss(ssim, scale_xs[target_pos_id], scale_predictions)
        warp_loss = minimum(cat(auto_loss, pred_loss; dims=3); dims=3)
        loss += mean(warp_loss)

        disparity = reshape(disparity, size(disparity)[2:end])
        disparity = disparity ./ (mean(disparity; dims=(1, 2)) .+ T(1e-7))
        disparity_loss = smooth_loss(disparity, scale_xs[target_pos_id]) .*
            T(disparity_smoothness) .* T(scale)
        loss += disparity_loss
    end
    loss / T(length(scales)), disparities, warped
end

function save_disparity(disparity, i)
    disparity = disparity[:, :, 1, 1]
    disparity = permutedims(disparity, (2, 1))
    println("Disp min/max: $(minimum(disparity)), $(maximum(disparity))")
    save("/home/pxl-th/projects/disp-$i.png", disparity)
end

function save_depth(disparity, i)
    depth = disparity_to_depth(disparity, 0.1, 100.0)
    depth = depth[:, :, 1, 1]
    depth = permutedims(depth, (2, 1)) ./ 100.0
    println("Depth min/max: $(minimum(depth)), $(maximum(depth))")
    save("/home/pxl-th/projects/depth-$i.png", depth)
end

function save_warped(warped, i)
    warped = warped[:, :, 1, 1]
    warped = permutedims(warped, (2, 1))
    save("/home/pxl-th/projects/warped-$i.png", warped)
end

function nn()
    device = cpu
    precision = f32

    disparity_smoothness = 1e-1
    min_depth, max_depth = 0.1, 100.0
    original_resolution = (376, 376)
    target_size = (192, 192) # in (height, width) format.

    batch_size = 2
    dataset = KittyDataset(
        "/home/pxl-th/Downloads/kitty-dataset", "00";
        original_resolution, target_size, frame_ids=[1, 2, 3], n_frames=4541)
    seq_length = length(dataset.frame_ids)
    loader = DataLoader(dataset, batch_size)
    target_pos_id = dataset.target_pos_id
    source_ids = dataset.source_ids

    max_scale = 5
    scale_levels = collect(2:5)

    # In (width, height) format.
    scales = Float64[]
    scale_sizes = Tuple{Int64, Int64}[]
    projections, backprojections = Project[], Backproject[]

    for scale_level in scale_levels
        scale = 1.0 / 2.0^(max_scale - scale_level)
        scale_size = ceil.(Int64, target_size[[2, 1]] .* scale)

        push!(scales, scale)
        push!(scale_sizes, scale_size)
        push!(projections, Project(Float32; width=scale_size[1], height=scale_size[2]))
        push!(backprojections, Backproject(Float32; width=scale_size[1], height=scale_size[2]))
    end
    Ks, invKs = scale_intrinsics(dataset, scales)

    # Transfer to the device.
    Ks, invKs = map(device ∘ precision ∘ Array, Ks), map(device ∘ precision ∘ Array, invKs)
    projections = map(device ∘ precision, projections)
    backprojections = map(device ∘ precision, backprojections)

    encoder = EffNet("efficientnet-b0"; include_head=false, in_channels=1)
    encoder_channels = collect(encoder.stages_channels)
    depth_decoder = DepthDecoder(;encoder_channels, scale_levels)
    pose_decoder = PoseDecoder(encoder_channels[end], 2, 1)
    model = Model(encoder, depth_decoder, pose_decoder) |> precision |> device

    ssim = SSIM() |> precision |> device
    θ = model |> params
    optimizer = ADAM(3e-4) |> precision

    # TODO separate pose cnn
    # TODO compute warp error by upsampling lover-level to top level

    i = 1
    for _ in 1:100, images in loader
        x = images |> precision |> device

        model |> trainmode!
        loss_cpu = 0.0
        disparity = nothing
        warped = nothing

        ∇ = gradient(θ) do
            loss, disparities, warped = train_loss(
                model, x, projections, backprojections, ssim;
                Ks, invKs, source_ids, target_pos_id, seq_length,
                scales, scale_sizes, min_depth, max_depth, disparity_smoothness)
            disparity = cpu(disparities[end])
            loss_cpu = loss |> cpu
            loss
        end
        Flux.Optimise.update!(optimizer, θ, ∇)

        println("$i | Loss: $loss_cpu")
        save_disparity(disparity, i)
        save_depth(disparity, i)
        save_warped(warped, i)
        i += 1
    end
end
nn()

end
