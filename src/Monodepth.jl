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
using VideoIO
using ProgressMeter
using Plots
using StaticArrays
gr()

import ChainRulesCore: rrule
using ChainRulesCore
using Zygote
using CUDA
using Flux
using ResNet

import Random
Random.seed!(42)

Zygote.@nograd CUDA.ones
Zygote.@nograd CUDA.zeros

CUDA.allowscalar(false)

Base.@kwdef struct Params
    min_depth::Float64 = 0.1
    max_depth::Float64 = 100.0
    disparity_smoothness::Float64 = 1e-3
    frame_ids::Vector{Int64} = [1, 2, 3]

    automasking::Bool = true

    target_size::Tuple{Int64, Int64} # (width, height)
    batch_size::Int64
end

struct TrainCache{S, B, P, I}
    ssim::S
    backprojections::B
    projections::P

    K::I
    invK::I

    target_id::Int64
    source_ids::Vector{Int64}
    scales::Vector{Float64}
end

include("dtk.jl")
include("kitty.jl")
include("dchain.jl")

include("io_utils.jl")
include("utils.jl")
include("depth_decoder.jl")
include("pose_decoder.jl")
include("model.jl")
include("simple_depth.jl")

include("training.jl")

function train()
    device = gpu
    precision = f32
    transfer = device ∘ precision

    log_dir = "/home/pxl-th/projects/Monodepth2.jl/logs"
    save_dir = "/home/pxl-th/projects/Monodepth2.jl/models"

    isdir(log_dir) || mkdir(log_dir)
    isdir(save_dir) || mkdir(save_dir)

    grayscale = true
    in_channels = grayscale ? 1 : 3
    augmentations = FlipX(0.5)
    target_size=(128, 416)

    kitty_dir = "/home/pxl-th/projects/datasets/kitty-dataset"
    datasets = [
        KittyDataset(kitty_dir, s; target_size, augmentations)
        for s in map(i -> @sprintf("%02d", i), 0:21)]

    # dtk_dir = "/home/pxl-th/projects/datasets/depth10k"
    # dtk_dataset = Depth10k(
    #     joinpath(dtk_dir, "imgs"),
    #     readlines(joinpath(dtk_dir, "trainable-nonstatic"));
    #     augmentations, grayscale)
    # push!(datasets, dtk_dataset)

    dchain = DChain(datasets)
    dataset = datasets[begin]

    width, height = dataset.resolution
    parameters = Params(;
        batch_size=4, target_size=dataset.resolution,
        disparity_smoothness=1e-3, automasking=false)
    max_scale, scale_levels = 5, collect(2:5)
    scales = [1.0 / 2.0^(max_scale - level) for level in scale_levels]
    println(parameters)

    train_cache = TrainCache(
        transfer(SSIM()),
        transfer(Backproject(; width, height)),
        transfer(Project(; width, height)),
        transfer(Array(dataset.K)), transfer(Array(dataset.invK)),
        dataset.target_id, dataset.source_ids, scales)

    encoder = ResidualNetwork(18; in_channels, classes=nothing)
    encoder_channels = collect(encoder.stages)
    model = transfer(Model(
        encoder,
        DepthDecoder(;encoder_channels, scale_levels),
        PoseDecoder(encoder_channels[end])))

    θ = Flux.params(model)
    optimizer = ADAM(1e-4)
    trainmode!(model)

    # Perform first gradient computation using small batch size.
    println("Precompile grads...")
    for x_host in DataLoader(dchain, 1)
        x = transfer(x_host)

        println("Forward timing:")
        @time train_loss(model, x, nothing, train_cache, parameters, false)[1]

        println("Backward timing:")
        @time begin
            ∇ = gradient(θ) do
                train_loss(model, x, nothing, train_cache, parameters, false)[1]
            end
        end

        println(mean(∇[model.pose_decoder.pose[end].weight]))
        break
    end
    GC.gc()

    # Do regular training.
    n_epochs, log_iter, save_iter = 20, 50, 500

    println("Training...")
    for epoch in 1:n_epochs
        loader = DataLoader(shuffleobs(dchain), parameters.batch_size)
        bar = get_pb(length(loader), "Epoch $epoch / $n_epochs: ")

        for (i, x_host) in enumerate(loader)
            x = transfer(x_host)

            auto_loss = nothing
            if parameters.automasking
                auto_loss = automasking_loss(
                    train_cache.ssim, x, x[:, :, :, train_cache.target_id, :];
                    source_ids=train_cache.source_ids)
            end

            loss_cpu = 0.0
            disparity, warped, vis_loss = nothing, nothing, nothing
            do_visualization = i % log_iter == 0 || i == 1

            Flux.Optimise.update!(optimizer, θ, gradient(θ) do
                loss, disparity, warped, vis_loss = train_loss(
                    model, x, auto_loss, train_cache,
                    parameters, do_visualization)
                loss_cpu = cpu(loss)
                loss
            end)

            if do_visualization
                save_disparity(
                    disparity[:, :, 1, 1],
                    joinpath(log_dir, "disp-$epoch-$i.png"))
                # save(
                #     joinpath(log_dir, "loss-$epoch-$i.png"),
                #     permutedims(vis_loss[:, :, 1, 1], (2, 1)))
                for sid in 1:length(warped)
                    save_warped(
                        warped[sid][:, :, :, 1],
                        joinpath(log_dir, "warp-$epoch-$i-$sid.png"))
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

function eval_image()
    device = gpu
    precision = f32
    target_resolution = (128, 416)

    image_dir = "/home/pxl-th/projects/datasets/kitty-dataset/sequences/00/image_0"
    model_path = "/home/pxl-th/projects/Monodepth2.jl/models/9-2500-0.12209008.bson"
    model = BSON.load(model_path, @__MODULE__)[:model_host]
    model = testmode!(device(precision(model)))

    bar = get_pb(4541, "Inference: ")
    for i in 0:4540
        image_path = joinpath(image_dir, @sprintf("%.06d.png", i))

        x = load(image_path)
        x = imresize(Gray{Float32}.(x), target_resolution)

        x = Float32.(channelview(x))
        x = permutedims(Flux.unsqueeze(x, 1), (3, 2, 1))
        x = device(Flux.unsqueeze(x, 4))

        disparity = cpu(eval_disparity(model, x)[end])
        save_disparity(disparity[:, :, 1, 1], "/home/pxl-th/d-$i.png")

        next!(bar)
    end
end

function eval_video()
    device = gpu
    target_resolution = (128, 416) # depth10k resolution
    precision = f32

    video_path = "/home/pxl-th/projects/datasets/calib_challenge/labeled/4.hevc"
    model_path = "/home/pxl-th/projects/Monodepth2.jl/models/6-500-0.026155949.bson"
    model = BSON.load(model_path, @__MODULE__)[:model_host]
    model = testmode!(device(precision(model)))

    for (i, frame) in enumerate(VideoIO.openvideo(video_path))
        x = imresize(Gray{Float32}.(frame), target_resolution)
        x = Float32.(channelview(x))
        x = permutedims(Flux.unsqueeze(x, 1), (3, 2, 1))
        x = device(Flux.unsqueeze(x, 4))

        disparity = cpu(eval_disparity(model, x)[end])
        save_disparity(disparity[:, :, 1, 1], "/home/pxl-th/d-$i.png")
    end
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

# train()
# simple_depth()
# eval_video()
# eval_image()

end
