struct Depth10k{A}
    K::SMatrix{3, 3, Float64, 9}
    dir::String
    files::Vector{String}
    resolution::Tuple{Int64, Int64}

    source_ids::Vector{Int64}
    target_pos_id::Int64

    flip_augmentation::A
end
function Depth10k(image_dir, image_files; flip_augmentation = nothing)
    focal = 2648.0 / 4.63461538462
    width, height = 416, 128
    K = SMatrix{3, 3, Float64, 9}(
        focal, 0, 0,
        0, focal, 0,
        width / 2.0, height / 2.0, 1)
    Depth10k(K, image_dir, image_files, (width, height), [1, 3], 2, flip_augmentation)
end

Base.length(d::Depth10k) = length(d.files)
function Base.getindex(d::Depth10k, i)
    width = d.resolution[1]
    frames = joinpath(d.dir, d.files[i]) |> load

    frames = [frames[:, (width * j + 1):(width * (j + 1))] for j in 0:2]
    if d.flip_augmentation ≢ nothing
        frames = d.flip_augmentation(frames)
    end
    frames = cat(channelview.(frames)...; dims=4)
    Float32.(permutedims(frames, (3, 2, 1, 4)))
end

@inline DataLoaders.nobs(d::Depth10k) = length(d.files)
@inline DataLoaders.getobs(d::Depth10k, i) = d[i]

function find_static(dataset::Depth10k, α)
    ssim = SSIM()
    non_static = String[]
    bar = get_pb(length(dataset), "Detecting static: ")
    for i in 1:length(dataset)
        x = dataset[i]
        x = Flux.unsqueeze(x, 5)

        auto_loss = mean(automasking_loss(
            ssim, x, x[:, :, :, dataset.target_pos_id, :];
            source_ids=dataset.source_ids))
        if auto_loss > α
            push!(non_static, dataset.files[i])
        end
        next!(bar; showvalues=[
            (:loss, auto_loss), (:non_static, length(non_static))])
    end
    non_static
end

struct Pose{R, T}
    rvec::R
    tvec::T
end
Flux.@functor Pose

function slow_depth(
    x, ssim, backprojections, projections, invKs, Ks, transfer;
    target_id, source_ids, min_depth, max_depth, log_dir,
)
    T = eltype(x)
    width, height = size(x)[1:2]

    disp = transfer(fill(T(0.5), (1, width, height, 1)))
    poses = [
        transfer(Pose(
            reshape(Float32[0, 0, 0.01], (3, 1)),
            zeros(T, (3, 1, 1))))
        for _ in 1:length(source_ids)]
    θ = params(disp, poses)

    optimizer = ADAM(3e-4)
    target_x = x[:, :, :, target_id, :]
    inverse_transform = source_ids .< target_id

    vw = nothing
    log_step = 5
    for iter in 1:500
        do_visualization = iter % log_step == 0 || iter == 1

        ∇ = gradient(θ) do
            Ps = map(
                p -> _get_transformation(p[1].rvec, p[1].tvec, p[2]),
                zip(poses, inverse_transform))

            warped_images = warp(
                disp, x, Ps, backprojections, projections, invKs, Ks;
                min_depth, max_depth, source_ids)

            if do_visualization
                vw = cpu.(warped_images)
            end

            warp_loss = mean(prediction_loss(ssim, warped_images, target_x))
            depth_loss = smooth_loss(reshape(disp, (width, height, 1)), target_x)
            warp_loss + depth_loss
        end
        Flux.Optimise.update!(optimizer, θ, ∇)

        if do_visualization
            save_disparity(
                reshape(cpu(disp), (width, height)),
                joinpath(log_dir, "d-$iter.png"))
            # save_warped(
            #     vw[1][:, :, :, 1],
            #     joinpath(log_dir, "w1-$iter.png"))
            # save_warped(
            #     vw[2][:, :, :, 1],
            #     joinpath(log_dir, "w2-$iter.png"))

            println(iter, " ", mean(disp))
            p1 = cpu(poses[1])
            @show p1.rvec, p1.tvec
            p2 = cpu(poses[2])
            @show p2.rvec, p2.tvec
        end
    end
end
