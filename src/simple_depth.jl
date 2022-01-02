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

function simple_depth()
    device = gpu
    precision = f32

    dtk_dir = "/home/pxl-th/projects/datasets/depth10k"
    log_dir = "/home/pxl-th/projects/Monodepth2.jl/logs"
    save_dir = "/home/pxl-th/projects/Monodepth2.jl/models"

    isdir(log_dir) || mkdir(log_dir)
    isdir(save_dir) || mkdir(save_dir)

    image_dir = joinpath(dtk_dir, "imgs")
    image_files = readlines(joinpath(dtk_dir, "trainable-nonstatic"))

    dataset = Depth10k(image_dir, image_files)
    width, height = dataset.resolution

    transfer = device ∘ precision

    projections = Project(; width, height) |> transfer
    backprojections = Backproject(; width, height) |> transfer
    Ks = Array(dataset.K) |> transfer
    invKs = inv(Array(dataset.K)) |> transfer
    ssim = SSIM() |> transfer

    k = 8
    println(dataset.files[k])

    x = Flux.unsqueeze(dataset[k], 5) |> transfer
    slow_depth(
        x, ssim, backprojections, projections, invKs, Ks, transfer;
        target_id=dataset.target_id, source_ids=dataset.source_ids,
        min_depth=0.1, max_depth=100.0, log_dir="/home/pxl-th/")
end
