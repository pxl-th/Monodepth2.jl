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
    target_x = x[:, :, :, cache.target_id, :]
    disparities, poses = model(x, cache.source_ids, cache.target_id)

    # TODO pass as parameter to function
    inverse_transform = cache.source_ids .< cache.target_id
    Ps = map(
        p -> composeT(p[1].rvec, p[1].tvec, p[2]),
        zip(poses, inverse_transform))

    vis_warped, vis_loss, vis_disparity = nothing, nothing, nothing
    if do_visualization
        vis_disparity = cpu(disparities[end])
    end

    loss = zero(T)
    width, height = parameters.target_size

    for (i, (disparity, scale)) in enumerate(zip(disparities, cache.scales))
        dw, dh, _, dn = size(disparity)
        if dw != width || dh != height
            disparity = upsample_bilinear(disparity; size=(width, height))
        end

        depth = disparity_to_depth(
            disparity, parameters.min_depth, parameters.max_depth)
        coordinates = cache.backprojections(
            reshape(depth, (1, width * height, dn)), cache.invK)
        warped_images = map(zip(Ps, cache.source_ids)) do t
            uvs = reshape(
                cache.projections(coordinates, cache.K, t[1]...),
                (2, width, height, dn))
            grid_sample(x[:, :, :, t[2], :], uvs; padding_mode=:border)
        end

        warp_loss = prediction_loss(cache.ssim, warped_images, target_x)
        if parameters.automasking
            warp_loss = minimum(cat(auto_loss, warp_loss; dims=3); dims=3)
        end

        normalized_disparity = (
            disparity ./ (mean(disparity; dims=(1, 2)) .+ T(1e-7)))[:, :, 1, :]
        disparity_loss = smooth_loss(normalized_disparity, target_x) .*
            T(parameters.disparity_smoothness) .* T(scale)

        loss += mean(warp_loss) + disparity_loss

        if do_visualization && i == length(cache.scales)
            vis_warped = cpu.(warped_images)
            vis_loss = cpu(warp_loss)
        end
    end

    loss / T(length(cache.scales)), vis_disparity, vis_warped, vis_loss
end
