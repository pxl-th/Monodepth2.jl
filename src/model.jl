struct Model{E, D, P}
    encoder::E
    depth_decoder::D
    pose_decoder::P
end
Flux.@functor Model

function (m::Model)(x, source_ids, target_id)
    w, h, c, l, n = size(x)
    x_flattened = reshape(x, (w, h, c, l * n))

    features = map(
        f -> reshape(f, (size(f, 1), size(f, 2), size(f, 3), l, n)),
        m.encoder(x_flattened, Val(:stages)))

    depth_decoder_x = map(f -> f[:, :, :, target_id, :], features)
    disparities = m.depth_decoder(depth_decoder_x)
    poses = eval_poses(m, features[end], source_ids, target_id)
    disparities, poses
end

function eval_poses(m::Model, features, source_ids, target_id)
    map(
        i -> m.pose_decoder(_get_pose_features(features, i, target_id)),
        source_ids)
end

eval_disparity(m::Model, x) = m.depth_decoder(m.encoder(x, Val(:stages)))

function _get_pose_features(features, i, target_id)
    if i < target_id
        return features[:, :, :, i, :], features[:, :, :, target_id, :]
    end
    features[:, :, :, target_id, :], features[:, :, :, i, :]
end
