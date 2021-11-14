struct Model{E, D, P}
    encoder::E
    depth_decoder::D
    pose_decoder::P
end
Flux.@functor Model

function (m::Model)(images; source_ids, target_pos_id)
    w, h, c, l, n = size(images)
    images = reshape(images, (w, h, c, l * n))

    features = map(
        f -> reshape(f, (size(f, 1), size(f, 2), size(f, 3), l, n)),
        m.encoder(images, Val(:stages)))

    disparities = m.depth_decoder(map(f -> f[:, :, :, target_pos_id, :], features))
    poses = map(sid -> m.pose_decoder(get_pose_features(features[end], sid, target_pos_id)), source_ids)
    disparities, poses
end

function get_pose_features(features, i, target_pos_id)
    if i < target_pos_id
        return features[:, :, :, i, :], features[:, :, :, target_pos_id, :]
    end
    features[:, :, :, target_pos_id, :], features[:, :, :, i, :]
end
