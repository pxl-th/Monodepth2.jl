struct Pose{R, T}
    rvec::R # 3xN - so3 rotation
    tvec::T # 3x1xN - translation
end
Flux.@functor Pose

struct PoseDecoder{S, P}
    squeezer::S
    pose::P
end
Flux.@functor PoseDecoder

function PoseDecoder(encoder_out_channels)
    n_input_features = 2
    squeezer = Conv((1, 1), encoder_out_channels=>256, relu)
    pose = Chain(
        Conv((3, 3), (n_input_features * 256)=>256, relu; pad=1),
        Conv((3, 3), 256=>256, relu; pad=1),
        Conv((1, 1), 256=>6))
    PoseDecoder(squeezer, pose)
end

function _squeeze(squeezer, features::NTuple{N, T})::T where {N, T}
    cat(map(squeezer, features)...; dims=3)
end

function (decoder::PoseDecoder)(features::NTuple{N, T}) where {N, T}
    squeezed = _squeeze(decoder.squeezer, features)
    pose = mean(decoder.pose(squeezed); dims=(1, 2))
    pose = eltype(T)(1e-2) .* reshape(pose, (6, 1, size(pose, 4)))
    Pose(pose[1:3, 1, :], pose[4:6, :, :])
end
