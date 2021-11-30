struct PoseDecoder{S, P, R, T}
    squeezer::S
    pose::P
    rotation::R
    translation::T
    n_predictions::Int64
end
Flux.@functor PoseDecoder
function PoseDecoder(encoder_out_channels, n_input_features::Int, n_predictions::Int)
    squeezer = Conv((1, 1), encoder_out_channels=>256, relu)
    pose = Chain(
        Conv((3, 3), (n_input_features * 256)=>256, relu; pad=1),
        Conv((3, 3), 256=>256, relu; pad=1))
    rotation = Conv((1, 1), 256=>(3 * n_predictions))
    translation = Conv((1, 1), 256=>(3 * n_predictions))
    PoseDecoder(squeezer, pose, rotation, translation, n_predictions)
end

function (decoder::PoseDecoder)(features)
    squeezed = cat(map(decoder.squeezer, features)...; dims=3)
    pose = decoder.pose(squeezed)
    rotation = mean(decoder.rotation(pose); dims=(1, 2))
    translation = mean(decoder.translation(pose); dims=(1, 2))

    ϵ = eltype(rotation)(1e-2)
    shape = (3, decoder.n_predictions, size(rotation, 4))
    ϵ .* reshape(rotation, shape), ϵ .* reshape(translation, shape)
end
