struct PoseDecoder{S, P1, P2, P3}
    squeezer::S
    p1::P1
    p2::P2
    p3::P3
    n_predictions::Int64
end
Flux.@functor PoseDecoder

function PoseDecoder(encoder_out_channels, n_input_features::Int, n_predictions::Int)
    squeezer = Conv((1, 1), encoder_out_channels=>256, relu)
    p1 = Conv((3, 3), (n_input_features * 256)=>256, relu; pad=1)
    p2 = Conv((3, 3), 256=>256, relu; pad=1)
    p3 = Conv((1, 1), 256=>(6 * n_predictions))
    PoseDecoder(squeezer, p1, p2, p3, n_predictions)
end

function (decoder::PoseDecoder)(features)
    squeezed = cat(map(decoder.squeezer, features)...; dims=3)
    pose = mean(decoder.p3(decoder.p2(decoder.p1(squeezed))); dims=(1, 2))

    shape = (6, decoder.n_predictions, size(pose, 4))
    pose = eltype(pose)(1e-2) .* reshape(pose, shape)
    pose[1:3, :, :], pose[4:6, :, :]
end
