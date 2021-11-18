struct DecoderBlock{D}
    decoder::D
end
Flux.@functor DecoderBlock
function DecoderBlock(in_channels, skip_channels, out_channels)
    DecoderBlock(Chain(
        Conv((3, 3), (in_channels + skip_channels)=>out_channels, elu; pad=1, bias=false),
        Conv((3, 3), out_channels=>out_channels, elu; pad=1, bias=false)))
end
function (block::DecoderBlock)(x, skip)
    o = upsample_bilinear(x, (2, 2))
    if skip ≢ nothing
        o = cat(o, skip; dims=3)
    end
    block.decoder(o)
end

struct DepthDecoder{P, S, D}
    partitions::P
    scale_convolutions_ids::Dict{Int64, Int64}
    scale_convolutions::Vector{S}
    decoder_blocks::D
end
Flux.@functor DepthDecoder
function DepthDecoder(;encoder_channels, scale_levels)
    if length(scale_levels) > 5 || minimum(scale_levels) < 1 || maximum(scale_levels) > 5
        error("`scale_levels` should be at most of length 5 and have values in [1, 5] range.")
    end

    decoder_channels = [256, 128, 64, 32, 16] # TODO parametrize
    encoder_channels = encoder_channels[end:-1:1]
    head_channels = encoder_channels[1]
    in_channels = [head_channels, decoder_channels[1:end - 1]...]
    skip_channels = [encoder_channels[2:end]..., 0]

    scale_convolutions_ids = Dict{Int64, Int64}(
        s => si for (si, s) in enumerate(scale_levels))
    scale_convolutions = [
        Conv((3, 3), decoder_channels[s]=>1, σ; pad=1, bias=false) for s in scale_levels]

    bstart = 1
    partitions = Tuple{Int64, Int64}[]
    for scale_level in scale_levels
        push!(partitions, (bstart, scale_level))
        bstart = scale_level + 1
    end
    partitions = tuple(partitions...)

    decoder_blocks = [
        DecoderBlock(inc, sc, oc)
        for (inc, sc, oc) in zip(in_channels, skip_channels, decoder_channels)]
    DepthDecoder(
        partitions, scale_convolutions_ids, scale_convolutions, decoder_blocks)
end

function (decoder::DepthDecoder)(features)
    features = features[end:-1:1]
    head, skips = features[1], features[2:end]
    x = head

    function runner(block_range)
        for i in (block_range[1]):(block_range[2])
            skip = nothing
            if i ≤ length(skips)
                skip = skips[i]
            end
            x = decoder.decoder_blocks[i](x, skip)
        end
        sid = decoder.scale_convolutions_ids[block_range[end]]
        decoder.scale_convolutions[sid](x)
    end
    map(runner, decoder.partitions)
end
