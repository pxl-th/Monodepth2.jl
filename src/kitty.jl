struct KittyDataset{A}
    frames_dir::String

    K::SMatrix{3, 3, Float64, 9}
    resolution::Tuple{Int64, Int64}

    source_ids::Vector{Int64}
    target_id::Int64
    frame_ids::Vector{Int64}
    total_length::Int64

    augmentations::A
end

"""
- `target_size`: Size in `(height, width)` format.
"""
function KittyDataset(image_dir, sequence; target_size, augmentations = nothing)
    frames_dir = joinpath(image_dir, "sequences", sequence)
    Ks = readlines(joinpath(frames_dir, "calib.txt"))
    K = parse_matrix(Ks[1][5:end])

    frames_dir = joinpath(frames_dir, "image_0")
    n_frames, original_size = _get_seq_info(frames_dir)

    fx = mean(target_size ./ original_size) * K[1, 1]
    K = construct_intrinsic(fx, fx, target_size[2] ÷ 2, target_size[1] ÷ 2)

    target_id = 2
    source_ids = [1, 3]
    frame_ids = [1, 2, 3]
    total_length = n_frames ÷ length(frame_ids)

    height, width = target_size

    KittyDataset(
        frames_dir, K, (width, height),
        source_ids, target_id, frame_ids, total_length,
        augmentations)
end

@inline Base.length(dataset::KittyDataset) = dataset.total_length
function Base.getindex(d::KittyDataset, i)
    sid = (i - 1) * length(d.frame_ids)
    images = map(
        x -> load(joinpath(d.frames_dir, @sprintf("%.06d.png", sid + x - 1))),
        d.frame_ids)

    width, height = d.resolution
    images = map(x -> imresize(x, (height, width)), images)
    if d.augmentations ≢ nothing
        images = d.augmentations(images)
    end

    images = map(
        x -> Flux.unsqueeze(permutedims(Float32.(channelview(x)), (2, 1)), 3),
        images)
    cat(images...; dims=4)
end

@inline DataLoaders.nobs(dataset::KittyDataset) = length(dataset)
@inline DataLoaders.getobs(dataset::KittyDataset, i) = dataset[i]

function _get_seq_info(seq_dir::String)
    files = readdir(seq_dir; sort=false)
    n_frames = length(files)
    original_size = size(load(joinpath(seq_dir, files[begin])))
    n_frames, original_size
end

function parse_matrix(line)
    m = parse.(Float64, split(line, " "))
    K = SMatrix{4, 4, Float64}(m..., 0, 0, 0, 1)'
    K[1:3, 1:3]
end

@inline function construct_intrinsic(fx, fy, cx, cy)
    SMatrix{3, 3, Float64, 9}(
        fx, 0, 0,
        0, fy, 0,
        cx, cy, 1)
end
