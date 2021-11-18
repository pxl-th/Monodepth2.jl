struct KittyDataset
    K::SMatrix{3, 3, Float64, 9}
    frames_dir::String

    source_ids::Vector{Int64}
    target_pos_id::Int64

    target_id::Int64
    frame_ids::Vector{Int64}

    target_size::Tuple{Int64, Int64}
    sequence_length::Int64
    total_length::Int64
end

"""
- `target_size`: Size in `(height, width)` format.
"""
function KittyDataset(base_dir, sequence; original_resolution, target_size, frame_ids, n_frames)
    frames_dir = joinpath(base_dir, "sequences", sequence)
    Ks = readlines(joinpath(frames_dir, "calib.txt"))
    K = parse_matrix(Ks[1][5:end])
    K[1, 3] = K[2, 3]
    K = scale_intrinsic(K, mean(target_size ./ original_resolution))
    display(K); println()

    frames_dir = joinpath(frames_dir, "image_0")

    target_pos_id = ceil(Int, (length(frame_ids) + 1.0) / 2.0)
    source_ids = collect(setdiff(Set{Int64}(1:length(frame_ids)), target_pos_id))

    target_id = frame_ids[target_pos_id]
    sequence_length = maximum(frame_ids)
    total_length = n_frames ÷ sequence_length

    KittyDataset(
        K, frames_dir, source_ids, target_pos_id, target_id, frame_ids,
        target_size, sequence_length, total_length)
end

@inline Base.length(dataset::KittyDataset) = dataset.total_length
function Base.getindex(dataset::KittyDataset, i)
    start_id = (i - 1) * dataset.sequence_length
    images = cat([
        load_image(dataset, joinpath(
            dataset.frames_dir, @sprintf("%.06d.png", start_id + fid - 1)))
        for fid in dataset.frame_ids]...; dims=4)
    images
end

DataLoaders.nobs(dataset::KittyDataset) = length(dataset)
DataLoaders.getobs(dataset::KittyDataset, i) = dataset[i]

function parse_matrix(line)
    m = parse.(Float64, split(line, " "))
    K = MMatrix{4, 4, Float64}(m..., 0, 0, 0, 1)'
    K[1:3, 1:3]
end

function load_image(dataset::KittyDataset, image_path)
    image = load(image_path) |> channelview .|> Float32
    h, w = size(image)
    hp, wp = h ÷ 2, w ÷ 2
    image = image[:, (wp - hp):(wp + hp)]
    image = imresize(image, dataset.target_size)
    Flux.unsqueeze(permutedims(image, (2, 1)), 3)
end

function scale_intrinsic(K, scale = 1)
    fx, fy, cx, cy = K[1, 1], K[2, 2], K[1, 3], K[2, 3]
    MMatrix{3, 3, Float64, 9}(
        fx * scale, 0, 0,
        0, fy * scale, 0,
        cx * scale, cy * scale, 1)
end
