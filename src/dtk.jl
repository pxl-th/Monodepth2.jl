struct Depth10k{A}
    K::SMatrix{3, 3, Float64, 9}
    invK::SMatrix{3, 3, Float64, 9}

    dir::String
    files::Vector{String}
    resolution::Tuple{Int64, Int64} # width, height

    source_ids::Vector{Int64}
    target_id::Int64

    augmentations::A
    grayscale::Bool
end
function Depth10k(image_dir, image_files; augmentations = nothing, grayscale = false)
    focal = 2648.0 / 4.63461538462
    width, height = 416, 128
    K = SMatrix{3, 3, Float64, 9}(
        focal, 0, 0,
        0, focal, 0,
        width / 2.0, height / 2.0, 1)
    invK = inv(K)
    Depth10k(
        K, invK, image_dir, image_files, (width, height), [1, 3], 2,
        augmentations, grayscale)
end

Base.length(d::Depth10k) = length(d.files)
function Base.getindex(d::Depth10k, i)
    width = d.resolution[1]
    frames = load(joinpath(d.dir, d.files[i]))
    if d.grayscale
        frames = Gray{Float32}.(frames)
    end

    frames = [frames[:, (width * j + 1):(width * (j + 1))] for j in 0:2]
    if d.augmentations ≢ nothing
        frames = d.augmentations(frames)
    end
    frames = channelview.(frames)
    if d.grayscale
        frames = Flux.unsqueeze.(frames, 1)
    end
    frames = cat(frames...; dims=4)
    Float32.(permutedims(frames, (3, 2, 1, 4)))
end

@inline DataLoaders.nobs(d::Depth10k) = length(d.files)
@inline DataLoaders.getobs(d::Depth10k, i) = d[i]

function find_static(dataset::Depth10k, α)
    ssim = SSIM()
    non_static = String[]
    bar = get_pb(length(dataset), "Detecting static: ")
    for i in 1:length(dataset)
        x = dataset[i]
        x = Flux.unsqueeze(x, 5)

        auto_loss = mean(automasking_loss(
            ssim, x, x[:, :, :, dataset.target_id, :];
            source_ids=dataset.source_ids))
        if auto_loss > α
            push!(non_static, dataset.files[i])
        end
        next!(bar; showvalues=[
            (:loss, auto_loss), (:non_static, length(non_static))])
    end
    non_static
end

