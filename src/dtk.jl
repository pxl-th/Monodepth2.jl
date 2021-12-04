struct Depth10k
    K::SMatrix{3, 3, Float64, 9}
    dir::String
    files::Vector{String}
    resolution::Tuple{Int64, Int64}

    source_ids::Vector{Int64}
    target_pos_id::Int64
end
function Depth10k(image_dir, image_files)
    focal = 2648.0 / 4.63461538462
    width, height = 416, 128
    K = SMatrix{3, 3, Float64, 9}(
        focal, 0, 0,
        0, focal, 0,
        width / 2.0, height / 2.0, 1)
    Depth10k(K, image_dir, image_files, (width, height), [1, 3], 2)
end

Base.length(d::Depth10k) = length(d.files)
function Base.getindex(d::Depth10k, i)
    width = d.resolution[1]
    frames = joinpath(d.dir, d.files[i]) |> load |> channelview .|> Float32
    frames = permutedims(frames, (3, 2, 1))
    cat([frames[(width * j + 1):(width * (j + 1)), :, :] for j in 0:2]...; dims=4)
end

@inline DataLoaders.nobs(d::Depth10k) = length(d.files)
@inline DataLoaders.getobs(d::Depth10k, i) = d[i]

function find_static(dataset::Depth10k)
    ssim = SSIM()
    non_static = String[]
    for i in 1:length(dataset)
        x = dataset[i]
        xs = map(i -> x[:, :, :, [i]], 1:(length(dataset.source_ids) + 1))
        auto_loss = mean(automasking_loss(
            ssim, xs, xs[dataset.target_pos_id]; source_ids=dataset.source_ids))
        if auto_loss > 0.02
            println(i, " - ", auto_loss)
            push!(non_static, dataset.files[i])
        end
    end
    @info "Non-static amount: $(length(non_static))"
    non_static
end
