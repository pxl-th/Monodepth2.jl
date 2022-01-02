struct DChain{D}
    datasets::D
    bins::Vector{Int64}
end

function DChain(datasets)
    lengths = length.(datasets)
    bins = cumsum(lengths)
    DChain(datasets, bins)
end

@inline Base.length(d::DChain) = d.bins[end]
function Base.getindex(d::DChain, i)
    bid = 1
    for b in d.bins
        if i > b
            bid += 1
        else
            break
        end
    end

    if bid > 1
        i -= d.bins[bid - 1]
    end
    d.datasets[bid][i]
end

@inline DataLoaders.nobs(d::DChain) = length(d)
@inline DataLoaders.getobs(d::DChain, i) = d[i]
