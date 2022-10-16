function obj(u::Vector{T},grad::Vector{T}) where T<:Real
    C = VarianceComponents2(K,log1pexp.(u))
    it = bcg_iterator!(C_,yb,maxiter=30)

    for (iteration, item) in enumerate(it)
    end
    ld = estimate_logdet(it)
end