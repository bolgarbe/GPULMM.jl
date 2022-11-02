struct BCGIterable{T<:Real,
        DMatT<:AbstractArray{T,2},
        VCT<:VarianceComponents{T,DMatT},
        CMatT<:Array{T,2},
        DTsrT<:AbstractArray{T,3}}
    A::VCT
    b::DMatT
    x::DMatT
    r::DMatT
    c::DMatT
    u::DMatT
    reltol::T
    maxiter::Int
    α_::DMatT
    β_::DMatT
    α::CMatT
    β::CMatT
    Q::DTsrT
    residual::DMatT
    prev_residual::DMatT
end

@inline converged(it::BCGIterable) = all(@. sqrt(it.residual) ≤ it.reltol)
@inline start(it::BCGIterable) = 0
@inline done(it::BCGIterable, iteration::Int) = iteration ≥ it.maxiter  || converged(it)

function iterate(it::BCGIterable, iteration::Int=start(it))
    if done(it, iteration) return nothing end

    mul!(it.c,it.A,it.u)
    it.α_   .= it.residual ./ sum(it.u .* it.c,dims=1)
    @. it.x += it.α_*it.u
    @. it.r -= it.α_*it.c
    
    it.r .-= batched_vec(it.Q,batched_vec(batched_transpose(it.Q),it.r))

    @. it.prev_residual = it.residual
    it.residual .= sum(it.r.^2,dims=1)
    
    @. it.Q[:,iteration+1,:] = it.r / sqrt(it.residual)

    @. it.β_ = it.residual / it.prev_residual
    @. it.u  = it.r+it.β_*it.u

    it.α[iteration+1,:] .= Array(it.α_)[:]
    it.β[iteration+1,:] .= Array(it.β_)[:]
    
    sqrt.(it.residual), iteration+1
end

function reset!(it::BCGIterable)
    @. it.x = 0
    @. it.c = 0
    copyto!(it.r,it.b)
    copyto!(it.u,it.b)
    it.residual .= sum(it.r.^2,dims=1)
    @. it.prev_residual = 1
    @. it.α = 0
    @. it.β = 0
    @. it.Q = 0
end

function _Q_tensor(b::CuArray{T,2},maxiter::Int) where {T<:Real}
    CuArray{T,3}(undef,size(b,1),maxiter,size(b,2))
end

function _Q_tensor(b::Array{T,2},maxiter::Int) where {T<:Real}
    zeros(T,size(b,1),maxiter,size(b,2))
end

function BCGIterable(A::VarianceComponents{T,MatT}, b::MatT;
    tol = sqrt(eps(T)),
    maxiter::Int = size(A, 2),
) where {T<:Real, MatT <: AbstractArray{T}}
    x,c,u,r = zero(b),zero(b),similar(b),similar(b)
    copyto!(r,b)
    copyto!(u,b)

    residual      = sum(r.^2,dims=1)
    prev_residual = one.(residual)
    
    α_,β_ = similar(residual),similar(residual)
    α,β   = zeros(T,maxiter,size(b,2)),zeros(T,maxiter,size(b,2))
    Q     = _Q_tensor(b,maxiter)

    return BCGIterable(A,b,x,r,c,u,
        tol,maxiter,α_,β_,
        α,β,Q,
        residual,prev_residual
    )
end

function estimate_logdet(α::MatT,β::MatT,N::Int) where {T<:Real, MatT<:AbstractArray{T}}
    td = 1 ./ α
    td[2:end,:] .+= (β./α)[1:end-1,:]
    tl = (sqrt.(β)./α)[1:end-1,:]
    ld_est = 0.
    num_est = 0
    for i in axes(td,2)
        TD = SymTridiagonal(td[:,i],tl[:,i])
        Te,Tv = eigen(TD)
        ld_est += dot(log.(Te),Tv[1,:].^2)
        num_est += 1
    end
    ld_est * N/num_est
end