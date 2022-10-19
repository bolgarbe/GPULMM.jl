struct Objective{T<:Real,DVecT<:AbstractArray{T,1},DMatT<:AbstractArray{T,2}}
    it::BCGIterable{T,DMatT,VarianceComponents2{T,DMatT}}
    Ksol::DVecT
    Ksmp::DMatT
end

function Objective(K::Array{T,2},y::Array{T,1}; num_samples=30, maxiter=30) where {T<:Real}
    C  = VarianceComponents2(K,log1pexp.(T[0,0]))
    b  = randn(T,size(K,1),num_samples)
    yb = hcat(y,b)
    it = BCGIterable(C,yb,maxiter=maxiter)
    
    Ksol = similar(y)
    Ksmp = similar(b)
    Objective(it,Ksol,Ksmp)
end

function Objective(K::CuArray{T,2},y::CuArray{T,1}; num_samples=30, maxiter=30) where {T<:Real}
    C  = VarianceComponents2(K,log1pexp.(T[0,0]))
    b  = CUDA.randn(T,size(K,1),num_samples)
    yb = hcat(y,b)
    it = BCGIterable(C,yb,maxiter=maxiter)
    
    Ksol = similar(y)
    Ksmp = similar(b)
    Objective(it,Ksol,Ksmp)
end


function (o::Objective{T})(u::Vector{T},grad::Vector{T}) where T<:Real
    reset!(o.it)
    o.it.A.comp .= log1pexp.(u)

    for (iteration, item) in enumerate(o.it)
    end
    ld = estimate_logdet(o.it)
    
    p1,p2 = logistic.(u)
    sol   = @view o.it.x[:,1]
    y     = @view o.it.b[:,1]
    ssmp  = @view o.it.x[:,2:end]
    smp   = @view o.it.b[:,2:end]
    
    #mul!(o.Ksol,o.it.A.K,sol)
    #mul!(o.Ksmp,o.it.A.K,smp)
    o.Ksol .= o.it.A.K * sol
    o.Ksmp .= o.it.A.K * smp
    
    grad[1] = -p1*(dot(sol,o.Ksol) - dot(ssmp,o.Ksmp)/size(smp,2))
    grad[2] = -p2*(dot(sol,sol) - dot(ssmp,smp)/size(smp,2))
    
    ld + dot(y,sol)
end