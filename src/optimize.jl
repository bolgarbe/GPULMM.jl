function fit_base(K::AbstractArray{T,2},y::AbstractArray{T,1};
    optimizer = Descent(0.0001),
    optimizer_iters = 500,
    solver_iters = 30,
    solver_samples = 30) where {T<:Real}

    obj  = Objective(K,y;num_samples=solver_samples,maxiter=solver_iters)
    u0   = zeros(T,2)
    grad = similar(u0)

    progress = Progress(optimizer_iters,showspeed=true)

    for i in 1:optimizer_iters
        loss = obj(u0,grad)
        update!(optimizer,u0,grad)
        u = log1pexp.(u0)

        ProgressMeter.next!(progress; showvalues=[(:loss,loss),(:components,u)])
    end
    log1pexp.(u0)
end