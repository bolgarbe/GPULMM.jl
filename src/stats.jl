struct SetTest{T<:Real,DVecT<:AbstractArray{T,1},DMatT<:AbstractArray{T,2}}
    C::Cholesky{T,DMatT}
    y::DVecT
end

function SetTest(vc::VarianceComponents2{T,CuArray{T,2}}) where {T<:Real}

end

function (st::SetTest{T,DMatT})(S::DMatT)

end

function saddle(x,λ)
    d = maximum(λ)
    @. λ /= d
    x /= d

    k0(ζ)   = -sum(@. log(1-2*ζ*λ))/2
    dk0(ζ)  = sum(@. λ/(1-2*ζ*λ))
    ddk0(ζ) = sum(@. 2*λ^2/(1-2*ζ*λ)^2)

    lmin = x > sum(λ) ? - 0.01 : -length(λ)/(2*x)
    lmax = minimum(1 ./ (2 .* λ))*0.99999
    ζopt = find_zero(ζ -> dk0(ζ)-x,(lmin,lmax))
    m = k0(ζopt)
    w = sign(ζopt)*sqrt(2*(ζopt*x-m))
    v = ζopt*sqrt(ddk0(ζopt))
    return abs(ζopt)<1e-4 ? NaN : 1-cdf(Normal(),(w+log(v/w)/w))
end

function pval_saddle(x,a)
    tr1 = mean(a)
    tr2 = mean(a.^2)/tr1
    scl = tr1*tr2
    df  = length(a)/tr2
    res = saddle(x,a)
    isnan(res) ? 1 - cdf(Chisq(df),x/scl) : 1-res
end

function pval_sumchi2(vc::VarianceComponents2{T,CuArray{T,2}},u::Vector{T},S::CuArray{T,2}) where {T<:Real}
    #C = vc.comp[1]*vc.K + vc.comp2[2]*I
    #CiS = bcg_solve(cov,S)
    #A   = CiS'obj.y

    #grad = dot(A,A)/2
    #e    = eigvals(Array(S'CiS)./2)
    #return grad,1-pval_saddle(grad,e)
end

