struct SetTest{T<:Real,DVecT<:AbstractArray{T,1},DMatT<:AbstractArray{T,2}}
    C::Cholesky{T,DMatT}
    y::DVecT
end

function SetTest(vc::VarianceComponents2{T,DMatT},y::DVecT) where {T<:Real, DVecT<:AbstractArray{T,1}, DMatT<:AbstractArray{T,2}}
    C = vc.comp[1] .* vc.K + vc.comp[2]*I
    chol = cholesky!(C)
    SetTest(chol,y)
end

function p_value(st::SetTest{T,DVecT,DMatT}, S::DMatT) where {T<:Real,DVecT<:CuArray{T,1},DMatT<:CuArray{T,2}}
    CiS  = st.C\S
    A    = CiS'st.y
    grad = dot(A,A)/2
    ev   = Array(CUDA.CUSOLVER.syevd!('N','U',S'CiS)) ./ 2
    return grad,1-pval_saddle(grad,ev)
end

function p_value(st::SetTest{T,Array{T,1},Array{T,2}},S::Array{T,2}) where {T<:Real}
    CiS  = st.C\S
    A    = CiS'st.y
    grad = dot(A,A)/2
    ev   = eigvals(S'CiS) ./ 2
    return grad,1-pval_saddle(grad,ev)
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
