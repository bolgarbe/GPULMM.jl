abstract type VarianceComponents{T,DMatT} end

struct VarianceComponents2{T<:Real,DMatT<:AbstractArray{T,2}} <: VarianceComponents{T,DMatT}
    K::DMatT
    comp::Vector{T}
end

struct VarianceComponents3{T<:Real,DMatT<:AbstractArray{T,2}} <: VarianceComponents{T,DMatT}
    K::DMatT
    S::DMatT
    comp::Vector{T}
end

struct VarianceComponents3_LR{T<:Real,DMatT<:AbstractArray{T,2}} <: VarianceComponents{T,DMatT}
    K::DMatT
    G::DMatT
    comp::Vector{T}
end

function mul!(res::DMatT,vc::VarianceComponents2{T,DMatT},rhs::DMatT) where {T<:Real, DMatT<:AbstractArray{T,2}}
    res .= vc.comp[1] .* (vc.K*rhs) .+ vc.comp[2].*rhs
end

function mul!(res::DMatT,vc::VarianceComponents3{T,DMatT},rhs::DMatT) where {T<:Real, DMatT<:AbstractArray{T,2}}
    res .= vc.comp[1] .* (vc.K*rhs) .+ vc.comp[2] .* (vc.S*rhs) .+ vc.comp[3].*rhs
end

function mul!(res::DMatT,vc::VarianceComponents3_LR{T,DMatT},rhs::DMatT) where {T<:Real, DMatT<:AbstractArray{T,2}}
    res .= vc.comp[1] .* (vc.K*rhs) .+ (vc.comp[2]/(2*size(vc.G,2))) .* (vc.G*(vc.G'rhs)) .+ vc.comp[3].*rhs
end