module GPULMM

export VarianceComponents
export bcg_iterator!
export iterate

using SnpArrays
using LinearAlgebra
using CUDA
using NNlib
using NNlibCUDA

import Base.iterate

include("variance_components.jl")
include("solver.jl")
include("optimize.jl")

end
