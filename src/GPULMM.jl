module GPULMM

export fit_base

using SnpArrays
using LinearAlgebra
using CUDA
using NNlib
using NNlibCUDA
using LogExpFunctions
using Flux
using ProgressMeter

import Base.iterate
import Flux.update!

include("variance_components.jl")
include("solver.jl")
include("objective.jl")
include("optimize.jl")

end
