module GPULMM

export fit_components
export SetTest
export p_value

using SnpArrays
using LinearAlgebra
using CUDA
using NNlib
using NNlibCUDA
using LogExpFunctions
using Flux
using ProgressMeter
using Roots
using Distributions

import Base.iterate
import Flux.update!

include("variance_components.jl")
include("solver.jl")
include("objective.jl")
include("stats.jl")

end
