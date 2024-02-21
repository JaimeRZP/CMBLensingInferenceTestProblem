
module CMBLensingInferenceTestProblem

using CMBLensing
using ComponentArrays
using LogDensityProblems
using LinearAlgebra
using NamedTupleTools
using Zygote
using MicroCanonicalHMC

export load_cmb_lensing_problem, CMBLensingTarget

include("problem.jl")
include("mchmc.jl")

end