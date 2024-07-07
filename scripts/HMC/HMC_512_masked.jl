# Dependencies
using Pkg
Pkg.activate(".")
Pkg.instantiate()

using CMBLensing, CMBLensingInferenceTestProblem, CUDA, LaTeXStrings, MCMCDiagnosticTools, Plots, ProgressMeter, Random, Zygote
ENV["LINES"] = 10;

# Settings
Nside = 512
T = Float32;
masking=true

# Make problem
prob = load_cmb_lensing_problem(
    storage = CuArray,
    T = T,
    Nside=Nside,
    masking=masking,
);

#init_params
map = true
if map
    Ω = prob.Ωtrue
else
    Ω = prob.start
end

#Sampler
chain = []
rng = Xoshiro(1)
ϵ=0.005
n_samples = 1500
prob.ncalls[] = 0
_samples_hmc = []
@showprogress for i=1:n_samples
    Ω, = state = hmc_step(rng, prob, Ω, prob.Λmass; symp_kwargs=[(N=25, ϵ=ϵ)], progress=false, always_accept=(i<10))
    push!(_samples_hmc, Array(Ω[:]))
end
prob.ncalls

_samples_hmc = permutedims(hcat(_samples_hmc...))

# Name of files
using NPZ, JLD2
file_name = string("/pscratch/sd/j/jaimerz/marius_chains/HMC_use_map_",map,
    "_masking_", masking,
    "_Nside_", Nside,
    "_N_", n_samples,
    "_ϵ_", ϵ)
@save file_name _samples_hmc

hmc_hyperparams = permutedims(reduce(hcat, [[exp.(sample[end-1:end]);] for sample in eachrow(_samples_hmc) if all(isfinite.(sample))]))

last_n = 0 
if isdir(fol_name)
    fol_files = readdir(fol_name)
    println("Found existing file ", fol_name)
    if length(fol_files) != 0
        last_chain = last([file for file in fol_files if occursin("chain", file)])
        last_n = parse(Int, last_chain[end])
    end
else
    mkdir(fol_name)
    println(string("Created new folder ", fol_name))
end

file_name = string(fol_name, "/chain_", last_n+1, "_", n_mchmc);

# Sample
prob.ncalls[] = 0
samples_mchmc = MicroCanonicalHMC.Sample(spl, target, n_mchmc; init_params=init_params, include_latent=true, thinning=15, file_name=file_name)
ncalls_mchmc = prob.ncalls[]
