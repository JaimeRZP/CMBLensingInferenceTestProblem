#   Lab CMBLens
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

using Adapt, CMBLensing, CMBLensingInferenceTestProblem, CUDA, JLD2, LaTeXStrings, NPZ, 
    LinearAlgebra, MCMCChains, MCMCDiagnosticTools, MuseInference, MicroCanonicalHMC,
    Plots, ProgressMeter, Random, Statistics, Zygote

ENV["GKSwstype"] = "100"
Plots.default(fmt=:png, dpi=120, size=(500,300), legendfontsize=10)

SCRATCHDIR = ENV["SCRATCH"]
println("Writing to scratch directory: $(SCRATCHDIR)")

Nside = 128
T = Float64;
use_map = true
masking = false
t = nothing
precond_path = string("../pixel_preconditioners/pp_nside_", Nside, "_t_", t)
println("Nside: ", Nside)
println("Use Map: ", use_map)
println("Masking: ", masking)


prob = load_cmb_lensing_problem(;storage=CuArray, T, Nside,
    masking=masking, global_parameters=true);
d = length(prob.Ωstart)

prob_cpu = load_cmb_lensing_problem(;storage=Array, T, Nside,
    masking=masking, global_parameters=true);
to_vec, from_vec = CMBLensingInferenceTestProblem.to_from_vec(prob_cpu.Ωstart);

cl = get_Cℓ(prob.Ωstart[:ϕ°][:I]);
println("Built problem")

# Precond
prob.Λmass.diag.θ.r *= 5.85
prob.Λmass.diag.θ.Aϕ *= 112.09

if t == nothing
    precond = one(simulate(Diagonal(one(LenseBasis(diag(prob.Λmass))))));
else
    precond = load(precond_path, "dist_mat_precond")
    precond = adapt(CuArray, precond)
    precond = from_vec(precond);
end
;

#init_params
if use_map
    println("Starting from the map")
    init_params = prob.Ωtrue
else
    println("Starting from the starting point")
    init_params = prob.Ωstart
end
;

#   MCHMC
#   ≡≡≡≡≡≡≡

function CMBLensingTarget(prob; kwargs...)
    θ_start = prob.Ωstart
    Λmass = prob.Λmass
    sqrtΛmass = sqrt(Λmass)
    inv_sqrtΛmass = pinv(sqrtΛmass)

    transform(θ) = CMBLensing.LenseBasis(sqrtΛmass * θ)
    inv_transform(x) = CMBLensing.LenseBasis(inv_sqrtΛmass * x)
    ℓπ(x) = prob(inv_transform(x))
    ∂lπ∂x(x) = (ℓπ(x), CMBLensing.LenseBasis(Zygote.gradient(ℓπ, x)[1]))

    return MicroCanonicalHMC.CustomTarget(
        ℓπ,
        ∂lπ∂x,
        θ_start;
        transform=transform,
        inv_transform=inv_transform,
        kwargs...)
end

target = CMBLensingTarget(prob);

#Sampler
TEV = 0.000001
nadapts = 0
n_mchmc = 100_000
L=70.0
spl = MCHMC(nadapts, TEV;
    adaptive=true, eps=10.0, L=L, sigma=precond,
    tune_L=false, tune_sigma=false);

folder_name = string("MCHMC",
    "_Nside_", Nside,
    "_use_map_", use_map,
    "_masking_", masking,
    "_precond_", t,
    "_L_", L,
    "_TEV_", TEV)
fol_name=joinpath(SCRATCHDIR, "$(Nside)", folder_name)

last_n = 0 
if isdir(fol_name)
    fol_files = readdir(fol_name)
    println("Found existing file ", fol_name)
    if length(fol_files) != 0
        last_chain = last([file for file in fol_files if occursin("chain", file)])
        last_n = parse(Int, last_chain[end])
    end
else
    mkpath(fol_name)
    println(string("Created new folder ", fol_name))
end

file_name = string(fol_name, "/chain_", last_n+1, "_", n_mchmc)

prob.ncalls[] = 0
samples_mchmc = MicroCanonicalHMC.Sample(spl, target, n_mchmc; init_params=init_params, include_latent=true, thinning=5, file_name=file_name)
ncalls_mchmc = prob.ncalls[]

chain_mchmc = Chains(permutedims(reduce(hcat, [Array([exp.(sample[end÷2-2:end÷2-1]); sample[end-2:end]]) for sample in eachcol(samples_mchmc) if all(isfinite.(sample))])),  [:r, :Aϕ, :eps, :dE, :logpdf]);


ess_mchmc = MCMCDiagnosticTools.ess(chain_mchmc)[[:r,:Aϕ],:ess]

ess_per_call_mchmc = ess_mchmc ./ n_mchmc #ncalls_mchmc

folder_name=string("MCHMC_summaries",
    "_Nside_", Nside,
    "_use_map_", use_map,
    "_masking_", masking,
    "_precond_", t,
    "_L_", L,
    "_TEV_", TEV)
fol_name=joinpath(SCRATCHDIR, "summaries", "unmasked", folder_name)

if isdir(fol_name)
    fol_files = readdir(fol_name)
    println("Found existing file ", fol_name)
    if length(fol_files) != 0
        last_chain = last([file for file in fol_files if occursin("hyperparams", file)])
        last_n = parse(Int, last_chain[end-4])
    end
else
    mkpath(fol_name)
    println(string("Created new folder ", fol_name))
end

file_name = string(fol_name, "/hyperparams_", last_n+1, "_", n_mchmc, ".npz" )
npzwrite(file_name, Dict(
        "r"=> vec(chain_mchmc[:r])[:],
        "Aphi"=> vec(chain_mchmc[:Aϕ])[:],
        "eps"=> vec(chain_mchmc[:eps])[:]))

file_name = string(fol_name, "/ESS_", last_n+1, "_", n_mchmc, ".npz" )
npzwrite(file_name,
    Dict("ESS"=> ess_mchmc,
         "ESS_per_lkl" =>  ess_per_call_mchmc))

plot(chain_mchmc[:r], label=L"r", xlabel="step")
plot!(chain_mchmc[:Aϕ], label=L"A_\phi")

histogram2d(chain_mchmc[:r], chain_mchmc[:Aϕ], bins=(40, 20), show_empty_bins=true,
    normalize=:pdf, color=:plasma)
title!(string("Masked - NSIDE=", Nside))
ylabel!("Aϕ")
xlabel!("r")
savefig("plots/Masked_histogram.pdf") 

plot(chain_mchmc[:dE]/d, label=L"\mathrm{Energy}/d", xlabel="step")
savefig("plots/chain_mchmc_Energy_step.pdf") 

ps = map([(:ϕ°,:I,L"L",L"\phi^\circ"), (:f°,:E,"L\ell",L"E^\circ"), (:f°,:B,L"\ell",L"B^\circ")]) do (k1, k2, xlabel, title)
    plot(get_Cℓ(prob.Ωtrue[k1][k2]); label="true", xlabel, title)
    plot!(get_Cℓ(prob.Ωstart[k1][k2]); label="start", xlabel, title)
    plot!(get_Cℓ(adapt(Array, from_vec(samples_mchmc[:, end]))[k1][k2]); label="last sample", xlabel, title)
end
plot(ps..., layout=(1,3), xscale=:log10, yscale=:log10, size=(1000,300), legend=:bottomleft)
savefig("plots/C_ell.pdf") 
     

cls = zeros(3, length(get_Cℓ(prob.Ωstart[:ϕ°])[:]), size(samples_mchmc)[2])
fields = [[:ϕ°, :I], [:f°,:E,], [:f°,:B]]
for j in 1:3
    f1, f2 = fields[j]
    for i in 1:size(samples_mchmc)[2]
        cls[j,:,i] = get_Cℓ(adapt(Array, from_vec(samples_mchmc[:, i]))[f1][f2])[:]
    end
end 

phi_cls, E_cls, B_cls = cls[1,:,:], cls[2,:,:], cls[3,:,:]
phi_cls_m, phi_cls_s = mean(phi_cls, dims=2)[:], std(phi_cls, dims=2)[:]
E_cls_m, E_cls_s = mean(E_cls, dims=2)[:], std(E_cls, dims=2)[:]
B_cls_m, B_cls_s = mean(B_cls, dims=2)[:], std(B_cls, dims=2)[:]


file_name = string(fol_name, "/cls_", last_n+1, "_", n_mchmc, ".npz" )
npzwrite(file_name,
    Dict("phi_cls_m"=> phi_cls_m,
          "phi_cls_s"=> phi_cls_s,
          "E_cls_m"=> E_cls_m,
          "E_cls_s"=> E_cls_s,
          "B_cls_m"=> B_cls_m,
          "B_cls_s"=> B_cls_s))
