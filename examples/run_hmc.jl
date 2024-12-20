#   Lab CMBLens
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

using Adapt, CMBLensing, CMBLensingInferenceTestProblem, CUDA, JLD2, LaTeXStrings, NPZ, 
    LinearAlgebra, MCMCChains, MCMCDiagnosticTools, MuseInference, MicroCanonicalHMC,
    Plots, ProgressMeter, Random, Statistics, Zygote

ENV["GKSwstype"] = "100"
Plots.default(fmt=:png, dpi=120, size=(500,300), legendfontsize=10)

const SCRATCHDIR = joinpath(ENV["SCRATCH"], "cmblensing")
println("Writing to scratch directory: $(SCRATCHDIR)")

#  13000 for nside 128 at 3.2 hours
function run(T=Float64; masking=false, Nside=256, ϵ=0.01, n_hmc = 1000)
    # Nside = 128
    # T = Float64;
    use_map = true
    # masking = false
    t = nothing
    global_parameters = true
    precond_path = string("../pixel_preconditioners/pp_nside_", Nside, "_t_", t)
    println("Nside: ", Nside)
    println("Use Map: ", use_map)
    println("Masking: ", masking)


    prob = load_cmb_lensing_problem(;storage=CuArray, T, Nside,
        masking=masking, global_parameters=global_parameters);
    d = length(prob.Ωstart)

    prob_cpu = load_cmb_lensing_problem(;storage=Array, T, Nside,
        masking=masking, global_parameters=global_parameters);
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


    #   HMC
    #   ≡≡≡≡≡

    samples_hmc = []
    rng = Xoshiro(1)
    prob.ncalls[] = 0
    
    N=25
    
    @showprogress for i=1:n_hmc
        Ω, = state = hmc_step(rng, prob, init_params, prob.Λmass; symp_kwargs=[(N=N, ϵ=ϵ)], progress=false, always_accept=(i<10))
        push!(samples_hmc, adapt(Array, state))
    end
    ncalls_hmc = prob.ncalls[]

    chain_hmc = Chains(
        permutedims(reduce(hcat, [exp.(sample[1].θ) for sample in samples_hmc])),
        [:r, :Aϕ],
    );

    _samples_hmc = zeros(n_hmc, 3*Nside^2+2)
    for i in 1:n_hmc
        _samples_hmc[i, :]  = samples_hmc[i][1][:]
    end


    folder_name=string("HMC",
        "_use_map_", use_map,
        "_masking_", masking,
        "_Nside_", Nside,
        "_N_", N,
        "_ϵ_", ϵ)
    fol_name=joinpath(SCRATCHDIR, "new_chains", "$(Nside)", folder_name)


    if isdir(fol_name)
        fol_files = readdir(fol_name)
        println("Found existing file ", fol_name)
        if length(fol_files) != 0
            chain_files = ([file for file in fol_files if occursin("chain", file)])
            chain_numbers = [parse(Int, split(c, "_")[2]) for c in chain_files]
            last_n = maximum(chain_numbers)
        else
            println("Starting new chain")
            last_n = 0
        end
    else
        mkpath(fol_name)
        println(string("Created new folder ", fol_name))
        last_n = 0
    end

    file_name = string(fol_name, "/chain_", lpad(last_n+1, 4, "0"), "_", (ncalls_hmc/2))

    @save file_name _samples_hmc

    ess_hmc = MCMCDiagnosticTools.ess(chain_hmc)[[:r,:Aϕ],:ess]

    ess_per_call_hmc = ess_hmc ./ (ncalls_hmc/2)

    folder_name=string("HMC_summaries",
        "_Nside_", Nside,
        "_use_map_", use_map,
        "_masking_", masking,
        "_precond_", t)
    fol_name=joinpath(SCRATCHDIR, "summaries", "unmasked", folder_name)

    if isdir(fol_name)
        fol_files = readdir(fol_name)
        println("Found existing file ", fol_name)
        if length(fol_files) != 0
            chain_files = ([file for file in fol_files if occursin("hyperparams", file)])
            chain_numbers = [parse(Int, split(c, "_")[2]) for c in chain_files]
            last_n = maximum(chain_numbers)
        end
    else
        mkpath(fol_name)
        println(string("Created new folder ", fol_name))
    end

    file_name = string(fol_name, "/hyperparams_", lpad(last_n+1, 4, "0"), "_", Int(ncalls_hmc/2), ".npz" )
    npzwrite(file_name,
        Dict("r"=> vec(chain_hmc[:r])[:],
        "Aphi"=> vec(chain_hmc[:Aϕ])[:]))

    file_name = string(fol_name, "/ESS_", lpad(last_n+1, 4, "0"), "_", Int(ncalls_hmc/2), ".npz" )
    npzwrite(file_name,
        Dict("ESS"=> ess_hmc,
            "ESS_per_lkl" =>  ess_per_call_hmc))

    #   Plot
    #   ======

    plot([exp(Ω.θ.r) for (Ω,) in samples_hmc], label=L"r", xlabel="step")
    plot!([exp(Ω.θ.Aϕ) for (Ω,) in samples_hmc], label=L"A_\phi")     
    savefig("plots/A_phi_r.pdf") 


    histogram2d(chain_hmc[:r], chain_hmc[:Aϕ], bins=(40, 20), show_empty_bins=true,
        normalize=:pdf, color=:plasma)
    title!(string("Masked: ", masking, "- NSIDE=", Nside))
    ylabel!("Aϕ")
    xlabel!("r")
    savefig("plots/hmc_histogram_Aphi_r.pdf") 

    plot(getindex.(samples_hmc, 2), label=nothing, xlabel="step", ylabel=L"\Delta H")     

    savefig("plots/step_Delta_H.pdf") 


    ps = map([(:ϕ°,:I,L"L",L"\phi^\circ"), (:f°,:E,"L\ell",L"E^\circ"), (:f°,:B,L"\ell",L"B^\circ")]) do (k1, k2, xlabel, title)
        plot(get_Cℓ(prob.Ωtrue[k1][k2]); label="true", xlabel, title)
        plot!(get_Cℓ(prob.Ωstart[k1][k2]); label="start", xlabel, title)
        plot!(get_Cℓ(samples_hmc[end][1][k1][k2]); label="last sample", xlabel, title)
    end
    plot(ps..., layout=(1,3), xscale=:log10, yscale=:log10, size=(1000,300), legend=:bottomleft)
        
    savefig("plots/C_ell_hmc.pdf") 


    cls = zeros(3, length(get_Cℓ(prob.Ωstart[:ϕ°])[:]), length(samples_hmc))
    fields = [[:ϕ°, :I], [:f°,:E,], [:f°,:B]]
    for j in 1:3
        f1, f2 = fields[j]
        for i in 1:length(samples_hmc)
            cls[j,:,i] = get_Cℓ(samples_hmc[i][1][f1][f2])[:]
        end
    end 

    phi_cls, E_cls, B_cls = cls[1,:,:], cls[2,:,:], cls[3,:,:]
    phi_cls_m, phi_cls_s = mean(phi_cls, dims=2)[:], std(phi_cls, dims=2)[:]
    E_cls_m, E_cls_s = mean(E_cls, dims=2)[:], std(E_cls, dims=2)[:]
    B_cls_m, B_cls_s = mean(B_cls, dims=2)[:], std(B_cls, dims=2)[:]


    file_name = string(fol_name, "/cls_", lpad(last_n+1, 4, "0"), "_", Int(ncalls_hmc/2), ".npz" )
    npzwrite(file_name,
        Dict("phi_cls_m"=> phi_cls_m,
            "phi_cls_s"=> phi_cls_s,
            "E_cls_m"=> E_cls_m,
            "E_cls_s"=> E_cls_s,
            "B_cls_m"=> B_cls_m,
            "B_cls_s"=> B_cls_s))

    cl = get_Cℓ(prob.Ωstart[:ϕ°])
    cl.ℓ
end


# function run(T=Float64; masking=false, Nside=256, ϵ=0.01, n_hmc = 1000)
arg_masking = ARGS[1] == "masked"
arg_Nside = parse(Int, ARGS[2])
arg_eps = parse(Float64, ARGS[3])
arg_n_hmc = parse(Int, ARGS[4])

for _ in 1:1000000
    run(Float64, masking=arg_masking, Nside=arg_Nside, ϵ=arg_eps, n_hmc=arg_n_hmc)
end
