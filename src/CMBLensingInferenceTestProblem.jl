
module CMBLensingInferenceTestProblem

using CMBLensing
using ComponentArrays
using LogDensityProblems
using LinearAlgebra
using NamedTupleTools
using Zygote
using MicroCanonicalHMC

export load_cmb_lensing_problem

include("mchmc.jl")

struct CMBLensingLogDensityProblem
    ds
    Ωstart
    Ωtrue
    Λmass
    ncalls
    ncalls_logpdf
    ncalls_grad_logpdf
end

function load_cmb_lensing_problem(;
    storage, 
    T, 
    Nside, 
    masking = false,
    global_parameters = true
)

    θpix = 3
    field_deg = θpix*Nside/60
    pixel_mask_kwargs = masking ? (edge_padding_deg=0.15*field_deg, apodization_deg=0, num_ptsrcs=0) : nothing
    (;f,  ϕ, ds) = load_sim(; pixel_mask_kwargs, storage, T, Nside, θpix, pol=:P, μKarcminT=1, beamFWHM=1, seed=0)
    (;f°, ϕ°) = mix(ds; f, ϕ)
    θ = ComponentVector(r=T(0.2), Aϕ=T(1))
    Ωtrue = LenseBasis(FieldTuple(;f°, ϕ°, θ=log.(θ)))

    Ωkeys = global_parameters ? (:f°, :ϕ°, :θ) : (:f°, :ϕ°)

    Ωstart = let
        (;f, ϕ) = MAP_joint(ds, progress=false)
        (;f°, ϕ°) = mix(ds; f, ϕ)
        LenseBasis(select(FieldTuple(;f°, ϕ°, θ=log.(θ)), Ωkeys))
    end

    Λmass = let
        (;Cf, Cϕ, Nϕ, D, G, B̂, M̂, Cn̂) = ds
        Diagonal(select(FieldTuple(
            f° = diag(pinv(D)^2 * (pinv(Cf) + B̂'*M̂'*pinv(Cn̂)*M̂*B̂)),
            ϕ° = diag(pinv(G)^2 * (pinv(Cϕ) + pinv(Nϕ))),
            θ = ComponentVector(r=T(1), Aϕ=T(1))
        ),Ωkeys))
    end

    ncalls = Ref(0)
    ncalls_logpdf = Ref(0)
    ncalls_grad_logpdf = Ref(0)
    ds.logprior = let orig_logprior = ds.logprior
        function (;Ω...)
            ncalls[] += 1
            isderiving() ? (ncalls_grad_logpdf[] += 1) : (ncalls_logpdf[] += 1)
            orig_logprior(;Ω...)
        end
    end

    return CMBLensingLogDensityProblem(ds, Ωstart, Ωtrue, Λmass, ncalls, ncalls_logpdf, ncalls_grad_logpdf)

end

function (prob::CMBLensingLogDensityProblem)(Ω::FieldTuple)
    if haskey(Ω,:θ)
        logpdf(Mixed(prob.ds); Ω.f°, Ω.ϕ°, θ=exp.(Ω.θ)) + sum(Ω.θ)
    else
        logpdf(Mixed(prob.ds); Ω.f°, Ω.ϕ°)
    end
end
LogDensityProblems.logdensity(prob::CMBLensingLogDensityProblem, Ω::FieldTuple) = prob(Ω)
LogDensityProblems.capabilities(prob::CMBLensingLogDensityProblem) = 0
LogDensityProblems.dimension(prob::CMBLensingLogDensityProblem) = length(prob.Ωstart)

to_from_vec(prob::CMBLensingLogDensityProblem) = to_from_vec(prob.Ωstart)
function to_from_vec(Ω_template::FieldTuple)
    Ω_template₀ = LenseBasis(Ω_template)
    to_vec(ft) = LenseBasis(ft)[:]
    from_vec(vec) = identity.(first(promote(vec, Ω_template₀)))
    return to_vec, from_vec
end

NamedTupleTools.select(ft::FieldTuple, ks) = FieldTuple(select(ft.fs, ks))

isderiving() = false
Zygote.@adjoint isderiving() = true, _ -> nothing


end