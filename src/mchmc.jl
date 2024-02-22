function CMBLensingTarget(prob::CMBLensingLogDensityProblem)
    θ_start = prob.Ωstart
    d = length(θ_start)
    θ_names = [string("θ_", i) for i=1:d]
    Λmass = prob.Λmass
    sqrtΛmass = sqrt(Λmass)
    inv_sqrtΛmass = pinv(sqrtΛmass)

    function transform(θ)
        x = CMBLensing.LenseBasis(sqrtΛmass * θ)
        return x
    end

    function inv_transform(x)
        θ = CMBLensing.LenseBasis(inv_sqrtΛmass * x)
        return θ
    end

    ℓπ(θ) = prob(θ)
    ∂lπ∂θ(θ)= (ℓπ(θ), CMBLensing.LenseBasis(Zygote.gradient(ℓπ, θ)[1]))

    return MicroCanonicalHMC.Target(
        d,
        MicroCanonicalHMC.Hamiltonian(ℓπ, ∂lπ∂θ, inv_transform),
        transform,
        inv_transform,
        θ_start,
        θ_names)
end