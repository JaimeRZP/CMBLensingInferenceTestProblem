function CMBLensingTarget(prob::CMBLensingLogDensityProblem)
    θ_start = prob.Ωstart
    d = length(θ_start)
    θ_names = [string("θ_", i) for i=1:d]
    sqrtΛmass = sqrt(Λmass)
    inv_sqrtΛmass = pinv(sqrtΛmass)

    function transform(x)
        xt = CMBLensing.LenseBasis(sqrtΛmass * x)
        return xt
    end

    function inv_transform(xt)
        x = CMBLensing.LenseBasis(inv_sqrtΛmass * xt)
        return x
    end

    function ℓπ(xt)
        x = inv_transform(xt)
        return -1.0 .* prob(x)
    end

    function ∂lπ∂θ(xt)
        return (ℓπ(xt), CMBLensing.LenseBasis(Zygote.gradient(nlogp, xt)[1]))
    end

    return MicroCanonicalHMC.Target(
        d,
        Hamiltonian(ℓπ, ∂lπ∂θ),
        transform,
        inv_transform,
        θ_start,
        θ_names)
end