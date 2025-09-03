function k_phi(φ, a; K0=50.0)
    return a^2 * φ^3 / (K0 * (1 - φ)^2 )
end

function ζ_phi(φ, ηs)
   return ηs / φ
end

function ηs_phi(φ, ηl)
    l = 25.7 
    φstar = 1.0 - 0.569
    ξ = 1.17e-9 
    γ = 5.0 

    Theta = (1 - φ)/(1 - φstar)
    F = (1 - ξ) * erf(sqrt(π)/(2(1-ξ)) * Theta * (1 + Theta^γ) )

    return ηl * (1+Theta^l) / (1 - F)^(5/2 * (1 - φstar))
end

function α_phi(φ, b; φ0 = 1e-3)
    return 1.0 - (φ0 / φ)^b
end

function φ_alpha(α, b; φ0 = 1e-3)
    φ = φ0 / (1 - α)^(1/b)
    if φ < 0.0
        return 0.0
    elseif φ > 0.3
        return NaN
    end
    return φ
end

function κ_phi(φ, κs, b=1.0; φ0 = 1e-3)
    return κs * (1 - α_phi(φ, b, φ0=φ0))^2
end

function b_from_a_phi(φ, α; φ0 = 1e-3)
    return log(1-α) / log(φ0 / φ)
end
