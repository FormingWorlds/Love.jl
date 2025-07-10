module SphericalHarmonics
    using AssociatedLegendrePolynomials


    export Ynm, Ynmc
    export Snm, Snmc

    function Snmc(n,m,theta,phi)
        if (m <= n + 1)
            GradY_v = n*(n-m+1)/(2n + 1) * Ynmc(n+1,m, theta, phi)./sin.(theta) 
        end 
        if (m <= n - 1)
            GradY_v -= (n+1)*(n+m)/(2n + 1) .* Ynmc(n-1,m, theta, phi)./sin.(theta) 
        end
        GradY_u = -1im * m * Ynmc(n,m,theta,phi)./sin.(theta)
        
        return GradY_v, GradY_u
    end

    function Snm(n,m,theta,phi)
        if (m <= n + 1)
            GradY_v = n*(n-m+1)/(2n + 1) * Ynm(n+1,m, theta, phi)./sin.(theta) 
        end 
        if (m <= n - 1)
            GradY_v -= (n+1)*(n+m)/(2n + 1) .* Ynm(n-1,m, theta, phi)./sin.(theta) 
        end
        GradY_u = 1im * m * Ynm(n,m,theta,phi)./sin.(theta)
        
        return GradY_v, GradY_u
    end

    function Ynmc(n,m,theta,phi)
        return Plm.(n,m,cos.(theta)) .* exp.(-1im * m .* phi)
    end

    function Ynm(n,m,theta,phi)
        return Plm.(n,m,cos.(theta)) .* exp.(1im * m .* phi)
    end

    # function dYnmdθ(n,m,theta,phi)
    #     return 1/sin.(theta) * (n*(n-m+1) * Ynm(n+1,m,theta,phi) - n*(n-m+1) * Ynm(n+1,m,theta,phi)) 
    # end

end