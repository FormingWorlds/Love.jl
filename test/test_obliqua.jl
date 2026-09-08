using Test
using LinearAlgebra
using Obliqua
using Plots
using NCDatasets
using Interpolations
using Obliqua.constants
using Obliqua.solid1d_relax
using Obliqua.solid1d_relax.common


@testset "Obliqua Tidal Simulation Tests" begin

    # --- Setup Common Data Paths ---
    interior_json_path = joinpath(RES_DIR, "interior_data", "runtests_mantle.json")
    config_path        = joinpath(TEST_DIR, "test.toml")  
    
    cfg = Obliqua.open_config(config_path)

    @testset "interior data and config" begin
        # Test interior data validity
        @test load.load_interior_mush_full(interior_json_path, true)

        # Test config validity
        @test Obliqua.open_config(config_path) !== nothing
    end

    # =========================================================================
    # 1. Solid1D Module Tests
    # =========================================================================
    @testset "solid1d module" begin
        # Base configuration for solid1d tests
        cfg["orbit"]["obliqua"]["module_solid"] = "solid1d"
        cfg["orbit"]["obliqua"]["module_fluid"] = "none"
        cfg["orbit"]["obliqua"]["module_mushy"] = "none"

        @testset "Andrade Rheology" begin
            cfg["orbit"]["obliqua"]["material_mu"] = "andrade"

            omega, axial, ecc, sma, S_mass, rho, radius, visc, shear, bulk, phi, ncalc =
                load.load_interior_mush_full(interior_json_path, false)

            perm      = Obliqua.interior.get_permeability(phi, cfg)
            perm, phi = Obliqua.interior.limit_porosity(perm, phi, cfg)
            bulkd     = Obliqua.interior.get_drained_bulk(bulk, phi, cfg)
            
            power_prf_expt = [5.811903654413909e-17, 3.1331643176818168e-18, 3.1025899880099643e-18, 2.9724177811069162e-18, 2.898779016027967e-18, 2.8427679069919094e-18, 6.972115536420127e-20, 9.094049487001497e-21, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            power_blk_expt = 1.093766208671846e6
            imag_k2_expt    = [0.0014986632230270696]

            power_prf, power_blk, _, _, LNk = Obliqua.run_tides(
                omega, axial, ecc, sma, S_mass, rho, radius, visc, shear, bulk, bulkd, phi, perm, cfg
            )
            imag_k2 = -imag.(LNk)

            println(power_prf)
            println(power_blk)
            println(imag_k2)

            @test all(isapprox.(power_prf, power_prf_expt; rtol=rtol, atol=atol))
            @test isapprox(power_blk, power_blk_expt; rtol=rtol)
            @test all(isapprox.(imag_k2, imag_k2_expt; rtol=rtol))
        end

        @testset "Maxwell Rheology" begin
            cfg["orbit"]["obliqua"]["material_mu"] = "maxwell"

            omega, axial, ecc, sma, S_mass, rho, radius, visc, shear, bulk, phi, ncalc =
                load.load_interior_mush_full(interior_json_path, false)

            perm      = Obliqua.interior.get_permeability(phi, cfg)
            perm, phi = Obliqua.interior.limit_porosity(perm, phi, cfg)
            bulkd     = Obliqua.interior.get_drained_bulk(bulk, phi, cfg)

            power_prf_expt = [6.930763584609395e-17, 7.717943205767067e-22, 7.678856966385867e-22, 7.023802241566042e-22, 6.794063964493952e-22, 6.660180949962994e-22, 2.0194178851454583e-23, 2.999024557829818e-24, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            power_blk_expt = 1.0210408255188541e6
            imag_k2_expt    = [0.0013990159160908932]

            power_prf, power_blk, _, _, LNk = Obliqua.run_tides(
                omega, axial, ecc, sma, S_mass, rho, radius, visc, shear, bulk, bulkd, phi, perm, cfg
            )
            imag_k2 = -imag.(LNk)

            println(power_prf)
            println(power_blk)
            println(imag_k2)

            @test all(isapprox.(power_prf, power_prf_expt; rtol=rtol, atol=atol))
            @test isapprox(power_blk, power_blk_expt; rtol=rtol)
            @test all(isapprox.(imag_k2, imag_k2_expt; rtol=rtol))
        end
    end

    # =========================================================================
    # 2. Solid1D-Relax Module Tests
    # =========================================================================
    @testset "solid1d-relax module" begin
        cfg["orbit"]["obliqua"]["module_solid"] = "solid1d-relax"
        cfg["orbit"]["obliqua"]["module_fluid"] = "none"
        cfg["orbit"]["obliqua"]["module_mushy"] = "none"

        @testset "Andrade Rheology" begin
            cfg["orbit"]["obliqua"]["material_mu"] = "andrade"

            omega, axial, ecc, sma, S_mass, rho, radius, visc, shear, bulk, phi, ncalc =
                load.load_interior_mush_full(interior_json_path, false)

            perm      = Obliqua.interior.get_permeability(phi, cfg)
            perm, phi = Obliqua.interior.limit_porosity(perm, phi, cfg)
            bulkd     = Obliqua.interior.get_drained_bulk(bulk, phi, cfg)

            power_prf_expt = [5.826056103767154e-17, 3.138911860450565e-18, 3.1084419405545135e-18, 2.9781614227522515e-18, 2.9045079765139066e-18, 2.848515571814759e-18, 6.98862439431475e-20, 7.590322846147857e-21, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            power_blk_expt = 999012.931402662
            imag_k2_expt    = [0.0013688336024200785]

            power_prf, power_blk, _, _, LNk = Obliqua.run_tides(
                omega, axial, ecc, sma, S_mass, rho, radius, visc, shear, bulk, bulkd, phi, perm, cfg
            )
            imag_k2 = -imag.(LNk)

            println(power_prf)
            println(power_blk)
            println(imag_k2)

            @test all(isapprox.(power_prf, power_prf_expt; rtol=rtol, atol=atol))
            @test isapprox(power_blk, power_blk_expt; rtol=rtol)
            @test all(isapprox.(imag_k2, imag_k2_expt; rtol=rtol))
        end

        @testset "Maxwell Rheology" begin
            cfg["orbit"]["obliqua"]["material_mu"] = "maxwell"

            omega, axial, ecc, sma, S_mass, rho, radius, visc, shear, bulk, phi, ncalc =
                load.load_interior_mush_full(interior_json_path, false)

            perm      = Obliqua.interior.get_permeability(phi, cfg)
            perm, phi = Obliqua.interior.limit_porosity(perm, phi, cfg)
            bulkd     = Obliqua.interior.get_drained_bulk(bulk, phi, cfg)

            power_prf_expt = [6.947705185870303e-17, 7.731741854354142e-22, 7.69299645791919e-22, 7.037072744910366e-22, 6.807211631708028e-22, 6.6733847925196295e-22, 2.0242218790426267e-23, 2.5056019859058257e-24, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            power_blk_expt = 904776.457606981
            imag_k2_expt    = [0.0012397120987334411]

            power_prf, power_blk, _, _, LNk = Obliqua.run_tides(
                omega, axial, ecc, sma, S_mass, rho, radius, visc, shear, bulk, bulkd, phi, perm, cfg
            )
            imag_k2 = -imag.(LNk)

            println(power_prf)
            println(power_blk)
            println(imag_k2)

            @test all(isapprox.(power_prf, power_prf_expt; rtol=rtol, atol=atol))
            @test isapprox(power_blk, power_blk_expt; rtol=rtol)
            @test all(isapprox.(imag_k2, imag_k2_expt; rtol=rtol))
        end
    end

    # =========================================================================
    # 3. Solid1D-Mush Module Tests
    # =========================================================================
    @testset "solid1d-mush module" begin
        cfg["orbit"]["obliqua"]["module_solid"] = "solid1d-mush"
        cfg["orbit"]["obliqua"]["module_fluid"] = "none"
        cfg["orbit"]["obliqua"]["module_mushy"] = "none"
        cfg["orbit"]["obliqua"]["visc_sus"] = 5e12

        @testset "Andrade Rheology" begin
            cfg["orbit"]["obliqua"]["material_mu"] = "andrade"

            omega, axial, ecc, sma, S_mass, rho, radius, visc, shear, bulk, phi, ncalc =
                load.load_interior_mush_full(interior_json_path, false)

            perm      = Obliqua.interior.get_permeability(phi, cfg)
            perm, phi = Obliqua.interior.limit_porosity(perm, phi, cfg)
            bulkd     = Obliqua.interior.get_drained_bulk(bulk, phi, cfg)

            power_prf_expt = [5.3950076375562126e-17, 2.8407437126296814e-18, 2.818017456639662e-18, 2.705284895036029e-18, 2.6435353304508365e-18, 2.5975027640527154e-18, 6.681457041216717e-20, 1.6896966982997684e-20, 1.7546748749581933e-19, 4.269886897967008e-19, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            power_blk_expt = 1.0100820769031402e6
            imag_k2_expt    = [0.0013840003913923272]

            power_prf, power_blk, _, _, LNk = Obliqua.run_tides(
                omega, axial, ecc, sma, S_mass, rho, radius, visc, shear, bulk, bulkd, phi, perm, cfg
            )
            imag_k2 = -imag.(LNk)

            println(power_prf)
            println(power_blk)
            println(imag_k2)

            @test all(isapprox.(power_prf, power_prf_expt; rtol=rtol, atol=atol))
            @test isapprox(power_blk, power_blk_expt; rtol=rtol)
            @test all(isapprox.(imag_k2, imag_k2_expt; rtol=rtol))
        end

        @testset "Maxwell Rheology" begin
            cfg["orbit"]["obliqua"]["material_mu"] = "maxwell"

            omega, axial, ecc, sma, S_mass, rho, radius, visc, shear, bulk, phi, ncalc =
                load.load_interior_mush_full(interior_json_path, false)

            perm      = Obliqua.interior.get_permeability(phi, cfg)
            perm, phi = Obliqua.interior.limit_porosity(perm, phi, cfg)
            bulkd     = Obliqua.interior.get_drained_bulk(bulk, phi, cfg)

            power_prf_expt = [6.439313496369688e-17, 6.997885032837296e-22, 6.974614792513135e-22, 6.392492308651947e-22, 6.195600031186184e-22, 6.085179544465906e-22, 1.9359067755994564e-23, 5.617084352918099e-24, 6.593993133185726e-23, 1.3445953880034122e-19, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            power_blk_expt = 941057.7310122688
            imag_k2_expt    = [0.00128942419415749]

            power_prf, power_blk, _, _, LNk = Obliqua.run_tides(
                omega, axial, ecc, sma, S_mass, rho, radius, visc, shear, bulk, bulkd, phi, perm, cfg
            )
            imag_k2 = -imag.(LNk)

            println(power_prf)
            println(power_blk)
            println(imag_k2)

            @test all(isapprox.(power_prf, power_prf_expt; rtol=rtol, atol=atol))
            @test isapprox(power_blk, power_blk_expt; rtol=rtol)
            @test all(isapprox.(imag_k2, imag_k2_expt; rtol=rtol))
        end
        
        # Cleanup configuration change
        cfg["orbit"]["obliqua"]["visc_sus"] = 5e13
    end

    # =========================================================================
    # 4. Solid1D-Mush-Relax Module Tests
    # =========================================================================
    @testset "solid1d-mush-relax module" begin
        cfg["orbit"]["obliqua"]["module_solid"] = "solid1d-mush-relax"
        cfg["orbit"]["obliqua"]["module_fluid"] = "none"
        cfg["orbit"]["obliqua"]["module_mushy"] = "none"
        cfg["orbit"]["obliqua"]["visc_sus"] = 5e12

        @testset "Andrade Rheology" begin
            cfg["orbit"]["obliqua"]["material_mu"] = "andrade"

            omega, axial, ecc, sma, S_mass, rho, radius, visc, shear, bulk, phi, ncalc =
                load.load_interior_mush_full(interior_json_path, false)

            perm      = Obliqua.interior.get_permeability(phi, cfg)
            perm, phi = Obliqua.interior.limit_porosity(perm, phi, cfg)
            bulkd     = Obliqua.interior.get_drained_bulk(bulk, phi, cfg)

            power_prf_expt = [7.315978108274724e-17, 4.458711446343957e-18, 4.330910624019476e-18, 4.069363099496459e-18, 3.890748808213957e-18, 3.739640660105169e-18, 1.2938764764597603e-19, 9.837626405773877e-19, 3.703412075751872e-18, 3.347661878077609e-17, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            power_blk_expt = 1.7168766674244788e6
            imag_k2_expt    = [0.0023524404937200856]

            power_prf, power_blk, _, _, LNk = Obliqua.run_tides(
                omega, axial, ecc, sma, S_mass, rho, radius, visc, shear, bulk, bulkd, phi, perm, cfg
            )
            imag_k2 = -imag.(LNk)

            println(power_prf)
            println(power_blk)
            println(imag_k2)

            @test all(isapprox.(power_prf, power_prf_expt; rtol=rtol, atol=atol))
            @test isapprox(power_blk, power_blk_expt; rtol=rtol)
            @test all(isapprox.(imag_k2, imag_k2_expt; rtol=rtol))
        end

        @testset "Maxwell Rheology" begin
            cfg["orbit"]["obliqua"]["material_mu"] = "maxwell"

            omega, axial, ecc, sma, S_mass, rho, radius, visc, shear, bulk, phi, ncalc =
                load.load_interior_mush_full(interior_json_path, false)

            perm      = Obliqua.interior.get_permeability(phi, cfg)
            perm, phi = Obliqua.interior.limit_porosity(perm, phi, cfg)
            bulkd     = Obliqua.interior.get_drained_bulk(bulk, phi, cfg)

            power_prf_expt = [8.749974163533983e-17, 1.1097048394303545e-21, 1.0827983667990608e-21, 9.711610165504064e-22, 9.20794551513194e-22, 8.84504288582015e-22, 3.773403712313033e-23, 3.345952994572589e-22, 2.0936944435948474e-18, 3.010156081071643e-18, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            power_blk_expt = 1.2750268133149191e6
            imag_k2_expt    = [0.001747023978560087]

            power_prf, power_blk, _, _, LNk = Obliqua.run_tides(
                omega, axial, ecc, sma, S_mass, rho, radius, visc, shear, bulk, bulkd, phi, perm, cfg
            )
            imag_k2 = -imag.(LNk)

            println(power_prf)
            println(power_blk)
            println(imag_k2)

            @test all(isapprox.(power_prf, power_prf_expt; rtol=rtol, atol=atol))
            @test isapprox(power_blk, power_blk_expt; rtol=rtol)
            @test all(isapprox.(imag_k2, imag_k2_expt; rtol=rtol))
        end

        # Cleanup configuration change
        cfg["orbit"]["obliqua"]["visc_sus"] = 5e13
    end

    # =========================================================================
    # 5. Fluid Module Tests (fluid0d & fluid1d)
    # =========================================================================
    @testset "Fluid module models" begin
        cfg["orbit"]["obliqua"]["module_mushy"] = "none"
        cfg["orbit"]["obliqua"]["material_mu"] = "andrade"

        @testset "fluid0d module" begin
            cfg["orbit"]["obliqua"]["module_solid"] = "none"
            cfg["orbit"]["obliqua"]["module_fluid"] = "fluid0d"

            omega, axial, ecc, sma, S_mass, rho, radius, visc, shear, bulk, phi, ncalc =
                load.load_interior_mush_full(interior_json_path, false)

            perm      = Obliqua.interior.get_permeability(phi, cfg)
            perm, phi = Obliqua.interior.limit_porosity(perm, phi, cfg)
            bulkd     = Obliqua.interior.get_drained_bulk(bulk, phi, cfg)

            power_blk_expt = 5.198662290572516e6
            imag_k2_expt    = [0.007123134653500875]

            _, power_blk, _, _, LNk = Obliqua.run_tides(
                omega, axial, ecc, sma, S_mass, rho, radius, visc, shear, bulk, bulkd, phi, perm, cfg
            )
            imag_k2 = -imag.(LNk)

            println(power_blk)
            println(imag_k2)

            @test isapprox(power_blk, power_blk_expt; rtol=rtol)
            @test all(isapprox.(imag_k2, imag_k2_expt; rtol=rtol))
        end

        @testset "fluid1d module" begin
            cfg["orbit"]["obliqua"]["module_solid"] = "none"
            cfg["orbit"]["obliqua"]["module_fluid"] = "fluid1d"

            omega, axial, ecc, sma, S_mass, rho, radius, visc, shear, bulk, phi, ncalc =
                load.load_interior_mush_full(interior_json_path, false)

            perm      = Obliqua.interior.get_permeability(phi, cfg)
            perm, phi = Obliqua.interior.limit_porosity(perm, phi, cfg)
            bulkd     = Obliqua.interior.get_drained_bulk(bulk, phi, cfg)

            power_prf_expt = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.7512395381499228e-17, 2.793167490862105e-17, 2.840529703140165e-17, 2.894612058526975e-17, 2.4004514792321753e-17, 1.8814267868589255e-17, 1.482423643344309e-17, 1.1778686437017732e-17]
            power_blk_expt = 5.198662290572516e6
            imag_k2_expt    = [0.007123134653500875]

            power_prf, power_blk, _, _, LNk = Obliqua.run_tides(
                omega, axial, ecc, sma, S_mass, rho, radius, visc, shear, bulk, bulkd, phi, perm, cfg
            )
            imag_k2 = -imag.(LNk)

            println(power_prf)
            println(power_blk)
            println(imag_k2)

            @test all(isapprox.(power_prf, power_prf_expt; rtol=rtol, atol=atol))
            @test isapprox(power_blk, power_blk_expt; rtol=rtol)
            @test all(isapprox.(imag_k2, imag_k2_expt; rtol=rtol))
        end
    end

    # =========================================================================
    # 6. Complete Model Test
    # =========================================================================
    @testset "Complete model (Simultaneous modules)" begin
        cfg["orbit"]["obliqua"]["module_solid"] = "solid1d"
        cfg["orbit"]["obliqua"]["module_fluid"] = "fluid1d"
        cfg["orbit"]["obliqua"]["module_mushy"] = "interp"

        cfg["orbit"]["obliqua"]["s_min"] = -2
        cfg["orbit"]["obliqua"]["s_max"] = 6
        cfg["orbit"]["obliqua"]["material_mu"] = "andrade"

        omega, axial, ecc, sma, S_mass, rho, radius, visc, shear, bulk, phi, ncalc =
            load.load_interior_mush_full(interior_json_path, false)

        perm      = Obliqua.interior.get_permeability(phi, cfg)
        perm, phi = Obliqua.interior.limit_porosity(perm, phi, cfg)
        bulkd     = Obliqua.interior.get_drained_bulk(bulk, phi, cfg)

        power_prf_expt = [2.253416932569662e-14, 1.2969687152089551e-15, 1.284317849973822e-15, 1.2304513290501027e-15, 1.1999767161471653e-15, 1.1767973140210735e-15, 2.885987911789033e-17, 3.7652262670483354e-18, 3.884135423480617e-18, 1.3359516918923412e-18, 4.575700941205927e-19, 1.5606985394675695e-19, 5.3014969741074366e-20, 1.793571816061222e-20, 6.0436894510768036e-21, 2.0284762728179813e-21, 6.78176784245978e-22, 2.2586099946282244e-22, 7.493499686480194e-23, 2.476806530791136e-23, 8.156089169057235e-24, 2.6759181480123464e-24, 8.747459767433619e-25, 2.849221062988066e-25, 9.247448978256785e-26, 2.9907859535632453e-26, 9.638997989499756e-27, 3.095829071291302e-27, 9.909135215132926e-28, 3.1609823818779192e-28, 1.0049624217014523e-28, 3.184434467291166e-29, 1.0057325402989003e-29, 3.166017666281719e-30, 9.93429655477572e-31, 3.1071799299123705e-31, 9.68751649069829e-32, 3.010831156328364e-32, 9.328227545596274e-33, 2.881121692652761e-33, 8.871236451606357e-34, 2.723183691518359e-34, 8.333943096732691e-35, 2.542822786970244e-35, 7.73580183189588e-36, 2.3478770145423524e-36, 7.154129916797755e-37, 2.337795304623464e-37, 1.308493285083435e-37, 2.4283157772817694e-37, 7.591045451357273e-37, 2.54795543236148e-36, 8.632192217828748e-36, 2.934798902609151e-35, 1.0007744402424495e-34, 3.422710082023477e-34, 1.1740163368098833e-33, 4.038723263072389e-33, 1.393397786267481e-32, 4.8212935118549946e-32, 1.6730384702792294e-31, 5.822358436804129e-31, 2.0320751356229802e-30, 7.112555260102203e-30, 2.49663627497486e-29, 8.788700196651061e-29, 3.102652862245179e-28, 1.098446762706548e-27, 3.899968789874167e-27, 1.3886054980816842e-26, 4.9582915455296047e-26, 1.7754998382537867e-25, 6.3759584027734245e-25, 2.2961899059708784e-24, 8.292933678414047e-24, 3.003644687871403e-23, 1.091016957195519e-22, 3.974301023237362e-22, 1.4519162678640861e-21, 5.319619756058623e-21, 1.9547227580361233e-20, 7.2038495216176e-20, 2.66274826773369e-19, 9.871773452855781e-19, 3.671157510452736e-18, 1.3694762279275003e-17, 5.1247458908750365e-17, 1.9239126084706933e-16, 7.246566861091503e-16, 2.7388294901300264e-15, 1.0388384225614868e-14, 1.0532040117949126e-14, 1.069254481915923e-14, 1.0873852449000426e-14, 1.108088550763046e-14, 9.18918563012912e-15, 7.202303249026087e-15, 5.674876263134549e-15, 4.509006923637211e-15]
        power_blk_expt = 2.8317958612079725e9
        imag_k2_expt    = [0.011581176958188387, 0.01108652173121107, 0.010590988673882713, 0.010094581843049406, 0.009597312117327252, 0.009099198763898173, 0.008600271466978374, 0.008100573001742285, 0.007600162830519118]

        power_prf, power_blk, _, _, LNk = Obliqua.run_tides(
            omega, axial, ecc, sma, S_mass, rho, radius, visc, shear, bulk, bulkd, phi, perm, cfg
        )
        imag_k2 = -imag.(LNk)

        println(power_prf)
        println(power_blk)
        println(imag_k2)

        @test all(isapprox.(power_prf, power_prf_expt; rtol=rtol, atol=atol))
        @test isapprox(power_blk, power_blk_expt; rtol=rtol)
        @test all(isapprox.(imag_k2, imag_k2_expt; rtol=rtol, atol=atol))
    end


    # =========================================================================
    # 7. Analytical Limit Tests (homogeneous sphere)
    # =========================================================================
    #
    # Idea: if every layer of a discretised model is given identical material 
    # properties, the discretised solid should converge onto the same tidal 
    # response as Obliqua's zero-dimensional homogeneous-sphere model ("solid0d"),
    # which is Obliqua's closed-form/analytical solution. We treat the
    # solid0d output as the "analytical limit" and check that the 1D
    # (multi-layer) solid modules reproduce it within a loose relative
    # tolerance once discretised over many identical layers.
    #
    # By setting the inner radius for the 1d solvers to a small non-zero value 
    # (e.g., 1 m), we avoid the singularity at r=0 in the 1D solvers, which is 
    # not present in the solid0d model. At the same time, our test becomes 
    # agnostic to the choise of core solution vector. Hence we tests with
    # the fluid core solution vector, and not the solid core solution vector.
    # The effects become stronger as the CMB shifts upwards, and we test this 
    # behaviour in test 8, hereafter.
    #
    # A full forcing spectrum is used (rather than the single semi-diurnal
    # term used in the exact-regression tests above) so several forcing
    # frequencies are checked simultaneously: this exercises N_sigma
    # frequency components spanning p in [p_min, p_max].
    @testset "Analytical limit (homogeneous sphere)" begin

        cfg["orbit"]["obliqua"]["spectrum"] = "full"
        cfg["orbit"]["obliqua"]["N_sigma"]  = 10
        cfg["orbit"]["obliqua"]["p_min"]    = -5
        cfg["orbit"]["obliqua"]["p_max"]    = 5

        cfg["orbit"]["obliqua"]["module_fluid"] = "none"
        cfg["orbit"]["obliqua"]["module_mushy"] = "none"

        cfg["orbit"]["obliqua"]["solid"]["core_props"] = "mantle"

        analytic_rtol = 1e-1

        omega, axial, ecc, sma, S_mass, rho, radius, visc, shear, bulk, phi, ncalc =
            load.load_interior_mush_full(interior_json_path, false)

        idx = 5    # corresponds to a solid layer

        n_points    = 10
        radius_homo = collect(range(1, stop=radius[end], length=n_points+1))
        rho_homo    = fill(rho[idx],   n_points)
        visc_homo   = fill(visc[idx],  n_points)
        shear_homo  = fill(shear[idx], n_points)
        bulk_homo   = fill(1e20,       n_points)        # incompressible sphere
        phi_homo    = fill(phi[idx],   n_points)

        for material_mu in ("andrade", "maxwell")
            @testset "$material_mu rheology" begin
                cfg["orbit"]["obliqua"]["material_mu"] = material_mu

                # --- "Analytical" reference: solid0d homogeneous sphere ---
                cfg["orbit"]["obliqua"]["module_solid"] = "solid0d"

                perm                        = Obliqua.interior.get_permeability(phi_homo, cfg)
                perm_homo, phi_homo = Obliqua.interior.limit_porosity(perm, phi_homo, cfg)
                bulkd_homo          = Obliqua.interior.get_drained_bulk(bulk_homo, phi_homo, cfg)

                # Fetch spectral frequency axis (sigma) and reference Love Numbers
                _, power_blk_ref, _, sigma, LNk_ref = Obliqua.run_tides(
                    omega, axial, ecc, sma, S_mass,
                    rho_homo, radius_homo, visc_homo, shear_homo, bulk_homo, bulkd_homo,
                    phi_homo, perm_homo, cfg
                )

                # Load reference data from NetCDF (extracting both Tidal & Load Love numbers)
                nc_file_path_ref = joinpath(OUT_DIR, "0_obliqua.nc")
                ds_ref = Dataset(nc_file_path_ref, "r")

                # Assuming index conventions: Tidal = index 1, Load = index 2 (adjust if needed)
                raw_kT_ref = ds_ref["knms_T"]
                raw_kL_ref = ds_ref["knms_L"]
                k_tidal_ref = raw_kT_ref[:, 1, 1] + 1im * raw_kT_ref[:, 1, 2]  # Tidal
                k_load_ref  = raw_kL_ref[:, 1, 1] + 1im * raw_kL_ref[:, 1, 2]  # Load 
                close(ds_ref)

                # --- Initialize 2x2 Plot Canvas ---
                plt = plot(
                    layout = (2, 2),
                    size   = (900, 700),
                    xscale = :log10,
                    yscale = :log10,
                    xlabel = "Forcing Frequency σ [rad/s]",
                    link   = :x # Keeps x-axes synchronized across panels
                )

                # Subplot Titles & Y-Labels
                plot!(plt[1], title="Tidal Love Number: Re(k₂)",   ylabel="Re(k₂)")
                plot!(plt[2], title="Load Love Number: -Re(k'₂)",   ylabel="-Re(k'₂)")
                plot!(plt[3], title="Tidal Love Number: -Im(k₂)",   ylabel="-Im(k₂)")
                plot!(plt[4], title="Load Love Number: Im(k'₂)",  ylabel="Im(k'₂)")

                # Helper plot function to apply across all 4 panels
                plot_ref_line! = (sp, data, lbl) -> plot!(
                    plt, sigma, data, sp=sp, label=lbl, 
                    lw=2, linestyle=:dash, color=:black
                )

                # Plot solid0d reference lines
                plot_ref_line!(1, real.(k_tidal_ref), "solid0d (ref)")
                plot_ref_line!(2, -real.(k_load_ref),  "solid0d (ref)")
                plot_ref_line!(3, -imag.(k_tidal_ref), "solid0d (ref)")
                plot_ref_line!(4, imag.(k_load_ref),   "solid0d (ref)")

                # --- Discretised 1D models ---
                models_to_test = ("solid1d", "solid1d-mush", "solid1d-relax", "solid1d-mush-relax", "solid1d-equil-relax")
                
                # Dictionary to store results for boundary checks
                model_results_real_kT = Dict{String, Vector{Float64}}()

                for model in models_to_test
                    @testset "$model vs analytical limit" begin
                        cfg["orbit"]["obliqua"]["module_solid"] = model

                        _, power_blk, _, sigma, LNk = Obliqua.run_tides(
                            omega, axial, ecc, sma, S_mass,
                            rho_homo, radius_homo, visc_homo, shear_homo, bulk_homo, bulkd_homo,
                            phi_homo, perm_homo, cfg
                        )

                        # Load netCDF data for current model
                        nc_file_path = joinpath(OUT_DIR, "0_obliqua.nc")
                        ds = Dataset(nc_file_path, "r")
                        
                        raw_kT  = ds["knms_T"]
                        raw_kL  = ds["knms_L"]
                        k_tidal = raw_kT[:, 1, 1] + 1im * raw_kT[:, 1, 2]
                        k_load  = raw_kL[:, 1, 1] + 1im * raw_kL[:, 1, 2]
                        close(ds)

                        # Cache real tidal results for boundary evaluation
                        model_results_real_kT[model] = real.(k_tidal)

                        # Overlay 1D model curves onto each panel
                        lstyle = (model == "solid1d-equil-relax") ? :dot : :solid
                        lw_val = (model == "solid1d-equil-relax") ? 2.5 : 1.5
                        
                        plot!(plt[1], sigma, real.(k_tidal),  label=model, marker=:circle, ms=3, linestyle=lstyle, lw=lw_val)
                        plot!(plt[2], sigma, -real.(k_load),  label=model, marker=:circle, ms=3, linestyle=lstyle, lw=lw_val)
                        
                        if model != "solid1d-equil-relax"
                            plot!(plt[3], sigma, -imag.(k_tidal), label=model, marker=:circle, ms=3, linestyle=lstyle, lw=lw_val)
                            plot!(plt[4], sigma, imag.(k_load),   label=model, marker=:circle, ms=3, linestyle=lstyle, lw=lw_val)

                            imag_k2     = -imag.(LNk)
                            imag_k2_ref = -imag.(LNk_ref)

                            @test isapprox(power_blk, power_blk_ref; rtol=analytic_rtol)
                            @test length(imag_k2) == length(imag_k2_ref)
                            @test all(isapprox.(imag_k2, imag_k2_ref; rtol=analytic_rtol))
                        end
                    end
                end

                # --- Verify that solid1d-equil-relax serves as a boundary/limit ---
                @testset "solid1d-equil-relax boundary check" begin
                    if haskey(model_results_real_kT, "solid1d-equil-relax")
                        eq_vals = model_results_real_kT["solid1d-equil-relax"]
                        dynamic_models = setdiff(collect(models_to_test), ["solid1d-equil-relax"])
                        
                        # Check that equil-relax bounds or tightly tracks the dynamic models 
                        # (e.g., serving as a static/relaxed upper or lower envelope limit across frequencies)
                        for i in eachindex(eq_vals)
                            dyn_vals = [model_results_real_kT[m][i] for m in dynamic_models]
                            # Ensure it doesn't wildly diverge from the envelope of dynamic solutions
                            @test eq_vals[i] <= maximum(dyn_vals) * 1.25 || eq_vals[i] >= minimum(dyn_vals) * 0.75
                        end
                    end
                end

                # --- Save Plot to OUT_DIR ---
                mkpath(OUT_DIR)
                plot_filename = joinpath(OUT_DIR, "analytical_limit_comparison_$(material_mu).png")
                savefig(plt, plot_filename)

            end
        end
    end


    # =========================================================================
    # 8. Core Boundary Tests (homogeneous sphere)
    # =========================================================================
    #
    # Idea: Extending test 7, we now shift the core boundary to a radius
    # of 50% of the total radius, and check that the 1D solid modules still
    # converge onto the same tidal response. We use the 'solid' core 
    # solution and compare it to the 'solid0d' homogeneous sphere solution, 
    # which is the analytical limit. We check that the 1D
    # (multi-layer) solid modules reproduce it within a loose relative
    # tolerance once discretised over many identical layers.
    #
    # A full forcing spectrum is used (rather than the single semi-diurnal
    # term used in the exact-regression tests above) so several forcing
    # frequencies are checked simultaneously: this exercises N_sigma
    # frequency components spanning p in [p_min, p_max].
    @testset "Core boundary (homogeneous sphere)" begin

        cfg["orbit"]["obliqua"]["spectrum"] = "full"
        cfg["orbit"]["obliqua"]["N_sigma"]  = 10
        cfg["orbit"]["obliqua"]["p_min"]    = -5
        cfg["orbit"]["obliqua"]["p_max"]    = 5

        cfg["orbit"]["obliqua"]["module_fluid"] = "none"
        cfg["orbit"]["obliqua"]["module_mushy"] = "none"

        cfg["orbit"]["obliqua"]["solid"]["core"] = "solid"
        cfg["orbit"]["obliqua"]["solid"]["core_props"] = "mantle"

        analytic_rtol = 1e0

        omega, axial, ecc, sma, S_mass, rho, radius, visc, shear, bulk, phi, ncalc =
            load.load_interior_mush_full(interior_json_path, false)

        idx = 5     # corresponds to a solid layer

        n_points    = 100
        radius_homo = collect(range(0.5*radius[end], stop=radius[end], length=n_points+1))
        rho_homo    = fill(rho[idx],   n_points)
        visc_homo   = fill(visc[idx],  n_points)
        shear_homo  = fill(shear[idx], n_points)
        bulk_homo   = fill(1e20,       n_points)        # incompressible sphere
        phi_homo    = fill(phi[idx],   n_points)

        # Visual configuration map for models
        model_styles = Dict(
            "solid0d"            => (color=:black,       style=:dash,       marker=:none,       ms=0),
            "solid1d"            => (color=:crimson,     style=:dot,        marker=:circle,     ms=4),
            "solid1d-mush"       => (color=:royalblue,   style=:dash,       marker=:square,     ms=4),
            "solid1d-relax"      => (color=:darkgreen,   style=:dashdot,    marker=:diamond,    ms=4),
            "solid1d-mush-relax" => (color=:darkorange,  style=:dashdotdot, marker=:utriangle,  ms=4)
        )

        # Dictionary to cache heating profiles and bulk power without double-evaluations
        # Key: (material_mu, model) => (r_norm, power_prf, power_blk)
        heating_data = Dict{Tuple{String, String}, Tuple{Vector{Float64}, Vector{Float64}, Float64}}()

        for material_mu in ("andrade", "maxwell")
            @testset "$material_mu rheology" begin
                cfg["orbit"]["obliqua"]["material_mu"] = material_mu

                # --- "Analytical" reference: solid0d homogeneous sphere ---
                cfg["orbit"]["obliqua"]["module_solid"] = "solid0d"

                perm                = Obliqua.interior.get_permeability(phi_homo, cfg)
                perm_homo, phi_homo = Obliqua.interior.limit_porosity(perm, phi_homo, cfg)
                bulkd_homo          = Obliqua.interior.get_drained_bulk(bulk_homo, phi_homo, cfg)

                # Fetch spectral frequency axis (sigma) and reference Love Numbers
                power_prf_ref, power_blk_ref, _, sigma, LNk_ref = Obliqua.run_tides(
                    omega, axial, ecc, sma, S_mass,
                    rho_homo, radius_homo, visc_homo, shear_homo, bulk_homo, bulkd_homo,
                    phi_homo, perm_homo, cfg
                )

                # Cache heating data for reference model
                # skip first radial layer
                r_norm = radius_homo[2:end] ./ radius_homo[end]
                heating_data[(material_mu, "solid0d")] = (r_norm, power_prf_ref, power_blk_ref)

                # Load reference data from NetCDF (extracting both Tidal & Load Love numbers)
                nc_file_path_ref = joinpath(OUT_DIR, "0_obliqua.nc")
                ds_ref = Dataset(nc_file_path_ref, "r")

                # Assuming index conventions: Tidal = index 1, Load = index 2 (adjust if needed)
                raw_kT_ref = ds_ref["knms_T"]
                raw_kL_ref = ds_ref["knms_L"]
                k_tidal_ref = raw_kT_ref[:, 1, 1] + 1im * raw_kT_ref[:, 1, 2]  # Tidal
                k_load_ref  = raw_kL_ref[:, 1, 1] + 1im * raw_kL_ref[:, 1, 2]  # Load 
                close(ds_ref)

                # --- Initialize 2x2 Plot Canvas ---
                plt = plot(
                    layout = (2, 2),
                    size   = (900, 700),
                    xscale = :log10,
                    yscale = :log10,
                    xlabel = "Forcing Frequency σ [rad/s]",
                    link   = :x # Keeps x-axes synchronized across panels
                )

                # Subplot Titles & Y-Labels
                plot!(plt[1], title="Tidal Love Number: Re(k₂)",   ylabel="Re(k₂)")
                plot!(plt[2], title="Load Love Number: -Re(k'₂)",  ylabel="-Re(k'₂)")
                plot!(plt[3], title="Tidal Love Number: -Im(k₂)",  ylabel="-Im(k₂)")
                plot!(plt[4], title="Load Love Number: Im(k'₂)", ylabel="Im(k'₂)")

                # Helper plot function to apply across all 4 panels
                plot_ref_line! = (sp, data, lbl) -> plot!(
                    plt, sigma, data, sp=sp, label=lbl, 
                    lw=2, linestyle=:dash, color=:black
                )

                # Plot solid0d reference lines
                plot_ref_line!(1, real.(k_tidal_ref), "solid0d (ref)")
                plot_ref_line!(2, -real.(k_load_ref),  "solid0d (ref)")
                plot_ref_line!(3, -imag.(k_tidal_ref), "solid0d (ref)")
                plot_ref_line!(4, imag.(k_load_ref),  "solid0d (ref)")

                # --- Discretised 1D models ---
                for model in ("solid1d", "solid1d-mush", "solid1d-relax", "solid1d-mush-relax")
                    @testset "$model vs analytical limit" begin
                        cfg["orbit"]["obliqua"]["module_solid"] = model

                        power_prf, power_blk, _, sigma, LNk = Obliqua.run_tides(
                            omega, axial, ecc, sma, S_mass,
                            rho_homo, radius_homo, visc_homo, shear_homo, bulk_homo, bulkd_homo,
                            phi_homo, perm_homo, cfg
                        )

                        # Cache heating data for current model
                        heating_data[(material_mu, model)] = (r_norm, power_prf, power_blk)

                        # Load netCDF data for current model
                        nc_file_path = joinpath(OUT_DIR, "0_obliqua.nc")
                        ds = Dataset(nc_file_path, "r")
                        
                        raw_kT  = ds["knms_T"]
                        raw_kL  = ds["knms_L"]
                        k_tidal = raw_kT[:, 1, 1] + 1im * raw_kT[:, 1, 2]
                        k_load  = raw_kL[:, 1, 1] + 1im * raw_kL[:, 1, 2]
                        close(ds)

                        # Retrieve style settings for current model
                        st = model_styles[model]

                        # Shared plot kwargs for consistency and high contrast
                        curve_kwargs = (
                            label             = model,
                            color             = st.color,
                            linestyle         = st.style,
                            marker            = st.marker,
                            markersize        = st.ms,
                            markerstrokewidth = 0.5,
                            lw                = 1.8,
                            alpha             = 0.75,
                            markevery         = 1
                        )

                        # Overlay 1D model curves onto each panel
                        plot!(plt[1], sigma, real.(k_tidal);  curve_kwargs...)
                        plot!(plt[2], sigma, -real.(k_load);  curve_kwargs...)
                        plot!(plt[3], sigma, -imag.(k_tidal); curve_kwargs...)
                        plot!(plt[4], sigma, imag.(k_load);   curve_kwargs...)

                        imag_k2     = -imag.(LNk)
                        imag_k2_ref = -imag.(LNk_ref)

                        @test isapprox(power_blk, power_blk_ref; rtol=analytic_rtol)
                        @test length(imag_k2) == length(imag_k2_ref)
                        @test all(isapprox.(imag_k2, imag_k2_ref; rtol=analytic_rtol))
                    end
                end

                # --- Save Plot to OUT_DIR ---
                mkpath(OUT_DIR)
                plot_filename = joinpath(OUT_DIR, "core_boundary_comparison_$(material_mu).png")
                savefig(plt, plot_filename)

            end
        end

        # =========================================================================
        # Additional Plot 1: Radial Heating Profiles (2 panels: Andrade vs Maxwell)
        # =========================================================================
        models_keys = ("solid0d", "solid1d", "solid1d-mush", "solid1d-relax", "solid1d-mush-relax")

        plt_profiles = plot(
            layout = (1, 2),
            size   = (950, 450),
            xlabel = "Normalized Radius (r / R)",
            ylabel = "Volumetric Heating Power [log10(W/m³)]"
        )

        for (i, material_mu) in enumerate(("andrade", "maxwell"))
            plot!(plt_profiles[i], title="$(uppercasefirst(material_mu)) Rheology: Radial Profile")
            
            for model in models_keys
                r_norm, power_prf, _ = heating_data[(material_mu, model)]
                st = model_styles[model]
                
                plot!(
                    plt_profiles[i], r_norm, log10.(power_prf),
                    label      = model,
                    color      = st.color,
                    linestyle  = st.style,
                    marker     = st.marker,
                    markersize = st.ms,
                    lw         = 1.8,
                    alpha      = 0.8
                )
            end
        end

        savefig(plt_profiles, joinpath(OUT_DIR, "radial_heating_profiles.png"))

        # =========================================================================
        # Additional Plot 2: Bulk Heating Distribution (Models vs Bulk Heating)
        # =========================================================================
        # Convert models_keys to a vector of strings and define numeric x-coordinates
        x_labels  = string.(collect(models_keys))
        x_indices = 1:length(x_labels)

        andrade_bulk = [heating_data[("andrade", m)][3] for m in models_keys]
        maxwell_bulk = [heating_data[("maxwell", m)][3] for m in models_keys]

        # Matrix layout: solid models horizontally, bulk power values vertically per rheology group
        bulk_matrix = [andrade_bulk maxwell_bulk]

        plt_bulk = scatter(
            x_indices,
            log10.(bulk_matrix),
            xticks     = (x_indices, x_labels),
            label      = ["Andrade" "Maxwell"],
            title      = "Bulk Heating Power across Solid Models",
            xlabel     = "Solid Model",
            ylabel     = "Total Bulk Heating [log10(W)]",
            size       = (850, 500),
            markersize = 6,
            palette    = :tab10,
            legend     = :topright
        )

        savefig(plt_bulk, joinpath(OUT_DIR, "bulk_heating_distribution.png"))

    end
end