using MLUtilities
using SGMCMC
using PyPlot

include("../src/utils/probability_utils.jl")
include("../src/utils/chain_stats.jl")

function banana_logdensity(x,y)
    -1.0/20.0*(100*(y-x^2)^2 + (1-x)^2)
end

banana_logdensity(x) = banana_logdensity(x...)

function banana_gradient(x,y;noisevar=0.0)
    n1 = noisevar > 0 ? randn()*sqrt(noisevar) : 0.0
    n2 = noisevar > 0 ? randn()*sqrt(noisevar) : 0.0
    [-1.0/20.0*( -400x*(y-x^2) - 2(1-x)) + n1, -1.0/20.0*200*(y-x^2) + n2]
end

banana_gradient(x) = banana_gradient(x...)
banana_gradient(x; noisevar=0.0) = banana_gradient(x..., noisevar=noisevar)

function sample(num_iterations=1000, B=1.0; sampler = SGMCMC.relhmc, eps=0.1, mass=[1.0], c=[1.0], lsteps=1, independent_momenta=true, noisevar=0.0, refresh_steps=50, plot=false, final_plot=false)

    xx = zeros(2)
    pp = zeros(2)

    samples = zeros(num_iterations, 2)

    if sampler == SGMCMC.sglda! || sampler == SGMCMC.sgldaTheta! ||  sampler == SGMCMC.sgldaSExact!
        mass = length(mass) == 1 ? mass[1]*ones(2) : mass
        specs = SGLDA_specs(eps, mass=mass, masscount=100, niters=1)
    elseif sampler == SGMCMC.sgld!
        mass = length(mass) == 1 ? mass[1]*ones(2) : mass
        specs = SGLD_specs(eps, mass=mass, niters=1)
    elseif sampler == SGMCMC.sgrhmc!
        specs = SGRHMC_specs(eps, mass=mass, c=c, niters = 1, independent_momenta=independent_momenta, D=[1.0])
        pp = SGMCMC.sample_rel_p(mass, c, length(xx))
    elseif sampler == SGMCMC.sgo!
        specs = SGO_specs(eps, niters=1)
    elseif sampler == SGMCMC.rmsprop!
        specs = RMSprop_specs(eps*ones(2), niters=1, epsincfactor=0.1*eps)
    end

    grad = x -> banana_gradient(x, noisevar=noisevar)

    for i = 1:num_iterations

        if sampler == SGMCMC.hmc || sampler == SGMCMC.relhmc
            xx, aa = sampler(xx, grad, banana_logdensity, eps=eps, niters=lsteps, mass = mass)
        elseif sampler == SGMCMC.sglda! || sampler == SGMCMC.sgo! || sampler == SGMCMC.rmsprop! || sampler == SGMCMC.sgld!
            sampler(xx, grad, specs)
        elseif sampler == SGMCMC.sgrhmc!

            if mod(i,refresh_steps) == 0
                pp = SGMCMC.sample_rel_p(mass, c, length(xx))
            end
            sampler(xx, pp, grad, specs)
        elseif sampler == SGMCMC.sgldaTheta! || sampler == SGMCMC.sgldaSExact!
            sampler(xx, 0.0, sigma_prior, grad, specs)
        else
            error("Unknown sampler specified")
        end

        samples[i,:] = xx

        if plot
            if mod(i, 50) == 1
                PyPlot.clf()
                plot_contour(banana_logdensity, -5:.05:6, -1:.05:32)
                PyPlot.scatter(samples[1:i,1], samples[1:i,2])
            end
            PyPlot.scatter(xx[1],xx[2])

            p_grad = independent_momenta ? eps .* pp ./ (mass .* sqrt(pp.*pp ./ (mass.^2 .* c.^2) + 1)) : eps .* pp ./ (mass .* sqrt(pp'pp ./ (mass.^2 .* c.^2) + 1))
            PyPlot.plot(xx[1] + [0, p_grad[1]], xx[2] + [0, p_grad[2]])
            if mod(i,10) == 0
                sleep(0.001)
            end
        else
            if mod(i, 1000) == 0
                println("Iteration $i")
            end
        end
    end

    if final_plot
        PyPlot.clf()
        plot_contour(banana_logdensity, -5:.05:6, -1:.05:32)
        PyPlot.scatter(samples[:,1], samples[:,2])
    end

    samples
end

function plot_contour(f, range_x, range_y)
    grid_x = [i for i in range_x, j in range_y]
    grid_y = [j for i in range_x, j in range_y]

    grid_f = [exp(f(i,j)) for i in range_x, j in range_y]

    PyPlot.contour(grid_x', grid_y', grid_f', 1)
end
