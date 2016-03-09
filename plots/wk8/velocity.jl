#@everywhere push!(LOAD_PATH,"/homes/xlu/Documents/RelHMC/src")
#@everywhere push!(LOAD_PATH,"/homes/xlu/Documents/RelHMC/models/")
@everywhere push!(LOAD_PATH,"/Users/Xiaoyu Lu/Documents/RelHMC/src")
@everywhere push!(LOAD_PATH,"/Users/Xiaoyu Lu/Documents/RelHMC/models/")
@everywhere using SGMCMC
@everywhere using DataModel
@everywhere using Banana
@everywhere using PyPlot
@everywhere import StatsBase.autocor


@everywhere dm = BananaModel()
@everywhere function plot_contour(f, range_x, range_y)
    grid_x = [i for i in range_x, j in range_y]
    grid_y = [j for i in range_x, j in range_y]

    grid_f = [exp(f(i,j)) for i in range_x, j in range_y]

    PyPlot.contour(grid_x', grid_y', grid_f', 1)
end
@everywhere function run(s::SamplerState,dm::AbstractDataModel;num_iterations=1000, final_plot=false)
    grad = getgrad(dm)
    llik = getllik(dm)
    samples = zeros(num_iterations, length(s.x))
    velocity = zeros(num_iterations, length(s.x))

    for i = 1:num_iterations
        sample!(s,llik,grad)
        samples[i,:] = s.x
        velocity[i,:] = s.p
        if typeof(s) <: SGMCMC.SGNHTRelHMCState  zeta[i] = s.zeta[1]  end
    end

    if final_plot
        if length(s.x) == 2
            figure()
            PyPlot.clf()
            llik(x,y) = llik([x,y])
           # subplot(131);
	    plot_contour(llik, -5:.05:6, -1:.05:32)
            PyPlot.scatter(samples[:,1], samples[:,2]);
            #plot(samples[:,1]);plot(samples[:,2]);title("traceplots of components")
        end
    end
    samples,velocity
end

	shmc = HMCState(zeros(2),stepsize=0.01);
	hmc,vhmc = run(shmc,dm,num_iterations=1000000, final_plot=false)
  srhmc = RelHMCState(zeros(2),stepsize=0.01,c=9.3, mass=0.4);
	rhmc,rvhmc = run(srhmc,dm,num_iterations=1000000, final_plot=false)

vhmc1 = sqrt(vhmc[:,1].^2 + vhmc[:,2].^2 )
rvhmc1 = sqrt(rvhmc[:,1].^2 + rvhmc[:,2].^2 )

figure()
plt[:hist](vhmc1,10,label="HMC");plt[:hist](rvhmc1,10,label="RHMC")
title("histogram of velocity, stepsize=0.5")
legend()
