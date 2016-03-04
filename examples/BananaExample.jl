push!(LOAD_PATH,"/homes/xlu/Documents/RelHMC/src")
push!(LOAD_PATH,"/homes/xlu/Documents/RelHMC/models/")
using SGMCMC
using DataModel
using Banana
using PyPlot
import StatsBase
dm = BananaModel()
shmc = HMCState(zeros(2),stepsize=0.1)
function plot_contour(f, range_x, range_y)
    grid_x = [i for i in range_x, j in range_y]
    grid_y = [j for i in range_x, j in range_y]

    grid_f = [exp(f(i,j)) for i in range_x, j in range_y]

    PyPlot.contour(grid_x', grid_y', grid_f', 1)
end
function run(s::SamplerState,dm::AbstractDataModel;num_iterations=1000, final_plot=false)
    grad = getgrad(dm)
    llik = getllik(dm)
    samples = zeros(num_iterations, length(s.x))
    zeta = zeros(num_iterations)
    ESS = zeros(num_iterations,length(s.x))
    for i = 1:num_iterations

        sample!(s,llik,grad)

        samples[i,:] = s.x
        if typeof(s) <: SGMCMC.SGNHTRelHMCState  zeta[i] = s.zeta[1]  end
        arf = StatsBase.autocor(samples[1:i,:])
        ESS[i,:] = [i/(1+2*sum(arf[:,j])) for j=1:length(s.x)]
    end
     
    if final_plot
        if length(s.x) == 2
           # figure()
            PyPlot.clf()
            llik(x,y) = llik([x,y])
            subplot(131);plot_contour(llik, -5:.05:6, -1:.05:32)
	    eps = s.stepsize
            PyPlot.scatter(samples[:,1], samples[:,2]);
            subplot(132)
            plot(ESS[:,1]);plot(ESS[:,2]);title("ESS")
            subplot(133)
            plot(samples[:,1]);plot(samples[:,2]);title("traceplots of components")
        end
    end

    samples,zeta    
end

shmc = HMCState(zeros(2),stepsize=0.1);hmc = run(shmc,dm,final_plot=true);
srhmc = RelHMCState(zeros(2),stepsize=0.1);rhmc = run(srhmc,dm,final_plot=true);
ssgrhmc = SGRelHMCState(zeros(2),stepsize=0.1);sgrhmc = run(ssgrhmc, dm, final_plot=true);
ssgrnhthmc = SGNHTRelHMCState(zeros(2),stepsize=0.1);sgrnhthmc = run(ssgrnhthmc, dm, final_plot=true)

function traceplot(samplestats)
	for i=1:5
		res = run(samplestats,dm,final_plot=false);
		plot(res[1][:,1])
	end
	title("traceplots pf x[1], RHMC for muptiple chains")
end



function myani(res)
	pygui(true)
	for i=1:size(res[1],1)
	    PyPlot.cla()
	    llik = getllik(dm)
	    llik(x,y) = llik([x,y])
	    plot_contour(llik, -5:.05:6, -1:.05:32)
	    PyPlot.scatter(res[1][1:i,1], res[1][1:i,2])
	    pause(0.001)
	end
end














