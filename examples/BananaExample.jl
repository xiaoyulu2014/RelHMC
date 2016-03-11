@everywhere push!(LOAD_PATH,"/homes/xlu/Documents/RelHMC/src")
@everywhere push!(LOAD_PATH,"/homes/xlu/Documents/RelHMC/models/")
#@everywhere push!(LOAD_PATH,"/Users/Xiaoyu Lu/Documents/RelHMC/src")
#@everywhere push!(LOAD_PATH,"/Users/Xiaoyu Lu/Documents/RelHMC/models/")
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
    zeta = zeros(num_iterations)
    for i = 1:num_iterations

        sample!(s,llik,grad)
        samples[i,:] = s.x
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
    samples
end

##function to plot ESS as the number of iterations
@everywhere function ESS_func(s::Array;plot=false)
	n,d=size(s)
	ESS=Array(Float64,n,d)
	for i=1:n
       arf = StatsBase.autocor(s[1:i,:])
  	   ESS[i,:] = [i/(1+2*sum(arf[:,j])) for j=1:2]
	end
	if plot  plot(ESS) end
	return(ESS)
end

#=
function myani(s::SamplerState)
  samples = run(s,dm,num_iterations=1000)
  llik = getllik(dm)
  llik1(x,y) = llik([x,y])
  range_x=-5:.05:6;range_y=-1:.05:32
  grid_x = [i for i in range_x, j in range_y];
  grid_y = [j for i in range_x, j in range_y];
  grid_f = [exp(llik1(i,j)) for i in range_x, j in range_y];
  fig=figure();
  ax=fig[:add_subplot](111)
  ax[:contour](grid_x',grid_y',grid_f',1)
  line=ax[:plot](samples[1,1], samples[1,2],"ro")
  for i=1:size(samples,1)
    line[1][:set_data](samples[i,1], samples[i,2])
    fig[:canvas][:draw]()
  end
end


myani(HMCState(zeros(2),stepsize=0.1))
myani(RelHMCState(zeros(2),stepsize=0.1,c=9, mass=0.4))
myani(SGRelHMCState(zeros(2),stepsize=0.2))
myani(SGNHTRelHMCState(zeros(2),stepsize=0.2))
myani(SGRelHMCState(zeros(2),stepsize=0.1,c=[9], mass=[0.4]))
myani(SGNHTRelHMCState(zeros(2),stepsize=0.1,c=[9], mass=[0.4]))

#=














