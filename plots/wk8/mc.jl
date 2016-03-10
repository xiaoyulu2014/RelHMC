#file to find appropriate m and c for the banana example
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
    accratio = zeros(num_iterations)
    zeta = zeros(num_iterations)
    for i = 1:num_iterations

        accratio[i]=sample!(s,llik,grad)[2]
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
    samples,accratio
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
##examine the effect of m and c for rHMC
@everywhere cvec=linspace(0.01,10.0,15);@everywhere mvec=linspace(0.01,5.0,15);
accvec=SharedArray(Float64,15,15);ESSvec=SharedArray(Float64,15,15)
@sync @parallel for i=1:15
  for j=1:15
  srhmc = RelHMCState(zeros(2),stepsize=0.1,c = cvec[i],mass=mvec[j]);
  rhmc,raccratio = run(srhmc,dm,num_iterations=1000000, final_plot=false)
  accvec[i,j]=median(raccratio)
  arf = StatsBase.autocor(rhmc)
  ESSvec[i,j]= (1000000/(1+2*sum(arf[:,1]))+1000000/(1+2*sum(arf[:,2])))/2
  end
end

using HDF5

ESSvec=convert(Array,ESSvec);accvec=convert(Array,accvec)
h5open("ESSmc.h5", "w") do file
    write(file, "ESSvec", ESSvec,"accvec", accvec,"cvec", cvec,"mvec", mvec )  # alternatively, say "@write file A"
end


ESSvec,accvec,cvec,mvec = h5open("ESSmc.h5", "r") do file
    read(file, "ESSvec","accvec","cvec","mvec")
end



fig=figure()
ax=fig[:add_axes]((0.1,0.1,0.9,0.9))
ax[:set_ylim]([0,1])
for i=1:15
     ax[:set_ylim]([0,1])
       m=round(mvec[i],2)
       plot(cvec,accvec[:,i],label="m = $m")
end
legend(loc="lower left")
xlabel("c");axis([0,10,0,1])
title("acceptance ratio vs c for different m, rHMC")

for i=1:15
       c=round(cvec[i],2)
       plot(mvec,accvec[i,:][:],label="c = $c")
end
legend(loc="lower right")
xlabel("m");axis([0,5,0,1])
title("acceptance ratio vs m for different c, rHMC")

for i=1:15
       c=round(cvec[i],2)
       plot(mvec,ESSvec[i,:][:],label="c = $c")
end
legend(loc="upper right")
xlabel("m");
title("ESS vs m for different c, rHMC")

for i=1:15
       m=round(mvec[i],2)
       plot(cvec,ESSvec[:,i],label="m = $m")
end
legend(loc="upper left")
xlabel("c");
title("ESS vs c for different m, rHMC")
=#
#=outfile=open("rhmc_mc","a") #append to file
    println(outfile,"accvec=",accvec,"; ESSvec=",ESSvec, "; cvec=", cvec, "; mvec=", mvec)
close(outfile)
=#
#optimal m and c: c=9.3; m=0.4;
