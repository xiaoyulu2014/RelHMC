@everywhere push!(LOAD_PATH,"/homes/xlu/Documents/RelHMC/src")
@everywhere push!(LOAD_PATH,"/homes/xlu/Documents/RelHMC/models/")
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

cvec=linspace(0.1,2,10);mvec=linspace(0.1,10,10);accvec=Array(Float64,10,10)
for i=1:10
  for j=1:10
  srhmc = RelHMCState(zeros(2),stepsize=0.5,c = cvec[i],mass=mvec[j]);
  rhmc,raccratio = run(srhmc,dm,num_iterations=10000, final_plot=false)
  accvec[i,j]=mean(raccratio)
end
end

mvec=linspace(0.1,10,10);accvec=Array(Float64,10)
for i=1:10
  srhmc = RelHMCState(zeros(2),stepsize=0.5,mass=mvec[i]);
  rhmc,raccratio = run(srhmc,dm,num_iterations=10000, final_plot=false)
  accvec[i]=mean(raccratio)
end


stepsizevec = linspace(0.001,0.5,32);ESS=SharedArray(Float64,length(stepsizevec),2);accratiovec=SharedArray(Float64,length(stepsizevec))
@sync @parallel for i=1:length(stepsizevec)
	shmc = HMCState(zeros(2),stepsize=stepsizevec[i]);
	hmc,accratio = run(shmc,dm,num_iterations=1000, final_plot=false)
  arf = StatsBase.autocor(hmc)
	ESS[i,:] = [1000/(1+2*sum(arf[:,j])) for j=1:size(hmc,2)]
  accratiovec[i] = exp(mean(accratio))
end
plot(stepsizevec,ESS);
plot(stepsizevec,accratiovec);

stepsizevec = linspace(0.001,0.5,32);rESS=SharedArray(Float64,length(stepsizevec),2);raccratiovec=SharedArray(Float64,length(stepsizevec))
@sync @parallel for i=1:length(stepsizevec)
	srhmc = RelHMCState(zeros(2),stepsize=stepsizevec[i]);
	rhmc,raccratio = run(srhmc,dm,num_iterations=1000, final_plot=false)
  arf = StatsBase.autocor(rhmc)
	rESS[i,:] = [1000/(1+2*sum(arf[:,j])) for j=1:size(rhmc,2)]
  raccratiovec[i] = exp(mean(raccratio))
end


@sync @parallel for i=1:length(stepsizevec)
	srhmc = RelHMCState(zeros(2),stepsize=stepsizevec[i]);
	rhmc = run(srhmc,dm,num_iterations=1000000, final_plot=false)
        arf = StatsBase.autocor(rhmc)
	ESS[i,:] = [1000000/(1+2*sum(arf[:,j])) for j=1:size(rhmc,2)]
end




#=

srhmc = RelHMCState(zeros(2),stepsize=0.1);rhmc = run(srhmc,dm,final_plot=true);
ssgrhmc = SGRelHMCState(zeros(2),stepsize=0.1);sgrhmc = run(ssgrhmc, dm, final_plot=true);
ssgrnhthmc = SGNHTRelHMCState(zeros(2),stepsize=0.1);sgrnhthmc = run(ssgrnhthmc, dm, final_plot=true)

function traceplot(samplestats)
	for i=1:5
		res = run(samplestats,dm,final_plot=false);
		plot(res[1][:,1])
	end
end
title("traceplots pf x[1], SGRHMC for muptiple chains")


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

=#













