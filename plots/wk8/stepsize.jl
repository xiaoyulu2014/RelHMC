##file to plot ESS vs stepsize for Newtonian and relativistic HMC
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


##examine the effect of stepsize
stepsizevec = linspace(0.001,0.5,32);ESS=SharedArray(Float64,length(stepsizevec),2);accratiovec=SharedArray(Float64,length(stepsizevec))
@sync @parallel for i=1:length(stepsizevec)
	shmc = HMCState(zeros(2),stepsize=stepsizevec[i]);
	hmc,accratio = run(shmc,dm,num_iterations=1000000, final_plot=false)
  arf = StatsBase.autocor(hmc)
	ESS[i,:] = [1000000/(1+2*sum(arf[:,j])) for j=1:size(hmc,2)]
  accratiovec[i] = mean(accratio)
end
plot(stepsizevec,ESS);title("ESS vs stepsize, Newtonian HMC")
plot(stepsizevec,accratiovec);title("acceptance prob vs stepsize, Newtonian HMC")


outfile=open("hmc_stepsize","a") #append to file
    println(outfile,"ESS=",ESS,"; accratiovec=",accratiovec, "; stepsizevec=", stepsizevec)
close(outfile)


stepsizevec = linspace(0.001,0.5,32);rESS=SharedArray(Float64,length(stepsizevec),2);raccratiovec=SharedArray(Float64,length(stepsizevec))
@sync @parallel for i=1:length(stepsizevec)
	srhmc = RelHMCState(zeros(2),stepsize=stepsizevec[i],c=9.3, mass=0.4);
	rhmc,raccratio = run(srhmc,dm,num_iterations=1000000, final_plot=false)
  arf = StatsBase.autocor(rhmc)
	rESS[i,:] = [1000000/(1+2*sum(arf[:,j])) for j=1:size(rhmc,2)]
  raccratiovec[i] = mean(raccratio)
end

outfile=open("rhmc_stepsize","a") #append to file
    println(outfile,"rESS=",rESS,"; raccratiovec=",raccratiovec, "; stepsizevec=", stepsizevec)
close(outfile)

plot(stepsizevec,rESS,label="RHMC,m=0.4, c=9.3");title("ESS vs stepsize")
plot(stepsizevec,ESS,label="HMC");
