@everywhere push!(LOAD_PATH,"/homes/xlu/Documents/RelHMC-group/src")
@everywhere push!(LOAD_PATH,"/homes/xlu/Documents/RelHMC-group/xiaoyu/models/")
@everywhere using SGMCMC
@everywhere using DataModel
@everywhere using Banana
#@everywhere using PyPlot
@everywhere using JLD
@everywhere using Lora
@everywhere using Iterators
#wk = "/data/greypartridge/oxwasp/oxwasp14/xlu/RelHMC/Banana_stepsize/"
#@everywhere wk = "/homes/xlu/Documents/RelHMC-group/xiaoyu/plots/wk12/Banana_stepsize_zoom/"


function banana(b::Real)
    function banana_logdensity(x)
        @assert length(x) == 2
        -1/2*(x[1]^2/100+(x[2]+b*x[1]^2-100*b)^2) 
    end
    return banana_logdensity
end

##plot the density
@everywhere function plot_surface(f, range_x, range_y)
    grid_x = [i for i in range_x, j in range_y]
    grid_y = [j for i in range_x, j in range_y]

    grid_f = [exp(f([i,j])) for i in range_x, j in range_y]

    PyPlot.plot_surface(grid_x', grid_y', grid_f', rstride=3,edgecolors="k", cmap=ColorMap("Greys"),cstride=1, alpha=0.5, linewidth=0.25)
end    
    
#plot_surface(banana(0.1),-20:.05:20, -10:.05:15)
    
@everywhere dm = BananaModel()

@everywhere function plot_contour(f, range_x, range_y)
    grid_x = [i for i in range_x, j in range_y]
    grid_y = [j for i in range_x, j in range_y]

    grid_f = [exp(f(i,j)) for i in range_x, j in range_y]

    PyPlot.contour(grid_x', grid_y', grid_f', 1)
end

#llik1(x,y) = getllik(dm)([x,y])
#plot_contour(llik1, -20:.05:20, -10:.05:15)


@everywhere function myrun(s::SamplerState,dm::AbstractDataModel;num_iterations=1000, final_plot=false)
    grad = getgrad(dm)
    llik = getllik(dm)
    samples = zeros(num_iterations, length(s.x))
    zeta = zeros(num_iterations)
    for i = 1:num_iterations

        sample!(s,llik,grad)
        samples[i,:] = s.x
        if typeof(s) <: SGNHTRelHMCState  zeta[i] = s.zeta[1]  end
    end

    if final_plot
        if length(s.x) == 2
            figure()
            PyPlot.clf()
            llik(x,y) = llik([x,y])
	    plot_contour(llik, -20:.05:20, -10:.05:15)
            PyPlot.scatter(samples[:,1], samples[:,2]);
        end
    end
    samples
end
###ESS contour plot
@everywhere function ESS_func(s::SamplerState,dm::AbstractDataModel;num_iterations=50000,burnin=1000,num_chain=20)
	ESS = 0;
	for k=1:num_chain
		grad = getgrad(dm)
		llik = getllik(dm)
		samples = zeros(num_iterations, length(s.x))
		for i=1:burnin
			curx = s.x
			sample!(s,llik,grad)
		end
		for i = 1:num_iterations
			curx = s.x
			if (typeof(s) <: SGNHTRelHMCState ||  typeof(s) <: SGRelHMCState)
		        dm.subobs=M
		        grad = getgrad(dm)
		        llik = getllik(dm)
		    end
			sample!(s,llik,grad)
			samples[i,:] = s.x
		end
		ESS += mean(ess(samples))
	end
    return(ESS/num_chain)
end

##stein discrepancy
#optimal parameters
#big stepsize, HMC cannot explore 
#=
samples=myrun(HMCState([0.0,10.0],stepsize=1,niters=50,mass=1),dm,num_iterations=1000)
rsamples=myrun(RelHMCState([0.0,10.0],stepsize=1,c=0.4,niters=50,mass=1),dm,num_iterations=1000)
subplot(221);plot(samples[:,1],label="HMC");plot(rsamples[:,1],label="Rel HMC");legend(loc="best");xlabel("number of samples");ylabel("x_1")
subplot(222);plot(samples[:,2],label="HMC");plot(rsamples[:,2],label="Rel HMC");legend(loc="best");xlabel("number of samples");ylabel("x_2")
subplot(223);plt[:hist](samples[:,1],10,label="HMC",alpha=0.5);plt[:hist](rsamples[:,1],25,label="Rel HMC",alpha=0.5);legend(loc="best");xlabel("number of samples");ylabel("x_1")
subplot(224);plt[:hist](samples[:,2],10,label="HMC",alpha=0.5);plt[:hist](rsamples[:,2],25,label="Rel HMC",alpha=0.5);legend(loc="best");xlabel("number of samples");ylabel("x_2")
suptitle("traceplots and histogram of banana samples, stepsize=1, L=50")
=#

#stein discrepancy vs stepsize
@everywhere mass,L=  1,50
@everywhere num_iter=10000
@everywhere num_chain=10
@everywhere stepsizevec = linspace(0.1,0.4,16) #linspace(0.01,2,20)
@everywhere cvec = exp(linspace(-4,2,32))

#=
samples,rsamples,srsamples,srnhtsamples = [SharedArray(Float64,num_iter,2,num_chain*length(stepsizevec)) for i=1:4]
@sync @parallel for i=1:length(stepsizevec)
	 for j=1:num_chain
		samples[:,:,(i-1)*num_chain+j] = myrun(HMCState([0.0,10.0],stepsize=stepsizevec[i],niters=L,mass=mass),dm,num_iterations=num_iter)
		rsamples[:,:,(i-1)*num_chain+j] = myrun(RelHMCState([0.0,10.0],stepsize=stepsizevec[i],niters=L,mass=mass,c=1),dm,num_iterations=num_iter)
		#srsamples[:,:,(i-1)*num_chain+j] = myrun(SGRelHMCState([0.0,10.0],stepsize=stepsizevec[i],niters=L,mass=[mass],c=[1]),dm,num_iterations=num_iter)
		#srnhtsamples[:,:,(i-1)*num_chain+j] = myrun(SGNHTRelHMCState([0.0,10.0],stepsize=stepsizevec[i],niters=L,mass=[mass],c=[1]),dm,num_iterations=num_iter)
	end
end
=#

rcsamples = SharedArray(Float64,num_iter,2,num_chain*length(stepsizevec))
@sync @parallel for i=1:length(cvec)   #optimal stepsize is 0.32 for Rel HMC
	 for j=1:num_chain
		rcsamples[:,:,(i-1)*num_chain+j] = myrun(RelHMCState([0.0,10.0],stepsize=0.32,niters=L,mass=mass,c=cvec[i]),dm,num_iterations=num_iter)
	end
end

sigma=1;b=0.1
phi(x)=[sigma/10*x[1],sigma*(x[2]+b*x[1]^2-100*b)]

y,ry,sry,srnhty,rcy=[Array(Float64,2,num_iter,num_chain*length(stepsizevec)) for i=1:5]
for i=1:num_iter,j=1:num_chain*length(stepsizevec)
	y[:,i,j]=phi(samples[i,:,j]) 
	ry[:,i,j]=phi(rsamples[i,:,j]))
end


#thinning
idx=round(linspace(1,num_iter,1000))
#idx=1:1000 #unthinned small datasets
jobs,rjobs,srjobs,srnhtjobs,rcjobs=[],[],[],[],[];;subsize=[1000]
for i=1:num_chain*length(stepsizevec)
	#jobs=[jobs,["d"=>2,"samples"=>y[:,idx,i],"sd"=>sigma,"subsize"=>subsize]]
	#rjobs=[rjobs,["d"=>2,"samples"=>ry[:,idx,i],"sd"=>sigma,"subsize"=>subsize]]
	rcjobs=[rcjobs,["d"=>2,"samples"=>rcy[:,idx,i],"sd"=>sigma,"subsize"=>subsize]]
end

for i=1:num_iter, j=1:num_chain*length(cvec)  rcy[:,i,j]=phi(rcsamples[i,:,j])  end
for i=1:num_chain*length(cvec) 	rcjobs=[rcjobs,["d"=>2,"samples"=>rcy[:,idx,i],"sd"=>sigma,"subsize"=>subsize]] end
#gjobs=[["d"=>2,"samples"=>randn(2,length(idx)),"sd"=>sigma,"subsize"=>subsize] for i=1:10]
#thinned samples
#save("$(wk)banana.jld","jobs",jobs)
#save("$(wk)banana_r.jld","jobs",rjobs)
save("$(wk)banana_rc.jld","jobs",rcjobs)
#save(string(wk,"gaussian.jld"),"jobs",gjobs)
#save(string(wk,"banana_sr.jld"),"jobs",srjobs)
#save(string(wk,"banana_srnht.jld"),"jobs",srnhtjobs)

##plot
#=
dic=load(string(wk,"banana.jld"))   
jobs=dic["jobs"]
rdic=load(string(wk,"banana_r.jld"))
rjobs=rdic["jobs"]
#srdic=load(string(wk,"banana_sr.jld"))
#srjobs=srdic["jobs"]
gdic=load(string(wk,"gaussian.jld"))
gjobs=gdic["jobs"]
rcdic=load(string(wk,"banana_rc.jld"))
rcjobs=rcdic["jobs"]

stein = reshape(float([ jobs[i]["stein_discrepancys"][1] for i=1:length(jobs) ]),num_chain,length(stepsizevec))
stein_r = reshape(float([ rjobs[i]["stein_discrepancys"][1] for i=1:length(jobs) ]),num_chain,length(stepsizevec))
stein_rc = reshape(float([ rcjobs[i]["stein_discrepancys"][1] for i=1:length(rcjobs) ]),num_chain,length(cvec))
#stein_g = mean(float([ gjobs[i]["stein_discrepancys"][1] for i=1:length(gjobs) ]))

for i=1:num_chain
	plot(stepsizevec,stein[i,:][:],linestyle="--",color="green",alpha=0.5)
	plot(stepsizevec,stein_r[i,:][:],linestyle="--",color="red",alpha=0.5)
	#plot(stepsizevec,stein_sr[i,:][:],linestyle="--",color="yellow",alpha=0.5)
	#plot(stepsizevec,stein_srnht[i,:][:],linestyle="--",color="pink",alpha=0.5)
end
plot(stepsizevec,mean(stein,1)',label="HMC",marker="o",color="green")
plot(stepsizevec,mean(stein_r,1)',label="Rel HMC",marker="o",color="red")
#plot(stepsizevec,mean(stein_sr,1)',label="stochastic Rel HMC",marker="o",color="yellow")
#plot(stepsizevec,mean(stein_srnht,1)',label="stochastic thermostats Rel HMC",marker="o",color="pink")
#plot([0,2],[stein_g,stein_g],label="Gaussian",marker="o",color="blue")
title("stein discrepancy vs stepsize, banana example")
xlabel("stepsize");ylabel("stein discrepancy")
legend()

##stein vs c
for i=1:num_chain
	plot(cvec,stein_rc[i,:][:],linestyle="--",alpha=0.5)
end
plot(cvec,mean(stein_rc,1)',label="Rel HMC",marker="o",color="green")
title("stein discrepancy vs c, banana example")
xlabel("c");ylabel("stein discrepancy")
legend()
=#
#ESS for optimal parameters
#=
@everywhere mass,L,stepsizehmc,stepsizerhmc,c=  1,50,0.2,0.32,1
@everywhere function ESS_func(s::SamplerState,dm::AbstractDataModel;num_iterations=10000)
    grad = getgrad(dm)
    llik = getllik(dm)
    samples = zeros(num_iterations, length(s.x))
    ESS,accratio = zeros(2);  
    for i = 1:num_iterations
		curx = s.x
		if (typeof(s) <: SGNHTRelHMCState ||  typeof(s) <: SGRelHMCState)
            dm.subobs=M
            grad = getgrad(dm)
            llik = getllik(dm)
        end
		sample!(s,llik,grad)
		if s.x!=curx accratio += 1 end
		samples[i,:] = s.x
		arf = StatsBase.autocor(samples)
		ESS = (num_iterations/(1+2*sum(arf[:,1]))+num_iterations/(1+2*sum(arf[:,2])))/2
	end
    return(ESS,accratio/(num_iterations))
end

ESS,rESS,acc,racc = [SharedArray(Float64,10) for i=1:4]
@sync @parallel for i=1:10
	ESS[i],acc[i] = ESS_func(HMCState(zeros(2),stepsize=stepsizehmc,niters=L,mass=mass),dm)
	rESS[i],racc[i] = ESS_func(RelHMCState(zeros(2),stepsize=stepsizerhmc,niters=L,mass=mass,c=c),dm)
end
plt[:hist](ESS,30,alpha=0.5,label="HMC");plt[:hist](rESS,30,alpha=0.5,label="Rel HMC");legend()
xlabel("ESS");title("histogram of ESS")
=#



#########################################################################################################################################################

#stein discrepancy vs sample size
#=
samples = pmap(s-> myrun(s,dm,num_iterations=num_iter),[HMCState([0.0,10.0],stepsize=stepsize,mass=mass,niters=L) for i=1:10])
rsamples = pmap(s-> myrun(s,dm,num_iterations=num_iter),[RelHMCState([0.0,10.0],stepsize=stepsize,mass=mass,niters=L,c=c) for i=1:10])
srsamples = pmap(s-> myrun(s,dm,num_iterations=num_iter),[SGRelHMCState([0.0,10.0],stepsize=stepsize,mass=[mass],niters=L,c=[c]) for i=1:10])
srnhtsamples = pmap(s-> myrun(s,dm,num_iterations=num_iter),[SGNHTRelHMCState([0.0,10.0],stepsize=stepsize,mass=[mass],niters=L,c=[c]) for i=1:10])
num_chain=length(samples)
original_sample=["HMCsamples" => samples, "rHMCsamples" => rsamples, "srHMCsamples" => srsamples, "srnhtHMCsamples" => srnhtsamples]
save("banana_mulchain_orignalsamples_small.jld","original_sample",original_sample)
=#

#=
dict = load("banana_mulchain_orignalsamples_small.jld","original_sample")
samples,rsamples,srsamples,srnhtsamples = dict["HMCsamples"],dict["rHMCsamples"],dict["srHMCsamples"],dict["srnhtHMCsamples"]

#transform to gaussian rvs
sigma=1;b=0.1
phi(x)=[sigma/10*x[1],sigma*(x[2]+b*x[1]^2-100*b)]

y=Array(Float64,2,num_iter,num_chain);ry=Array(Float64,2,num_iter,num_chain);sry=Array(Float64,2,num_iter,num_chain);srnhty=Array(Float64,2,num_iter,num_chain)
for i=1:size(samples[1],1),j=1:num_chain
	y[:,i,j]=phi(samples[j][i,:]) 
	ry[:,i,j]=phi(rsamples[j][i,:]) 
	sry[:,i,j]=phi(srsamples[j][i,:]) 
	srnhty[:,i,j]=phi(srnhtsamples[j][i,:]) 
end

                   
fig, axs = plt[:subplots](2,3)
axs[1][:scatter](y[1,:,1],y[2,:,1],alpha=0.8);axs[1][:set_title]("HMC")
axs[2][:scatter](ry[1,:,1],ry[2,:,1],alpha=0.8);axs[2][:set_title]("Rel HMC")
axs[3][:scatter](sry[1,:,1],sry[2,:,1],alpha=0.8);axs[3][:set_title]("stochastic Rel HMC")
axs[4][:scatter](srnhty[1,:,1],srnhty[2,:,1],alpha=0.8);axs[3][:set_title]("stochastic thermostats Rel HMC")
axs[5][:scatter](sigma*randn(size(y[1,:,1],2)),sigma*randn(size(y[1,:,1],2)),alpha=0.8);axs[5][:set_title]("standard normal")
[axs[i][:set_xlim](-5,5) for i=1:5]
[axs[i][:set_ylim](-5,5) for i=1:5]
suptitle("transformation of banana sampels to standard normal variables")


###save jobs for stein discrepancy

jobs,rjobs,gjobs,srjobs,srnhtjobs=[],[],[],[],[];subsize=round(exp(linspace(1,log(size(y,2)),10)))
for i=1:10
	jobs=[jobs,["d"=>2,"samples"=>y[:,:,i],"sd"=>sigma,"subsize"=>subsize]]
	rjobs=[rjobs,["d"=>2,"samples"=>ry[:,:,i],"sd"=>sigma,"subsize"=>subsize]]
	gjobs=[gjobs,["d"=>2,"samples"=>randn(2,size(y,2)),"sd"=>sigma,"subsize"=>subsize]]
	srjobs=[srjobs,["d"=>2,"samples"=>sry[:,:,i],"sd"=>sigma,"subsize"=>subsize]]
	srnhtjobs=[srnhtjobs,["d"=>2,"samples"=>srnhty[:,:,i],"sd"=>sigma,"subsize"=>subsize]]
end
save("banana_small.jld","jobs",jobs)
save("banana_rsmall.jld","jobs",rjobs)
save("gaussian_small.jld","jobs",gjobs)
save("banana_srsmall.jld","jobs",srjobs)
save("banana_srnhtsmall.jld","jobs",srnhtjobs)


dic=load("/data/greyheron/oxwasp/oxwasp14/xlu/RelHMC/wk12/banana_small.jld") 
jobs=dic["jobs"]
rdic=load("/data/greyheron/oxwasp/oxwasp14/xlu/RelHMC/wk12/banana_rsmall.jld")
rjobs=rdic["jobs"]
srdic=load("/data/greyheron/oxwasp/oxwasp14/xlu/RelHMC/wk12/banana_srsmall.jld")
srjobs=srdic["jobs"]
gdic=load("/data/greyheron/oxwasp/oxwasp14/xlu/RelHMC/wk12/gaussian_small.jld") 
gjobs=gdic["jobs"]
srnhtdic=load("/data/greyheron/oxwasp/oxwasp14/xlu/RelHMC/wk12/banana_srnhtsmall.jld") 
srnhtjobs=srnhtdic["jobs"]

figure()
subplot(121)
scatter(sigma*randn(size(y,2)),sigma*randn(size(y,2)),alpha=0.5,label="standard normal",color="blue")
scatter(y[1,:,1],y[2,:,1],alpha=0.5,label="HMC",color="green")
scatter(ry[1,:,1],ry[2,:,1],alpha=0.5,label="Rel HMC",color="red")
scatter(sry[1,:,1],sry[2,:,1],alpha=0.5,label="stochastic Rel HMC",color="yellow")
scatter(srnhty[1,:,1],srnhty[2,:,1],alpha=0.5,label="stochastic thermostats Rel HMC",color="pink")
title("transformation of banana sampels to standard normal variables")
legend()
axs=subplot(122)
axs[:set_ylim]([-0.7,1.5])
for i=1:length(gjobs)
        plot(log(gjobs[i]["subsize"]),log(gjobs[i]["stein_discrepancys"]),linestyle="--",color="blue",alpha=0.5)
        plot(log(jobs[i]["subsize"]),log(jobs[i]["stein_discrepancys"]),linestyle="--",color="green",alpha=0.5)
        plot(log(rjobs[i]["subsize"]),log(rjobs[i]["stein_discrepancys"]),linestyle="--",color="red",alpha=0.5)
        plot(log(srjobs[i]["subsize"]),log(srjobs[i]["stein_discrepancys"]),linestyle="--",color="yellow",alpha=0.5)
        plot(log(srnhtjobs[i]["subsize"]),log(srnhtjobs[i]["stein_discrepancys"]),linestyle="--",color="pink",alpha=0.7)
end
gmean = mean([gjobs[i]["stein_discrepancys"] for i=1:num_chain]) 
hmean = mean([jobs[i]["stein_discrepancys"] for i=1:num_chain]) 
rmean = mean([rjobs[i]["stein_discrepancys"] for i=1:num_chain]) 
srmean = mean([srjobs[i]["stein_discrepancys"] for i=1:num_chain]) 
srnhtmean = mean([srnhtjobs[i]["stein_discrepancys"] for i=1:num_chain]) 
plot(log(gjobs[1]["subsize"]),log(gmean),label="Gaussian",marker="o",color="blue")
plot(log(gjobs[1]["subsize"]),log(hmean),label="HMC",marker="o",color="green")
plot(log(gjobs[1]["subsize"]),log(rmean),label="Rel HMC",marker="o",color="red")
plot(log(gjobs[1]["subsize"]),log(srmean),label="stochastic Rel HMC",marker="o",color="yellow")
plot(log(gjobs[1]["subsize"]),log(srnhtmean),label="stochastic thermostats Rel HMC",marker="o",color="pink")
title("comparison of stein discrepancy on log scale")
xlabel("log(number of samples)");ylabel("log(stein discrepancy)")
legend()

=#



