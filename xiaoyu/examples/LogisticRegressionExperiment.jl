@everywhere push!(LOAD_PATH,"/homes/xlu/Documents/RelHMC-group/src")
@everywhere push!(LOAD_PATH,"/homes/xlu/Documents/RelHMC-group/xiaoyu/models/")
#@everywhere push!(LOAD_PATH,"C:\\Users\\Xiaoyu Lu\\Documents\\RelHMC-group\\xiaoyu\\src")
#@everywhere push!(LOAD_PATH,"C:\\Users\\Xiaoyu Lu\\Documents\\RelHMC-group\\xiaoyu\\models")
@everywhere using SGMCMC
@everywhere using DataModel
@everywhere using LogisticRegression
#@everywhere using PyPlot
@everywhere import Lora.ess
@everywhere import Distributions.MvNormal
@everywhere using JLD
@everywhere using Iterators
#@everywhere wk = "/data/greypartridge/oxwasp/oxwasp14/xlu/RelHMC/LR_stepsize/"
#@everywhere wk = "/homes/xlu/Documents/RelHMC-group/xiaoyu/plots/wk12/LR_cvec/"

#data generation
@everywhere nobs=500
@everywhere d=3;
@everywhere C = eye(d);
@everywhere Cinv = inv(C)
#@everywhere beta = reshape(rand(MvNormal(zeros(d),C)),(d,1))
@everywhere beta = load("/homes/xlu/Documents/RelHMC-group/xiaoyu/examples/LR_param.jld")["LR_param"]["beta"]
@everywhere x = load("/homes/xlu/Documents/RelHMC-group/xiaoyu/examples/LR_param.jld")["LR_param"]["x"]
@everywhere y = load("/homes/xlu/Documents/RelHMC-group/xiaoyu/examples/LR_param.jld")["LR_param"]["y"]
#=LR_param=["beta" => beta, "x" => x, "y" => y,
"beta" => beta, "x" => x, "y" => y,"cvec" => cvec]
save("/homes/xlu/Documents/RelHMC-group/xiaoyu/examples/LR_param.jld","LR_param",LR_param)
=#

@everywhere dm = LogisticRegressionModel(d,x,y,Cinv,nobs)
@everywhere M = 100 # minibatch
@everywhere function myrun(s::SamplerState,dm::AbstractDataModel;num_iterations=1000,burnin=1000)
    grad = getgrad(dm)
    llik = getllik(dm)
    samples = zeros(num_iterations, length(s.x))
    for i = 1:(num_iterations+burnin)
        if (typeof(s) <: SGNHTRelHMCState ||  typeof(s) <: SGRelHMCState ||  typeof(s) <: SGHMCState ||  typeof(s) <: SGNHTHMCState)
            dm.subobs=M
            grad = getgrad(dm)
            llik = getllik(dm)
        end
        SGMCMC.sample!(s,llik,grad)
        if i > burnin samples[i-burnin,:] = s.x end
    end
    samples
end

##traceplots and histogram
#=
shmc=RelHMCState(zeros(d),stepsize=0.1,niters=50,independent_momenta=false)
res=myrun(shmc,dm,num_iterations=1000);
color=["blue","red","purple"]
fig, axe = plt[:subplots](1,2)
for i=1:size(x,2)
    axe[1][:plot](res[:,i],color=color[i]);axe[1][:axhline](beta[i],0,size(x,1),color=color[i],label="true beta$i")
end
axe[1][:legend](loc="best")
for i=1:size(x,2)
    axe[2][:hist](res[:,i],color=color[i],alpha=0.5);axe[2][:axvline](beta[i],color=color[i],label="true beta$i")
end
axe[2][:legend](loc="best")
suptitle("$d-d logistic regression")
=#

#=
@everywhere begin
cd("/homes/xlu/Documents/RelHMC-group/stein_discrepancy/") 
include("src/startup.jl")
using StatsBase: logistic
using SteinDiscrepancy: stein_discrepancy
using SteinDistributions: SteinLogisticRegressionGaussianPrior
solver = "clp"
function eval_logisticgp(dict::Dict{ASCIIString,Array{Float64,2}})
	samples=dict["samples"]
    stein_discrepancys=0
    target = SteinLogisticRegressionGaussianPrior(x, y)

	try
	    result=stein_discrepancy(points=samples,target=target,solver=solver)
	    stein_discrepancys=sum(result.objectivevalue)
	catch
		stein_discrepancys=NaN
	end
    return(stein_discrepancys)
end
end

@everywhere mass,L=  1,50
@everywhere num_iter=1000
@everywhere num_chain=5

#stein discrepancy vs stepsize
@everywhere stepsizevec = exp(linspace(-5,0,16))   #exp(linspace(0.1,2,8))  #just for thermostats 
#@everywhere c=0.07
samples,rsamples,srsamples,srnhtsamples = [SharedArray(Float64,num_iter,d,num_chain*length(stepsizevec)) for i=1:4]
#ssamples,snhtsamples = [SharedArray(Float64,num_iter,d,num_chain*length(stepsizevec)) for i=1:2]
@sync @parallel for i=1:length(stepsizevec)
	 for j=1:num_chain
		#samples[:,:,(i-1)*num_chain+j] = myrun(HMCState(zeros(d),stepsize=stepsizevec[i],niters=L,mass=mass),dm,num_iterations=num_iter)
		rsamples[:,:,(i-1)*num_chain+j] = myrun(RelHMCState(zeros(d),stepsize=stepsizevec[i],niters=L,mass=mass,c=0.01/stepsizevec[i]),dm,num_iterations=num_iter)
		#srsamples[:,:,(i-1)*num_chain+j] = myrun(SGRelHMCState(zeros(d),stepsize=stepsizevec[i],niters=L,mass=[mass],c=[c]),dm,num_iterations=num_iter)
		#srnhtsamples[:,:,(i-1)*num_chain+j] = myrun(SGNHTRelHMCState(zeros(d),stepsize=stepsizevec[i],niters=L,mass=[mass],c=[c]),dm,num_iterations=num_iter)
		#ssamples[:,:,(i-1)*num_chain+j] = myrun(SGHMCState(zeros(d),stepsize=stepsizevec[i],niters=L,mass=[mass]),dm,num_iterations=num_iter)
		#snhtsamples[:,:,(i-1)*num_chain+j] = myrun(SGNHTHMCState(zeros(d),stepsize=stepsizevec[i],niters=L,mass=[mass]),dm,num_iterations=num_iter)
	end
end

idx=1:num_iter
jobs,rjobs,srjobs,srnhtjobs=[],[],[],[];
sjobs,snhtjobs=[],[]
for i=1:num_chain*length(stepsizevec)
	jobs=[jobs,["samples"=>samples[idx,:,i]]]
	rjobs=[rjobs,["samples"=>rsamples[idx,:,i]]]
end
result=pmap(eval_logisticgp,jobs)
stein = reshape(float(result),num_chain,length(stepsizevec))
result_r=pmap(eval_logisticgp,rjobs)
stein_r = reshape(float(result_r),num_chain,length(stepsizevec))


LR_stein=["stein" => stein, "stein_r" => stein_r, "burnin" => 1000, "num_iterations" => 1000,"c" => 0.07, "stepsizevec" => stepsizevec]
save("/homes/xlu/Documents/RelHMC-group/xiaoyu/plots/wk1/LR_stein.jld","LR_stein",LR_stein)
LR_stein_reparam=["stein" => stein, "stein_r" => stein_r, "burnin" => 1000, "num_iterations" => 1000,"c*stepsize" => 0.01, "stepsizevec" => stepsizevec]
save("/homes/xlu/Documents/RelHMC-group/xiaoyu/plots/wk1/LR_stein_reparam.jld","LR_stein_reparam",LR_stein_reparam)
stein = load("/homes/xlu/Documents/RelHMC-group/xiaoyu/plots/wk1/LR_stein_reparam.jld")["LR_stein_reparam"]["stein"]
stein_r = load("/homes/xlu/Documents/RelHMC-group/xiaoyu/plots/wk1/LR_stein_reparam.jld")["LR_stein_reparam"]["stein_r"]
stepsizevec = load("/homes/xlu/Documents/RelHMC-group/xiaoyu/plots/wk1/LR_stein_reparam.jld")["LR_stein_reparam"]["stepsizevec"]

=#


#vs stepsize
#=
stein = reshape(float([ jobs[i]["stein_discrepancys"][1] for i=1:length(jobs) ]),num_chain,length(stepsizevec))
stein_r = reshape(float([ rjobs[i]["stein_discrepancys"][1] for i=1:length(rjobs) ]),num_chain,length(stepsizevec))
stein_sr = reshape(float([ srjobs[i]["stein_discrepancys"][1] for i=1:length(srjobs) ]),num_chain,length(stepsizevec))
stein_srnht = reshape(float([ srnhtjobs[i]["stein_discrepancys"][1] for i=1:length(srnhtjobs) ]),num_chain,length(stepsizevec))
stein_snht = reshape(float([ snhtjobs[i]["stein_discrepancys"][1] for i=1:length(snhtjobs) ]),num_chain,length(stepsizevec))
stein_s = reshape(float([ sjobs[i]["stein_discrepancys"][1] for i=1:length(sjobs) ]),num_chain,length(stepsizevec))

for i=1:num_chain
	plot(log(stepsizevec),log(stein[i,:][:]),linestyle="--",color="green",alpha=0.5)
	plot(log(stepsizevec),log(stein_s[i,:][:]),linestyle="--",color="purple",alpha=0.5)
	plot(log(stepsizevec),log(stein_snht[i,:][:]),linestyle="--",color="blue",alpha=0.5)
	plot(log(stepsizevec),log(stein_r[i,:][:]),linestyle="--",color="red",alpha=0.5)
	plot(log(stepsizevec),log(stein_sr[i,:][:]),linestyle="--",color="yellow",alpha=0.5)
	plot(log(stepsizevec),log(stein_srnht[i,:][:]),linestyle="--",color="pink",alpha=0.5)
end
plot(stepsizevec,mean(stein_s,1)',label="stochastic HMC",marker="o",color="purple")
plot(stepsizevec,mean(stein_snht,1)',label="stochastic thermostats HMC",marker="o",color="blue")
plot(stepsizevec,mean(stein_sr,1)',label="stochastic Rel HMC",marker="o",color="yellow")
plot(stepsizevec,mean(stein_srnht,1)',label="stochastic thermostats Rel HMC",marker="o",color="pink")
plot(stepsizevec,mean(stein,1)',label="HMC",marker="o",color="green")
plot(stepsizevec,mean(stein_r,1)',label="Rel HMC",marker="o",color="red")
title("stein discrepancy vs stepsize, 3-d logistic regression, \n stepsize*c=0.01, burnin=1000, num_sample=1000")
xlabel("stepsize");ylabel("stein discrepancy")
xscale("log");yscale("log")
legend()
=#

###contour plot ESS
@everywhere function ESS_func(s::SamplerState,dm::AbstractDataModel;num_iterations=50000,burnin=5000,num_chain=20)
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
			sample!(s,llik,grad)
			samples[i,:] = s.x
		end
		ESS += mean(ess(samples))
	end
    return(ESS/num_chain)
end


@everywhere n1,n2=7,9
@everywhere avec = exp(linspace(-3,0,n1))
@everywhere epsvec = exp(linspace(-5,-1,n2))
@everywhere cvec = exp(linspace(-3,0,n1))
@everywhere t=Iterators.product(avec,epsvec)
@everywhere t1=Iterators.product(cvec,epsvec)
@everywhere myt=Array(Any,n1*n2);
@everywhere it=1;
@everywhere for prod in t
	myt[it]=prod;
    it+=1;
end
@everywhere myt1=Array(Any,n1*n2);
@everywhere it=1;
@everywhere for prod in t1
	myt1[it]=prod;
    it+=1;
end
ESS = SharedArray(Float64,n1*n2)
@sync @parallel for i=1:n1*n2
	a,eps=myt[i]
	c=a/eps
	ESS[i] = ESS_func(RelHMCState(zeros(d),stepsize=eps,niters=50,mass=0.5,c=c),dm)
end
ESS=reshape(ESS,n1,n2)

ESS1 = SharedArray(Float64,n1*n2)
@sync @parallel for i=1:n1*n2
	c,eps=myt1[i]
	ESS1[i] = ESS_func(RelHMCState(zeros(d),stepsize=eps,niters=50,mass=0.5,c=c,dm)
end
ESS1=reshape(ESS1,n1,n2)
	
xgrid=repmat(epsvec',n1,1)
ygrid=repmat(avec,1,n2)
fig = figure("pyplot_surfaceplot")
ax = fig[:add_subplot](1,1,1) 
cp = ax[:contour](xgrid, ygrid, ESS, linewidth=2.0) 
ax[:clabel](cp, inline=1, fontsize=10) 
xscale("log");yscale("log")
xlabel("stepsize"); ylabel("stepsize*c")
title("contour plot of ESS, LR")


fig = figure("pyplot_surfaceplot",figsize=(10,10))
ax = fig[:add_subplot](1,1,1, projection = "3d") 
ax[:plot_surface](xgrid,ygrid, ESS, rstride=2,edgecolors="k", cstride=2, cmap=ColorMap("gray"), alpha=0.8, linewidth=0.25) 
xlabel("X") 
ylabel("Y")
title("Surface Plot")

plot(z=ESS, x=xgrid, y=ygrid, Geom.contour)

ax1 = fig[:add_subplot](1,2,2) 
cp = ax1[:contour](epsvec, cvec, ESS1, linewidth=2.0) 
ax1[:clabel](cp, inline=1, fontsize=10) 
xscale("log");yscale("log")
xlabel("stepsize"); ylabel("c")
title("contour plot of ESS, banana example")


#stein discrepancy vs c
#=
@everywhere cvec = exp(linspace(-4,0,16))
@everywhere stepsize = 0.1
rsamples,srsamples,srnhtsamples = [SharedArray(Float64,num_iter,d,num_chain*length(cvec)) for i=1:3]
@sync @parallel for i=1:length(cvec)
	 for j=1:num_chain
		rsamples[:,:,(i-1)*num_chain+j] = myrun(RelHMCState(zeros(d),stepsize=stepsize,niters=L,mass=mass,c=cvec[i]),dm,num_iterations=num_iter)
		srsamples[:,:,(i-1)*num_chain+j] = myrun(SGRelHMCState(zeros(d),stepsize=stepsize,niters=L,mass=[mass],c=[cvec[i]]),dm,num_iterations=num_iter)
		srnhtsamples[:,:,(i-1)*num_chain+j] = myrun(SGNHTRelHMCState(zeros(d),stepsize=stepsize,niters=L,mass=[mass],c=[cvec[i]]),dm,num_iterations=num_iter)
	end
end
=#


#=
#thinning
idx=round(linspace(1,num_iter,1000))
sigma=1;
subsize=[1000]
#jobs,rjobs,srjobs,srnhtjobs=[],[],[],[];
sjobs,snhtjobs=[],[]
for i=1:num_chain*length(stepsizevec)
	#jobs=[jobs,["d"=>d,"samples"=>samples[idx,:,i]',"sd"=>sigma,"subsize"=>subsize,"X"=>x,"y"=>y]]
	#rjobs=[rjobs,["d"=>d,"samples"=>rsamples[idx,:,i]',"sd"=>sigma,"subsize"=>subsize,"X"=>x,"y"=>y]]
	#srjobs=[srjobs,["d"=>d,"samples"=>srsamples[idx,:,i]',"sd"=>sigma,"subsize"=>subsize,"X"=>x,"y"=>y]]
	#srnhtjobs=[srnhtjobs,["d"=>d,"samples"=>srnhtsamples[idx,:,i]',"sd"=>sigma,"subsize"=>subsize,"X"=>x,"y"=>y]]
	sjobs=[sjobs,["d"=>d,"samples"=>ssamples[idx,:,i]',"sd"=>sigma,"subsize"=>subsize,"X"=>x,"y"=>y]]
    snhtjobs=[snhtjobs,["d"=>d,"samples"=>snhtsamples[idx,:,i]',"sd"=>sigma,"subsize"=>subsize,"X"=>x,"y"=>y]]
end
#save("$(wk)LR_s.jld","jobs",jobs)
#save("$(wk)LR_rs.jld","jobs",rjobs)
#save("$(wk)LR_srs.jld","jobs",srjobs)
#save("$(wk)LR_srnhts_largerstepsize.jld","jobs",srnhtjobs)
save("$(wk)LR_ss.jld","jobs",sjobs)
save("$(wk)LR_snhts.jld","jobs",snhtjobs)


dic=load("$(wk)LR_s.jld") 
jobs=dic["jobs"]
rdic=load("$(wk)LR_rs.jld")
rjobs=rdic["jobs"]
srdic=load("$(wk)LR_srs.jld")
srjobs=srdic["jobs"]
srnhtdic=load("$(wk)LR_srnhts.jld") 
srnhtjobs=srnhtdic["jobs"]
sdic=load("$(wk)LR_ss.jld") 
sjobs=sdic["jobs"]
snhtdic=load("$(wk)LR_snhts.jld") 
snhtjobs=snhtdic["jobs"]
srnhtdic1=load("$(wk)LR_srnhts_largerstepsize.jld") 
srnhtjobs1=srnhtdic1["jobs"]

#vs c
stein_r = reshape(float([ rjobs[i]["stein_discrepancys"][1] for i=1:length(rjobs) ]),num_chain,length(cvec))
stein_sr = reshape(float([ srjobs[i]["stein_discrepancys"][1] for i=1:length(srjobs) ]),num_chain,length(cvec))
stein_srnht = reshape(float([ srnhtjobs[i]["stein_discrepancys"][1] for i=1:length(srnhtjobs) ]),num_chain,length(cvec))


for i=1:num_chain
	plot(log(cvec),log(stein_r[i,:][:]),linestyle="--",color="red",alpha=0.5)
	plot(log(cvec),log(stein_sr[i,:][:]),linestyle="--",color="green",alpha=0.5)
	plot(log(cvec),log(stein_srnht[i,:][:]),linestyle="--",color="blue",alpha=0.5)
end
plot(log(cvec),log(mean(stein_r,1))',label="Rel HMC",marker="o",color="red")
plot(log(cvec),log(mean(stein_sr,1))',label="stochastic Rel HMC",marker="o",color="green")
plot(log(cvec),log(mean(stein_srnht,1))',label="stochastic thermostats Rel HMC",marker="o",color="blue")
#axhline(0.151,0,label="optimal HMC")
title("stein discrepancy vs c on log scale, \n 3-d logistic regression, stepsize=$stepsize")
xlabel("log(c)");ylabel("log(stein discrepancy)")
legend()
=#


##larger stepsize for srnht
stepsizevec1 = [exp(linspace(-5,0,16)),exp(linspace(0.1,2,8))]
stein_srnht1 = hcat(reshape(float([ srnhtjobs[i]["stein_discrepancys"][1] for i=1:length(srnhtjobs) ]),num_chain,16),
				reshape(float([ srnhtjobs1[i]["stein_discrepancys"][1] for i=1:length(srnhtjobs1) ]),num_chain,8))
plot(log(stepsizevec1),log(mean(stein_srnht1,1))',label="stochastic thermostats Rel HMC",marker="o",color="pink")
xlabel("log(stepsize)");ylabel("log(stein discrepancy)")
legend()
title("stein discrepancy vs stepsize on log scale, 3-d logistic regression, c=0.07")
#min(stein)=0.151 with stepsize < 0.0067
=#


#ESS for optimal parameters
#=
@everywhere mass,L,stepsizehmc,stepsizerhmc,c=  1,50,0.013,0.1,0.1
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
	ESS[i],acc[i] = ESS_func(HMCState(zeros(d),stepsize=stepsizehmc,niters=L,mass=mass),dm)
	rESS[i],racc[i] = ESS_func(RelHMCState(zeros(d),stepsize=stepsizerhmc,niters=L,mass=mass,c=c),dm)
end
plt[:hist](ESS,15,alpha=0.5,label="HMC");plt[:hist](rESS,10,alpha=0.5,label="Rel HMC");legend()
xlabel("ESS");title("histogram of ESS")
=#


#= 
stein vs sample size
##histogram

color=["blue","red","purple"]
fig, axe = plt[:subplots](1,2)
for i=1:size(x,2)
    axe[1][:hist](samples[:,i,1],color=color[i],alpha=0.5);axe[1][:axvline](beta[i],color=color[i],label="true beta$i")
end
axe[1][:legend](loc="best");axe[1][:set_title]("HMC")
for i=1:size(x,2)
    axe[2][:hist](rsamples[:,i,1],color=color[i],alpha=0.5);axe[2][:axvline](beta[i],color=color[i],label="true beta$i")
end
axe[2][:legend](loc="best");axe[2][:set_title]("Rel HMC")
suptitle("$d-d logistic regression")


for i=1:length(jobs)
        plot(log(jobs[i]["subsize"]),log(jobs[i]["stein_discrepancys"]),linestyle="--",color="purple",alpha=0.5)
        plot(log(rjobs[i]["subsize"]),log(rjobs[i]["stein_discrepancys"]),linestyle="--",color="green",alpha=0.5)
        plot(log(srjobs[i]["subsize"]),log(srjobs[i]["stein_discrepancys"]),linestyle="--",color="blue",alpha=0.5)
        plot(log(srnhtjobs[i]["subsize"]),log(srnhtjobs[i]["stein_discrepancys"]),linestyle="--",color="red",alpha=0.7)
end
hmean = mean([jobs[i]["stein_discrepancys"] for i=1:num_chain]) 
rmean = mean([rjobs[i]["stein_discrepancys"] for i=1:num_chain]) 
srmean = mean([srjobs[i]["stein_discrepancys"] for i=1:num_chain]) 
srnhtmean = mean([srnhtjobs[i]["stein_discrepancys"] for i=1:num_chain]) 
plot(log(jobs[1]["subsize"]),log(hmean),label="HMC",marker="o",color="purple")
plot(log(jobs[1]["subsize"]),log(rmean),label="Rel HMC",marker="o",color="green")
plot(log(jobs[1]["subsize"]),log(srmean),label="stochastic Rel HMC",marker="o",color="blue")
plot(log(jobs[1]["subsize"]),log(srnhtmean),label="stochastic thermostats Rel HMC",marker="o",color="red")
legend(loc="best")
title("stein discrepancy on log scale, 3-d logistic regression, N=$nobs")
xlabel("log(number of samples)");ylabel("log(stein discrepancy)")
=#
