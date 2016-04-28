@everywhere push!(LOAD_PATH,"/homes/xlu/Documents/RelHMC-group/xiaoyu/src")
@everywhere push!(LOAD_PATH,"/homes/xlu/Documents/RelHMC-group/xiaoyu/models/")
#@everywhere push!(LOAD_PATH,"C:\\Users\\Xiaoyu Lu\\Documents\\RelHMC-group\\xiaoyu\\src")
#@everywhere push!(LOAD_PATH,"C:\\Users\\Xiaoyu Lu\\Documents\\RelHMC-group\\xiaoyu\\models")
@everywhere using SGMCMC
@everywhere using DataModel
@everywhere using LogisticRegression
@everywhere using PyPlot
@everywhere import StatsBase.autocor
@everywhere import Lora.ess
@everywhere import Distributions.MvNormal
using JLD
wk = "/data/greypartridge/oxwasp/oxwasp14/xlu/RelHMC/LR_stepsize/"


#data generation
@everywhere function logit(z)
    1.0./(1.0.+exp(-z))
end

@everywhere nobs=500
@everywhere d=3;
@everywhere C = eye(d);
@everywhere Cinv = inv(C)
@everywhere beta = reshape(rand(MvNormal(zeros(d),C)),(d,1))
@everywhere x= [randn(nobs,d-1) ones(nobs,1)]
@everywhere y=(rand(nobs).<reshape(logit(x *beta),nobs))*2.0-1.0;

#=
color = [((item+2)/4,0.1,(item+2)/3) for item in y]        
scatter(x[:,1],x[:,2],color=color)
=#
@everywhere dm = LogisticRegressionModel(d,x,y,Cinv,nobs)
@everywhere M = 100 # minibatch
@everywhere function myrun(s::SamplerState,dm::AbstractDataModel;num_iterations=1000)
    grad = getgrad(dm)
    llik = getllik(dm)
    samples = zeros(num_iterations, length(s.x))
    zeta = zeros(num_iterations)
    for i = 1:num_iterations
        if (typeof(s) <: SGNHTRelHMCState ||  typeof(s) <: SGRelHMCState)
            dm.subobs=M
            grad = getgrad(dm)
            llik = getllik(dm)
        end
        SGMCMC.sample!(s,llik,grad)
        samples[i,:] = s.x
    end
    samples
end

##stein discrepancy
#optimal parameters
#=
shmc=HMCState(zeros(d),stepsize=0.01,niters=50)
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

#posterior density
if length(beta)==2
    posterior(beta) = exp(getllik(dm)(collect(beta)))
    b1 = linspace(-2, 2, 50);
    b2 = linspace(-1, 2, 50);

    simpost = zeros(50,50);
    for i = 1:length(b1)
        for j = 1:length(b2)
            simpost[i,j] = exp(getllik(dm)([ b1[i],b2[j] ]))
        end;
    end;
    mesh(b1,b2,simpost)
    xlabel("beta1")
    ylabel("beta2")
    zlabel("Posterior density")
end
    
    
    
#stein discrepancy vs stepsize
@everywhere mass,L=  1,50
@everywhere num_iter=10000
@everywhere num_chain=5
@everywhere stepsizevec = exp(linspace(-5,0,16))

samples,rsamples,srsamples,srnhtsamples = [SharedArray(Float64,num_iter,d,num_chain*length(stepsizevec)) for i=1:4]
@sync @parallel for i=1:length(stepsizevec)
	 for j=1:num_chain
		samples[:,:,(i-1)*num_chain+j] = myrun(HMCState(zeros(d),stepsize=stepsizevec[i],niters=L,mass=mass),dm,num_iterations=num_iter)
		rsamples[:,:,(i-1)*num_chain+j] = myrun(RelHMCState(zeros(d),stepsize=stepsizevec[i],niters=L,mass=mass,c=0.1),dm,num_iterations=num_iter)
		srsamples[:,:,(i-1)*num_chain+j] = myrun(SGRelHMCState(zeros(d),stepsize=stepsizevec[i],niters=L,mass=[mass],c=[0.1]),dm,num_iterations=num_iter)
		srnhtsamples[:,:,(i-1)*num_chain+j] = myrun(SGNHTRelHMCState(zeros(d),stepsize=stepsizevec[i],niters=L,mass=[mass],c=[0.1]),dm,num_iterations=num_iter)
	end
end
original_sample=["HMCsamples" => samples, "rHMCsamples" => rsamples, "srHMCsamples" => srsamples, "srnhtHMCsamples" => srnhtsamples,
"beta" => beta, "x" => x, "y" => y]
save(string(wk,"LR_orignalsamples.jld"),"original_sample",original_sample)

#=
dict = load(string(wk,"LR_originalsamples.jld"),"original_sample")
samples,rsamples,srsamples,srnhtsamples,beta,x,y = dict["HMCsamples"],dict["rHMCsamples"],dict["srHMCsamples"],dict["srnhtHMCsamples"],dict["beta"],dict["x"],dict["y"]

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
=#
#thinning
idx=round(linspace(1,num_iter,1000))
sigma=1;
subsize=[1000]
jobs,rjobs,srjobs,srnhtjobs=[],[],[],[];
for i=1:num_chain*length(stepsizevec)
	jobs=[jobs,["d"=>d,"samples"=>samples[idx,:,i]',"sd"=>sigma,"subsize"=>subsize,"X"=>x,"y"=>y]]
	rjobs=[rjobs,["d"=>d,"samples"=>rsamples[idx,:,i]',"sd"=>sigma,"subsize"=>subsize,"X"=>x,"y"=>y]]
	srjobs=[srjobs,["d"=>d,"samples"=>srsamples[idx,:,i]',"sd"=>sigma,"subsize"=>subsize,"X"=>x,"y"=>y]]
	srnhtjobs=[srnhtjobs,["d"=>d,"samples"=>srnhtsamples[idx,:,i]',"sd"=>sigma,"subsize"=>subsize,"X"=>x,"y"=>y]]
end
save(string(wk,"LR_s.jld"),"jobs",jobs)
save(string(wk,"LR_rs.jld"),"jobs",rjobs)
save(string(wk,"LR_srs.jld"),"jobs",srjobs)
save(string(wk,"LR_srnhts.jld"),"jobs",srnhtjobs)

#=
dic=load("LR_s.jld") 
jobs=dic["jobs"]
rdic=load("LR_rs.jld")
rjobs=rdic["jobs"]
srdic=load("LR_srs.jld")
srjobs=srdic["jobs"]
srnhtdic=load("LR_srnhts.jld") 
srnhtjobs=srnhtdic["jobs"]


stein = reshape(float([ jobs[i]["stein_discrepancys"][1] for i=1:length(jobs) ]),num_chain,length(stepsizevec))
stein_r = reshape(float([ rjobs[i]["stein_discrepancys"][1] for i=1:length(jobs) ]),num_chain,length(stepsizevec))
stein_sr = reshape(float([ srjobs[i]["stein_discrepancys"][1] for i=1:length(jobs) ]),num_chain,length(stepsizevec))
stein_srnht = reshape(float([ srnhtjobs[i]["stein_discrepancys"][1] for i=1:length(jobs) ]),num_chain,length(stepsizevec))


for i=1:num_chain
	plot(log(stepsizevec),log(stein[i,:][:]),linestyle="--",color="green",alpha=0.5)
	plot(log(stepsizevec),log(stein_r[i,:][:]),linestyle="--",color="red",alpha=0.5)
	plot(log(stepsizevec),log(stein_sr[i,:][:]),linestyle="--",color="yellow",alpha=0.5)
	plot(log(stepsizevec),log(stein_srnht[i,:][:]),linestyle="--",color="pink",alpha=0.5)
end
plot(log(stepsizevec),log(mean(stein,1))',label="HMC",marker="o",color="green")
plot(log(stepsizevec),log(mean(stein_r,1))',label="Rel HMC",marker="o",color="red")
plot(log(stepsizevec),log(mean(stein_sr,1))',label="stochastic Rel HMC",marker="o",color="yellow")
plot(log(stepsizevec),log(mean(stein_srnht,1))',label="stochastic thermostats Rel HMC",marker="o",color="pink")
title("stein discrepancy vs stepsize, $d-d logistic regression, log scale, c=0.1")
xlabel("stepsize");ylabel("stein discrepancy")
legend()





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
title("stein discrepancy on log scale, $d-d logistic regression, N=$nobs")
xlabel("log(number of samples)");ylabel("log(stein discrepancy)")
=#
