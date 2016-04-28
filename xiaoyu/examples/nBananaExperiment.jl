@everywhere push!(LOAD_PATH,"/homes/xlu/Documents/RelHMC-group/xiaoyu/src")
@everywhere push!(LOAD_PATH,"/homes/xlu/Documents/RelHMC-group/xiaoyu/models/")
@everywhere using SGMCMC
@everywhere using DataModel
@everywhere using Banana_n
@everywhere using PyPlot
@everywhere using JLD
#@everywhere wk = "/data/greypartridge/oxwasp/oxwasp14/xlu/RelHMC/nBanana_stepsize/"
#@everywhere wk = "/data/greypartridge/oxwasp/oxwasp14/xlu/RelHMC/nBanana_c/"

@everywhere wk = "/homes/xlu/Documents/RelHMC-group/xiaoyu/plots/wk12/nBanana_stepsize/"

#@everywhere b=rand(10).*sign(randn(10))
#@everywhere b = load("/homes/xlu/Documents/RelHMC-group/xiaoyu/examples/nBanana_param.jld")["nbanana_param"]["b"]
@everywhere b = [0.01 0.5]
@everywhere d=2*length(b)
@everywhere dm = BananaModel_n(b)
@everywhere function myrun(s::SamplerState,dm::AbstractDataModel;num_iterations=1000)
    grad = getgrad(dm)
    llik = getllik(dm)
    samples = zeros(num_iterations, length(s.x))
    zeta = zeros(num_iterations)
    for i = 1:num_iterations

        sample!(s,llik,grad)
        samples[i,:] = s.x
        if typeof(s) <: SGNHTRelHMCState  zeta[i] = s.zeta[1]  end
    end
    samples
end


#=
stepsize=0.05
samples=myrun(HMCState(zeros(d),stepsize=stepsize,niters=50,mass=1),dm,num_iterations=1000)
rsamples=myrun(RelHMCState(zeros(d),stepsize=stepsize,c=1,niters=50,mass=1),dm,num_iterations=1000)
subplot(221);plot(samples[:,1],label="HMC");plot(rsamples[:,1],label="Rel HMC");legend(loc="best");xlabel("number of samples");ylabel("x_1")
subplot(222);plot(samples[:,2],label="HMC");plot(rsamples[:,2],label="Rel HMC");legend(loc="best");xlabel("number of samples");ylabel("x_2")
subplot(223);plot(samples[:,3],label="HMC");plot(rsamples[:,3],label="Rel HMC");legend(loc="best");xlabel("number of samples");ylabel("x_1")
subplot(224);plot(samples[:,4],label="HMC");plot(rsamples[:,4],label="Rel HMC");legend(loc="best");xlabel("number of samples");ylabel("x_2")
suptitle("traceplots of $d-d banana samples, stepsize=1, L=50")

@everywhere mass,L=  1,50
@everywhere num_iter=10000
@everywhere num_chain=5
#stein discrepancy vs c
@everywhere cvec = exp(linspace(-4,4,16))
@everywhere stepsize = 0.013
rsamples = SharedArray(Float64,num_iter,d,num_chain*length(cvec)) 
@sync @parallel for i=1:length(cvec)
	 for j=1:num_chain
		rsamples[:,:,(i-1)*num_chain+j] = myrun(RelHMCState(zeros(d),stepsize=stepsize,niters=L,mass=mass,c=cvec[i]),dm,num_iterations=num_iter)
	end
end
original_sample=["rHMCsamples" => rsamples,"cvec" => cvec,"b" => b]
save("$(wk)nBanana_originalsamples.jld","original_sample",original_sample)
=#


#stein discrepancy vs stepsize
@everywhere mass,L=  1,50
@everywhere num_iter=10000
@everywhere num_chain=5
@everywhere stepsizevec = exp(linspace(-5,0,16))
@everywhere c = 1 #2

samples,rsamples = [SharedArray(Float64,num_iter,d,num_chain*length(stepsizevec)) for i=1:2]
@sync @parallel for i=1:length(stepsizevec)
	 for j=1:num_chain
		samples[:,:,(i-1)*num_chain+j] = myrun(HMCState(zeros(d),stepsize=stepsizevec[i],niters=L,mass=mass),dm,num_iterations=num_iter)
		rsamples[:,:,(i-1)*num_chain+j] = myrun(RelHMCState(zeros(d),stepsize=stepsizevec[i],niters=L,mass=mass,c=c),dm,num_iterations=num_iter)
	end
end
#=nBanana_param=["b" => b, "stepsizevec" => stepsizevec, "cvec" => cvec]
save("/homes/xlu/Documents/RelHMC-group/xiaoyu/examples/nBanana_param.jld","nBanana_param",nBanana_param)
=#
sigma=1;
function phi(x)
	y=Float64[]
	for i=1:d/2 y=[y,[sigma/10*x[2*(i-1)+1],sigma*(x[2*i]+b[i]*x[2*(i-1)+1]^2-100*b[i])]] end
	return(y)
end

y,ry=[Array(Float64,d,num_iter,num_chain*length(cvec)) for i=1:2]
for i=1:num_iter,j=1:num_chain*length(cvec)
	y[:,i,j]=phi(samples[i,:,j]) 
	ry[:,i,j]=phi(rsamples[i,:,j]) 
end

#thinning
idx=round(linspace(1,num_iter,1000))
#idx=1:1000 #unthinned small datasets
jobs,rjobs,srjobs,srnhtjobs=[],[],[],[];;subsize=[1000]
for i=1:num_chain*length(cvec)
	jobs=[jobs,["d"=>d,"samples"=>y[:,idx,i],"sd"=>sigma,"subsize"=>subsize]]
	rjobs=[rjobs,["d"=>d,"samples"=>ry[:,idx,i],"sd"=>sigma,"subsize"=>subsize]]
end
gjobs=[["d"=>d,"samples"=>randn(d,length(idx)),"sd"=>sigma,"subsize"=>subsize] for i=1:num_chain]
save("$(wk)nbanana.jld","jobs",jobs)
save("$(wk)nbanana_r.jld","jobs",rjobs)
save("$(wk)ngaussian.jld","jobs",gjobs)

##plot vs stepsize
dic=load("$(wk)nbanana.jld") 
jobs=dic["jobs"]
rdic=load("$(wk)nbanana_r.jld")
rjobs=rdic["jobs"]
gdic=load("$(wk)ngaussian.jld")
gjobs=gdic["jobs"]

stein = reshape(float([ jobs[i]["stein_discrepancys"][1] for i=1:length(jobs) ]),num_chain,length(stepsizevec))
stein_r = reshape(float([ rjobs[i]["stein_discrepancys"][1] for i=1:length(rjobs) ]),num_chain,length(stepsizevec))
stein_g = mean(float([ gjobs[i]["stein_discrepancys"][1] for i=1:length(gjobs) ]))

for i=1:num_chain
	plot(log(stepsizevec),log(stein[i,:][:]),linestyle="--",color="green",alpha=0.5)
	plot(log(stepsizevec),log(stein_r[i,:][:]),linestyle="--",color="red",alpha=0.5)
end
plot(log(stepsizevec),log(mean(stein,1))',label="HMC",marker="o",color="green")
plot(log(stepsizevec),log(mean(stein_r,1))',label="Rel HMC",marker="o",color="red")
axhline(log(stein_g),0,label="20-d gaussian")
title("stein discrepancy vs stepsize on log scale, with 10 different banana, c=1")
xlabel("log(stepsize)");ylabel("log(stein discrepancy)")
legend(loc="best")

##plot vs c
stein_r = reshape(float([ rjobs[i]["stein_discrepancys"][1] for i=1:length(rjobs) ]),num_chain,length(cvec))
plot(log(cvec),log(mean(stein_r,1))',label="Rel HMC",marker="o",color="red")
