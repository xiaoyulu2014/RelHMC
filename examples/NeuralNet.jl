addprocs(20)
@everywhere begin
push!(LOAD_PATH,"/homes/xlu/Documents/RelHMC-group/src")
push!(LOAD_PATH,"/homes/xlu/Documents/RelHMC-group/models/")

using SGMCMC
using DataModel
using Mocha
using MochaDataModel
using MochaWrapper2
using JLD
using HDF5
using Iterators

#data_path = "/data/greyheron/oxwasp/oxwasp14/xlu/MNIST/"
data_path = "/homes/xlu/Documents/RelHMC-group/MNIST/"
trainFile = h5open( "$(data_path)mnist_train.hdf5", "r" )
testFile  = h5open( "$(data_path)mnist_test.hdf5", "r" )
images    = convert(Array{Float64,4},trainFile["data"][:,:,:,:])
dlabel    = convert(Array{Float64,2},trainFile["label"][:,:])
timages   = convert(Array{Float64,4},testFile["data"][:,:,:,:])
tdlabel   = convert(Array{Float64,2},testFile["label"][:,:])

backend = initMochaBackend(false)
include("/homes/xlu/Documents/RelHMC-group/models/MochaModelFactories/models.jl")
model,name = make_dense_nn([100],10)
dm = MochaSGMCMCDataModel(images,dlabel,model,backend)
dmtest = MochaSGMCMCDataModel(timages,tdlabel,model,backend,do_accuracy=true)# for test set accuracy
dmtraintest = MochaSGMCMCDataModel(images,dlabel,model,backend,do_accuracy=true)# for training set accuracy
nparam=MochaDataModel.fetchnparams(dm)

MochaDataModel.fetchnparams(dm)
x = MochaDataModel.fetchparams(dm)
y = deepcopy(x)

function myrun(s::SamplerState,dm::AbstractDataModel;num_iterations=10001,annealing=false)  
 		grad = getgrad(dm)
    	llik = getllik(dm)
 		acc_train=Float64[]; acc_test=Float64[];
       for i = 1:num_iterations
       	
       	if annealing s.beta=i^0.5 end
       (typeof(s) <: AdamState || typeof(s) <: AdagradState || typeof(s) <: SGLDState) ? SGMCMC.sample!(s,grad) :  SGMCMC.sample!(s,llik,grad)
       	   
		   if rem(i,60000/100) == 1
		        push!(acc_train,MochaDataModel.evaluate(dmtraintest,s.x)[:accuracy])
		        push!(acc_test, MochaDataModel.evaluate(dmtest,s.x)[:accuracy])
		        println(acc_test[end])
		    end

    end
    return acc_train,acc_test
end
end

##tuning RelHMC
@everywhere stepsizevec, mvec, avec, Dvec = [0.0001,0.001],[0.05],[0.001],[1e-7,1e-6,1e-5,1e-4] #0.003]
@everywhere myn=length(stepsizevec)*length(mvec)*length(avec)*length(Dvec)
@everywhere t=Iterators.product(stepsizevec,mvec,avec,Dvec)
@everywhere myt=Array(Any,myn);
@everywhere it=1;
@everywhere for prod in t
	myt[it]=prod;
    it+=1;
end

@everywhere function funcr_thermo_ind(param)
	stepsize,m,a,D = param
	x=deepcopy(y)
	return(myrun(SGNHTRelHMCState(x,stepsize=stepsize,c=[a/stepsize],mass=[m],D=[D],niters=1,independent_momenta=true),dm,annealing=true)[:])
end
resultr_thermo_ind=pmap(funcr_thermo_ind,myt)
acctest_rt_ind=Array(Float64,myn)
for i=1:myn acctest_rt_ind[i]=resultr_thermo_ind[i][2][end] end
acctest_rt_ind=reshape(acctest_rt_ind,1,1,2,4)



#result independent
resultr_thermo_ind=funcr_thermo_ind([0.1,0.05,0.001])
resultr_thermo_ind=([0.125133,0.7342,0.81625,0.8415,0.8462,0.8453,0.841133,0.838783,0.82245,0.82025,0.806367,0.79925,0.7967,0.79155,0.79275,0.778617,0.778983],[0.1322,0.7442,0.8214,0.8455,0.8464,0.8452,0.843,0.8372,0.8219,0.8254,0.8086,0.8007,0.8027,0.7975,0.7959,0.7838,0.7851])
=#


x=deepcopy(y)
mystate=SGNHTRelHMCState(x,stepsize=0.00045,c=[0.01],mass=[0.05],D=[20],independent_momenta=true)
out=myrun(mystate,dm)[:]


















