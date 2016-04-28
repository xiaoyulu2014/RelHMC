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
model,name = make_dense_nn([500,300],10)
dm = MochaSGMCMCDataModel(images,dlabel,model,backend)
dmtest = MochaSGMCMCDataModel(timages,tdlabel,model,backend,do_accuracy=true)# for test set accuracy
dmtraintest = MochaSGMCMCDataModel(images,dlabel,model,backend,do_accuracy=true)# for training set accuracy
nparam=MochaDataModel.fetchnparams(dm)

MochaDataModel.fetchnparams(dm)
x = MochaDataModel.fetchparams(dm)
y = deepcopy(x)
end

@everywhere function myrun(s::SamplerState,dm::AbstractDataModel;num_iterations=50001)  #### 10001 for adam
 		grad = getgrad(dm)
    	llik = getllik(dm)
 		acc_train=Float64[]; acc_test=Float64[];
       for i = 1:num_iterations
       (typeof(s) <: AdamState || typeof(s) <: AdagradState || typeof(s) <: SGLDState) ? SGMCMC.sample!(s,grad) :  SGMCMC.sample!(s,llik,grad)
       if rem(i,60000/100) == 1
            push!(acc_train,MochaDataModel.evaluate(dmtraintest,s.x)[:accuracy])
            push!(acc_test, MochaDataModel.evaluate(dmtest,s.x)[:accuracy])
            println(acc_test[end])
        end

    end
    return acc_train,acc_test
end


@everywhere stepsizevec, mvec, avec = [0.01,0.05,0.1],[0.005,0.01,0.05],[0.0005,0.001,0.005]
#@everywhere stepsizevec, mvec, avec = [0.1,0.5],[0.1,0.5,1],[3,5,10]
@everywhere myn=length(stepsizevec)*length(mvec)*length(avec)
@everywhere t=Iterators.product(stepsizevec,mvec,avec)
@everywhere myt=Array(Any,myn);
@everywhere it=1;
@everywhere for prod in t
	myt[it]=prod;
    it+=1;
end

@everywhere function funcr_thermo_ind(param)
	stepsize,m,a = param
	x=deepcopy(y)
	return(myrun(SGNHTRelHMCState(x,stepsize=stepsize,c=[a/stepsize],mass=[m],independent_momenta=true),dm)[:])
end
resultr_thermo_ind=pmap(funcr_thermo_ind,myt)
acctest_rt_ind=Array(Float64,myn)
for i=1:myn acctest_rt_ind[i]=resultr_thermo_ind[i][2][end] end
acctest_rt_ind=reshape(acctest_rt_ind,3,3,3)



####compare my zero temp with adam######
@everywhere function myfunc(param)
	stepsize,m,a = param
	x=deepcopy(y)
	return(myrun(myState(x,stepsize=stepsize,c=[a/stepsize],mass=[m],independent_momenta=true),dm)[:])
end

@everywhere hyp=[0.5,0.5,5];
acc_ind=pmap(myfunc,[hyp for i=1:10])
@everywhere function adamfunc(i)
	x=deepcopy(y)
	sadam=AdamState(x)
	return(myrun(sadam,dm)[:])
end
adam=pmap(adamfunc,[i for i=1:10])
adam_mean_train,adam_mean_test,my_mean_train,my_mean_test = [Array(Float64,length(adam[1][1])) for i=1:4]
for i=1:10
	adam_mean_train += adam[i][1]
	adam_mean_test += adam[i][2]
	my_mean_train += acc_ind[i][1]
	my_mean_test += acc_ind[i][2]
end
adam_mean_train /= 10
adam_mean_test /= 10
my_mean_train /= 10
my_mean_test /= 10

Adam_my=["adam" => adam, "acc_ind" => acc_ind, "hyp" => hyp]
#save("/homes/xlu/Documents/RelHMC-group/xiaoyu/plots/wk1/Adam_my.jld","Adam_my",Adam_my)
subplot(121)
for i=1:10
plot(adam[i][1],color="blue",linestyle="--",alpha=0.5)
plot(acc_ind[i][1],color="red",linestyle="--",alpha=0.5)
end
plot(adam_mean_train,color="blue",label="Adam")
plot(my_mean_train,color="red",label="zero temp")
legend(loc="3");title("training set")
subplot(122)
for i=1:10
plot(adam[i][2],color="blue",linestyle="--",alpha=0.5)
plot(acc_ind[i][2],color="red",linestyle="--",alpha=0.5)
end
plot(adam_mean_test,color="blue",label="Adam")
plot(my_mean_test,color="red",label="zero temp")
legend(loc="3");title("test set")
suptitle("MNIST with 1 hidden layer and 100 hidden units")
subplot(121)
for i=1:10
plot(adam[i][1],color="blue",linestyle="--",alpha=0.5)
plot(acc_ind[i][1],color="red",linestyle="--",alpha=0.5)
end
plot(adam_mean_train,color="blue",label="Adam")
plot(my_mean_train,color="red",label="zero temp")
legend(loc="3");title("training set")
ylim([0.994,1])
subplot(122)
for i=1:10
plot(adam[i][2],color="blue",linestyle="--",alpha=0.5)
plot(acc_ind[i][2],color="red",linestyle="--",alpha=0.5)
end
plot(adam_mean_test,color="blue",label="Adam")
plot(my_mean_test,color="red",label="zero temp")
legend(loc="3");title("test set")
ylim([0.95,1])
suptitle("MNIST with 1 hidden layer and 100 hidden units")





@everywhere function funcr_thermo(param)
	stepsize,m,a = param
	x=deepcopy(y)
	return(myrun(SGNHTRelHMCState(x,stepsize=stepsize,c=[a/stepsize],mass=[m],independent_momenta=false),dm)[:])
end
resultr_thermo=pmap(funcr_thermo,myt)
acctest_rt=Array(Float64,myn)
for i=1:myn acctest_rt[i]=resultr_thermo[i][2][end] end
acctest_rt=reshape(acctest_rt,3,4,4)


##Adam
x=deepcopy(y)
sadam=AdamState(x,stepsize=0.005)
result_adam=myrun(sadam,dm)
x=deepcopy(y)
sadagrad=AdagradState(x)
result_adagrad=myrun(sadagrad,dm)
x=deepcopy(y)
smy=myState(x,stepsize=0.001)
result_my=myrun(smy,dm)

Result=["Adam" => Adam, "myalg" => myalg, "Adagrad" => Adagrad, "myparam" => myt[30]]
#save("/homes/xlu/Documents/RelHMC-group/xiaoyu/plots/wk1/NeuralNet.jld","Result",Result)
Result=load("/homes/xlu/Documents/RelHMC-group/xiaoyu/plots/wk1/NeuralNet.jld")["Result"]
subplot(121)
plot(Adam[1],label="Adam");plot(Adagrad[1],label="Adagrad");plot(myalg[1],label="zero temperature");plot(myalg_ind1[1],label="zero temperature, indep momenta");legend(loc="best")
title("training accuracy")
subplot(122)
plot(Adam[2],label="Adam");plot(Adagrad[2],label="Adagrad");plot(myalg[2],label="zero temperature");plot(myalg_ind1[2],label="zero temperature, indep momenta");legend(loc="best")
title("test accuracy")

@everywhere myn1=length(stepsizevec)*length(mvec)
@everywhere t1=Iterators.product(stepsizevec,mvec)
@everywhere myt1=Array(Any,myn1);
@everywhere it=1;
@everywhere for prod in t1
	myt1[it]=prod;
    it+=1;
end

@everywhere function func(param)
	stepsize,m = param
	x=deepcopy(y)
	return(myrun(SGHMCState(x,stepsize=stepsize,mass=[m]),dm)[:])
end
@everywhere function func_thermo(param)
	stepsize,m = param
	x=deepcopy(y)
	return(myrun(SGNHTHMCState(x,stepsize=stepsize,mass=[m]),dm)[:])
end

result=pmap(func,myt1)
result_thermo=pmap(func_thermo,myt1)

#=
acc_srnht, acc_snht, acc_sr, acc_s = [SharedArray(Float64,myn,2) for i=1:4];
X=Array(Float64,myn,length(y))
for i=1:myn X[i,:] = deepcopy(y) end



@sync @parallel for i=1:myn
    stepsize,m,c = myt[i]
    acc_srnht[i,:] = myrun(SGNHTRelHMCState(X[i,:][:],stepsize=stepsize,c=[c],mass=[m]),dm)[:]
   # srhmc = SGRelHMCState(x,stepsize=stepsize,c=[c],mass=[m]);
   # acc_sr[i,:] = myrun(srhmc,dm)[:]
    println("stepsize = ", stepsize, "; m =", m, "; c=", c, "; acc_rel thermo=", round(acc_srnht[i,:],2))
end



@everywhere x = deepcopy(y)
for i=1:myn
    stepsize,m,c = myt[i]
    x = deepcopy(y)
    shmc = SGHMCState(x,stepsize=stepsize,mass=[m]);
    acc_s[i,:] =  myrun(shmc,dm)[:]
    x = deepcopy(y)
    snhthmc = SGNHTHMCState(x,stepsize=stepsize,mass=[m]);
    acc_snht[i,:] = myrun(snhthmc,dm)[:]
    println("stepsize = ", stepsize, "; m =", m, "; RMSE = ", round(acc_s[i,:],2), "; RMSE thermo=", round(acc_snht[i,:],2))
end

=#




#=
x = deepcopy(y)
s = SGNHTRelHMCState(x,stepsize=0.5,mass=[1.0]);
run(s,dm)
=#
