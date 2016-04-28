@everywhere push!(LOAD_PATH,"../src");
@everywhere push!(LOAD_PATH,"../src/utils");
@everywhere push!(LOAD_PATH,"../models");

@everywhere using Gadfly;
@everywhere using DataModel;
@everywhere using GaussianMixture;
@everywhere using MLUtilities;
@everywhere using SGMCMC;
@everywhere using JLD;
@everywhere using Compat;

@everywhere var_ratio = 1
@everywhere dm = GaussianMixtureModel([-5,0,5],[1/var_ratio,var_ratio,1/var_ratio],[0.3,0.3,0.3])

@everywhere function run_acc(s::SamplerState,dm::AbstractDataModel;num_iterations=1000, final_plot=false)
    grad = getgrad(dm)
    llik = getllik(dm)
    samples = zeros(num_iterations, length(s.x))
    acc = zeros(num_iterations)

    for i = 1:num_iterations
        acc[i] = sample!(s,llik,grad)[2][]
        samples[i,:] = s.x
    end

    if final_plot
        if length(s.x) == 2
            PyPlot.clf()
            llik(x,y) = llik([x,y])
            plot_contour(llik, -5:.05:6, -1:.05:32)
            PyPlot.scatter(samples[:,1], samples[:,2])
        end
    end

    samples, mean(acc)
end


srand(100)
num_it = 1000
c = 1.0
mass= 1.0
step = 1
sa_rel = SharedArray(Float64,num_it);

srhmc = RelHMCState([1.],stepsize = step,c = c,mass=mass)
out_rel = run_acc(srhmc,dm,num_iterations = num_it)
samples = out_rel[1]
samples = transpose(samples)


subsize=[1,10,50,100,200,500,1000]

jobs=[@compat Dict("weights"=>1,"var"=>dm.var, "mu"=>dm.mu,"weights"=>dm.weights,"subsize"=>subsize,"samples"=>samples)]
save("test.jld","jobs",jobs)
run(`julia ../stein_discrepancy/steingmmjld.jl test.jld`)
#Gadfly.plot(x=samples,Geom.histogram)
