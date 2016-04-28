# ARGUMENT one JLD file with one variable called jobs
# jobs is an array of Dicts\
# each dict is of the form ["d"=>2,"samples"=>randn(2,20),"sd"=>1.0,"subsize"=>[10,20]]
# d: dimension
# samples: which should be N(0,sd^2 I )
# sd: standard deviation
# subsize: computes stein_discrepancys of samples[1:subsize[i]]
# OUTPUT: discrepancies that are written to the same JLD file. 
#jobs=[["d"=>2,"samples"=>randn(2,20),"sd"=>1.0,"subsize"=>[10,20]],["d"=>2,"samples"=>randn(2,20),"sd"=>1.0,"subsize"=>[10,20]]]
using JLD
dic=load(ARGS[1]) ## filename in jld
jobs=dic["jobs"]

addprocs(40)
@everywhere begin
    #cd("/homes/vollmer/projects/stein_discrepancy/") ASSUMES to be in Stein folder
include("src/startup.jl")
using SteinDistributions: SteinGaussian, SteinScaleLocationStudentT
using SteinDiscrepancy: stein_discrepancy
solver = "clp"
function eval_isogaussian(dic::Dict{ASCIIString,Any})
    d=dic["d"]
    samples=dic["samples"]/dic["sd"]
    subsize=dic["subsize"]
    stein_discrepancys=zeros(length(subsize))
    gaussian_target = SteinGaussian(d)

    for i=1:length(subsize)
        result=stein_discrepancy(points=samples[:,1:subsize[i]]',target=gaussian_target,solver=solver)
        stein_discrepancys[i]=sum(result.objectivevalue)
    end

    stein_discrepancys
end
end
result=pmap(eval_isogaussian,jobs)
for i=1:length(jobs)
    jobs[i]["stein_discrepancys"]=result[i]
end
save(ARGS[1],"jobs",jobs)
