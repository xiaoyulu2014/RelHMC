# ARGUMENT one JLD file with one variable called jobs
# jobs is an array of Dicts
# each dict is of the form ["d"=>2,"samples"=>randn(2,20),"mu"=>[1:3],"var"=>ones(3),"weights"=>ones(3)/3.0,"subsize"=>[10,20]]
# samples: which should be N(0,sd^2 I )
# mu::Array{Float64}
#var::Array{Float64}
#weights::Array{Float64}
# subsize: computes stein_discrepancys of samples[1:subsize[i]]
# OUTPUT: discrepancies that are written to the same JLD file.
#jobs=[["d"=>2,"samples"=>randn(2,20),"sd"=>1.0,"subsize"=>[10,20]],["d"=>2,"samples"=>randn(2,20),"sd"=>1.0,"subsize"=>[10,20]]]
using JLD
dic=load(ARGS[1]) ## filename in jld
jobs=dic["jobs"]
np=40
if length(ARGS)>=2
  np=parse(Int64,ARGS[2])
end

addprocs(np)
@everywhere begin
    #cd("/homes/vollmer/projects/stein_discrepancy/") ASSUMES to be in Stein folder
include("src/startup.jl")
using SteinDistributions: SteinGMM
using SteinDiscrepancy: stein_discrepancy
solver = "clp"
function eval_gmm(dic::Dict{ASCIIString,Any})
    samples=dic["samples"]
    subsize=dic["subsize"]
    mus=dic["mu"]
    vars=dic["var"]
    weights=dic["weights"]
    stein_discrepancys=zeros(length(subsize))
    target = SteinGMM(mus,vars,weights)

    for i=1:length(subsize)
        result=stein_discrepancy(points=samples[:,1:subsize[i]]',target=target,solver=solver)
        stein_discrepancys[i]=sum(result.objectivevalue)
    end

    stein_discrepancys
end
end
result=pmap(eval_gmm,jobs)
for i=1:length(jobs)
    jobs[i]["stein_discrepancys"]=result[i]
end
save(ARGS[1],"jobs",jobs)
