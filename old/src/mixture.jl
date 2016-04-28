cd("/Users/sjv/syncthing/computations/relativistic")
## check sign in normlogpdf
include("utils/probability_utils.jl")
include("MLUtilities.jl")
include("mixtureLib.jl")

#simplest case known weights 0.5 0.5
using Distributions
using StatsBase

comp=2
d=3
w=ones(comp)/comp
nu=5.0
precprior=Wishart(nu,eye(3))
mu0s=zeros(d, comp)
precs = Array(Any, comp)
for i = 1:comp
    precs[i] = rand(precprior)
    mu0s[:,i] = rand(MvNormal(inv(precs[i])))
end

dataDist=[ MvNormal(mu0s[:,i],inv(precs[i])) for i = 1:comp]
#generate data
N=100
z=StatsBase.sample([1:comp],WeightVec(w),N)
global y=zeros(N,d) # IMP: possible speed up
for i=1:N
  y[i,:]=rand(dataDist[z[i]])
end
nmix=NormMix(comp,3,w,nu,mu0s,y)

#setup parameter vector


mus=deepcopy(mu0s)
# evaluate gradient
L=Array[(A=rand(d,d);A-triu(A,1)) for i = 1:nmix.comp]
L=convert(Array{Array{Float64,2},1},L)
D=ones(d,comp)
th=[vec_matrix(D), vec_matrix(mus), copy(w), vec_L(L)]

################## function for vectorisaiton and gradient


g1_LL, g2_LL = MLUtilities.checkgrad( th,x-> (Lpost(x,nmix)),x-> Lpost_gradient(x,nmix), eps=1e-6)


# function grad(para,batchsize)
#     (mus,L,D)=dvecMix(para)
# end

addprocs(4)
function sendTo(p::Int; args...)
    for (nm, val) in args
        @spawnat(p, eval(Main, Expr(:(=), nm, val)))
    end
end
@everywhere begin
  cd("/Users/sjv/syncthing/computations/relativistic/src/utils")
  include("utils/probability_utils.jl")
  include("MLUtilities.jl")
  include("mixtureLib.jl.jl")

  #simplest case known weights 0.5 0.5
  using Distributions
  using StatsBase
  include("SGMCMC.jl")
end
include("SGMCMC.jl")

nnparms=nmix.pd

ss=100
global fails=0
function valid(th)
  global fails
  res=!any(th[1:6].<=0) && !any(isnan(th))
  if !res
    fails+=1
  end
  return res
end

th=[vec_matrix(D), vec_matrix(mus), vec_L(L)]
h=1.0e-3
sgldspecs=SGMCMC.SGLDR_specs(h,valid;mass=[1.0],niters=10)
ss=100000
[copy(th) for i=1:10]
SGMCMC.sgldr!(th,p->LpostFW_gradient(p,nmix,w),sgldspecs)
map(SGMCMC.sgldr!,[copy(th) for i=1:10],

