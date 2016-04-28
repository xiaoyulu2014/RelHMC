## check sign in normlogpdf
include("utils/probability_utils.jl")
include("MLUtilities.jl")

#simplest case known weights 0.5 0.5
using Distributions
using StatsBase

global comp=2
global d=3
global tril_ind=tril_inds(eye(3),-1)
global ntril=length(tril_ind)
# generate model
global w=ones(comp)/comp
global pd=comp*d+ntril*comp+d*comp # dimension
nu=5.0
precprior=Wishart(nu,eye(3))
mu0s=zeros(d, comp)
precs = Array(Any, comp)
for i = 1:comp
    precs[i] = rand(precprior)
    mu0s[:,i] = rand(MvNormal(inv(precs[i])))
end



type NormMix
    comp::Int64
    d::Int64
    tril_ind::Array{Int64,1}
    ntril::Int64
    w::Array{Float64,1}
    pd::Int64
    nu::Float64
    function NormMix(comp,d,w,nu,mu0s)
        tril_ind=tril_inds(eye(3),-1)
        nrtil=length(tril_ind)
        pd=comp*d+ntril*comp+d*comp
        new(comp,d,tril_ind,ntril,w,pd,nu,mu0s)
    end
    mu0s::Array{Float64,2}
end
nmix=NormMix(comp,3,w,nu,mu0s)
mus=deepcopy(mu0s)


dataDist=[ MvNormal(mus[:,i],inv(precs[i])) for i = 1:comp]
#generate data
N=100
z=StatsBase.sample([1:comp],WeightVec(w),N)
global y=zeros(N,d) # IMP: possible speed up
for i=1:N
  y[i,:]=rand(dataDist[z[i]])
end


#setup vector - do we use precision






# evaluate gradient
L=Array[(A=rand(d,d);A-triu(A,1)) for i = 1:nmix.comp]
L=convert(Array{Array{Float64,2},1},L)
D=ones(d,comp)
################## function for vectorisaiton and gradient


function normal_dmu_logpdf(x, mu, L::Matrix{Float64}, D::Vector{Float64})
    diffs = get_diff(x,mu')
    diffs * L' * diagm(1./D) * L
end
function normal_logpdf(x, mu, L::Matrix{Float64}, D::Vector{Float64})
    diffs = get_diff(x,mu')
    -0.5*(diffs * L') * diagm(1./D) *( L *diffs')
end

vec_matrix = th -> th[:]
devec_matrix = (th, y) -> th[:] = y[:]

# g1_D, g2_D = MLUtilities.checkgrad( vec_matrix(D), D_prior, D_prior_gradient)




D_prior = th -> (D = zeros(nmix.d, nmix.comp); devec_matrix(D, th); GMM_D_prior(D, nmix.nu, nmix.comp))
D_prior_gradient = th -> (D = zeros(nmix.d, nmix.comp); devec_matrix(D, th); GMM_D_prior_gradient(D, nmix.nu, nmix.comp))

g1_D, g2_D = MLUtilities.checkgrad( vec_matrix(D), D_prior, D_prior_gradient)

vec_L = th -> cat(1, [vectorize_L(LL) for LL in th]...)
function devec_L(th, y)
    A = reshape(y, (int(nmix.d*(nmix.d-1)/2), nmix.comp))
    for i = 1:nmix.comp
        devectorize_L(th[i], A[:,i])
    end
end




function Lpost(th,nmix)
    D = zeros(nmix.d, nmix.comp)
    mus = zeros(nmix.d, nmix.comp)
    L::Array{Array{Float64,2},1} = [eye(d)::Array{Float64,2} for i=1:nmix.comp]

    nD = int(nmix.d*nmix.comp)
    K = nmix.comp

    thD = th[1:nD]
    thmu = th[nD+1:2nD]
    thw = th[2nD+1:2nD+K]
    thL = th[2nD+K+1:end]

    devec_matrix(D, thD)
    devec_matrix(mus, thmu)
    w = copy(thw)
    devec_L(L, thL)

    res=GMM_likelihood(y, w, mus, L,D)+GMM_L_prior(L,D,nmix.comp)+GMM_D_prior(D, nmix.nu, nmix.comp) #prior on covariance

    return res
end




function Lpost_gradient(th,nmix)
    D = zeros(nmix.d, nmix.comp)
    mus = zeros(nmix.d, nmix.comp)
    L::Array{Array{Float64,2},1} = [eye(d)::Array{Float64,2} for i=1:nmix.comp]

    K = nmix.comp
    nD = int(nmix.d*nmix.comp)

    thD = th[1:nD]
    thmu = th[nD+1:2nD]
    thw = th[2nD+1:2nD+K]
    thL = th[2nD+K+1:end]

    devec_matrix(D, thD)
    devec_matrix(mus, thmu)
    w = copy(thw)
    devec_L(L, thL)


    (dw, dmu, dL, dD) = GMM_likelihood_gradient(y, w, mus, L,D)
    (dL1, dD1) = GMM_L_prior_gradient(L,D,nmix.comp);dL+=dL1;dD+=dD1#prior on covariance
    dD+=GMM_D_prior_gradient(D, nmix.nu, nmix.comp) #prior on covariance

    [vec_matrix(dD), vec_matrix(dmu), dw, vec_L(dL)]
end
function LpostFW_gradient(th,nmix,thw) # fixed weights
    D = zeros(nmix.d, nmix.comp)
    mus = zeros(nmix.d, nmix.comp)
    L::Array{Array{Float64,2},1} = [eye(d)::Array{Float64,2} for i=1:nmix.comp]

    K = nmix.comp
    nD = int(nmix.d*nmix.comp)

    thD = th[1:nD]
    thmu = th[nD+1:2nD]
    thL = th[2nD+1:end]

    devec_matrix(D, thD)
    devec_matrix(mus, thmu)
    w = copy(thw)
    devec_L(L, thL)


    (dw, dmu, dL, dD) = GMM_likelihood_gradient(y, w, mus, L,D)
    dw=0.0*dw
    (dL1, dD1) = GMM_L_prior_gradient(L,D,nmix.comp);dL+=dL1;dD+=dD1#prior on covariance
    dD+=GMM_D_prior_gradient(D, nmix.nu, nmix.comp) #prior on covariance

    [vec_matrix(dD), vec_matrix(dmu),  vec_L(dL)]
end

th=[vec_matrix(D), vec_matrix(mus), copy(w), vec_L(L)]

g1_LL, g2_LL = MLUtilities.checkgrad( th,x-> (Lpost(x,nmix)),x-> Lpost_gradient(x,nmix), eps=1e-6)
g1_LL
batchsize=10

# function grad(para,batchsize)
#     (mus,L,D)=dvecMix(para)
# end

addprocs(4)

@everywhere begin
  cd("/Users/sjv/syncthing/computations/relativistic/src/utils")
  include("utils/probability_utils.jl")
  include("MLUtilities.jl")

  #simplest case known weights 0.5 0.5
  using Distributions
  using StatsBase
  include("SGMCMC.jl")
end
nnparams=pd



ss=100
function valid(th)
    return !any(th[1:6].<=0) && !any(isnan(th))
end

th=[vec_matrix(D), vec_matrix(mus), vec_L(L)]
h=1.0e-3
sgldspecs=SGMCMC.SGLDR_specs(h,valid;mass=[1.0],niters=10)
ss=1000
[copy(th) for i=1:10]
sgldr!(th,p->LpostFW_gradient(p,nmix,w),sgldspecs)
map(sgldr!,[copy(th) for i=1:10],
