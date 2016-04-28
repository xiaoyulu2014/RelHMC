type NormMix ## type for mixture of normals
    comp::Int64
    d::Int64
    tril_ind::Array{Int64,1}
    ntril::Int64
    w::Array{Float64,1}
    pd::Int64
    nu::Float64
    mu0s::Array{Float64,2}
    y::Array{Float64,2}
    function NormMix(comp,d,w,nu,mu0s,y)
        tril_ind=tril_inds(eye(3),-1)
        nrtil=length(tril_ind)
        pd=comp*d+nrtil*comp+d*comp
        new(comp,d,tril_ind,nrtil,w,pd,nu,mu0s,y)
    end

end

# vectorization function
vec_matrix = th -> th[:]
devec_matrix = (th, y) -> th[:] = y[:]

# g1_D, g2_D = MLUtilities.checkgrad( vec_matrix(D), D_prior, D_prior_gradient


vec_L = th -> cat(1, [vectorize_L(LL) for LL in th]...)
function devec_L(th, y)
    A = reshape(y, (int(nmix.d*(nmix.d-1)/2), nmix.comp))
    for i = 1:nmix.comp
        devectorize_L(th[i], A[:,i])
    end
end





function normal_dmu_logpdf(x, mu, L::Matrix{Float64}, D::Vector{Float64})
    diffs = get_diff(x,mu')
    diffs * L' * diagm(1./D) * L
end
function normal_logpdf(x, mu, L::Matrix{Float64}, D::Vector{Float64})
    diffs = get_diff(x,mu')
    -0.5*(diffs * L') * diagm(1./D) *( L *diffs')
end
function Lpost(th,nmix)
    D = zeros(nmix.d, nmix.comp)
    mus = zeros(nmix.d, nmix.comp)
    L::Array{Array{Float64,2},1} = [eye(d)::Array{Float64,2} for i=1:nmix.comp]
    y=nmix.y
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

    @show GMM_likelihood(y, w, mus, L,D)+GMM_L_prior(L,D,nmix.comp)+GMM_D_prior(D, nmix.nu, nmix.comp) #prior on covariance

#     return res
end


D_prior = th -> (D = zeros(nmix.d, nmix.comp); devec_matrix(D, th); GMM_D_prior(D, nmix.nu, nmix.comp))
D_prior_gradient = th -> (D = zeros(nmix.d, nmix.comp); devec_matrix(D, th); GMM_D_prior_gradient(D, nmix.nu, nmix.comp))


function Lpost_gradient(th,nmix)
    D = zeros(nmix.d, nmix.comp)
    mus = zeros(nmix.d, nmix.comp)
    L::Array{Array{Float64,2},1} = [eye(d)::Array{Float64,2} for i=1:nmix.comp]
    y=nmix.y
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
    y=nmix.y
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