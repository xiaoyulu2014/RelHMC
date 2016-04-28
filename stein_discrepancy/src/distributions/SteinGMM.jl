# SteinGMM (alternative version)
using Distributions: Multinomial,  Normal
import Distributions

type SteinGMM <: SteinDistribution
    # Model parameters
    mu::Array{Float64}
    var::Array{Float64}
    weights::Array{Float64}
    # Stein factors
    c1::Float64
    c2::Float64
    c3::Float64
end


#useful functions
normldf(x,mu,var) =  -.5*log(2.0*pi*var) -.5*((x-mu).*(x-mu))/var
normpdf(x,mu,var) = exp(normldf(x,mu,var))
function logsumexp(x...)
  m = max(x...)
  return m + log(+([exp(a-m) for a in x]...))
end


#Constructor
SteinGMM(mu::Array{Float64}, var::Array{Float64}, weights::Array{Float64}) =
SteinGMM(mu, var, weights, 1.0, 1.0, 1.0)


###Draw n independent samples from the distribution
function rand(d::SteinGMM, n::Int64)
    x = Array(Float64, n)
    num_comp = length(d.mu)
    norm_weights = d.weights/sum(d.weights)
    index = mapslices(find, rand(Multinomial(1, norm_weights), n), 1) #indices of the gaussians to sample from
    for i=1:num_comp
        ind = find(index.==i) #a vector with the positions of the i-th gaussian
        len_ind = length(ind)
        x[ind] = d.mu[i] + sqrt(d.var[i]) * randn(len_ind)
    end
    x
end


function supportlowerbound(d::SteinGMM, j::Int64)
    -Inf
end

function supportupperbound(d::SteinGMM, j::Int64)
    Inf
end


function gradlogdensity(d::SteinGMM, x)
    z = exp(logsumexp([log(d.weights[i])+normldf(x,d.mu[i],d.var[i]) for i in 1:length(d.mu)]...))
    d = sum([d.weights[i].*normpdf(x,d.mu[i],d.var[i]).*(d.mu[i]-x)./d.var[i] for i in 1:length(d.mu)])
    d./z
end


function logdensity(d::SteinGMM, x)
     logsumexp([log(d.weights[i])+normldf(x,d.mu[i],d.var[i]) for i in 1:length(d.mu)]...)
end


# Cumulative distribution function
function cdf(d::SteinGMM, t)
   norm_weights = d.weights/sum(d.weights)
   num_comp = length(d.mu)
   cdf_mix = 0
   for i=1:num_comp
       cdf_mix += norm_weights[i] * Distributions.cdf(Normal(d.mu[i],sqrt(d.var[i])),t);
   end
   cdf_mix;
end


function numdimensions(d::SteinGMM)
    1
end
