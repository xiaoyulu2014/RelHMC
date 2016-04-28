
module Gaussian

export NatParam, MeanParam, mean, var, suffstats

import Base.+, Base.-, Base.*, Base./, Base.==, Base.size, Base.zeros, Base.ones, Base.mean

suffstats(x::Array{Float64}) = MeanParam(x,.5*(x.*x))

immutable NatParam
  muPrec::Array{Float64}
  negPrec::Array{Float64}

  function NatParam(muprec::Array{Float64},negprec::Array{Float64})
    if any(negprec .> 0.0)
      #warn("Negative precision detected.")
    elseif any(isinf(negprec))
      error("Infinite precision detected.")
    elseif any(isnan(negprec)) | any(isnan(muprec))
      error("NaN precision detected.")
    end
    new(muprec,negprec)
  end
end
NatParam() = NatParam([0.0],[0.0])
NatParam(muPrec::Number,negPrec::Number) = NatParam([convert(Float64,muPrec)],[convert(Float64,negPrec)])
#NatParam(dims::Int...) = NatParam(zeros(Float64,dims),zeros(Float64,dims))


immutable MeanParam
  mu::Array{Float64}
  mu2var::Array{Float64}

  function MeanParam(mu::Array{Float64},mu2var::Array{Float64})
    if any(2.0*mu2var .< mu.*mu)
      #warn("Negative variance detected.")
    elseif any(isinf(mu)) | any(isinf(mu2var))
      error("Infinite variance detected.")
    elseif any(isnan(mu)) | any(isnan(mu2var))
      error("NaN variance detected.")
    end
    new(mu,mu2var)
  end
end
MeanParam() = MeanParam(0.0,Inf)
MeanParam(muPrec::Number,negPrec::Number) = MeanParam([convert(Float64,muPrec)],[convert(Float64,negPrec)])
#MeanParam(dims::Int...) = MeanParam(zeros(Float64,dims),fill(Inf,Float64,dims))

Base.size(x::NatParam) = size(x.muPrec)
Base.size(x::MeanParam) = size(x.mu)

Base.length(x::NatParam) = length(x.muPrec)
Base.length(x::MeanParam) = length(x.mu)

Base.zeros(::Type{MeanParam},dims...) = MeanParam(zeros(Float64,dims),fill(Inf,dims))
Base.zeros(::Type{NatParam},dims...)  = NatParam(zeros(Float64,dims),zeros(Float64,dims))
Base.ones(::Type{MeanParam},dims...) = MeanParam(zeros(Float64,dims),fill(0.5,dims))
Base.ones(::Type{NatParam},dims...)  = NatParam(zeros(Float64,dims),fill(-1.0,dims))

project!(x::NatParam) = (x.negPrec[:] = min(-1e-6,x.negPrec))
project!(x::MeanParam) = (x.mu2var[:] = max(x.mu2var,.5*(1.0000000001)*x.mu.*x.mu))

reflect!(x::NatParam) = broadcast!(z->-abs(z),x.negPrec,x.negPrec)
reflect!(x::MeanParam) = broadcast!((a,b)->b+abs(a-b),x.mu2var,x.mu2var,.5*x.mu.*x.mu)

function limit!(x::MeanParam,minmu::Float64,maxmu::Float64,minvar::Float64,maxvar::Float64)
  mm = max(minmu,min(maxmu,mean(x)))
  vv = max(minvar,min(maxvar,var(x)))
  x.mu[:]= mm
  x.mu2var[:] = .5*(vv+mm.*mm)
end

isvalid(x::NatParam) = x.negPrec .<= 0.0
isvalid(x::MeanParam) = 2.0*x.mu2var .>= x.mu.*x.mu

function average!(x::MeanParam,y::MeanParam,factor::Float64)
  #stdx = sqrt(var(x))
  #stdy = sqrt(var(y))
  #stdx *= (1.0-factor)
  #stdx += factor*stdy
  x.mu2var[:] *= (1.0-factor)
  x.mu2var[:] += factor*y.mu2var[:]
  x.mu[:] *= (1.0-factor)
  x.mu[:] += factor*y.mu[:]
  #broadcast!((s,m)->.5*(s*s + m*m),x.mu2var,stdx,x.mu)
end

function Base.convert(::Type{MeanParam}, x::NatParam)
  m = - x.muPrec ./ x.negPrec
  MeanParam(m, .5*(m.*m -1.0./x.negPrec))
end

function Base.convert(::Type{NatParam}, x::MeanParam)
  var = 2.0*x.mu2var - x.mu.*x.mu
  NatParam(x.mu./var,-1.0./var)
end

mean(x::NatParam) = - x.muPrec./x.negPrec
mean(x::MeanParam) = x.mu

var(x::NatParam) = -1.0./x.negPrec
var(x::MeanParam) = 2.0*x.mu2var - x.mu.*x.mu

std(x) = sqrt(var(x))

randn(x::NatParam) = mean(x) + std(x).*Base.randn(size(x.muPrec))
randn(x::MeanParam) = mean(x) + std(x).*Base.randn(size(x.mu))



+(x::MeanParam, y::MeanParam) = MeanParam(x.mu + y.mu, x.mu2var + y.mu2var)
-(x::MeanParam, y::MeanParam) = MeanParam(x.mu - y.mu, x.mu2var - y.mu2var)

+(x::NatParam, y::NatParam) = NatParam(x.muPrec + y.muPrec, x.negPrec + y.negPrec)
-(x::NatParam, y::NatParam) = NatParam(x.muPrec - y.muPrec, x.negPrec - y.negPrec)

*(x::Number, y::MeanParam) = MeanParam(x*y.mu, x*y.mu2var)
*(x::Number, y::NatParam)  = NatParam(x*y.muPrec, x*y.negPrec)

*(y::MeanParam, x::Number) = MeanParam(x*y.mu, x*y.mu2var)
*(y::NatParam, x::Number)  = NatParam(x*y.muPrec, x*y.negPrec)

/(y::MeanParam, x::Number) = MeanParam(y.mu/x, y.mu2var/x)
/(y::NatParam, x::Number)  = NatParam(y.muPrec/x, y.negPrec/x)

==(x::MeanParam,y::MeanParam) = all((x.mu==y.mu) & (x.mu2var==y.mu2var))
==(x::NatParam, y::NatParam)  = all((x.muPrec==y.muPrec) & (x.negPrec==y.negPrec))


# don't automatically promote
# *(x::Gaussian, y::Gaussian) = *(promote(x,y)...)
# /(x::Gaussian, y::Gaussian) = /(promote(x,y)...)

# Base.promote_rule(::Type{NatParam},::Type{MeanParam}) = NatParam

Base.show(io::IO, x::NatParam) = print(io, "NatParam(size=$(size(x)))")
Base.show(io::IO, x::MeanParam) = print(io, "MeanParam(size$(size(x)))")

end
