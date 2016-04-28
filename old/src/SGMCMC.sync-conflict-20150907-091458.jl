module SGMCMC

export SGO_specs,RMSprop_specs,SGLD_specs,SGLDA_specs,SGHMC_specs,SGNHTN_specs,SGNHTS_specs,HMC_specs,
       sgo!, rmsprop!, sgld!, sglda!, sghmc!, sgnhtn!, sgnhts!, sgnhta!, hmc!

type SGO_specs
  niters::Int32
  eps::Float64
  precond::Array{Float64,1}
  SGO_specs(eps=.001,precond=[1.0];niters=10) = new(niters,eps,precond)
end
function sgo!(x::Array{Float64,1}, grad::Function, s::SGO_specs)
  # stochastic gradient optimizer
  for iter=1:s.niters
    x[:] += s.eps * s.precond .* grad(x)
  end
  x
end

type RMSprop_specs
  niters::Int32
  eps::Array{Float64,1}
  msgrad::Array{Float64,1}
  prevgrad::Array{Float64,1}
  epsgrad::Float64
  epsincfactor::Float64
  epsdecfactor::Float64
  function RMSprop_specs(eps=[.001];niters=10,msgrad=[0.0],epsgrad=.1,epsincfactor=.01,epsdecfactor=.99)
    if msgrad[1]==0.0
      msgrad = ones(size(eps))
    end
    prevgrad = zeros(size(eps))
    new(niters,eps,msgrad,prevgrad,epsgrad,epsincfactor,epsdecfactor)
  end
end
function rmsprop!(x::Array{Float64,1}, grad::Function, s::RMSprop_specs)
  # RMSProp procedure
  for iter = 1:s.niters
    g = grad(x)
    # mean-squared gradients
    s.msgrad *= 1.0-s.epsgrad
    s.msgrad += s.epsgrad * g.*g
    # whether gradients pointed in same direction
    inc = g.*s.prevgrad .> 0.0
    s.eps[inc]  += s.epsincfactor
    s.eps[!inc] *= s.epsdecfactor
    s.prevgrad = g
    x[:] += s.eps ./ sqrt(s.msgrad) .* g
  end
  x
end

type SGLD_specs
  niters::Int32
  eps::Float64
  mass::Array{Float64,1}
  function SGLD_specs(eps;mass=[1.0],niters=10)
    new(niters,eps,mass)
  end
end
function sgld!(x::Array{Float64,1}, grad, s::SGLD_specs)
  # stochastic gradient langevin dynamics
  for iter=1:s.niters
    x[:] += (s.eps./s.mass) .* grad(x) + sqrt(2.0*s.eps./s.mass).*randn(length(x))
  end
  x
end

type SGLDA_specs
  niters::Int32
  eps::Float64
  mass::Array{Float64,1}
  minmass::Float64
  iter::Int32
  function SGLDA_specs(eps;mass=[1.0],niters=1,masscount=0,minmass=1e-10)
    new(niters,eps,mass*masscount,minmass,masscount)
  end
end
function sglda!(x, grad, s::SGLDA_specs)
  # stochastic gradient langevin dynamics
  # adapt mass according to adagrad
  for iter=1:s.niters
    g = grad(x)
    s.mass += g.*g 
    s.iter += 1
    mm = sqrt(max(s.minmass,s.mass/s.iter))
    x[:] += s.eps./mm .* g + sqrt(2.0*s.eps./mm).*randn(length(x))
    # whether mean or exponential decay didn't matters as much.
    # using sqrt on mass really helps! (rather, removing sqrt makes it not work)
  end
  x
end

function sghmc(x, mom, grad, eps; niters=10, precond=1.0)
  # stochastic gradient hamiltonian monte carlo
  error("SGHMC not done: use SGNHT variant instead")
  x[:] += (eps)*mom
  mom = (eps*precond)*grad(x) - precond*mom + sqrt(2.0*eps*(precond - Best))*randn(length(x))
  (x,mom)
end

function sgnhts(xx, pp, xi, grad; eps=.001, niters=10, mass=1.0, epsxi=0.1, A=1.0)
  # stochastic gradient nose-hoover thermostat, Leimkhuler & Shang version
  # A is sigma_A^2 in Leimkuhler & Shang, but multidimensional
  # epsxi is 1/mu in Leimkuhler & Shang
  nparams = length(xx)
  epsxi /= nparams
  if isa(mass,Number)
    mass = mass * ones(nparams)
  end
  if isa(A,Number)
    A = A * mass
  end

  pp += .5*eps * grad(xx)
  for iter = 1:niters
    xx += .5*eps * pp./mass
    xi += .5*eps*epsxi * (sum(pp.*pp./mass) - nparams)
    if abs(eps*xi) > 1e-6
      pp = exp(-eps*xi)*pp + sqrt(-.5*expm1(-2.0*eps*xi)/xi * A) .* randn(nparams)
    else
      pp = pp + sqrt(eps*A) .* randn(nparams)
    end
    xi += .5*eps*epsxi * (sum(pp.*pp./mass) - nparams)
    xx += .5*eps * pp./mass
    pp += (iter<niters ? eps : .5*eps) * grad(xx)
  end
  (xx,pp,xi)
end

function sgnhtt(xx, pp, xi, grad; eps=.001, niters=10, mass=1.0, epsxi=0.1, A=1.0)
  # stochastic gradient nose-hoover thermostat, individual thermostats per dim
  # A is sigma_A^2 in Leimkuhler & Shang, but multidimensional
  # epsxi is 1/mu in Leimkuhler & Shang
  nparams = length(xx)
  if isa(mass,Number)
    mass = mass * ones(nparams)
  end
  if isa(A,Number)
    A = A * mass
  end

  pp += .5*eps * grad(xx)
  for iter = 1:niters
    xx += .5*eps * pp./mass
    xi += .5*eps*epsxi * (pp.*pp./mass - 1.0)
    ll = abs(eps*xi) .> 1e-6
    ss = !ll
    pp[ll] = exp(-eps*xi[ll]).*pp[ll] + sqrt(-.5*expm1(-2.0*eps*xi[ll])./xi[ll] .* A[ll]) .* randn(sum(ll))
    pp[ss] = pp[ss] + sqrt(eps*A[ss]) .* randn(sum(ss))
    xi += .5*eps*epsxi * (pp.*pp./mass - 1.0)
    xx += .5*eps * pp./mass
    pp += (iter<niters ? eps : .5*eps) * grad(xx)
  end
  (xx,pp,xi)
end

function sgnhtr(xx, pp, xi, grad; eps=.001, niters=10, mass=1.0, epsxi=0.1, A=1.0)
  # DOESN'T WORK
  # stochastic gradient nose-hoover thermostat, individual thermostats per dim, rearranged steps
  # symmetric steps are B DO A OD B DO A OD B
  # A is sigma_A^2 in Leimkuhler & Shang, but multidimensional
  # epsxi is 1/mu in Leimkuhler & Shang
  nparams = length(xx)
  if isa(mass,Number)
    mass = mass * ones(nparams)
  end
  if isa(A,Number)
    A = A * mass
  end

    xx += .25*eps * pp./mass
  for iter = 1:niters
    ll = abs(.25*eps*xi) .> 1e-6
    ss = !ll
    pp[ll] = exp(-.25*eps*xi[ll]).*pp[ll] + sqrt(-.5*expm1(-0.5*eps*xi[ll])./xi[ll] .* A[ll]) .* randn(sum(ll))
    pp[ss] = pp[ss] + sqrt(.25*eps*A[ss]) .* randn(sum(ss))

    xi += .25*eps*epsxi * (pp.*pp./mass - 1.0)

    pp += .5*eps * grad(xx)

    xi += .25*eps*epsxi * (pp.*pp./mass - 1.0)

    ll = abs(.25*eps*xi) .> 1e-6
    ss = !ll
    pp[ll] = exp(-.25*eps*xi[ll]).*pp[ll] + sqrt(-.5*expm1(-0.5*eps*xi[ll])./xi[ll] .* A[ll]) .* randn(sum(ll))
    pp[ss] = pp[ss] + sqrt(.25*eps*A[ss]) .* randn(sum(ss))

    xx += .5*eps * pp./mass

    ll = abs(.25*eps*xi) .> 1e-6
    ss = !ll
    pp[ll] = exp(-.25*eps*xi[ll]).*pp[ll] + sqrt(-.5*expm1(-0.5*eps*xi[ll])./xi[ll] .* A[ll]) .* randn(sum(ll))
    pp[ss] = pp[ss] + sqrt(.25*eps*A[ss]) .* randn(sum(ss))

    xi += .25*eps*epsxi * (pp.*pp./mass - 1.0)

    pp += .5*eps * grad(xx)

    xi += .25*eps*epsxi * (pp.*pp./mass - 1.0)

    ll = abs(.25*eps*xi) .> 1e-6
    ss = !ll
    pp[ll] = exp(-.25*eps*xi[ll]).*pp[ll] + sqrt(-.5*expm1(-0.5*eps*xi[ll])./xi[ll] .* A[ll]) .* randn(sum(ll))
    pp[ss] = pp[ss] + sqrt(.25*eps*A[ss]) .* randn(sum(ss))

    xx += (iter<niters ? .5 : .25) * eps * pp./mass
  end
  (xx,pp,xi)
end

type SGNHTA_specs
  niters::Int32
  pp::Array{Float64,1}
  xi::Float64
  mass::Array{Float64,1}

  eps::Float64
  epsxi::Float64
  varA::Float64
  iter::Int32
  function SGNHTA_specs(eps::Float64;niters=10,epsxi=.3,varA=1.0,mass=[1.0],masscount=10)
    pp = randn(size(mass)).*sqrt(mass)
    xi = varA
    epsxi /= length(mass)
    iter = 0
    new(niters,pp,xi,mass*masscount,eps,epsxi,varA,masscount)
  end
end
function sgnhta!(xx, grad, s::SGNHTA_specs)
  # stochastic gradient nose-hoover thermostat, Leimkhuler & Shang version
  # A is sigma_A^2, but multidimensional
  # adaptive estimation of mass, which is diagonal, using adagrad idea
  nparams = length(xx)

  gg = grad(xx)
  s.pp += .5*s.eps * gg
  s.mass += gg.*gg
  s.iter += 1
  for iter = 1:s.niters
    mm = sqrt(s.mass/s.iter)
    xx[:] += .5*s.eps * s.pp./mm

    s.xi += .5*s.eps*s.epsxi * (sum(s.pp.*s.pp./mm) - nparams)
    if abs(s.eps*s.xi) > 1e-6
      s.pp *= exp(-s.eps*s.xi)
      s.pp += sqrt(-.5*expm1(-2.0*s.eps*s.xi)/s.xi * s.varA * mm) .* randn(nparams)
    else
      s.pp += sqrt(s.eps*s.varA * mm) .* randn(nparams)
    end
    s.xi += .5*s.eps*s.epsxi * (sum(s.pp.*s.pp./mm) - nparams)

    xx[:] += .5*s.eps * s.pp./mm

    ee = (iter<s.niters ? s.eps : .5*s.eps)
    gg = grad(xx)
    s.pp += ee * gg
    # this is an ad hoc mass update
    s.mass += gg.*gg
    s.iter += 1
  end
  xx
end

function hmc(xx,grad,mixldf; eps=.001, niters=10, mass=1.0)
  # hamiltonian monte carlo (radford neal's version)
  nparams = length(xx)
  if isa(mass,Number)
    mass = mass * ones(nparams)
  end

  pp = sqrt(mass).*randn(nparams)
  curx = xx
  curp = pp
  pp += .5*eps * grad(xx)
  for iter = 1:niters
    xx += eps * pp./mass
    pp += (iter<niters ? eps : .5*eps) * grad(xx)
  end

  accratio = mixldf(xx) - mixldf(curx) -.5*sum((pp.*pp - curp.*curp)./mass)[1]
  return (0.0 < accratio - log(rand()) ? xx : curx, min(1.0,exp(accratio)))
end

end

