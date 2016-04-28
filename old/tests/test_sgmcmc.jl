using Gadfly
using MLUtilities
using SGMCMC

normldf(x,mu,var) =  -.5*log(2.0*pi*var) -.5*((x-mu).*(x-mu))/var
normpdf(x,mu,var) = exp(normldf(x,mu,var))

mu = 2.5
var = 1.0
rel = .03
mu2 = mu*rel
var2 = var*rel*rel
mixldf(x) = log(.5) + logsumexp(normldf(x,mu,var), normldf(x,-mu,var))[1]
function mixgrad(x)
  lp1 = normldf(x,-mu,var)
  lp2 = normldf(x,mu,var)
  pp = exp(lp1 - logsumexp(lp1,lp2))
  (pp.*(-mu-x) + (1.0-pp).*(mu-x))/var
end
function mixldf2(x)
  2*log(.5) + logsumexp(normldf(x[1],-mu,var), normldf(x[1],+mu,var)) +
  logsumexp(normldf(x[2],-mu2,var2), normldf(x[2],+mu2,var2))
end
function mixgrad2(x)
  g = zeros(2)
  lp1 = normldf(x[1],-mu,var)
  lp2 = normldf(x[1],mu,var)
  pp = exp(lp1 - logsumexp(lp1,lp2))
  g[1] = (pp*(-mu-x[1]) + (1.0-pp)*(mu-x[1]))/var
  lp1 = normldf(x[2],-mu2,var2)
  lp2 = normldf(x[2],mu2,var2)
  pp = exp(lp1 - logsumexp(lp1,lp2))
  g[2] = (pp*(-mu2-x[2]) + (1.0-pp)*(mu2-x[2]))/var2
  g
end
randgrad(x) = mixgrad(x) + randn(1)
randgrad2(x) = mixgrad2(x) + randn(2)

println("Checking that gradients are correct")
MLUtilities.checkgrad([.1],mixldf,mixgrad)
MLUtilities.checkgrad([.1 .1],mixldf2,mixgrad2)

# 1 D case
println("1D HMC")
niters = 1000
nburnin = 100
eps = 1.0
xx = zeros(1)
tsamples = zeros(1,niters)
accprobs = zeros(niters)
for iter = 1:nburnin
  (xx,aa) = SGMCMC.hmc(xx,mixgrad,mixldf,eps=eps)
end
for iter = 1:niters
  (xx,aa) = SGMCMC.hmc(xx,mixgrad,mixldf,eps=eps)
  tsamples[:,iter] = xx
  accprobs[iter] = aa
end
plot(x=1:niters,y=tsamples, Guide.title("HMC"))
plot(x=tsamples, Geom.histogram(bincount=50), Guide.xlabel("x"), Guide.title("HMC"))
plot(x=accprobs, Geom.histogram(bincount=50), Guide.xlabel("accprobs"), Guide.title("HMC"))
readline()

println("1D SGNHTS")
eps = 1.0
niters = 1000
nburnin = 100
xx = zeros(1)
pp = ones(1)
xi = 1.0
xxsamples = zeros(1,niters)
ppsamples = zeros(1,niters)
xisamples = zeros(1,niters)
for iter = 1:nburnin
  (xx,pp,xi) = SGMCMC.sgnhts(xx,pp,xi,mixgrad,eps=eps)
end
for iter = 1:niters
  (xx,pp,xi) = SGMCMC.sgnhts(xx,pp,xi,mixgrad,eps=eps)
  xxsamples[:,iter] = xx
  ppsamples[:,iter] = pp
  xisamples[:,iter] = xi
end

plot(x=1:niters,y=xxsamples, Guide.xlabel("x"), Guide.title("SGNHTS"))
plot(x=1:niters,y=xisamples, Guide.xlabel("xi"), Guide.title("SGNHTS"))
plot(x=xxsamples, Geom.histogram(bincount=50), Guide.xlabel("x"), Guide.title("SGNHTS"))
plot(x=ppsamples, Geom.histogram(bincount=50), Guide.xlabel("pp"), Guide.title("SGNHTS"))
plot(x=xisamples, Geom.histogram(bincount=50), Guide.xlabel("xi"), Guide.title("SGNHTS"))
readline()

println("2D HMC")
niters = 1000
nburnin = 100
eps = .01
xx = zeros(2)
tsamples = zeros(2,niters)
accprobs = zeros(niters)
for iter = 1:nburnin
  (xx,aa) = SGMCMC.hmc(xx,mixgrad2,mixldf2,eps=eps)
end
for iter = 1:niters
  (xx,aa) = SGMCMC.hmc(xx,mixgrad2,mixldf2,eps=eps)
  tsamples[:,iter] = xx
  accprobs[iter] = aa
end
plot(x=tsamples[1,:], y=tsamples[2,:], Guide.xlabel("x"), Guide.title("HMC"))
plot(x=accprobs, Geom.histogram(bincount=50), Guide.xlabel("accprob"), Guide.title("HMC"))
plot(x=tsamples[1,:], Geom.histogram(bincount=50), Guide.xlabel("x1"), Guide.title("HMC"))
plot(x=tsamples[2,:], Geom.histogram(bincount=50), Guide.xlabel("x2"), Guide.title("HMC"))
readline()

# need differing mass
println("2D HMC with masses")
eps = 1.0
mass = [1.0, 1.0/rel/rel]
xx = zeros(2)
tsamples = zeros(2,niters)
accprobs = zeros(niters)
for iter = 1:nburnin
  (xx,aa) = SGMCMC.hmc(xx,mixgrad2,mixldf2,eps=eps, mass=mass)
end
for iter = 1:niters
  (xx,aa) = SGMCMC.hmc(xx,mixgrad2,mixldf2,eps=eps,mass=mass)
  tsamples[:,iter] = xx
  accprobs[iter] = aa
end
plot(x=1:niters, y=tsamples[1,:], Guide.xlabel("x1"), Guide.title("HMC"))
plot(x=1:niters, y=tsamples[2,:], Guide.xlabel("x2"), Guide.title("HMC"))
plot(x=accprobs, Geom.histogram(bincount=50), Guide.xlabel("accprob"), Guide.title("HMC"))
plot(x=tsamples[1,:], Geom.histogram(bincount=50), Guide.xlabel("x1"), Guide.title("HMC"))
plot(x=tsamples[2,:], Geom.histogram(bincount=50), Guide.xlabel("x2"), Guide.title("HMC"))
readline()

println("2D SGNHTS")
niters = 1000
nburnin = 100
eps = 0.01
xx = zeros(2)
pp = ones(2)
xi = 1.0
xxsamples = zeros(2,niters)
ppsamples = zeros(2,niters)
xisamples = zeros(1,niters)
for iter = 1:nburnin
  (xx,pp,xi) = SGMCMC.sgnhts(xx,pp,xi,mixgrad2,eps=eps)
end
for iter = 1:niters
  (xx,pp,xi) = SGMCMC.sgnhts(xx,pp,xi,mixgrad2,eps=eps)
  xxsamples[:,iter] = xx
  ppsamples[:,iter] = pp
  xisamples[:,iter] = xi
end

plot(x=1:niters, y=xxsamples[1,:], Guide.xlabel("x1"), Guide.title("SGNHTS"))
plot(x=1:niters, y=xxsamples[2,:], Guide.xlabel("x2"), Guide.title("SGNHTS"))
plot(x=xxsamples[1,:], Geom.histogram(bincount=50), Guide.xlabel("x1"), Guide.title("SGNHTS"))
plot(x=xxsamples[2,:], Geom.histogram(bincount=50), Guide.xlabel("x2"), Guide.title("SGNHTS"))
plot(x=1:niters,y=xisamples, Guide.xlabel("xi"), Guide.title("SGNHTS"))
plot(x=ppsamples, Geom.histogram(bincount=50), Guide.xlabel("pp"), Guide.title("SGNHTS"))
plot(x=xisamples, Geom.histogram(bincount=50), Guide.xlabel("xi"), Guide.title("SGNHTS"))
readline()

println("2D SGNHTS with masses")
niters = 1000
nburnin = 100
eps = 1.0
mass = [1.0, 1.0/rel/rel]
xx = zeros(2)
pp = ones(2)
xi = 1.0
xxsamples = zeros(2,niters)
ppsamples = zeros(2,niters)
xisamples = zeros(1,niters)
for iter = 1:nburnin
  (xx,pp,xi) = SGMCMC.sgnhts(xx,pp,xi,mixgrad2,eps=eps,mass=mass)
end
for iter = 1:niters
  (xx,pp,xi) = SGMCMC.sgnhts(xx,pp,xi,mixgrad2,eps=eps,mass=mass)
  xxsamples[:,iter] = xx
  ppsamples[:,iter] = pp
  xisamples[:,iter] = xi
end

plot(x=1:niters, y=xxsamples[1,:], Guide.xlabel("x1"), Guide.title("SGNHTS"))
plot(x=1:niters, y=xxsamples[2,:], Guide.xlabel("x2"), Guide.title("SGNHTS"))
plot(x=xxsamples[1,:], Geom.histogram(bincount=50), Guide.xlabel("x1"), Guide.title("SGNHTS"))
plot(x=xxsamples[2,:], Geom.histogram(bincount=50), Guide.xlabel("x2"), Guide.title("SGNHTS"))
plot(x=1:niters,y=xisamples, Guide.xlabel("xi"), Guide.title("SGNHTS"))
plot(x=ppsamples, Geom.histogram(bincount=50), Guide.xlabel("pp"), Guide.title("SGNHTS"))
plot(x=xisamples, Geom.histogram(bincount=50), Guide.xlabel("xi"), Guide.title("SGNHTS"))
readline()

println("2D SGNHTA with adaptive masses")
niters = 2000
nburnin = 0
eps = 1.0
mass = [1.0, 1.0]
xx = zeros(2)
pp = ones(2)
xi = 1.0
xxsamples = zeros(2,niters)
ppsamples = zeros(2,niters)
xisamples = zeros(1,niters)
masssamples = 0.01*ones(2,niters)
for iter = 1:nburnin
  (xx,pp,xi,mass) = SGMCMC.sgnhta(xx,pp,xi,mass,mixgrad2,eps=eps)
end
for iter = 1:niters
  (xx,pp,xi,mass) = SGMCMC.sgnhta(xx,pp,xi,mass,mixgrad2,eps=eps)
  xxsamples[:,iter] = xx
  ppsamples[:,iter] = pp
  xisamples[:,iter] = xi
  masssamples[:,iter] = mass
end

plot(x=1:niters, y=xxsamples[1,:], Guide.xlabel("x1"), Guide.title("SGNHTA"))
plot(x=1:niters, y=xxsamples[2,:], Guide.xlabel("x2"), Guide.title("SGNHTA"))
plot(x=xxsamples[1,:], Geom.histogram(bincount=50), Guide.xlabel("x1"), Guide.title("SGNHTA"))
plot(x=xxsamples[2,:], Geom.histogram(bincount=50), Guide.xlabel("x2"), Guide.title("SGNHTA"))
plot(x=1:niters,y=xisamples, Guide.xlabel("xi"), Guide.title("SGNHTA"))
plot(x=ppsamples, Geom.histogram(bincount=50), Guide.xlabel("pp"), Guide.title("SGNHTA"))
plot(x=xisamples, Geom.histogram(bincount=50), Guide.xlabel("xi"), Guide.title("SGNHTA"))
plot(x=1:niters, y=1.0./sqrt(masssamples[1,:]), Guide.xlabel("mass1"), Guide.title("SGNHTA"))
plot(x=1:niters, y=1.0./sqrt(masssamples[2,:]), Guide.xlabel("mass2"), Guide.title("SGNHTA"))
readline()

println("2D SGNHTA with adaptive masses, random gradients")
niters = 2000
nburnin = 0
eps = 1.0
mass = [1.0, 1.0]
xx = zeros(2)
pp = ones(2)
xi = 1.0
xxsamples = zeros(2,niters)
ppsamples = zeros(2,niters)
xisamples = zeros(1,niters)
masssamples = 0.01*ones(2,niters)
for iter = 1:nburnin
  (xx,pp,xi,mass) = SGMCMC.sgnhta(xx,pp,xi,mass,randgrad2,eps=eps)
end
for iter = 1:niters
  (xx,pp,xi,mass) = SGMCMC.sgnhta(xx,pp,xi,mass,randgrad2,eps=eps)
  xxsamples[:,iter] = xx
  ppsamples[:,iter] = pp
  xisamples[:,iter] = xi
  masssamples[:,iter] = mass
end

plot(x=1:niters, y=xxsamples[1,:], Guide.xlabel("x1"), Guide.title("SGNHTA random gradients"))
plot(x=1:niters, y=xxsamples[2,:], Guide.xlabel("x2"), Guide.title("SGNHTA random gradients"))
plot(x=xxsamples[1,:], Geom.histogram(bincount=50), Guide.xlabel("x1"), Guide.title("SGNHTA random gradients"))
plot(x=xxsamples[2,:], Geom.histogram(bincount=50), Guide.xlabel("x2"), Guide.title("SGNHTA random gradients"))
plot(x=1:niters,y=xisamples, Guide.xlabel("xi"), Guide.title("SGNHTA random gradients"))
plot(x=ppsamples, Geom.histogram(bincount=50), Guide.xlabel("pp"), Guide.title("SGNHTA random gradients"))
plot(x=xisamples, Geom.histogram(bincount=50), Guide.xlabel("xi"), Guide.title("SGNHTA random gradients"))
plot(x=1:niters, y=1.0./sqrt(masssamples[1,:]), Guide.xlabel("mass1"), Guide.title("SGNHTA random gradients"))
plot(x=1:niters, y=1.0./sqrt(masssamples[2,:]), Guide.xlabel("mass2"), Guide.title("SGNHTA random gradients"))
readline()

println("2D SGNHTA with adaptive masses, dim-specific thermostats")
niters = 2000
nburnin = 0
eps = 1.0
epsxi = .1
mass = [1.0, 1.0/rel/rel]
xx = zeros(2)
pp = ones(2)
xi = ones(2)
xxsamples = zeros(2,niters)
ppsamples = zeros(2,niters)
xisamples = zeros(2,niters)
for iter = 1:nburnin
  (xx,pp,xi) = SGMCMC.sgnhtt(xx,pp,xi,randgrad2,eps=eps, mass=mass, epsxi=epsxi, niters=1)
end
for iter = 1:niters
  (xx,pp,xi) = SGMCMC.sgnhtt(xx,pp,xi,randgrad2,eps=eps, mass=mass, epsxi=epsxi, niters=1)
  xxsamples[:,iter] = xx
  ppsamples[:,iter] = pp
  xisamples[:,iter] = xi
end

plot(x=1:niters, y=xxsamples[1,:], Guide.xlabel("x1"), Guide.title("SGNHTA random gradients"))
plot(x=1:niters, y=xxsamples[2,:], Guide.xlabel("x2"), Guide.title("SGNHTA random gradients"))
plot(x=xxsamples[1,:], Geom.histogram(bincount=50), Guide.xlabel("x1"), Guide.title("SGNHTA random gradients"))
plot(x=xxsamples[2,:], Geom.histogram(bincount=50), Guide.xlabel("x2"), Guide.title("SGNHTA random gradients"))
plot(x=1:niters, y=xisamples[1,:], Guide.xlabel("xi1"), Guide.title("SGNHTA random gradients"))
plot(x=1:niters, y=xisamples[2,:], Guide.xlabel("xi2"), Guide.title("SGNHTA random gradients"))
plot(x=1:niters, y=ppsamples[1,:], Guide.xlabel("pp1"), Guide.title("SGNHTA random gradients"))
plot(x=1:niters, y=ppsamples[2,:], Guide.xlabel("pp2"), Guide.title("SGNHTA random gradients"))

println("2D SGNHTA with adaptive masses, dim-specific thermostats, reordered steps")
niters = 2000
nburnin = 0
eps = 1.0
epsxi = .01
mass = [1.0, 1.0/rel]
xx = zeros(2)
pp = ones(2)
xi = ones(2)
xxsamples = zeros(2,niters)
ppsamples = zeros(2,niters)
xisamples = zeros(2,niters)
for iter = 1:nburnin
  (xx,pp,xi) = SGMCMC.sgnhtr(xx,pp,xi,randgrad2,eps=eps, mass=mass, epsxi=epsxi, niters=1)
end
for iter = 1:niters
  (xx,pp,xi) = SGMCMC.sgnhtr(xx,pp,xi,randgrad2,eps=eps, mass=mass, epsxi=epsxi, niters=1)
  xxsamples[:,iter] = xx
  ppsamples[:,iter] = pp
  xisamples[:,iter] = xi
end

plot(x=1:niters, y=xxsamples[1,:], Geom.line, Guide.xlabel("x1"), Guide.title("SGNHTA random gradients"))
plot(x=1:niters, y=xxsamples[2,:], Geom.line, Guide.xlabel("x2"), Guide.title("SGNHTA random gradients"))
plot(x=xxsamples[1,:], Geom.histogram(bincount=50), Guide.xlabel("x1"), Guide.title("SGNHTA random gradients"))
plot(x=xxsamples[2,:], Geom.histogram(bincount=50), Guide.xlabel("x2"), Guide.title("SGNHTA random gradients"))
plot(x=1:niters, y=xisamples[1,:], Geom.line, Guide.xlabel("xi1"), Guide.title("SGNHTA random gradients"))
plot(x=1:niters, y=xisamples[2,:], Geom.line, Guide.xlabel("xi2"), Guide.title("SGNHTA random gradients"))
plot(x=1:niters, y=ppsamples[1,:], Geom.line, Guide.xlabel("pp1"), Guide.title("SGNHTA random gradients"))
plot(x=1:niters, y=ppsamples[2,:], Geom.line, Guide.xlabel("pp2"), Guide.title("SGNHTA random gradients"))
