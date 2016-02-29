module SGMCMC

    abstract SamplerState

    export SamplerState,HMCState,RelHMCState,SGRelHMCState, sample!

    type HMCState <: SamplerState
        x::Array{Float64}
        p::Array{Float64} # included in sampler state to allow autoregressive updates.

        stepsize::Float64 #stepsize
        niters::Int64#L
        mass
        function HMCState(x::Array{Float64};p=randn(length(x)),stepsize=0.001,niters=10,mass=1.0)
            if isa(mass,Number)
              mass = mass * ones(length(x))
            end
            new(x,p,stepsize,niters,mass)
        end
    end

    function sample!(s::HMCState,llik,grad)
      # hamiltonian monte carlo (radford neal's version)
      nparams = length(s.x)
      mass = s.mass
      stepsize = s.stepsize
      niters = s.niters


      s.p = sqrt(mass).*randn(nparams)
      curx = s.x
      curp = s.p
      s.p += .5*stepsize * grad(s.x)
      for iter = 1:niters
        s.x += stepsize * s.p./mass
        s.p += (iter<niters ? stepsize : .5*stepsize) * grad(s.x) # two leapfrog steps rolled in one unless at the end.
      end

      logaccratio = llik(s.x) - llik(curx) -.5*sum((s.p.*s.p - curp.*curp)./mass)[1]
      if 0.0 > logaccratio - log(rand())
          #reject
          s.x = curx
          s.p = curp
      else
          #accept
          #negate momentum for symmetric Metropolis proposal
          s.p = -s.p
      end
      return s
    end

    #includes adaptive rejection sampling for momentum distribution.
    include("utils/ars.jl")

    type RelHMCState <: SamplerState
        x::Array{Float64}
        p::Array{Float64} # included in sampler state to allow autoregressive updates.

        stepsize::Float64 #stepsize
        niters::Int64#L
        c::Float64
        mass
        function RelHMCState(x::Array{Float64};stepsize=0.001,niters=10,c = 1.0,mass=1.0,p=:none)
            if isa(mass,Number)
              mass = mass * ones(length(x))
            end
            if p == :none
                p = sample_rel_p(mass, c, length(x))
            end
            new(x,p,stepsize,niters,c,mass)
        end
    end

    function sample!(s::RelHMCState, llik, grad)
      nparams = length(s.x)
      mass = s.mass
      stepsize = s.stepsize
      niters = s.niters
      c = s.c


      #resample relativistic momentum
      s.p = sample_rel_p(mass, c, nparams)

     #pp = sqrt(mass).*randn(nparams)
      curx = s.x
      curp = s.p
      s.p += .5*stepsize * grad(s.x)
      for iter = 1:niters
        s.x += stepsize * s.p./(mass .*sqrt(s.p.^2 ./ (mass.*c).^2 + 1))
        s.p += (iter<niters ? stepsize : .5*stepsize) * grad(s.x) # two leapfrog steps rolled in one unless at the end.
      end

      #current kinetic energy
      cur_ke = sum(mass.*c.^2 .* sqrt(curp.^2 ./(mass.*c).^2 +1))[1]
      ke = sum(mass.*c.^2 .* sqrt(s.p.^2 ./(mass.*c).^2 +1))[1]

      logaccratio = llik(s.x) - llik(curx) - ke + cur_ke
      if 0.0 > logaccratio - log(rand())
          #reject
          s.x = curx
          s.p = curp
      else
          #accept
          #negate momentum for symmetric Metropolis proposal
          s.p = -s.p
      end
      return s
    end

    # logpdf of the relhmc momentum variable
    function gen_rel_p_logpdf(m, c)
        function rel_p_logpdf(p)
            -m*c^2*sqrt(p^2/(m^2*c^2)+1)
        end

        rel_p_logpdf
    end


    # TODO choose ars endpoints more carefully, bookkeeping for ars
    function sample_rel_p(mass, c, nparams; bounds=[-Inf, Inf])

      if length(mass) == 1 && length(c) == 1
        p_logpdf = gen_rel_p_logpdf(mass[1], c[1])
        pp = ars(p_logpdf, -10.0, 10.0, bounds, nparams)
      else
        mass = length(mass) == 1 ? mass * ones(nparams) : mass
        c = length(c) == 1 ? c .* ones(nparams) : c

        pp = zeros(nparams)

        for i = 1:nparams
          p_logpdf = gen_rel_p_logpdf(mass[i],c[i])
          pp[i] = ars(p_logpdf, -10.0, 10.0, bounds, 1)[1]
        end

      end

      pp
    end

    type SGRelHMCState <: SamplerState
        x::Array{Float64}
        p::Array{Float64}

        niters::Int64
        stepsize::Float64
        mass::Array{Float64,1}
        c::Array{Float64,1}
        D::Array{Float64,1}
        Best::Array{Float64,1} # variance estimator Bhat_t in Levy's notes.
        independent_momenta::Bool
        function SGRelHMCState(x::Array{Float64};stepsize = 0.001, p=:none, mass=[1.0],niters=10,c=[1.0], D=[1.0], Best=[0.0], independent_momenta=false)
            if isa(mass,Number)
              mass = mass * ones(length(x))
            end
            if p == :none
                p = sample_rel_p(mass, c, length(x))
            end
            new(x,p,niters,stepsize,mass,c,D,Best,independent_momenta)
        end
    end


    function sample!(s::SGRelHMCState, llik, sgrad)
      # stochastic gradient relativistic langevin dynamics
      # using naive Euler updates.
      D = s.D
      niters = s.niters
      stepsize = s.stepsize
      Best = s.Best

      m = s.mass
      c = s.c
      for iter=1:s.niters

        p_grad = s.independent_momenta ? stepsize .* s.p ./ (m .* sqrt(s.p.*s.p ./ (m.^2 .* c.^2) + 1)) : stepsize .* s.p ./ (m .* sqrt(s.p's.p ./ (m.^2 .* c.^2) + 1))

        # equation 21 in Levy's notes
        n = sqrt(stepsize.*(2D.-stepsize.*Best)).*randn(length(s.x)) # noise term
        s.p[:] += stepsize.*sgrad(s.x) + n - D.*p_grad

        # new p_grad
        p_grad = s.independent_momenta ? stepsize .* s.p ./ (m .* sqrt(s.p.*s.p ./ (m.^2 .* c.^2) + 1)) : stepsize .* s.p ./ (m .* sqrt(s.p's.p ./ (m.^2 .* c.^2) + 1))

        # equation 22 in Levy's notes
        s.x[:] += p_grad
      end
      s
    end
end
