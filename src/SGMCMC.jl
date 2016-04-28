module SGMCMC

    abstract SamplerState

    export SamplerState,HMCState,RelHMCState,SGRelHMCState, SGNHTRelHMCState ,sample!, SGHMCState, SGNHTHMCState
    export AdamState, SGLDState, SGNHTState, AdagradState, myState
    
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

    function sample!(s::HMCState,llik,grad;v=false)
      # hamiltonian monte carlo (radford neal's version)
      nparams = length(s.x)
      mass = s.mass
      stepsize = s.stepsize
      niters = s.niters


      s.p = sqrt(mass).*randn(nparams)
      curx = s.x
      curp = s.p
      s.p += .5*stepsize * grad(s.x)
      velocity = zeros(2)

      iter=1;
      while iter<niters
	s.x += stepsize * s.p./mass
	if v velocity = (velocity*(iter-1) + s.p./mass)/iter end
	s.p += (iter<niters ? stepsize : .5*stepsize) * grad(s.x) # two leapfrog steps rolled in one unless at the end.
	maximum([abs(s.p),abs(s.p./mass)]) < Inf ?  iter += 1 : iter = niters
      end

      maximum([abs(s.p),abs(s.p./mass)]) < Inf ? logaccratio = llik(s.x) - llik(curx) -.5*sum((s.p.*s.p - curp.*curp)./mass) : logaccratio = -Inf
      if (0.0 > (logaccratio - log(rand()))[1]) 
	  #reject
	  s.x = curx
	  s.p = curp
      else
	  #accept
	  #negate momentum for symmetric Metropolis proposal
	  s.p = -s.p
      end

      v? (return s,velocity) : return s
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



   type SGNHTRelHMCState <: SamplerState
        x::Array{Float64}
        p::Array{Float64}
        zeta::Array{Float64,1}

        niters::Int64
        stepsize::Float64
        mass::Array{Float64,1}
        c::Array{Float64,1}
        D::Array{Float64,1}
        Best::Array{Float64,1} # variance estimator Bhat_t in Levy's notes.
        independent_momenta::Bool
        function SGNHTRelHMCState(x::Array{Float64};stepsize = 0.001, p=:none, mass=[1.0],niters=10,c=[1.0], D=[1.0], Best=[0.0], zeta=[0.0], independent_momenta=false)
            if isa(mass,Number)
              mass = mass * ones(length(x))
            end
            if p == :none
            	if independent_momenta  
            		p=zeros(length(x))
            		for i=1:length(x) p[i]=sample_rel_p(mass,c,1)[1]  end
            	else
                	p = sample_rel_p(mass, c, length(x))
                end
            end
        new(x,p,zeta,niters,stepsize,mass,c,D,Best,independent_momenta)
        end
    end


    function sample!(s::SGNHTRelHMCState, llik, sgrad)
      # stochastic gradient relativistic langevin dynamics
      # using naive Euler updates.
      D = s.D
      niters = s.niters
      stepsize = s.stepsize
      Best = s.Best
      zeta = s.zeta

      m = s.mass
      c = s.c
	  independent_momenta=s.independent_momenta

      for iter=1:s.niters

        #M(p)
        independent_momenta ? tmp = m .* sqrt(s.p.^2 ./ (m.^2 .* c.^2) + 1)  : tmp = m .* sqrt(s.p's.p ./ (m.^2 .* c.^2) + 1)
        #gradient of the theta(position)
        p_grad = zeta.*s.p./tmp

        # equation 21 in Levy's notes
        n = sqrt(stepsize.*(2D.-stepsize.*Best)).*randn(length(s.x)) # noise term
        s.p[:] += stepsize.*(sgrad(s.x)-p_grad) + n
        #update tmp
        independent_momenta ? tmp = m .* sqrt(s.p.^2 ./ (m.^2 .* c.^2) + 1)  : tmp = m .* sqrt(s.p's.p ./ (m.^2 .* c.^2) + 1)
        # equation 22 in Levy's notes
        s.x[:] += stepsize.*s.p./tmp
        #thermostats
        independent_momenta ? zeta[:]+=stepsize.*(s.p'*(s.p.*(1./(tmp.^2) + 1./(c.^2.*tmp.^3)))./length(s.x) - 1./length(s.x).*sum(1./tmp))  : zeta[:]+=stepsize.* (s.p'*s.p./length(s.x).*(1./tmp.^2 + 1./(tmp.^3 .* c.^2)) - 1./tmp)
       end

       s
   end
   
   
    type SGHMCState <: SamplerState
        x::Array{Float64}
        p::Array{Float64}

        niters::Int64
        stepsize::Float64
        mass::Array{Float64,1}
        D::Array{Float64,1}
        Best::Array{Float64,1} 
 	    function SGHMCState(x::Array{Float64}; stepsize = 0.001, p=:none, mass=[1.0],niters=10, D=[1.0], Best=[0.0])
            if isa(mass,Number)
              mass = mass * ones(length(x))
            end
            if p == :none
                p = sqrt(mass).*randn(length(x))
            end
        new(x,p,niters,stepsize,mass,D,Best)
        end
    end

    function sample!(s::SGHMCState, llik, sgrad)
      D = s.D
      niters = s.niters
      stepsize = s.stepsize
      Best = s.Best
      m = s.mass            

      for iter=1:s.niters
        p_grad = stepsize .* s.p ./ m
        n = sqrt(stepsize.*(2D.-stepsize.*Best)).*randn(length(s.x)) 
        s.p[:] += stepsize.*sgrad(s.x) + n - D.*p_grad
        p_grad = stepsize .* s.p ./ m
        s.x[:] += p_grad
      end
      s
    end

   type SGNHTHMCState <: SamplerState
        x::Array{Float64}
        p::Array{Float64}
        zeta::Array{Float64,1}

        niters::Int64
        stepsize::Float64
        mass::Array{Float64,1}
        D::Array{Float64,1}
        Best::Array{Float64,1} 
        function SGNHTHMCState(x::Array{Float64};stepsize = 0.001, p=:none, mass=[1.0],niters=10, D=[1.0], Best=[0.0], zeta=[0.0])
            if isa(mass,Number)
              mass = mass * ones(length(x))
            end
            if p == :none
                p = sqrt(mass).*randn(length(x))
            end
        new(x,p,zeta,niters,stepsize,mass,D,Best)
        end
    end


    function sample!(s::SGNHTHMCState, llik, sgrad)
      D = s.D
      niters = s.niters
      stepsize = s.stepsize
      Best = s.Best
      zeta = s.zeta
      m = s.mass

      for iter=1:s.niters

        p_grad = zeta.*s.p./ m
        n = sqrt(stepsize.*(2D.-stepsize.*Best)).*randn(length(s.x))
        s.p[:] += stepsize.*(sgrad(s.x)-p_grad) + n
        s.x[:] += stepsize.*s.p./m
        zeta[:] += stepsize.* (s.p'*(s.p./(length(s.x)*m.^2)) - mean(1./m))
       end
       s
   end
   
   type SGLDState <: SamplerState
        x::Array{Float64,1}
        t::Int
        stepsize::Function

        function SGLDState(x::Array{Float64,1},stepsize::Function)
          new(x,0,stepsize)
        end
        function SGLDState(x::Array{Float64,1},stepsize::Float64)
            eps = stepsize
            stepsize = niters::Int -> eps
          new(x,0,stepsize)
        end
    end

    function sample!(s::SGLDState,grad::Function)
        s.t += 1
        g = grad(s.x)
        Base.LinAlg.axpy!(s.stepsize(s.t),g,s.x)
        noise = sqrt(2.0*s.stepsize(s.t))
        Base.LinAlg.axpy!(float(noise),randn(length(s.x)),s.x)
        s
    end


    type AdamState <: SamplerState
      x::Array{Float64}
      t::Int
      mt::Array{Float64}
      vt::Array{Float64}

      stepsize::Float64
      beta1::Float64
      beta2::Float64
      minmass::Float64
      averagegrad::Bool
      injectnoise::Float64


      function AdamState(x::Array{Float64};
        stepsize=.001,beta1=.9,beta2=.999,minmass=1e-8,
        averagegrad=true,injectnoise=1.0)
        nparams = size(x)
        new(x,0,zeros(nparams),zeros(nparams),
        stepsize,beta1,beta2,minmass,
        averagegrad,injectnoise
        )
      end
    end


    function sample!(s::AdamState, grad::Function)
      # stochastic gradient langevin dynamics
      # adapt mass and average gradient according to Adam

      gt = grad(s.x)
      s.t += 1
      s.mt = s.beta1*s.mt + (1-s.beta1)*gt
      s.vt = s.beta2*s.vt + (1-s.beta2)*(gt.*gt)
      mt = s.mt/(1.0-s.beta1^s.t) # debiased
      vt = s.vt/(1.0-s.beta2^s.t) # debiased
      alphat = s.stepsize./(sqrt(vt)+s.minmass)
      s.x[:] += alphat .* (s.averagegrad ? mt : gt)
      if s.injectnoise>0.0
        noisestep = min(s.stepsize*s.injectnoise,sqrt(2.0*alphat)).*randn(size(s.x))
        s.x[:] += noisestep
      end

      if myid()<= 2 && rem(s.t,10)==0
        if false
          estvar = vt - mt.*mt
          factor = (1-s.beta1)^2/(1-s.beta1^(s.t))^2*(1-s.beta1^(2*s.t))/(1-s.beta1^2)
          @show factor
          @show mean(estvar)
          @show mean(alphat)
          estgradvar = factor*estvar.*alphat.*alphat
          noisevar = 2.0*alphat
          diffvar = estgradvar-noisevar
          difflogvar = log(1e-3+estgradvar) - log(1e-3+noisevar)
          @show mean(estgradvar),mean(noisevar)
          @show mean(diffvar),std(diffvar)
          @show mean(difflogvar),std(difflogvar)
          gradstep = alphat.*mt
          @show mean(abs(gradstep)),std(abs(gradstep))
          noisestep = sqrt(2.0*min(s.stepsize*s.injectnoise,alphat))
          @show mean(abs(noisestep)),std(abs(noisestep))
          fullnoisestep = sqrt(2.0*alphat)
          @show mean(abs(fullnoisestep)),std(abs(fullnoisestep))
        else
          x = mean(abs(s.x))
          grad = mean(abs(gt))
          grad2 = mean(abs(gt.*gt))
          gradstep = mean(abs(alphat .* (s.averagegrad ? mt : gt)))
          noisestep = s.injectnoise==0.0 ? 0.0 : mean(abs(noisestep))
          fullnoisestep = mean(abs(sqrt(2.0*alphat).*randn(size(s.x))))
          #@show x,grad,grad2,gradstep, noisestep, fullnoisestep
        end
      end
      s
    end

    type AdagradState <: SamplerState
      x::Array{Float64,1}
      mass::Array{Float64,1}
      masscount::Int32

      stepsize::Float64
      minmass::Float64
      boundnoise::Bool
      injectnoise::Float64

      function AdagradState(x::Array{Float64,1};
        stepsize=.001,
        mass=ones(size(x)),
        masscount=0,
        minmass=1e-10,
        boundnoise=false,
        injectnoise=1.0)

        new(x,mass*masscount,masscount,stepsize,minmass,boundnoise,injectnoise)
      end
    end

    function sample!(s::AdagradState, grad::Function)
      # stochastic gradient langevin dynamics
      # adapt mass according to adagrad
      g = grad(s.x)
      s.masscount += 1
      if s.masscount == 1
        s.mass = g.*g
      else
        meps = 1.0/sqrt(s.masscount)
        s.mass = (1.0-meps)*s.mass + meps*g.*g
      end
      if s.boundnoise
        minmasses = 2.*s.stepsize./((abs(s.x)*0.3).^2)
        mm = sqrt(max(s.minmass,s.mass,minmasses))
      else
        mm = sqrt(max(s.minmass,s.mass))
      end

      s.x[:] += s.stepsize./mm .* g

      if s.injectnoise > 0.
        s.x[:] += sqrt(2.0*s.stepsize./mm).*randn(length(s.x))
      end

      # whether mean or exponential decay didn't matters as much.
      # using sqrt on mass really helps! (rather, removing sqrt makes it not work)
    end

    type myState <: SamplerState
        x::Array{Float64}
        p::Array{Float64}

        stepsize::Float64
        mass::Array{Float64,1}
        c::Array{Float64,1}
        D::Array{Float64,1}
        Best::Array{Float64,1} # variance estimator Bhat_t in Levy's notes.
        independent_momenta::Bool
        function myState(x::Array{Float64};stepsize = 0.001, p=:none, mass=[1.0],c=[1.0], D=[1.0], Best=[0.0], independent_momenta=false)
            if isa(mass,Number)
              mass = mass * ones(length(x))
            end
            if p == :none
                p = sample_rel_p(mass, c, length(x))
            end
        new(x,p,stepsize,mass,c,D,Best,independent_momenta)
        end
    end

    function sample!(s::myState, llik, sgrad)
      D = s.D
      stepsize = s.stepsize
      Best = s.Best

      m = s.mass
      c = s.c
	  independent_momenta=s.independent_momenta
      independent_momenta ? tmp = m .* sqrt(s.p.^2 ./ (m.^2 .* c.^2) + 1) : tmp = m .* sqrt(s.p'*s.p ./ (m.^2 .* c.^2) + 1)
      p_grad = s.p./tmp
	  s.p[:] += stepsize.*(sgrad(s.x)-D[1]*p_grad) 
      independent_momenta ? tmp = m .* sqrt(s.p.^2 ./ (m.^2 .* c.^2) + 1)  :  tmp = m .* sqrt(s.p'*s.p ./ (m.^2 .* c.^2) + 1)
	  s.x[:] += stepsize.*s.p./tmp
   end

end






















