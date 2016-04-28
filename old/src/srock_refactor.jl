using Dierckx
# module SRockM


# stability / a

lstab=[7 ,58.9, 1074, 3840 ,7938 ,13584];
m=[3 , 10.0 , 50 ,100 ,150 ,200];
eta=[3.7, 5.9 ,10.8 ,13.5, 15.8 ,17];







#we will now define the mean and the variance of our posterior

spl = Spline1D(m,lstab )
splDamp= Spline1D(eta,m)

type SROCK_specs
  niters::Int32
  h::Float64
  a::Float64
  validate::bool
  valid
  function SROCK_specs(h,a;niters=10;validate=false,valid=false)
    new(niters,h,a,validate,valid)
  end
end
srocksp=SROCK_specs(0.01,200)

#further SROCK parameters
function srock(x,g,srocksp::SROCK_specsl)
  global spl, damp
  a=srocksp.a
  h=srocksp.h
  niters=srocksp.niters
  s=Dierckx.evaluate(spl, a)  # result = [2.375, 14.625]
  s=max(ceil(s),3);
  damp=Dierckx.evaluate(splDamp, s) ;      # adjust damping
  w0=1+damp/s^2;
  sw0=sqrt(w0^2-1);
  arg=s*log(w0+sw0);
  w1=cosh(arg)*sw0/(sinh(arg)*s);  # end method parameter
  t=0
  Xtemp=x
  for i=1:niters
    #SROCK-implemntation
    nu0=1;
    nu1=w0;
    @show (i,Xtemp)
    Xold=copy(Xtemp)
    Xdrift=0.5*g(Xtemp); ## Question where 0.5 from?
    Xtemp1=Xtemp+(w1/nu1)*h*Xdrift;  #SV: nu1 is w0
    nu2=2*w0*nu1-nu0;
    nut=nu1/nu2;
    for k = 2:(s-1)               # deterministic stages DEV needs to be modified
      if(validate && !valid(Xtemp1))
        Xtemp=Xold
        i=-1'
      end
      Xdrift=0.5*g(Xtemp1);
      Xtemp2=2*w1*nut*h*Xdrift+2*w0*nut*Xtemp1-(nu0/nu2)*Xtemp;
      Xtemp=Xtemp1;
      Xtemp1=Xtemp2;
      nu0=nu1;nu1=nu2;
      nu2=2*w0*nu1-nu0;
      nut=nu1/nu2;
    end

# end

    Winc = sqrt(h)*randn(size(x)); # last stage include the stochastic term
    Xtemp=2*w1*nut*h*g(Xtemp1)+2*w0*nut*Xtemp1-(nu0/nu2)*Xtemp+Winc;
    #storing the chain


  end
  return Xtemp


end



