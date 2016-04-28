include("Gaussian.jl")

x = Gaussian.NatParam()
x = Gaussian.NatParam(1.0,-1.0)
y = Gaussian.MeanParam(1.0,2.0)


z = convert(Gaussian.NatParam,y)
w = convert(Gaussian.MeanParam,z)
w == y

x = x + x
xx = Gaussian.NatParam(2,2)
xx = [x x]
[x] .+ xx
sum(xx)

x
y = convert(Gaussian.MeanParam,x)
Gaussian.var(x)
Gaussian.var(convert(Gaussian.MeanParam,x))
2 *x
x/2
y + y
y/2.0
3*y
x *= 2
x
show(x)
type tt
  a::Gaussian.NatParam
  b::Gaussian.MeanParam
end
v = tt(Gaussian.NatParam(),Gaussian.MeanParam())
u = [tt(Gaussian.NatParam(),Gaussian.MeanParam()) for i=1:2]
w = u[1]
w.a = Gaussian.NatParam(1.0,-1.0)
w
u
w=1.0
w

sqrt(-2.5)

isnan([NaN 1])
