using StatsBase
using RDatasets
using StatsBase: logistic

function checkgrad(x,func,grad; eps=1e-6)
  x = copy(x)
  ndim = length(x)
  f = func(x)
  g = grad(x)
  g2 = copy(g)
  for i=1:ndim
    x[i] += eps
    f2 = func(x)
    g2[i] = (f2-f)/eps
    x[i] -= eps
  end
  println("CheckGrad on $func with stepsize $eps")
  println("Maximum difference: $(maximum(abs(g2-g)))")
  println("Mean difference:    $(mean(abs(g2-g)))")
  (g,g2)
end

data_set = "nodal"
nodal = dataset("boot", "nodal")
y = 2 * convert(Array{Float64},nodal[:R]) - 1
# X includes an intercept term
X = array(nodal[[:M, :Aged, :Stage, :Grade, :Xray, :Acid]])
X = convert(Array{Float64, 2}, X)
using Distributions
srand(123)
function logit(z)
    1.0./(1.0.+exp(-z))
end

function fun(x::Array{Float64})
  x[1]^2
end
d=6;
C = eye(d);
Cinv = inv(C)
beta = reshape(rand(MvNormal(zeros(d),C)),(d,1))
x=X
xtp=x'
using LogisticRegression
lm=LogisticRegressionModel(d,x,y,Cinv,53)
ll=DataModel.getllik(lm)
gl=DataModel.getgrad(lm)
checkgrad(ones(6),ll,gl)
