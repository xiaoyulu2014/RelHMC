module MLUtilities

export logsumexp, checkgrad
function logsumexp(x...)
  m = max(x...)
  return m + log(+([exp(a-m) for a in x]...))
end

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

end
