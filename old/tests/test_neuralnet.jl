push!(LOAD_PATH,"/homes/teh/Research/DistBayes/juliaYW/src")

using NeuralNet
using MLUtilities

  arch = NNArch([2,3,2,3,4,3],[:linear,:relu,:leakyrelu,:sigmoid,:softmax])
  @show paramscale(arch)
  @show NNet(arch,cumsum(ones(nparams(arch))))
  @show nnet = randn(arch,1.0)
  params = vec(nnet)
  nnet2 = NNet(arch,params)
  params2 = vec(nnet2)
  @assert params == params2

  data = randn(2,2)
  resp = [1 0 0; 0 1 0]
  @show pred = predict(nnet,data)
  @show llik = loglik(nnet,data,resp)
  @show (llik,grad,gvar) = backprop(nnet,data,resp)

begin
  for nonlin1 in [:linear, :sigmoid, :softmax, :relu, :leakyrelu]
    for nonlin2 in [:linear, :sigmoid, :softmax, :relu, :leakyrelu]
      for nonlin3 in [:linear, :sigmoid, :softmax]
        arch = NNArch([2,3,4,2],[nonlin1,nonlin2,nonlin3])
        nnet = randn(arch,1.0)
        params = vec(nnet)
        resp = [1 0; 0 1]
        function f(x)
          NeuralNet.setparams!(nnet,x)
          return loglik(nnet,data,resp)
        end
        function g(x)
          setparams!(nnet,x)
          (llik,grad,gvar) = backprop(nnet,data,resp)
          return 2.0*vec(grad)
        end
        println("Non-linearities: $nonlin1 $nonlin2 $nonlin3")
        MLUtilities.checkgrad(params,f,g)
      end
    end
  end
end
