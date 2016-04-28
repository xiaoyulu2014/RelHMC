
module NeuralNet

using PyPlot
import Base.convert, Base.show, Base.randn, Base.zeros, Base.vec, Base.display
export NNArch, NNet, vec,nlayers, nunits, nparams, nweights, nbiases, paramscale, predict, loglik, backprop, randn, zeros
export setparams!, getparams!, display

# structure assumptions
# layer l weight has shape nunits[l-1] * nunits[l]
# layer l bias has shape 1 * nunits[l]
# layer l activations has shape n * nunits[l] (for n vectors of activations)
# data and activations are rows!
# serialization stores weights, then biases of layers in order

type NNArch
  nunits::Array{Int,1}
  nonlins::Array{Symbol,1}

  function NNArch(nunits::Array{Int64,1},nonlins::Array{Symbol,1})
    @assert length(nunits) == length(nonlins)+1
    @assert all(nunits.>0)
    for i=1:length(nonlins)
      if nonlins[i]!=:sigmoid &&
         nonlins[i]!=:relu &&
         nonlins[i]!=:leakyrelu &&
         nonlins[i]!=:softmax &&
         nonlins[i]!=:linear
        error("unknown $i$th nonlinearity $(nonlins[i])")
      end
    end
    new(nunits,nonlins)
  end
end

nlayers(arch::NNArch) = length(arch.nonlins)
nunits(arch::NNArch, l::Int) = arch.nunits[l+1]

nparams(arch::NNArch, l::Int) = (nunits(arch,l-1)+1) * nunits(arch,l)
nparams(arch::NNArch) = sum([nparams(arch,l) for l=1:nlayers(arch)])
nweights(arch::NNArch, l::Int) = nunits(arch,l-1) * nunits(arch,l)
nbiases(arch::NNArch, l::Int) = nunits(arch,l)

type NNet
  arch::NNArch
  x::Array{Float64,1}

  weights::Array{Array{Float64,2},1}
  biases::Array{Array{Float64,2},1}

  function NNet(arch::NNArch, x::Array{Float64})
    @assert length(x) == nparams(arch)
    nl = nlayers(arch)
    weights = cell(nl)
    biases = cell(nl)
    o = 0
    for l = 1:nl
      nw = nweights(arch,l)
      nb = nbiases(arch,l)
      weights[l] = reshape(x[o+1:o+nw],nunits(arch,l-1),nunits(arch,l))
      biases[l] = reshape(x[o+nw+1:o+nw+nb],1,nunits(arch,l))
      o += nw + nb
    end
    new(arch,x,weights,biases)
  end

end

zeros(arch::NNArch) = NNet(arch,zeros(nparams(arch)))
randn(arch::NNArch, sd::Float64) = NNet(arch, sd*randn(nparams(arch)).*vec(paramscale(arch)))

function setparams!(nnet::NNet, x::Array{Float64})
  arch = nnet.arch
  @assert length(x) == nparams(arch)
  nl = nlayers(arch)
  weights = cell(nl)
  biases = cell(nl)
  o = 0
  for l = 1:nl
    nw = nweights(arch,l)
    nb = nbiases(arch,l)
    nnet.weights[l][:] = reshape(x[o+1:o+nw],nunits(arch,l-1),nunits(arch,l))
    nnet.biases[l][:] = reshape(x[o+nw+1:o+nw+nb],1,nunits(arch,l))
    o += nw + nb
  end
  nnet
end

function getparams!(x::Array{Float64},nnet::NNet)
  arch = nnet.arch
  o = 0
  for l = 1:nlayers(arch)
    nw = nweights(arch,l)
    nb = nbiases(arch,l)
    x[o+1:o+nw] = nnet.weights[l]
    x[o+nw+1:o+nw+nb] = nnet.biases[l]
    o += nw + nb
  end
  x
end

function vec(nnet::NNet)
  arch = nnet.arch
  x = zeros(nparams(arch))
  getparams!(x,nnet)
end


const leakslope = .01

function nonlin(arch::NNArch, l::Int, x)
  f = arch.nonlins[l]
  if f == :sigmoid
    1.0./(1.0+exp(-x))
  elseif f == :relu
    return max(0.0,x)
  elseif f == :leakyrelu
    return max(leakslope*x,x)
  elseif f == :linear
    return x
  elseif f == :softmax
    m = maximum(x,2)
    y = exp(x .- m)
    return y ./ sum(y,2)
  else
    error("Unknown layer type")
  end
end

function gradient(arch::NNArch, l::Int, xin::Array{Float64,2},xout::Array{Float64,2},D::Array{Float64,2})
  # D is error derivative to be backpropagated
  f = arch.nonlins[l]
  if f == :sigmoid
    xout .* (1.0-xout) .* D
  elseif f == :relu
    g = copy(D)
    g[xin .< 0.0] = 0.0
    return g
  elseif f == :leakyrelu
    g = copy(D)
    g[xin .< 0.0] *= leakslope
    return g
  elseif f == :linear
    return D
  elseif f == :softmax
    return (D .- sum(D.*xout,2)) .* xout
  else
    error("Unknown layer type")
  end
end

function paramscale(arch::NNArch)
  ss = NNet(arch,ones(nparams(arch)))
  for l = 1:nlayers(arch)
    ss.weights[l][:] = 1.0/sqrt(nunits(arch,l-1))
  end
  ss
end


function predict(nnet::NNet, data::Array{Float64,2})
  arch = nnet.arch
  nl = nlayers(arch)
  @assert size(data,2) == nunits(arch,0)

  xout = data
  for l = 1:nl
    xin = xout * nnet.weights[l] .+ nnet.biases[l]
    xout = nonlin(arch,l,xin)
  end
  xout
end

function loglik(nnet::NNet, data, response)
  pred = predict(nnet,data)
  @assert size(data,1) == size(response,1)
  @assert size(response,2) == size(pred,2)
  fam = nnet.arch.nonlins[end]
  if fam == :sigmoid
    return sum(response.*log(pred+1e-6) + (1.0-response).*log(1.0+1e-6-pred))
  elseif fam == :softmax
    return sum(sum(response.*log(pred+1e-6)))
  elseif fam == :linear
    delta = response - pred
    return -.5*sum(sum(delta .* delta))
  else
    error("Unknown layer type")
  end
end

function accuracy(nnet::NNet, data, response)
  pred = predict(nnet,data)
  @assert size(data,1) == size(response,1)
  fam = nnet.arch.nonlins[end]
  @assert (size(response,2) == 1 || fam == :sigmoid || fam == :softmax)
  if fam == :sigmoid
    if size(response,2) == 1
        return 1.0-mean(abs(response-round(pred)))
    else # assumes multiclass accuracy 
        pred = pred .== maximum(pred,2)
        return mean(sum(pred.*response,2))
    end
  elseif fam == :softmax
    s = 0.0
    for i=1:size(data,1)
      c = indmax(pred[i,:])
      s += response[i] == c
    end
    return s/size(data,1)
  elseif fam == :linear
    delta = respnse - pred
    return -.5*sum(sum(delta .* delta))
  else
    error("Unknown layer type")
  end
end

function softmax_predictions(pred::Array{Float64,2})
    c = zeros(Integer,size(pred,1))
    for i=1:size(pred,1)
      c[i] = indmax(pred[i,:])
    end
    c
end

# assumes softmax
function softmax_accuracy(pred::Array{Float64,2}, data, response)
    s = 0.0
    for i=1:size(data,1)
      c = indmax(pred[i,:])
      s += response[i] == c
    end
    return s/size(data,1)
end

function backprop(nnet::NNet, data, response)
  # assume loss function is matched with final nonlinearity!
  # linear-squared loss, sigmoid, softmax-cross entropy
  # return mean log likelihood, mean (not sum) and var of gradient, structured as a network as well
  arch = nnet.arch
  nl = nlayers(arch)
  nd = size(data,1)
  @assert size(data,2) == nunits(arch,0)
  @assert size(response,2) == nunits(arch,nl)
  @assert nd == size(response,1)

  xin = cell(1,nl)
  xout = cell(1,nl)
  grad = NNet(arch,zeros(Float64,size(nnet.x)))
  gvar = NNet(arch,zeros(Float64,size(nnet.x)))
  # forward prop
  prevxout = data
  for l = 1:nlayers(arch)
    xin[l] = prevxout * nnet.weights[l] .+ nnet.biases[l]
    prevxout = xout[l] = nonlin(arch,l,xin[l])
  end
  # log likelihood
  fam = arch.nonlins[end]
  if fam == :sigmoid
    llik = mean(response.*log(prevxout+1e-6) + (1.0-response).*log(1.0+1e-6-prevxout))
  elseif fam == :softmax
    llik = mean(sum(response.*log(prevxout+1e-6)))
  elseif fam == :linear
    delta = response - prevxout
    llik = -.5*mean(sum(delta .* delta))
  else
    error("Unknown layer type")
  end
  # backward prop
  local D
  for l = nl:-1:1
    if l==nl
      D = response - xout[nl] # assume loss matched with output nonlinearity
    else
      D = gradient(arch,l,xin[l],xout[l],D * nnet.weights[l+1]')
    end
    if l==1
      xo = data
    else
      xo = xout[l-1]
    end
    grad.biases[l][:] = mean(D,1)
    gvar.biases[l][:] = var(D,1)
    grad.weights[l][:] = (xo' * D) / nd
    gvar.weights[l][:] = (((xo.*xo)' * (D.*D))/nd - grad.weights[l].*grad.weights[l])*nd/(nd-1) #CHECK!!
  end

  return (llik,grad,gvar)
end

function show(io::IO, nnet::NNet)
  arch = nnet.arch
  print(io,"NeuralNet($(nunits(arch,0))")
  for l=1:nlayers(arch)
    print(io,"-$(nunits(arch,l))")
    f = arch.nonlins[l]
    if f==:sigmoid
      print(io,"S")
    elseif f==:relu
      print(io,"R")
    elseif f==:softmax
      print(io,"P")
    elseif f==:linear
      print(io,"L")
    end
  end
  println(io,")")
  for l=1:nlayers(arch)
    println(io,"Layer $l weights:")
    show(io,nnet.weights[l])
    println(io)
    println(io,"Layer $l biases:")
    show(io,nnet.biases[l])
    println(io)
  end
end

function display(nnet::NNet; fignum = 1, threshold = 1.0)

  PyPlot.figure(fignum)
  PyPlot.clf()

  N = nlayers(nnet.arch)
  for l=1:N
    subplot(N, 2, l)
    show_weights(nnet.weights[l], threshold = threshold)
    subplot(N, 2, N+l)
    imshow(nnet.biases[l], cmap=ColorMap("gray"), interpolation="none")
  end

end

function show_weights(W; threshold = 1.0)
    N1 = size(W,1)
    sN1 = floor(Int, sqrt(N1))

    W = deepcopy(W)
    if sN1^2 == N1
        w1 = find(W .> threshold) 
        w2 = find(W .< -threshold)

        W[w1] = threshold
        W[w2] = -threshold

        N2 = size(W,2)
        npanels = min(N2,10)
        D1 = sN1*npanels
        D2 = sN1*ceil(Int, N2/npanels)

        A = zeros(D1,D2)

        for l = 0:N2-1
            i = mod(l,10)
            j = floor(Int, (l-i)/10)

            ii = i*sN1+1
            jj = j*sN1+1

            A[ii:ii+sN1-1,jj:jj+sN1-1] = W[:,l+1]
        end
        imshow(A, cmap=ColorMap("gray"), interpolation="none")
    else
        imshow(W, cmap=ColorMap("gray"), interpolation="none")
    end
end

end

