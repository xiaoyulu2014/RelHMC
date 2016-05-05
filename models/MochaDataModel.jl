module MochaDataModel

using Compat
using Mocha
using MochaWrapper2
using MLUtilities
using Utilities
using DataModel

export MochaSGMCMCDataModel



type MochaSGMCMCDataModel <: AbstractDataModel
  backend::Backend
  mochaNet::MochaWrapper2.MWrap
  labels::Array{Int64,1}
  ntrain::Float64 # effective training set size
  batchsize::Int64
  temperature::Float64
end


function MochaSGMCMCDataModel(
  datax,datac,
  modelfactory::Function,
  backend::Backend;
  ntrain::Int = length(datac),
  batchsize::Int = 100,
  temperature::Float64 = 1.0,
  do_shuffle::Bool = false,
  do_accuracy::Bool = true,
  do_predprob::Bool = false
  )

  data_layer = MemoryDataLayer(name = "data",
                               data = Array[datax,datac],
                               batch_size = batchsize,
                               shuffle = do_shuffle)

  mochaNet = MochaWrapper2.MWrap(data_layer,
                                 modelfactory,
                                 "MochaSGMCMCNet",
                                 do_accuracy,
                                 do_predprob,
                                 backend)
  MochaSGMCMCDataModel(backend,mochaNet,datac[:],ntrain,batchsize,temperature)
end

function Base.show(io::IO, x::MochaSGMCMCDataModel)
  print(io, "MochaSGMCMCDataModel($(x.ntrain),$(x.batchsize))")
end

function DataModel.getllik(dms::MochaSGMCMCDataModel)
    function llik(x)
        #@assert dms.ntrain == dms.batch_size
        (accuracy, loglikelihood) = MochaWrapper2.evaluateTestNN(dms.mochaNet, x, dms.batchsize)
        return loglikelihood
    end
end


function DataModel.getgrad(dms::MochaSGMCMCDataModel)
    function grad(x)
        (llik, grad) = MochaWrapper2.evaluateNN(dms.mochaNet, x, regu_coef = 0.)
        return grad
    end
end


function fetchparams(dms::MochaSGMCMCDataModel)
  MochaWrapper2.getparams(dms.mochaNet)
end

function fetchnparams(dms::MochaSGMCMCDataModel)
  MochaWrapper2.getnparams(dms.mochaNet)
end

function init_xavier(dms::MochaSGMCMCDataModel)
  MochaWrapper2.init_xavier(dms.mochaNet)
end

function init_simple_fanin(dms::MochaSGMCMCDataModel)
  MochaWrapper2.init_simple_fanin(dms.mochaNet)
end

function init_gaussian(dms::MochaSGMCMCDataModel, initvar::Float64)
  MochaWrapper2.init_gaussian(dms.mochaNet, initvar)
end

function init_uniform(dms::MochaSGMCMCDataModel, initvar::Float64)
  MochaWrapper2.init_uniform(dms.mochaNet, initvar)
end

function evaluateGrad(dms::MochaSGMCMCDataModel,
                                params::Vector{Float64};
                                regu_coef = 0.)
  (llik, grad) = MochaWrapper2.evaluateNN(dms.mochaNet, params, regu_coef = regu_coef)

  # TODO: Should I rescale gradient?
  return vec(grad) #* dms.ntrain
end

function evaluate(dms::MochaSGMCMCDataModel,
  predprobs::Array{Float64,2}
  )
  prediction =  (findmax(predprobs,1)[2]-1.) % size(predprobs)[1]
  #println("length of labels ", size(dms.labels))
  #println("length of prediction ", size(prediction))
  #println("matches ",sum(dms.labels .== prediction[:]))
  accuracy = sum(dms.labels .== prediction[:])./(length(dms.labels)+0.0)
  @dict(accuracy)
end


function evaluate(dms::MochaSGMCMCDataModel,
  x::Vector{Float64}
  )
  (accuracy, loglikelihood) = MochaWrapper2.evaluateTestNN(dms.mochaNet, x, dms.batchsize)
  @dict(accuracy, loglikelihood)
end

function evaluatePredProb(dms::MochaSGMCMCDataModel,
  x::Vector{Float64}
  )
  predictiveprobs = MochaWrapper2.evaluateTestNNPredProb(dms.mochaNet, x, dms.batchsize)
  @dict(predictiveprobs)
end

end
