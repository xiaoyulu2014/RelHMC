module MochaWrapper2

using Mocha
using Compat

export MWrap, evaluateNN, evaluateTestNN, initMochaBackend, evaluateTestNNPredProb

function initMochaBackend(use_gpu)
  if use_gpu == false
    backend = CPUBackend()
  else
    backend = GPUBackend()
  end
  init(backend)
  backend
end

# PUBLIC TYPE ==================================================================================
    type MWrap
        net          ::Net
        nparams      ::Int64
        paramRanges  ::Array{Int64,2}
        layersWParam ::Array{Int64,1}
        # Constructor
        function MWrap(
          data_layer::MemoryDataLayer,
          barebonesNN::Function,
          name::AbstractString,
          accuracylayer::Bool,
          predproblayer::Bool,
          bck::Backend)
        
            (common_layers,pred_prob_layer, storage_layer, loss_layer,acc_layer) = barebonesNN()

            if accuracylayer
                net = Net(name,bck,[data_layer, common_layers..., loss_layer, acc_layer])
            elseif predproblayer
                net = Net(name,bck,[data_layer,common_layers...,pred_prob_layer,storage_layer])
            else
                net = Net(name,bck,[data_layer, common_layers..., loss_layer])
            end
            init(net)
            (npara,paramRanges,layersWParam) = parameterisation(net)
            new(net,npara,paramRanges,layersWParam)
        end
    end

    function name(mw::MWrap)
      mw.net.name
    end
    #
    # PUBLIC FUNCTIONS =============================================================================
    function evaluateNN(mw::MWrap,para::Array{Float64,1}; regu_coef = 0.)

        mw = setparams!(mw,para)
        val = forward(mw.net, regu_coef) #solver.params[:regu_coef]
        backward(mw.net, regu_coef)
        # type needs to agree with para
        grad = getgrad(mw)

        #println("$(val)")

        return (-val, -grad)
    end
    #

    function forward_epoch_with_loss(net::Net)
        ll = 0.;
        epoch = get_epoch(net)
        count_batches = 0;
        while get_epoch(net) == epoch
            count_batches += 1
            result_forward = forward(net, 0.0)
            ll -= result_forward

            # DEBUG
            #println("LL from forward: $(result_forward)")
        end

        # DEBUG
        #println("Did $(count_batches) batches")

        return ll
    end


    function forward_epoch_pred_prob(net::Net)
        epoch = get_epoch(net)
        while get_epoch(net) == epoch
            result_forward = forward(net, 0.0)
        end
        #read out predictive probs
        result =  hcat(net.states[end].outputs[1]...)
        #reset network
        Mocha.reset_outputs(net.states[end])
        return  result

    end

    function evaluateTestNN(mw::MWrap, para::Array{Float64,1}, batch_size::Int64)

        mw = setparams!(mw,para)
        Mocha.reset_statistics(mw.net)
        ll = forward_epoch_with_loss(mw.net)

        #println(mw.states[end-1])
        return (mw.net.states[end].accuracy, ll * batch_size)
    end

    function evaluateTestNNPredProb(mw::MWrap, para::Array{Float64,1}, batch_size::Int64)

       mw = setparams!(mw,para)
       Mocha.reset_statistics(mw.net)
       pred_probs = forward_epoch_pred_prob(mw.net)

       return pred_probs
    end

    function setparams!(mw::MWrap,para::Array{Float64,1})
      r = 1
     for i in mw.layersWParam
          for j = 1:length(mw.net.states[i].parameters)
               copy!( mw.net.states[i].parameters[j].blob,
                       para[mw.paramRanges[1,r]:mw.paramRanges[2,r]] )
               r+=1
          end
     end
     mw
    end

    function getnparams(mw::MWrap)
      mw.nparams
    end

    function getparams(mw::MWrap)
      para::Array{Float64,1} = zeros(mw.nparams)
      r = 1
      for i in mw.layersWParam
          for j = 1:length(mw.net.states[i].parameters)
             x = zeros(mw.paramRanges[2,r]-mw.paramRanges[1,r]+1)
             copy!(x, mw.net.states[i].parameters[j].blob)
             para[mw.paramRanges[1,r]:mw.paramRanges[2,r]] = x
             r+=1
          end
      end
      para
    end

    function getgrad(mw::MWrap)
      # this looks like it won't work on a GPU.
      grad::Array{Float64,1} = zeros(mw.nparams)
      r=1
      for i in mw.layersWParam
          for j=1:length(mw.net.states[i].parameters)
              p = pointer_to_array( pointer(grad, mw.paramRanges[1,r] ), mw.paramRanges[2,r]-mw.paramRanges[1,r]+1)
              copy!(p, mw.net.states[i].parameters[j].gradient)
              r+=1
          end
      end
      grad
    end

    # DEBUG: Print out useful information
    function debug(net::Net)
      println("In MochaWrapper2.debug")

      for i  = 2:(length(net.states)-1)
        if Mocha.has_param(net.layers[i])
          for j = 1:length(net.states[i].parameters)
            this_blob = net.states[i].parameters[j].blob
            println("$this_blob")
            x = zeros(get_width(this_blob), get_height(this_blob))
            copy!(x, this_blob)
            print("$x")
          end
        end
      end
    end

    function init_xavier(mw::MWrap)
      for i  = 2:(length(mw.net.states)-1)
        if Mocha.has_param(mw.net.layers[i])
          #for j = 1:length(net.states[i].parameters)
          weights_blob = mw.net.states[i].parameters[1].blob
          bias_blob = mw.net.states[i].parameters[2].blob

          # Initialize weights
          # NOTE: Here is where I make a change from the Mocha code to match the equations in the Glorot and Bengio paper
          fan_in = get_fea_size(weights_blob) + size(weights_blob)[end]
          scale = sqrt(6.0 / fan_in)
          init_val = rand(eltype(weights_blob), size(weights_blob)) * 2scale - scale
          copy!(mw.net.states[i].parameters[1].blob, init_val)

          # Initialize biases
          fill!(mw.net.states[i].parameters[2].blob, 0)
        end
      end
    end

    function init_simple_fanin(mw::MWrap)
      for i  = 2:(length(mw.net.states)-1)
        if Mocha.has_param(mw.net.layers[i])
          #for j = 1:length(net.states[i].parameters)
          weights_blob = mw.net.states[i].parameters[1].blob
          bias_blob = mw.net.states[i].parameters[2].blob

          # Initialize weights
          # NOTE: Here is where I make a change from the Mocha code to match the equations in the Glorot and Bengio paper
          fan_in = get_fea_size(weights_blob)
          scale = sqrt(1.0 / fan_in)
          init_val = rand(eltype(weights_blob), size(weights_blob)) * 2scale - scale
          copy!(mw.net.states[i].parameters[1].blob, init_val)

          # Initialize biases
          fill!(mw.net.states[i].parameters[2].blob, 0)
        end
      end
    end

    # NOTE: This differs from previous version in that biases are initialized to zero
    function init_gaussian(mw::MWrap, initvar::Float64)
      for i  = 2:(length(mw.net.states)-1)
        if Mocha.has_param(mw.net.layers[i])
          #for j = 1:length(net.states[i].parameters)
          weights_blob = mw.net.states[i].parameters[1].blob
          bias_blob = mw.net.states[i].parameters[2].blob

          # Initialize weights
          init_val = randn(size(weights_blob)) * sqrt(initvar)
          copy!(mw.net.states[i].parameters[1].blob, init_val)

          # Initialize biases
          fill!(mw.net.states[i].parameters[2].blob, 0)
        end
      end
    end

    # NOTE: This differs from previous version in that biases are initialized to zero
    function init_uniform(mw::MWrap, initvar::Float64)
      for i = 2:(length(mw.net.states)-1)
        if Mocha.has_param(mw.net.layers[i])
          #for j = 1:length(net.states[i].parameters)
          weights_blob = mw.net.states[i].parameters[1].blob
          bias_blob = mw.net.states[i].parameters[2].blob

          # Initialize weights
          b = sqrt(12 * initvar)
          init_val = rand(eltype(weights_blob), size(weights_blob)) * b - 0.5 * b
          copy!(mw.net.states[i].parameters[1].blob, init_val)

          # Initialize biases
          fill!(mw.net.states[i].parameters[2].blob, 0)
        end
      end
    end

    #
    # PRIVATE FUNCTIONS ============================================================================
    function parameterisation(net::Net)
        istart = 1
        iend   = 1
        layersWParam=Int64[]
        paramRangesI=Int64[]
        for i  = 2:(length(net.states)-1)
            if Mocha.has_param(net.layers[i])
                push!(layersWParam,i)
                for j = 1:length(net.states[i].parameters)
                    iend   = istart+length(net.states[i].parameters[j].blob)-1
                    push!(paramRangesI,istart)
                    push!(paramRangesI,iend)
                    istart = iend+1
                end
            end
        end
        nparams = paramRangesI[end]
        paramRanges = reshape(paramRangesI,(2,div(length(paramRangesI),2)))
        return(nparams,paramRanges,layersWParam)
    end
end