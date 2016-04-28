using NeuralNet
using MLUtilities
using SGMCMC

using DataFrames
using HDF5
using JLD
using Images
using MultivariateStats

include("../src/utils/probability_utils.jl")
include("../src/utils/chain_stats.jl")

function load_mnist(binarize=false)
    @load "mnist.jld"
    if binarize
        N = size(mnisttrainx,1)
        Ntest = size(mnisttestx,1)
        return mnisttrainx, reshape(mnisttrainy[:,4], (N,1)), mnisttestx, reshape(mnisttesty[:,4], (Ntest,1))
    else
        return mnisttrainx, mnisttrainy, mnisttestx, mnisttesty
    end
end

function preprocess_data(train_x, train_y; kind=:resize)
    sx, _ = size(train_x)
    sy, _ = size(train_y)

    if kind == :resize
        tx = reshape(train_x, sx, 28, 28)
        ty = reshape(train_y, sy, 28, 28)

        rx = zeros(sx, 7, 7)
        ry = zeros(sy, 7, 7)

        for i = 1:sx
            rx[i,:,:] = Images.imresize(reshape(tx[i,:,:], 28,28), (7,7))
        end
        for i = 1:sy
            ry[i,:,:] = Images.imresize(reshape(ty[i,:,:], 28,28), (7,7))
        end

        return reshape(rx, sx, 49), reshape(ry, sy, 49)
    elseif kind == :filter
        tx = reshape(train_x, sx, 28, 28)
        ty = reshape(train_y, sy, 28, 28)

        rx = zeros(sx, 7, 7)
        ry = zeros(sy, 7, 7)

        for i = 1:sx
            rx[i,:,:] = Images.imresize(Images.imfilter_gaussian(reshape(tx[i,:,:],28,28), [4,4]), (7,7))
        end
        for i = 1:sy
            ry[i,:,:] = Images.imresize(Images.imfilter_gaussian(reshape(ty[i,:,:],28,28), [4,4]), (7,7))
        end

        return reshape(rx, sx, 49), reshape(ry, sy, 49)
    elseif kind == :pca
        M = fit(PCA, train_x'; maxoutdim=5)
        rx = transform(M, train_x')'
        mrx = std(rx,1);
        rx ./= mrx
 
        ry = transform(M, train_y')'
        ry ./= mrx
        return rx, ry 
    end
end

function gen_nnet_joint(nnet, trainx, trainy, batchsize, sigma)

    cur_index = 1
    N = size(trainx,1)
    cur_range = 1:batchsize

    function get_next_batch()
        end_index = cur_index + batchsize - 1

        if end_index >= N
            range = cur_index:N
            cur_index = 1
        else
            range = cur_index:end_index
            cur_index = end_index+1
        end

        cur_range = range
        range 
    end

    # batchsize not needed for prior scaling because likelihood is already the mean
    function nnet_joint(x; next_batch = true)
        range = next_batch ? get_next_batch() : cur_range

        setparams!(nnet, x)
        LL = loglik(nnet, trainx[range,:], trainy[range,:])
        prior = sum(normal_logpdf(x, 0, sigma))/N
        LL + prior
    end

    function nnet_joint_grad(x; next_batch = true)
        range = next_batch ? get_next_batch() : cur_range
        setparams!(nnet, x)
        grad_LL = vec(backprop(nnet, trainx[range,:], trainy[range,:])[2])
        grad_prior = normal_dx_logpdf(x, 0, sigma)/N

        if next_batch == false
            println("grad_LL: $(maximum(grad_LL))")
            println("grad_pr: $(grad_prior[1:20])")
        end

        grad_LL + grad_prior
    end

    nnet_joint, nnet_joint_grad
end


function train_nn(units, nonlins, trainx, trainy, testx, testy; num_iterations=20, lsteps=4, eps=0.1, sampler = SGMCMC.relhmc, mass = [1.0], batchsize=size(trainx,1), sigma_init=1.0, sigma_prior=1.0, c=[1.0], rmsprop_init=false, threshold=1.0, preprocess=:pca, plot_progress=false)

    nn_arch = NNArch(units, nonlins)
    nnet = randn(nn_arch, sigma_init)
    #nnet = zeros(nn_arch)

    sigma = sigma_prior

    if preprocess != Void && preprocess != :none
        trainx, testx = preprocess_data(trainx, testx, kind=preprocess) 
    end

    N, p = size(trainx)
    println("N,p: $N, $p")

    perm = randperm(N)
    trainx = trainx[perm,:]
    trainy = trainy[perm,:]

    local specs


#    nnet_LL = x -> (setparams!(nnet, x); loglik(nnet, trainx, trainy))
#    nnet_prior = x -> sum(normal_logpdf(x, 0, sigma))
# 
#    nnet_grad = x -> (setparams!(nnet, x); vec(backprop(nnet, trainx, trainy)[2]))
#    nnet_prior_grad = x -> normal_dx_logpdf(x, 0, sigma)
#
#    nnet_joint = x -> nnet_LL(x) + nnet_prior(x)
#    nnet_joint_grad = x -> nnet_grad(x) + nnet_prior_grad(x)


    nnet_joint, nnet_joint_grad = gen_nnet_joint(nnet, trainx, trainy, batchsize, sigma)

    train_LL = zeros(num_iterations)
    test_LL = zeros(num_iterations)
    test_acc = zeros(num_iterations)
    pred = Array(Any, num_iterations)
    avg_test_acc = zeros(num_iterations)

    if length(mass) == nlayers(nnet.arch)
        mass = cat(1,[mass[l]*ones(nparams(nnet.arch,l)) for l in 1:nlayers(nnet.arch)]...)
    end

    @assert length(mass) == nparams(nnet.arch) || length(mass) == 1
    local xx = vec(nnet)
    local pp

    if sampler == SGMCMC.sglda! || sampler == SGMCMC.sgldaTheta! ||  sampler == SGMCMC.sgldaSExact!
        mass = length(mass) == 1 ? mass[1]*ones(nparams(nnet.arch)) : mass
        specs = SGLDA_specs(eps, mass=mass, masscount=100, niters=round(Int,0.1*N/batchsize))
    elseif sampler == SGMCMC.sgld!
        mass = length(mass) == 1 ? mass[1]*ones(nparams(nnet.arch)) : mass
        specs = SGLD_specs(eps, mass=mass, niters=round(Int,0.1*N/batchsize))
    elseif sampler == SGMCMC.sgrhmc!
#        mass = length(mass) == 1 ? mass[1]*ones(nparams(nnet.arch)) : mass
#        c = length(c) == 1 ? c[1]*ones(nparams(nnet.arch)) : c
        specs = SGRHMC_specs(eps, mass=mass, c=c, niters = round(Int,0.1*N/batchsize), independent_momenta=true, D=[1.0])
        pp = SGMCMC.sample_rel_p(mass, c, length(xx))
    elseif sampler == SGMCMC.sgo!
        specs = SGO_specs(eps, niters=1)
    elseif sampler == SGMCMC.rmsprop!
        specs = RMSprop_specs(eps*ones(nparams(nnet.arch)), niters=1, epsincfactor=0.1*eps)
    end

    rms_eps = 0.01
    init_RMSprop_specs = RMSprop_specs(rms_eps*ones(nparams(nnet.arch)), niters=round(Int,0.1*N/batchsize), epsincfactor=0.1*rms_eps)

    XX = zeros(num_iterations+1,length(xx))

    for i = 1:num_iterations
        if mod(i, 1000) == 0
            println("i: $i")
        end

        XX[i,:] = xx

        if sampler == SGMCMC.hmc || sampler == SGMCMC.relhmc
            xx, aa = sampler(vec(nnet), nnet_joint_grad, nnet_joint, eps=eps, niters=lsteps, mass = mass)
        elseif sampler == SGMCMC.sglda! || sampler == SGMCMC.sgo! || sampler == SGMCMC.rmsprop! || sampler == SGMCMC.sgld!
            sampler(xx, nnet_joint_grad, specs)
        elseif sampler == SGMCMC.sgrhmc!

            if i <= 10 && rmsprop_init
                SGMCMC.rmsprop!(xx, nnet_joint_grad, init_RMSprop_specs)
            else
                if mod(i,2) == 0
                    pp = SGMCMC.sample_rel_p(mass, c, length(xx))
                end
                sampler(xx, pp, nnet_joint_grad, specs)
            end
        elseif sampler == SGMCMC.sgldaTheta! || sampler == SGMCMC.sgldaSExact!
            sampler(xx, 0.0, sigma_prior, nnet_joint_grad, specs)
        else
            error("Unknown sampler specified")
        end
        setparams!(nnet, xx)


        if plot_progress
            test_LL[i] = loglik(nnet, testx, testy)
            train_LL[i] = loglik(nnet, trainx, trainy)

            response = nonlins[end] == :softmax ? findn(testy)[2] : testy

            if nonlins[end] == :softmax
                probs = NeuralNet.predict(nnet,testx)
                test_acc[i] = NeuralNet.softmax_accuracy(probs, testx, response)
                pred[i] = NeuralNet.softmax_predictions(probs)
            else
                probs = NeuralNet.predict(nnet,testx)
                test_acc[i] = NeuralNet.accuracy(nnet, testx, response)
                pred[i] = round(Integer, probs)
            end

            pred_arr = [ pred[j][k]::Integer for j in 1:i, k in 1:length(response) ]

            start_ind = min(ceil(Int, i/2), floor(Int, num_iterations/2))

        
            if nonlins[end] == :softmax
                cts = [counts(pred_arr[start_ind:i, k],10) for k in 1:length(response)]
                avg_pred = [findmax(cts[k])[2] for k in 1:length(response)]
            else #sigmoid
                # voting
                avg_pred = [round(Integer, sum(pred_arr[start_ind:i, k])/length(start_ind:i)) for k in 1:length(response)] 
            end
            avg_test_acc[i] = mean(avg_pred .== response)


            display(nnet, fignum=1, threshold=threshold)
            PyPlot.figure(2)
            PyPlot.clf()
            PyPlot.subplot(411)
            PyPlot.plot(train_LL[1:i])
            PyPlot.ylabel("Train Log-Likelihood")


            PyPlot.subplot(412)
            PyPlot.plot(test_LL[1:i])
            PyPlot.ylabel("Test Log-Likelihood")


            PyPlot.subplot(413)
            PyPlot.plot(test_acc[1:i])
            PyPlot.ylabel("Test Accuracy")

            PyPlot.subplot(414)
            PyPlot.plot(avg_test_acc[1:i])
            PyPlot.ylabel("Avg. Test Accuracy")
        end 
    end
   
    XX[end,:] = xx

    return nnet, XX
end

function train_nn_eps_range(epsrange, args...; kwargs...)

    for eps in epsrange
        nnet, XX = train_nn(args..., eps=eps; kwargs...)
        kwdict = Dict(kwargs)
        sampler = kwdict[:sampler]
        mass = kwdict[:mass][1]
        c = kwdict[:c][1]

        outfile = "$sampler-$eps-$mass-$c.jld"
        save("results/$outfile", "chain", XX)
    end

end 

function eval_chains(epsrange; kwargs...)

    gt_chain = load("results/mnist_pca_hmc20K.jld")["W"]
    gt_mean = mean(gt_chain[1000:end,:], 1)
    gt_var = var(gt_chain[1000:end,:], 1)

    bias2s = Dict{Float64, Float64}()
    chain_vars = Dict{Float64, Float64}()

    for eps in epsrange
        kwdict = Dict(kwargs)
        sampler = kwdict[:sampler]
        mass = kwdict[:mass][1]
        c = kwdict[:c][1]

        file = "$sampler-$eps-$mass-$c.jld"

        chain = load("results/$file")["chain"]

        println(gt_mean)
        println(gt_var)

        println(mean(chain[1000:end,:],1))
        println(var(chain[1000:end,:],1))

        bias2s[eps], chain_vars[eps] = compute_bias2_var(chain[1000:end,:]', gt_mean', gt_var', x->x)

    end

    bias2s, chain_vars
end
