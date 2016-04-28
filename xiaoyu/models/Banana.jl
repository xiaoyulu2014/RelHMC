module Banana
    using DataModel

    export BananaModel

    type BananaModel <: AbstractDataModel
    end

    function DataModel.getgrad(dm::BananaModel;noisevar=0.0)
        # noise variance to simulate noisy gradient.
        function banana_gradient(x)
            @assert length(x) == 2
            n1 = noisevar > 0 ? randn()*sqrt(noisevar) : 0.0
            n2 = noisevar > 0 ? randn()*sqrt(noisevar) : 0.0
            [-(x[1]/100+0.2*x[1]*(x[2]+0.1*x[1]^2-10)), -(x[2]+0.1*x[1]^2-10)]
        end
        return banana_gradient
    end

    function DataModel.getllik(dm::BananaModel)
        function banana_logdensity(x)
            @assert length(x) == 2
            -1/2*(x[1]^2/100+(x[2]+0.1*x[1]^2-10)^2)
        end
        return banana_logdensity
    end

   #= function DataModel.getgrad(dm::BananaModel;noisevar=0.0)
        # noise variance to simulate noisy gradient.
        function banana_gradient(x)
            @assert length(x) == 2
            n1 = noisevar > 0 ? randn()*sqrt(noisevar) : 0.0
            n2 = noisevar > 0 ? randn()*sqrt(noisevar) : 0.0
            [-1.0/20.0*( -400x[1]*(x[2]-x[1]^2) - 2(1-x[1])) + n1, -1.0/20.0*200*(x[2]-x[1]^2) + n2]
        end
        return banana_gradient
    end

    function DataModel.getllik(dm::BananaModel)
        function banana_logdensity(x)
            @assert length(x) == 2
            -1.0/20.0*(100*(x[2]-x[1]^2)^2 + (1-x[1])^2)
        end
        return banana_logdensity
    end=#

end
