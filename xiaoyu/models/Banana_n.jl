module Banana_n
    using DataModel

    export BananaModel_n

    type BananaModel_n <: AbstractDataModel
    b::Array{Float64}
    end

    function DataModel.getgrad(dm::BananaModel_n;noisevar=0.0)
        # noise variance to simulate noisy gradient.
        function banana_gradient(x)
        	b=dm.b
            @assert length(x) == 2*length(b)
            n1 = noisevar > 0 ? randn()*sqrt(noisevar) : 0.0
            n2 = noisevar > 0 ? randn()*sqrt(noisevar) : 0.0
            grad = Float64[]
            for i=1:length(b) grad=[grad,-(x[2*(i-1) + 1]/100+2*b[i]*x[2*(i-1) + 1]*(x[2*i]+b[i]*x[2*(i-1) + 1]^2-100*b[i])),-(x[2*i]+b[i]*x[2*(i-1) + 1]^2-100*b[i])] end
            return(grad)
        end
        return banana_gradient
    end

    function DataModel.getllik(dm::BananaModel_n)
        function banana_logdensity(x)
        	b=dm.b
            @assert length(x) == 2*length(b)
            sum([-1/2*(x[2*(i-1) + 1]^2/100+(x[2*i]+b[i]*x[2*(i-1) + 1]^2-100*b[i])^2) for i=1:length(b)])
        end
        return banana_logdensity
    end

end
