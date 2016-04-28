
module LogisticRegression
using DataModel
using StatsBase
export LogisticRegressionModel
type LogisticRegressionModel <: AbstractDataModel
    d::Int64
    x::Array{Float64,2}
    y::Array{Float64}
    Cinv::Array{Float64,2}
    subobs::Int64
end

function logit(z)
    1.0./(1.0.+exp(-z))
end
log_logit(z)= -log(1.0 .+ exp(-z))
grad_log_logit(z)=1.0-logit(z)


function DataModel.getllik(dm::LogisticRegressionModel)
    function logdensity(beta)
        d=dm.d
        x=dm.x
        y=dm.y
        Cinv=dm.Cinv
        log_prior = -0.5 * dot(beta,Cinv*beta)
        log_like= sum(log_logit((x* beta).*y))
        return log_prior+log_like
    end
    return logdensity
end
function DataModel.getgrad(dm::LogisticRegressionModel)

    function grad_log_posterior_sub(beta)
        d=dm.d
        x=dm.x
        y=dm.y
        subobs=dm.subobs
        Cinv=dm.Cinv
        nobs=length(y)
        chosen_indices=sample(1:nobs,subobs,replace=false)
        log_prior_gradient= -Cinv* beta

        #first step: compute y*grad_log_logit(y*beta*x)
        weights = y[chosen_indices].*grad_log_logit((x[chosen_indices,:] * beta ).*y[chosen_indices])
        #second ztep: compute y*grad_log_logit(y*beta*x)*x

        return log_prior_gradient+(1.0*length(y))/subobs*(reshape(weights,1,subobs)* x[chosen_indices,:])[:]
    end
    return grad_log_posterior_sub
end

end
