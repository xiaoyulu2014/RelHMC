module DataModel


abstract AbstractDataModel

export AbstractDataModel, getgrad, getllik
function getgrad(dm::AbstractDataModel)
    # should return a function grad(x) that calculates the gradient at x
    error("getgrad not implemented for AbstractDataModel")
end

function getllik(dm::AbstractDataModel)
    # should return a function llik(x) that calculates the loglikelihood at x.
    error("getllik not implemented for AbstractDataModel")
end

end
