
module SGMCMC

    abstract SamplerState

    export SamplerState,HMCState,RelHMCState,SGRelHMCState, SGNHTRelHMCState, SGHMCState, SGNHTHMCState, sample!
    export AdamState, SGLDAdamState
    export RelFrozenState,RelFrozenThermoState
    export SGLDState, SGNHTState
    export SGLDAdagradState, AdagradState


    include("sgmcmc/hmc.jl")

    #includes adaptive rejection sampling for momentum distribution.
    include("utils/ars.jl")
    include("sgmcmc/relhmc.jl")
    include("sgmcmc/sgrelhmc.jl")
    include("sgmcmc/sgnhtrelhmc.jl")
    include("sgmcmc/sgnht.jl")
    include("sgmcmc/sghmc.jl")
    include("sgmcmc/sgnhthmc.jl")
    
    include("sgmcmc/sgld.jl")
    include("sgmcmc/adam.jl")
    include("sgmcmc/sgldadam.jl")
    include("sgmcmc/relfrozen.jl")
    include("sgmcmc/relfrozenthermo.jl")

    include("sgmcmc/sgldadagrad.jl")
    include("sgmcmc/adagrad.jl")


end



