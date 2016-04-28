# Relativistic-Stochastic-Gradient

# Examples

There are two examples IJulia notebooks in `examples/`
 - Banana example
 - Mixture of Gaussians
 
# Code structure
 - `src/` contains all the general code, it would be good to keep this separate from specific experiments
 - `src/SGMCMC.jl` contains the sampling code for one update. It implements sampler state objects that contain all the hyperparameters and the state itself. We found this set up useful in DistBayes. For every MCMC or SGMCMC algorithm there should be a `sample!` function that takes arguments `s::SamplerState`,`llik::Function` (a function to calculate the log likelihood at a specific point (for MH steps), and `grad::Function` which gives the gradient (noisy or not). Note that not all algorithms will necessarily need the loglikelihood or the gradient. Currently implemented are HMC, Relativistic HMC and a naive (Euler update) version of Stochastic Gradient Relativistic HMC. Sampling the momenta uses adaptive rejection sampling with complicated code that Levy adapted from Matlab. I haven't checked this but it seems to work.
 - `src/DataModel.jl` gives an abstract type `DataModel` and two functions `getgrad` and `getllik` which supply the functions that are needed for the samplers. This is an abstraction that encapsulates both the data and the model. Specific examples are in `src/models/Banana.jl` and `src/models/GaussianMixture.jl`. For the simple examples currently implemented this might seems a bit over the top but I think it will make plug & play experimentation much much easier when we move on to more involved models like neural networks. 
 - If you write new models to play around with, there's a `checkgrad` util in `src/MLUtilities.jl` that you can use to check that you implemented the right gradient (see Mixture of Gaussians example).
 - Everything seems to work in the example IJulia notebooks but I haven't done extensive testing.
 
# tmux
tmux is a command line tool that helps you to run jobs on the server without having to stay logged in the entire time. It is a better version of screen.
- if you log into a server via ssh, type `tmux new -s session1` to create a session called session1
- you can use this session just as you would you normal shell. There are a number of different command within tmux: you can find an exhaustive list by pressing `ctrl b + ?`. E.g. `ctrl b + c` creates another window and `ctrl b + 1` switches to window 1.
- `ctrl b + d` detaches the session. It will continue to run in the background. 
- To reattach a session use `tmux ls` to get a list of all session and then `tmux a -t sessionname` to reattach your session.
