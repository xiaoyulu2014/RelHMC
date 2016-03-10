push!(LOAD_PATH,"C:/Users/Xiaoyu Lu/Documents/RelHMC/src")
push!(LOAD_PATH,"C:/Users/Xiaoyu Lu/Documents/RelHMC/models")
#Pkg.add("PyPlot")
using SGMCMC
using DataModel
using Banana
using PyPlot
import StatsBase
dm = BananaModel()
shmc = HMCState(zeros(2),stepsize=0.1)
function plot_contour(f, range_x, range_y)
    grid_x = [i for i in range_x, j in range_y]
    grid_y = [j for i in range_x, j in range_y]

    grid_f = [exp(f(i,j)) for i in range_x, j in range_y]

    PyPlot.contour(grid_x', grid_y', grid_f', 1)
end
function run(s::SamplerState,dm::AbstractDataModel;num_iterations=1000, final_plot=false)
    grad = getgrad(dm)
    llik = getllik(dm)
    samples = zeros(num_iterations, length(s.x))
    zeta = zeros(num_iterations)
    ESS = zeros(num_iterations,length(s.x))
    for i = 1:num_iterations

        sample!(s,llik,grad)

        samples[i,:] = s.x
        if typeof(s) <: SGMCMC.SGNHTRelHMCState  zeta[i] = s.zeta[1]  end
        arf = StatsBase.autocor(samples[1:i,:])
        ESS[i,:] = [i/(1+2*sum(arf[:,j])) for j=1:length(s.x)]
    end

    if final_plot
        if length(s.x) == 2
           # figure()
           # subplot(121)
            PyPlot.clf()
            llik(x,y) = llik([x,y])
            plot_contour(llik, -5:.05:6, -1:.05:32)
            PyPlot.scatter(samples[:,1], samples[:,2])
           # subplot(122)
           # plot(ESS[:,1]);title("ESS")
           # subplot(133)
           # plot(samples[:,1]);title("traceplot of x1")
        end
    end

    samples,zeta
end

reshmc=run(shmc,dm,final_plot=true);
srhmc = RelHMCState(zeros(2),stepsize=0.1)
figure()
resrhmc = run(srhmc,dm,final_plot=true);
ssgrhmc = SGRelHMCState(zeros(2),stepsize=0.2)
figure()
ressgrhmc = run(ssgrhmc, dm, final_plot=true);
figure()
ssgrnhthmc = SGNHTRelHMCState(zeros(2),stepsize=0.2)
ressgrnhthmc = run(ssgrnhthmc, dm, final_plot=true)


#pygui(true)
llik = getllik(dm)
llik1(x,y) = llik([x,y])
plot_contour(llik1, -5:.05:6, -1:.05:32)
for i=1:size(res[1],1)
    PyPlot.scatter(reshmc[1][i,1], reshmc[1][i,2])
    pause(0.01)
end

plot_contour(llik1, -5:.05:6, -1:.05:32)
for i=1:size(res[1],1)
    PyPlot.scatter(resrhmc[1][i,1], resrhmc[1][i,2])
    pause(0.01)
end

plot_contour(llik1, -5:.05:6, -1:.05:32)
for i=1:size(res[1],1)
    PyPlot.scatter(ressgrhmc[1][i,1], ressgrhmc[1][i,2])
    pause(0.01)
end
plot_contour(llik1, -5:.05:6, -1:.05:32)
for i=1:size(res[1],1)
    PyPlot.scatter(ressgrnhthmc[1][i,1], ressgrnhthmc[1][i,2])
    pause(0.000000001)
end

using PyCall
using PyPlot
@pyimport matplotlib.animation as anim
fig=figure()
plot_contour(llik1, -5:.05:6, -1:.05:32)

ani = anim.ArtistAnimation(fig, ims, interval=25, blit=true)
function myscatter(x,y)
  plot_contour(llik1, -5:.05:6, -1:.05:32)
  PyPlot.scatter(x,y)
end

ims=[]
for i in 1:20
    im = (myscatter(ressgrnhthmc[1][i,1], ressgrnhthmc[1][i,2]))
    push!(ims, PyCall.PyObject[im])
end

ani = anim.ArtistAnimation(fig, ims, interval=5, blit=true, repeat_delay=10)
myanim[:save]("/tmp/sinplot.mp4", extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])

ani[:save]("ani.mp4", extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
