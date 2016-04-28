using DataFrames

mnisttrainx = convert(Array{Float64,2},readtable("mnisttrain.txt",separator=' ',header=false))
mnisttrainc = convert(Array{Int32,2},  readtable("mnisttrainc.txt",separator=' ',header=false))
mnisttestx  = convert(Array{Float64,2},readtable("mnisttest.txt",separator=' ',header=false))
mnisttestc  = convert(Array{Int32,2},  readtable("mnisttestc.txt",separator=' ',header=false))

mnisttrainy = zeros(Float64,60000,10)
for i=1:60000
  mnisttrainy[i,mnisttrainc[i]==0?10:mnisttrainc[i]] = 1.0
end

mnisttesty = zeros(Float64,10000,10)
for i=1:10000
  mnisttesty[i,mnisttestc[i]==0?10:mnisttestc[i]] = 1.0
end
