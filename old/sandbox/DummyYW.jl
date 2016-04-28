
addprocs(1)
workers()

### THIS WORKS!
@everywhere module dummy
  global a = 0
    function inita()
      global a = 1
    end
    function inca()
      return global a += 1
    end
end


@show remotecall_fetch(2,dummy.inita)
@show remotecall_fetch(2,dummy.inca)

dummy.a
dummy.inita()
dummy.inca()
@spawnat 2 begin
  @show remotecall_fetch(1,dummy.inca)
  @show remotecall_fetch(1,dummy.inca)
end

