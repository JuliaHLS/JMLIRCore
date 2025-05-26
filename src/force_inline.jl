const force_inline = Set([])

macro force_inline(f)
    sig = f.args[1].args[1].args[1]
    println("sig $sig")
    types = []

#     for arg in sig[2:end]
#         println("arg: $(arg.args[1])")
#     end

    push!(force_inline, eval(last(sig.args[1].args)))
    return @inline(f)
end


