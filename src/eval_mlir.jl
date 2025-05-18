include("code_mlir.jl")
include("compiler.jl")

function invoke(jit::IR.ExecutionEngine, name::String, arguments)
    fn = MLIR.API.mlirExecutionEngineInvokePacked(jit, name, arguments)
    return fn == C_NULL ? nothing : fn
end

"Macro @eval_mlir f(args...)"
macro eval_mlir(call)
    @assert Meta.isexpr(call, :call) "only calls are supported"

    f = esc(first(call.args))
    arg_types = esc(call.args)

    quote
        eval_mlir($f, $arg_types...)
    end
end

# externally lower TOSA to linalg (unsupported by MLIR.jl)
function external_lowering_mlir_opt!(op, passes::Vector{Cmd} , ctx)
    # extract str
    mlir_buffer = IOBuffer()
    print(mlir_buffer, op)
    mlir_str = String(take!(mlir_buffer))

    # write to temp file for mlir-opt to read
    open("/tmp/temp.mlir", "w") do io
        write(io, mlir_str)
    end

    # lower
    for pass in passes
        run(pass)
    end

    # read from file back into the pipeline
    ir = read("/tmp/temp_out.mlir", String)

    mod = parse(IR.Module, ir; context=ctx)
end

struct MatRes
  p1    :: Ptr{Cvoid}
  p2    :: Ptr{Cvoid}
  n     :: Int64
  dims1 :: NTuple{3,Int64}
  dims2 :: NTuple{3,Int64}
end

"Execute function using MLIR pipeline"
function eval_mlir(f, args...; ctx = IR.context())
    # registerAllUpstreamDialects!(ctx)
    for dialect in (:func, :cf, :scf, :llvm, :memref, :linalg, :tensor)
        IR.register_dialect!(IR.DialectHandle(dialect); context=ctx)
    end


    IR.allow_unregistered_dialects!(true; context=ctx)

    # preprocess arguments
    arg_types = eval(Expr(
            :curly,
            Tuple,
            map(arg -> :($(Core.Typeof)($arg)), args[(begin + 1):end])...,
           ))

    processed_arg_types_tuple = map(arg -> Core.Typeof(eval(arg)), args[(begin + 1):end])

    # TODO; consider integrating without running type inference twice without modifying fn code_mlir (check the return types function)
    interp = MLIRInterpreter()
    println("Created MLIRInterpreter")
    _, ret = only(CC.code_ircode(f, processed_arg_types_tuple; interp=interp))
    


#     # get the function ptr within the JIT
    # fptr = IR.context!(IR.Context()) do
        # get top-level mlir function call (MLIR.IR.Operation)
    mod = code_mlir(f, arg_types; ctx=ctx)

    GC.@preserve mod begin
        mod = external_lowering_mlir_opt!(mod, [`mlir-opt /tmp/temp.mlir --pass-pipeline="builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg))" -o /tmp/temp.mlir`, `mlir-opt /tmp/temp.mlir -one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" -o /tmp/temp_out.mlir`], ctx)
        mod = external_lowering_mlir_opt!(mod, [`mlir-opt /tmp/temp.mlir --canonicalize -o /tmp/temp_out.mlir`], ctx)

        mod = external_lowering_mlir_opt!(mod, [`mlir-opt /tmp/temp.mlir -one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" -o /tmp/temp_out.mlir`], ctx)
        mod = external_lowering_mlir_opt!(mod, [`mlir-opt /tmp/temp.mlir --convert-linalg-to-affine-loops -o /tmp/temp_out.mlir`], ctx)
        mod = external_lowering_mlir_opt!(mod, [`mlir-opt /tmp/temp.mlir --lower-affine -o /tmp/temp_out.mlir`], ctx)
        mod = external_lowering_mlir_opt!(mod, [`mlir-opt /tmp/temp.mlir --expand-strided-metadata -o /tmp/temp_out.mlir`], ctx)
        mod = external_lowering_mlir_opt!(mod, [`mlir-opt /tmp/temp.mlir --convert-scf-to-cf -o /tmp/temp_out.mlir`], ctx)
        mod = external_lowering_mlir_opt!(mod, [`mlir-opt /tmp/temp.mlir --convert-math-to-funcs -o /tmp/temp_out.mlir`], ctx)
        mod = external_lowering_mlir_opt!(mod, [`mlir-opt /tmp/temp.mlir --convert-to-llvm -o /tmp/temp_out.mlir`], ctx)

        # initialise PassManager
        pm::IR.PassManager = IR.PassManager()
        opm::IR.OpPassManager = IR.OpPassManager(pm)

        # verify the validity of the LLVM IR
        IR.enable_verifier!(pm, true)
        IR.verifyall(mod)

        println("Processing mod: $mod with name $(String(nameof(f)))")

        # create the jit (locally)
        jit = IR.ExecutionEngine(mod, 0)
        fptr = IR.lookup(jit, String(nameof(f)))


        expanded_args = eval(Expr(:tuple, args[2:end]...))
        expanded_types = Expr(:tuple, processed_arg_types_tuple...)

        original_ret = ret
        if ret <: AbstractArray
            ret = MatRes
        end

        dynamic_call = :(ccall($fptr, $ret, $(expanded_types), $(expanded_args...)))

        result = eval(dynamic_call)

        # extract intrinsic information (for 2d matrices)
        if result isa MatRes
            item_type = eltype(original_ret)
            ptr = Ptr{item_type}(result.p2)

            corrected_dims = reverse(result.dims1)
            result_out = unsafe_wrap(Array{item_type, 3}, ptr, Tuple(corrected_dims); own = false)
            result_out = permutedims(reshape(result_out, corrected_dims), (3, 2, 1))


            # find all dims of length 1
            sd = findall(x -> x==1, size(result_out))
            # drop them
            result_out = dropdims(result_out; dims = tuple(sd...))

            # for _ in 1:(length(size(result_out)) - length(size(original_ret)))
            #     for (idx, val) in enumerate(size(result_out))
            #         if val == 1
            #             result_out = dropdims(result_out; dims = idx) 
            #         end
            #     end
            # end

            GC.gc()

            result = convert(original_ret, result_out)
        end

        return result
    end

    return result
end

