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

# temp solution is just to run it externally, as it is not actually part of the main pipeline
function registerAllUpstreamDialects!(ctx)
    if LLVM.version() >= v"15"
        registry = MLIR.API.mlirDialectRegistryCreate()
        MLIR.API.mlirRegisterAllDialects(registry)
        MLIR.API.mlirContextAppendDialectRegistry(ctx, registry)
        MLIR.API.mlirDialectRegistryDestroy(registry)
    else
        MLIR.API.mlirRegisterAllDialects()
    end

    return nothing
end

# externally lower TOSA to linalg (unsupported by MLIR.jl)
function external_lowering_mlir_opt!(op, passes::Cmd , ctx)
    # extract str
    mlir_buffer = IOBuffer()
    print(mlir_buffer, op)
    mlir_str = String(take!(mlir_buffer))

    # write to temp file for mlir-opt to read
    open("/tmp/temp.mlir", "w") do io
        write(io, mlir_str)
    end

    # lower
    run(passes)

    # read from file back into the pipeline
    ir = read("/tmp/temp_out.mlir", String)

    registerAllUpstreamDialects!(ctx)

    mod = parse(IR.Module, ir)
    # ctx = IR.context(mod)

    # IR.Operation(mod)
    # get_op_with_ownership(mod)
    # return mod
end


"Execute function using MLIR pipeline"
function eval_mlir(f, args...; ctx = IR.context())
    # registerAllUpstreamDialects!(ctx)

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
    println("Running code_mlir")
    mod = code_mlir(f, arg_types; ctx=ctx)
    println("Produced IR Code: $mod")

    GC.@preserve mod begin
        # println("Running GC")

        # GC.gc()
        # println("Ran GC")
        # ctx = IR.Context(op)

        # lower to linalg
        mod = external_lowering_mlir_opt!(mod, `mlir-opt /tmp/temp.mlir --pass-pipeline="builtin.module(func.func(tosa-to-linalg))" -o /tmp/temp_out.mlir`, ctx)
        mod = external_lowering_mlir_opt!(mod, `mlir-opt /tmp/temp.mlir -one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" -o /tmp/temp_out.mlir`, ctx)
        mod = external_lowering_mlir_opt!(mod, `mlir-opt /tmp/temp.mlir --convert-linalg-to-affine-loops -o /tmp/temp_out.mlir`, ctx)
        mod = external_lowering_mlir_opt!(mod, `mlir-opt /tmp/temp.mlir --lower-affine -o /tmp/temp_out.mlir`, ctx)
        mod = external_lowering_mlir_opt!(mod, `mlir-opt /tmp/temp.mlir --expand-strided-metadata -o /tmp/temp_out.mlir`, ctx)
        mod = external_lowering_mlir_opt!(mod, `mlir-opt /tmp/temp.mlir --convert-scf-to-cf -o /tmp/temp_out.mlir`, ctx)
        mod = external_lowering_mlir_opt!(mod, `mlir-opt /tmp/temp.mlir --convert-to-llvm -o /tmp/temp_out.mlir`, ctx)

        # initialise PassManager
        pm::IR.PassManager = IR.PassManager()
        opm::IR.OpPassManager = IR.OpPassManager(pm)

        # verify the validity of the LLVM IR
        IR.enable_verifier!(pm, true)
        IR.verifyall(mod)

        # @atomic op.owned = true

        # mod = IR.Module(Location())
        # body = IR.body(mod)
        # push!(body, op)

        # mod = op
        println("Processing mod: $mod with name $(String(nameof(f)))")

        # create the jit (locally)
        jit = IR.ExecutionEngine(mod, 0)
        fptr = IR.lookup(jit, String(nameof(f)))

        println("got fptr $fptr")

        expanded_args = eval(Expr(:tuple, args[2:end]...))
        expanded_types = Expr(:tuple, processed_arg_types_tuple...)

        dynamic_call = :(ccall($fptr, $ret, $(expanded_types), $(expanded_args...)))

        result = eval(dynamic_call)

    end

    return result
end

