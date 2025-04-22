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
function external_lowering_mlir_opt!(op::IR.Operation, passes::Cmd , ctx)
    # extract str
    mlir_buffer = IOBuffer()
    print(mlir_buffer, op)
    mlir_str = String(take!(mlir_buffer))

    # write to temp file for mlir-opt to read
    open("/tmp/temp.mlir", "w") do io
        write(io, mlir_str)
    end

    # lower
    println("Running CMD: ", passes)
    run(passes)


    # read from file back into the pipeline
    ir = read("/tmp/temp.mlir", String)

    ctx = MLIR.API.mlirContextCreate()
    registerAllUpstreamDialects!(ctx)

    mod = MLIR.API.mlirModuleCreateParse(ctx , ir)

    old_op = op.operation
    op.operation = MLIR.API.mlirModuleGetOperation(mod)
    MLIR.API.mlirOperationDestroy(old_op)

    return 
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
    fptr = IR.context!(IR.Context()) do
        # get top-level mlir function call (MLIR.IR.Operation)
        println("Running code_mlir")
        op::IR.Operation = code_mlir(f, arg_types) 
        println("Produced IR Code")

        ctx = IR.context(op)

        # lower to linalg
        external_lowering_mlir_opt!(op, `mlir-opt /tmp/temp.mlir --pass-pipeline="builtin.module(func.func(tosa-to-linalg))" -o /tmp/temp_out.mlir`, ctx)

        external_lowering_mlir_opt!(op, `mlir-opt /tmp/temp.mlir -one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" -o /tmp/temp_out.mlir`, ctx) #--one-shot-bufferize --convert-linalg-to-loops --convert-to-llvm -o temp_out.mlir`, ctx)
        # # mod = external_lowering_mlir_opt!(op, `--one-shot-bufferize`, ctx)
        # op = IR.Operation(MLIR.API.mlirModuleGetOperation(mod))

        external_lowering_mlir_opt!(op, `mlir-opt /tmp/temp.mlir --convert-linalg-to-affine-loops -o /tmp/temp_out.mlir`, ctx)
        # op = IR.Operation(MLIR.API.mlirModuleGetOperation(mod))


        # # mod = external_lowering_mlir_opt!(op, `mlir-opt temp.mlir --convert-convert-memref-to-llvm -o /tmp/temp_out.mlir`, ctx)
        # # op = IR.Operation(MLIR.API.mlirModuleGetOperation(mod))

        external_lowering_mlir_opt!(op, `mlir-opt /tmp/temp.mlir --lower-affine -o /tmp/temp_out.mlir`, ctx)
        # op = IR.Operation(MLIR.API.mlirModuleGetOperation(mod))

        external_lowering_mlir_opt!(op, `mlir-opt /tmp/temp.mlir --expand-strided-metadata -o /tmp/temp_out.mlir`, ctx)


        external_lowering_mlir_opt!(op, `mlir-opt /tmp/temp.mlir --convert-scf-to-cf -o /tmp/temp_out.mlir`, ctx)
        # op = IR.Operation(MLIR.API.mlirModuleGetOperation(mod))


        external_lowering_mlir_opt!(op, `mlir-opt temp.mlir --convert-to-llvm -o /tmp/temp_out.mlir`, ctx)
        # op = IR.Operation(MLIR.API.mlirModuleGetOperation(fo.mod))
        

        # mod = external_lowering_mlir_opt!(op, `mlir-opt /tmp/temp.mlir --convert-to-llvm -o temp_out.mlir`, ctx)
        # op = IR.Operation(MLIR.API.mlirModuleGetOperation(mod))

        println("received: ", op)
        # encapsulate into a module
        # mod = IR.Module(Location())
        # body = IR.body(mod)
        # push!(body, op)

        println("Created body")


        # initialise PassManager
        pm::IR.PassManager = IR.PassManager()
        opm::IR.OpPassManager = IR.OpPassManager(pm)

        # lower mlir to linalg

        # register LLVM lowering passes
        # MLIR.API.mlirRegisterAllPasses()
        # MLIR.API.mlirRegisterAllLLVMTranslations(IR.context())
        # MLIR.API.mlirRegisterConversionConvertToLLVMPass()
        # MLIR.API.mlirRegisterConversionTosaToSCF()
        # MLIR.API.mlirRegisterConversionConvertLinalgToStandard()
        # # MLIR.API.mlirRegisterLinalgLinalgDetensorize()
        # MLIR.API.mlirRegisterLinalgLinalgBufferize()
        # MLIR.API.mlirRegisterConversionSCFToControlFlow()

        
        # MLIR.API.mlirCreateConversionTosaToLinalg()

        # add lowering pipeline
        # IR.add_pipeline!(opm, "convert-scf-to-cf,convert-arith-to-llvm,convert-func-to-llvm")
        # IR.add_pipeline!(opm, "linalg-bufferize")

        # run the lowering pipeline
        # IR.run!(pm, mod)
        # op = MLIR.API.mlirModuleGetOperation(mod)
        # MLIR.API.mlirPassManagerRunOnOp(pm, op)

        # verify the validity of the LLVM IR
        IR.enable_verifier!(pm, true)
        IR.verifyall(op)


        println("verified")

        # lowerToLinalg(ctx, mod)

        mod = IR.Module(op)
        # create the jit (locally)
        jit = IR.ExecutionEngine(mod, 0)
        IR.lookup(jit, String(nameof(f)))
    end

    println("compiled code successfully")

    # GC.gc()

    expanded_args = eval(Expr(:tuple, args[2:end]...))
    expanded_types = Expr(:tuple, processed_arg_types_tuple...)

    dynamic_call = :(ccall($fptr, $ret, $(expanded_types), $(expanded_args...)))

    return eval(dynamic_call)
end

