include("code_mlir.jl")
include("compiler.jl")

"Macro @eval_mlir f(args...)"
macro eval_mlir(call)
    @assert Meta.isexpr(call, :call) "only calls are supported"

    f = esc(first(call.args))
    arg_types = esc(call.args)

    quote
        eval_mlir($f, $arg_types...)
    end
end

# function lowerToLinalg(ctx, mod)
#     MLIR.API.mlirRegisterConversionTosaToSCF()
#     pm = MLIR.API.mlirPassManagerCreate(ctx)
#     if LLVM.version() >= v"15"
#         op = "func.func"
#     else
#         op = "builtin.func"
#     end
#     opm = MLIR.API.mlirPassManagerGetNestedUnder(pm, op)
#     # MLIR.API.mlirPassManagerAddOwnedPass(
#     #     pm, MLIR.API.mlirCreateConversionTosaToLinalg()
#     # )

#     MLIR.API.mlirOpPassManagerAddOwnedPass(
#     #     opm, MLIR.API.mlirCreateConversionArithToLLVMConversionPass()
#         opm, MLIR.API.mlirCreateConversionTosaToSCF()
#     )

#     op = MLIR.API.mlirModuleGetOperation(mod)
#    status = MLIR.API.mlirPassManagerRunOnOp(pm, op)

#     println("NEW MOD: ", mod)
#     println("NEW OP: ", op)
#     # undefined symbol: mlirLogicalResultIsFailure
#     if status.value == 0
#         error("Unexpected failure running pass failure")
#     end
#     return MLIR.API.mlirPassManagerDestroy(pm)
# end
# function transform_mlir_string_with_mlir_opt(mlir_string::String, pass_pipeline::String, output_file::String)
#     # Write the MLIR string to a temporary file
#     open("input.mlir", "w") do io
#         write(io, mlir_string)
#     end

#     # Execute mlir-opt
#     command = `mlir-opt input.mlir $pass_pipeline -o $output_file`
#     run(command)

#     # Parse output MLIR
#     output_mlir_string = read(output_file, String)
#     output_module = MLIR.parse(MLIR.Context(), output_mlir_string)

#     return output_module
# end
# temp solution is just to run it externally, as it is not actually part of the main pipeline
function registerAllUpstreamDialects!(ctx)
    if LLVM.version() >= v"15"
        registry = MLIR.API.mlirDialectRegistryCreate()
        MLIR.API.mlirRegisterAllDialects(registry)
        MLIR.API.mlirContextAppendDialectRegistry(ctx, registry)
        MLIR.API.mlirDialectRegistryDestroy(registry)
    else
        MLIR.API.mlirRegisterAllDialects(ctx)
    end

    return nothing
end

# externally lower TOSA to linalg (unsupported by MLIR.jl)
function external_lowering_mlir_opt!(ops, passes::Cmd , ctx)
    # extract str
    mlir_buffer = IOBuffer()
    print(mlir_buffer, ops)
    mlir_str = String(take!(mlir_buffer))

    # write to temp file for mlir-opt to read
    open("temp.mlir", "w") do io
        write(io, mlir_str)
    end

    println("str taken: ", mlir_str)


    # lower
    # cmd = `mlir-opt temp.mlir -o temp_out.mlir`
    # --one-shot-bufferize --convert-linalg-to-loops --convert-to-llvm -o temp_out.mlir`
    println("Running CMD: ", passes)
    run(passes)


    # read from file back into the pipeline
    ir = read("temp_out.mlir", String)

    println("MLIR: ", LLVM.version())
    println("processed: ", ir)
    

    ctx = MLIR.API.mlirContextCreate()
    println("processed: 1")
    registerAllUpstreamDialects!(ctx)
    println("processed: 2")
    mod = MLIR.API.mlirModuleCreateParse(ctx , ir)
    println("processed: 3")


    return IR.Module(mod)
end


"Execute function using MLIR pipeline"
function eval_mlir(f, args...)
    ctx = IR.Context()
    
    # preprocess arguments
    arg_types = eval(Expr(
            :curly,
            Tuple,
            map(arg -> :($(Core.Typeof)($arg)), args[(begin + 1):end])...,
           ))

    processed_arg_types_tuple = map(arg -> Core.Typeof(eval(arg)), args[(begin + 1):end])

    # TODO; consider integrating without running type inference twice without modifying fn code_mlir (check the return types function)
    interp = MLIRInterpreter()
    _, ret = only(CC.code_ircode(f, processed_arg_types_tuple; interp=interp))



#     # get the function ptr within the JIT
    fptr = IR.context!(IR.Context()) do
        # get top-level mlir function call (MLIR.IR.Operation)
        op::IR.Operation = code_mlir(f, arg_types) 

        # lower to linalg
        mod = external_lowering_mlir_opt!(op, `mlir-opt temp.mlir --pass-pipeline="builtin.module(func.func(tosa-to-linalg))" -o temp_out.mlir`, ctx)
        op = IR.Operation(MLIR.API.mlirModuleGetOperation(mod))

        mod = external_lowering_mlir_opt!(op, `mlir-opt temp.mlir --pass-pipeline="builtin.module(one-shot-bufferize{bufferize-function-boundaries})" -o temp_out.mlir`, ctx) #--one-shot-bufferize --convert-linalg-to-loops --convert-to-llvm -o temp_out.mlir`, ctx)
        # mod = external_lowering_mlir_opt!(op, `--one-shot-bufferize`, ctx)
        op = IR.Operation(MLIR.API.mlirModuleGetOperation(mod))

        mod = external_lowering_mlir_opt!(op, `mlir-opt temp.mlir --convert-linalg-to-affine-loops -o temp_out.mlir`, ctx)
        op = IR.Operation(MLIR.API.mlirModuleGetOperation(mod))


        # mod = external_lowering_mlir_opt!(op, `mlir-opt temp.mlir --convert-convert-memref-to-llvm -o temp_out.mlir`, ctx)
        # op = IR.Operation(MLIR.API.mlirModuleGetOperation(mod))

        mod = external_lowering_mlir_opt!(op, `mlir-opt temp.mlir --lower-affine -o temp_out.mlir`, ctx)
        op = IR.Operation(MLIR.API.mlirModuleGetOperation(mod))


        mod = external_lowering_mlir_opt!(op, `mlir-opt temp.mlir --convert-scf-to-cf -o temp_out.mlir`, ctx)
        op = IR.Operation(MLIR.API.mlirModuleGetOperation(mod))


        mod = external_lowering_mlir_opt!(op, `mlir-opt temp.mlir --convert-to-llvm -o temp_out.mlir`, ctx)
        op = IR.Operation(MLIR.API.mlirModuleGetOperation(mod))

        # mod = external_lowering_mlir_opt!(op, `mlir-opt temp.mlir --convert-to-llvm -o temp_out.mlir`, ctx)
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

        # println("GOT mod: ", mod)

        # create the jit (locally)
        jit = MLIR.API.mlirExecutionEngineCreate(mod, 0, 0, C_NULL, false)

        println("created jit")

        # register function call within the JIT
        MLIR.API.mlirExecutionEngineLookup(jit, nameof(f))
        println("created handle")
    end

    println("compiled code successfully")

    # expanded_args = eval(Expr(:tuple, args[2:end]...))
    # expanded_types = Expr(:tuple, processed_arg_types_tuple...)
    # dynamic_call = :(ccall($fptr, $ret, $(expanded_types), $(expanded_args...)))

    # return eval(dynamic_call)
    return -1
end

