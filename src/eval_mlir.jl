# using MLIR
using LLVM: LLVM
using Core: PhiNode, GotoNode, GotoIfNot, SSAValue, Argument, ReturnNode, PiNode
using MLIR.IR
using MLIR
using MLIR.Dialects: arith, func, cf

include("code_mlir.jl")

"Macro @eval_mlir f(args...)"
macro eval_mlir(call)
    @assert Meta.isexpr(call, :call) "only calls are supported"

    f = esc(first(call.args))
    arg_types = esc(call.args )

    quote
        eval_mlir($f, $arg_types...)
    end
end

"Execute function using MLIR pipeline"
function eval_mlir(f, args...)
    println("here")
    # preprocess arguments
    # arg_types = Tuple(map(arg-> Core.Typeof(arg), args[2:end]))# Tuple{Core.Typeof(5), Core.Typeof(10)}
    arg_types = eval(Expr(
            :curly,
            Tuple,
            map(arg -> :($(Core.Typeof)($arg)), args[(begin + 1):end])...,
           ))
    arg_types_tuple = map(arg -> typeof(arg), args[(begin + 1):end])
    println("a: ", arg_types_tuple, " -- ", typeof(arg_types_tuple))

    # get the function ptr within the JIT
    fptr = IR.context!(IR.Context()) do
        # get top-level mlir function call (MLIR.IR.Operation)
        op::IR.Operation = code_mlir(f, arg_types) 
        println("processed successfully")
        println(op)


        # encapsulate into a module
        mod = IR.Module(Location())
        body = IR.body(mod)
        push!(body, op)

        # initialise PassManager
        pm::IR.PassManager = IR.PassManager()
        opm::IR.OpPassManager = IR.OpPassManager(pm)

        # register LLVM lowering passes
        MLIR.API.mlirRegisterAllPasses()
        MLIR.API.mlirRegisterAllLLVMTranslations(IR.context())
        MLIR.API.mlirRegisterConversionConvertToLLVMPass()

        # add lowering pipeline
        IR.add_pipeline!(opm, "convert-arith-to-llvm,convert-func-to-llvm")

        # run the lowering pipeline
        IR.run!(pm, mod)

        # verify the validity of the LLVM IR
        IR.enable_verifier!(pm, true)

        # create the jit (locally)
        jit = MLIR.API.mlirExecutionEngineCreate(mod, 0, 0, C_NULL, false)

        # register function call within the JIT
        println("registered fn name: ", nameof(f))
        MLIR.API.mlirExecutionEngineLookup(jit, nameof(f))
    end
    println("finished")
    println(typeof(fptr))


    expanded_args = Expr(:tuple, args[2:end]...)
    expanded_types = Expr(:tuple, arg_types_tuple...)
    println(eval(expanded_types))
    println("a: ", arg_types_tuple, " -- ", typeof(arg_types_tuple))
    a = :(ccall($fptr, Int, $(expanded_types), $(expanded_args.args...)))
    # println(a)

    return eval(a)
end

