# using MLIR
using LLVM: LLVM
using Core: PhiNode, GotoNode, GotoIfNot, SSAValue, Argument, ReturnNode, PiNode
using MLIR.IR
using MLIR
using MLIR.Dialects: arith, func, cf


function lower_llvm(op::IR.Operation)
    mod = IR.Module(Location())
    body = IR.body(mod)
    push!(body, op)

    pm = IR.PassManager()
    opm = IR.OpPassManager(pm)

    MLIR.API.mlirRegisterAllPasses()
    MLIR.API.mlirRegisterAllLLVMTranslations(IR.context())
    MLIR.API.mlirRegisterConversionConvertToLLVMPass()

    IR.add_pipeline!(opm, "convert-arith-to-llvm,convert-func-to-llvm")

    IR.run!(pm, mod)
    println(mod)

    IR.enable_verifier!(pm, true)
end
