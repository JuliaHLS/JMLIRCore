module JMLIRCore

using LLVM: LLVM
using Core: PhiNode, GotoNode, GotoIfNot, SSAValue, Argument, ReturnNode, PiNode
using MLIR.IR
using MLIR
using MLIR.Dialects: arith, func, cf


# add files to import from
include("eval_mlir.jl")

# export code_mlir
export @code_mlir, code_mlir, @eval_mlir, code_mlir

end
