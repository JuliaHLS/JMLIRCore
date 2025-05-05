__precompile__(false)
module JMLIRCore

using MLIR.IR
ctx = IR.Context()

# add files to import from
include("eval_mlir.jl")

# export code_mlir
export @code_mlir, code_mlir, @eval_mlir, code_mlir

end
