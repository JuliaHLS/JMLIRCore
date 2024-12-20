# using MLIR
using LLVM: LLVM
using Core: PhiNode, GotoNode, GotoIfNot, SSAValue, Argument, ReturnNode, PiNode
using MLIR.IR
using MLIR
using MLIR.Dialects: arith, func, cf


mutable struct Context
  ir::Core.Compiler.IRCode
  values::Vector{Value}
  n_phi_nodes::Int
  sidx
  line
  stmt
end

mutable struct Blocks
  block_id
  current_block
  blocks::Array
  bb
end


