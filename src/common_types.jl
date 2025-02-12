# using MLIR
using LLVM: LLVM
using Core: PhiNode, GotoNode, GotoIfNot, SSAValue, Argument, ReturnNode, PiNode
using MLIR.IR
using MLIR
using MLIR.Dialects: arith, func, cf

include("compiler.jl")


mutable struct Context
  ir::Core.Compiler.IRCode
  values::Vector{Value}
  n_phi_nodes::Int
  phi_nodes_metadata::Dict{Int, Vector{Any}}
  sidx
  line
  stmt
end

mutable struct Blocks
  block_id
  current_block
  entry_block
  blocks::Array
  bb
end

# enforce that these types are not broadcastable
Base.broadcastable(c::Context) = Ref(c)
Base.broadcastable(b::Blocks) = Ref(b)


ScalarTypes = Union{Bool, UInt8, UInt64,Int64,UInt32,Int32,Float32,Float64}

function type_convert(ir)
    for i in 1:length(ir.argtypes)
        if ir.argtypes[i] == UInt64
            ir.argtypes[i] = Int64
        end
    end
end

# replace UInt with Int
