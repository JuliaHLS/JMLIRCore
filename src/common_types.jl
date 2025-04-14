# using MLIR
using LLVM: LLVM
using Core: PhiNode, GotoNode, GotoIfNot, SSAValue, Argument, ReturnNode, PiNode
using MLIR.IR
using MLIR
using MLIR.Dialects: arith, func, cf, linalg
using StaticArrays

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


ScalarTypes = Union{Bool, UInt8, UInt64,Int64,UInt32,Int32,Float32,Float64, SArray, MArray}

# Extend MLIR.jl
function IR.Type(T::Core.Type{<:Unsigned}; context::IR.Context=context())
    return IR.Type(MLIR.API.mlirIntegerTypeGet(context, sizeof(T) * 8))
end


## StaticArrays 
function IR.Type(T::Core.Type{<:MArray}; context::IR.Context=context())
    dims::Vector{Int64} = collect(T.parameters[1].parameters)
    type = IR.Type(T.parameters[2])

    return IR.TensorType(dims, type)
end


function IR.Type(T::Core.Type{<:SArray}; context::IR.Context=context())
    dims::Vector{Int64} = collect(T.parameters[1].parameters)
    type = IR.Type(T.parameters[2])

    return IR.TensorType(dims, type)
end



