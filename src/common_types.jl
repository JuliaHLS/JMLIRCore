# using MLIR
using LLVM: LLVM
using Core: PhiNode, GotoNode, GotoIfNot, SSAValue, Argument, ReturnNode, PiNode
using MLIR.IR
using MLIR
using MLIR.Dialects: arith, func, cf
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


ScalarTypes = Union{Bool, UInt8, UInt64,Int64,UInt32,Int32,Float32,Float64}

# Extend MLIR.jl
function IR.Type(T::Core.Type{<:Unsigned}; context::IR.Context=context())
    return IR.Type(MLIR.API.mlirIntegerTypeGet(context, sizeof(T) * 8))
end

# # Extend MLIR.jl
# function IR.Type(T::Core.Type{<:Integer}; context::IR.Context=context())
#     return IR.Type(MLIR.API.mlirIntegerTypeGet(context, sizeof(T) * 8))
# end

# function IR.Type(T::Core.Type{<:Signed}; context::IR.Context=context())
#     return IR.Type(MLIR.API.mlirIntegerTypeGet(context, sizeof(T) * 8))
# end


## StaticArrays 
function IR.Type(T::Core.Type{<:SVector}; context::IR.Context=context())
    dims::Vector{Int64} = collect(T.parameters[1].parameters)
    type = IR.Type(T.parameters[2])

    # TODO: check 0 is the correct memspace cfg
    memspace_cfg = IR.Attribute(0)

    return IR.MemRefType(type, dims, memspace_cfg)
end



