# using MLIR
using LLVM: LLVM
using Core: PhiNode, GotoNode, GotoIfNot, SSAValue, Argument, ReturnNode, PiNode
using MLIR.IR
using MLIR
using MLIR.Dialects: arith, func, cf, linalg, math
using StaticArrays
using LinearAlgebra
using FixedPointNumbers

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

# recast fixed as integers
function recast_fixed(x)
    if x isa Fixed
        type = typeof(x)
        underlying_datatype = first(type.parameters)
        return reinterpret(underlying_datatype, x)
    else
        return x
    end
end


ScalarTypes = Union{Bool, UInt8, UInt64,Int64,UInt32,Int32,Float32,Float64, SArray, MArray, Fixed}

# Extend MLIR.jl
function IR.Type(T::Core.Type{<:Unsigned}; context::IR.Context=context())
    return IR.Type(MLIR.API.mlirIntegerTypeGet(context, sizeof(T) * 8))
end

function IR.Type(T::Core.Type{<:Fixed}; context::IR.Context=context())
    println("Sizeof: $(sizeof(T) * 8)")
    container, frac_bits = T.parameters
    scale = 1.0 / (2.0 ^ frac_bits)
    zp = 0.0
    bw = 32

    flag = MLIR.API.mlirQuantizedTypeGetSignedFlag()

    storage_type = IR.Type(container)
    expressed_type = IR.Type(Float32)

    if container <: Signed 
        smin = -Int(1 << (bw - 1))
        smax =  Int(1 << (bw - 1)) - 1
    else
        smin = 0
        smax = Int(1 << bw) - 1
    end

    # return IR.Type(MLIR.API.mlirIntegerTypeGet(context, sizeof(T) * 8))
    return IR.Type(MLIR.API.mlirUniformQuantizedTypeGet(flag,storage_type, expressed_type, scale, zp, smin, smax))
end

get_op_with_ownership(module_::IR.Module) = IR.Operation(MLIR.API.mlirModuleGetOperation(module_), true)

## StaticArrays 
function IR.Type(T::Core.Type{<:AbstractArray}; context::IR.Context=context())
    dims::Vector{Int64} = collect(T.parameters[1].parameters)

    for _ in 1:(3 - length(dims))
        dims = reverse(dims)
        push!(dims, 1)
        dims = reverse(dims)
    end

    type = IR.Type(T.parameters[2])

    return IR.TensorType(dims, type)
end

# Adjoint
function IR.Type(T::Core.Type{<:LinearAlgebra.Adjoint}; context::IR.Context=context())
    dims = size(T)
    ret_type = IR.Type(eltype(T))

    return IR.TensorType(dims, ret_type)
end

function IR.Type(T::Core.Type{Any}; context::IR.Context=context())
    return nothing
end
