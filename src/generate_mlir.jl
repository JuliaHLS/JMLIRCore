using MLIR.Dialects: arith, func, cf, memref, linalg, tosa, tensor
using MLIR
using MLIR.IR

""" Method Wrapper """
struct MethodDetails
    sym::Symbol
    sig::Any
    rettype::DataType

    function MethodDetails(fn::Core.CodeInstance)
        new(clean_mangled_symbol(fn.def.def.name), fn.def.def.sig, fn.rettype)
    end
end


function generate_mlir(md::MethodDetails)
    return generate_mlir(Val(md.sym), (md.rettype), Val(md.sig))
end


""" clean mangled symbols """
function clean_mangled_symbol(sym::Symbol)::Symbol
    cleaned = lstrip(String(sym), '#')
    return Symbol(cleaned)
end

# This is an artifact of the inconsistency behind the MLIR.jl API
# There is no real pattern, and it has to be resolved partially
# manually

@inline function single_op_wrapper_with_result(fop)
    return (block::MLIR.IR.Block, args::Vector{MLIR.IR.Value}; result, location=Location()) ->
        push!(block, fop(args...; result, location))
end

@inline function single_op_wrapper_out_is_result(fop)
        return (block::MLIR.IR.Block, args::Vector{MLIR.IR.Value}; result, location=Location()) ->
        push!(block, fop(args...; out=result, location))
end

@inline function single_op_wrapper_no_result(fop)
    return (block::MLIR.IR.Block, args::Vector{MLIR.IR.Value}; result, location=Location()) ->
        push!(block, fop(args...; location))
end

@inline function single_op_wrapper_output_is_result(fop)
    return (block::MLIR.IR.Block, args::Vector{MLIR.IR.Value}; result, location=Location()) ->
        push!(block, fop(args...; output=result, location))
end


# Enumerate the Predicates
module Predicates
    const eq = 0
    const ne = 1
    const slt = 2
    const sle = 3
    const sgt = 4
    const sge = 5
    const ult = 6
    const ule = 7
    const ugt = 8
    const uge = 9
end


########################
###  GENERATE MLIR   ###
########################

""" Base method """
function generate_mlir(op, rettype, sig)
    error("Error: No mlir translation defined for $op with rettype: $rettype and signature: $sig. Please Modify `generate_mlir.jl`")
end


### INTEGERS ###
function generate_mlir(::Val{:+}, rettype::Type{<:Integer}, sig::Any)
    return single_op_wrapper_with_result(arith.addi)
end

function generate_mlir(::Val{:-}, rettype::Type{<:Integer}, sig::Any)
    return single_op_wrapper_with_result(arith.subi)
end

function generate_mlir(::Val{:*}, rettype::Type{<:Integer}, sig::Any)
    return single_op_wrapper_with_result(arith.muli)
end

function generate_mlir(::Val{Base.lshr_int}, rettype::Type{<:Integer}, sig::Any)
    return single_op_wrapper_out_is_result(arith.shrui)
end

function generate_mlir(::Val{Base.lshr_int}, rettype::Type{<:Integer}, sig::Any)
    return single_op_wrapper_out_is_result(arith.shrui)
end

function generate_mlir(::Val{Base.checked_srem_int}, rettype::Type{<:Integer}, sig::Any)
    return single_op_wrapper_out_is_result(arith.remsi)
end


### FLOAT ###
function generate_mlir(::Val{:+}, rettype::Type{<:AbstractFloat}, sig::Any)
    return single_op_wrapper_with_result(arith.addf)
end

function generate_mlir(::Val{:-}, rettype::Type{<:AbstractFloat}, sig::Any)
    return single_op_wrapper_with_result(arith.subf)
end

function generate_mlir(::Val{:*}, rettype::Type{<:AbstractFloat}, sig::Any)
    return single_op_wrapper_with_result(arith.mulf)
end

function generate_mlir(::Val{:/}, rettype::Type{<:AbstractFloat}, sig::Any)
    return single_op_wrapper_with_result(arith.divf)
end

function generate_mlir(::Val{Base.sitofp}, rettype::Type{<:AbstractFloat}, sig::Any)
    return single_op_wrapper_with_result(arith.sitofp)
end



### VECTORS ###
function generate_mlir(::Val{:+}, rettype::Type{<:MVector}, sig::Any)
    return single_op_wrapper_output_is_result(tosa.add)
end

function generate_mlir(::Val{:-}, rettype::Type{<:MVector}, sig::Any)
    return single_op_wrapper_output_is_result(tosa.sub)
end

function generate_mlir(::Val{:*}, rettype::Type{<:MVector}, sig::Any)
    return single_op_wrapper_output_is_result(tosa.matmul)
end



### PREDICATES ###

# predicate target helper
function cmpi_pred(predicate, rettype::Type)
    if rettype === float
        function (ops...; location=Location())
          return arith.cmpf(ops...; result=IR.Type(Bool), predicate, location)
        end
    elseif rettype <: Integer
        function (ops...; location=Location())
          return arith.cmpi(ops...; result=IR.Type(Bool), predicate, location)
        end
    else
        error("Unrecognized predicate type: $rettype")
    end
end

# Signed integer
function generate_mlir(::Val{:(<=)}, rettype::Type{<:Bool}, sig::Val{Tuple{typeof(<=), T, T} where T<:Union{Int128, Int16, Int32, Int64, Int8}})
    return single_op_wrapper_no_result(cmpi_pred(Predicates.sle, rettype))
end

function generate_mlir(::Val{:<}, rettype::Type{<:Bool}, sig::Val{Tuple{typeof(<), T, T} where T<:Union{Int128, Int16, Int32, Int64, Int8}})
    return single_op_wrapper_no_result(cmpi_pred(Predicates.slt, rettype))
end

function generate_mlir(::Val{:(>=)}, rettype::Type{<:Bool}, sig::Any)#sig::Val{Tuple{typeof(>=), T, T} where T<:Union{Int128, Int16, Int32, Int64, Int8}})
    return single_op_wrapper_no_result(cmpi_pred(Predicates.sge, rettype))
end

function generate_mlir(::Val{:>}, rettype::Type{<:Bool}, sig::Any)#sig::Tuple{typeof(>), T, T} where T<:Union{Int128, Int16, Int32, Int64, Int8})
    return single_op_wrapper_no_result(cmpi_pred(Predicates.sgt, rettype))
end


# unsigned
function generate_mlir(::Val{:(<=)}, rettype::Type{<:Bool}, sig::Val{Tuple{typeof(<=), T, T} where T<:Union{UInt128, UInt16, UInt32, UInt64, UInt8}})
    return single_op_wrapper_no_result(cmpi_pred(Predicates.ule, rettype))
end

function generate_mlir(::Val{:<}, rettype::Type{<:Bool}, sig::Val{Tuple{typeof(<), T, T} where T<:Union{UInt128, UInt16, UInt32, UInt64, UInt8}})
    return single_op_wrapper_no_result(cmpi_pred(Predicates.ult, rettype))
end

# TODO: can't currently be reached, as signature from Julia is wrong and doesn't take types into consideration
function generate_mlir(::Val{:(>=)}, rettype::Type{<:Bool}, sig::Val{Tuple{typeof(>=), T, T} where T<:Union{UInt128, UInt16, UInt32, UInt64, UInt8}})
    return single_op_wrapper_no_result(cmpi_pred(Predicates.uge, rettype))
end

# TODO: can't currently be reached, as signature from Julia is wrong and doesn't take types into consideration
function generate_mlir(::Val{:>}, rettype::Type{<:Bool}, sig::Val{Tuple{typeof(>), T, T} where T<:Union{UInt128, UInt16, UInt32, UInt64, UInt8}})
    return single_op_wrapper_no_result(cmpi_pred(Predicates.ugt, rettype))
end


# TODO: add float and Fixed Point


function generate_mlir(::Val{:(===)}, rettype::Type, sig::Any)
    return single_op_wrapper_no_result(cmpi_pred(Predicates.eq, rettype))
end

# TODO: check if this should take rettype as a float input, or do this generically
function generate_mlir(::Val{Base.ne_float}, rettype::Type{Bool}, sig::Any)
    return single_op_wrapper_no_result(cmpi_pred(Predicates.ne, rettype))
end

function generate_mlir(::Val{Base.not_int}, rettype::Type{Bool}, sig::Any)
    Base.not_int => function (block, args; location=Location())
        arg = only(args)
        mT = IR.type(arg)
        T = IR.julia_type(mT)
        ones = IR.result(
          push!(block, arith.constant(; value=typemax(UInt64) % T, result=mT, location)),
        )
    return push!(block, arith.xori(arg, ones; location))
  end
end




