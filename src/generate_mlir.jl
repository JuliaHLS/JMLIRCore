using MLIR.Dialects: arith, func, cf, memref, linalg, tosa, tensor
using MLIR
using MLIR.IR

include("dialect.jl")

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

########################
###  GENERATE MLIR   ###
########################

""" Base method """
function generate_mlir(op, rettype, sig)
    error("Error: No mlir translation defined for $op with rettype: $rettype and signature: $sig. Please Modify `generate_mlir.jl`")
end

#### GENERIC OPERATIONS ####

### ARITHMETIC ###
function generate_mlir(::Val{:+}, rettype::Type{<:Any}, sig::Any)
    return single_op_wrapper_with_result(julia.add)
end

function generate_mlir(::Val{:-}, rettype::Type{<:Any}, sig::Any)
    return single_op_wrapper_with_result(julia.sub)
end

function generate_mlir(::Val{:*}, rettype::Type{<:Any}, sig::Any)
    return single_op_wrapper_with_result(julia.mul)
end

function generate_mlir(::Val{:/}, rettype::Type{<:Any}, sig::Any)
    return single_op_wrapper_with_result(julia.div)
end


### PREDICATES ###
function cmpi_pred(predicate)
    function (ops...; location=Location())
      return julia.cmp(ops...; result=IR.Type(Bool), predicate, location)
    end
end

# Generic Comparators
function generate_mlir(::Val{:(<=)}, rettype::Type{<:Any}, sig::Any)
    return single_op_wrapper_no_result(cmpi_pred(julia.predicate.le))
end

function generate_mlir(::Val{:(<)}, rettype::Type{<:Any}, sig::Any)
    return single_op_wrapper_no_result(cmpi_pred(julia.predicate.lt))
end

function generate_mlir(::Val{:(>=)}, rettype::Type{<:Any}, sig::Any)
    return single_op_wrapper_no_result(cmpi_pred(julia.predicate.gt))
end

function generate_mlir(::Val{:(>)}, rettype::Type{<:Any}, sig::Any)
    return single_op_wrapper_no_result(cmpi_pred(julia.predicate.gt))
end

function generate_mlir(::Val{:(==)}, rettype::Type{<:Any}, sig::Any)
    return single_op_wrapper_no_result(cmpi_pred(julia.predicate.eq))
end

function generate_mlir(::Val{:(===)}, rettype::Type{<:Any}, sig::Any)
    return single_op_wrapper_no_result(cmpi_pred(julia.predicate.eq))
end

function generate_mlir(::Val{:(!=)}, rettype::Type{Any}, sig::Any)
    return single_op_wrapper_no_result(cmpi_pred(julia.predicate.ne))
end



#### SPECIALISED OPERATIONS ####

### Number/Dialects/Type Specific Operations ###
function generate_mlir(::Val{Base.lshr_int}, rettype::Type{<:Integer}, sig::Any)
    return single_op_wrapper_out_is_result(arith.shrui)
end

function generate_mlir(::Val{Base.lshr_int}, rettype::Type{<:Integer}, sig::Any)
    return single_op_wrapper_out_is_result(arith.shrui)
end

function generate_mlir(::Val{Base.checked_srem_int}, rettype::Type{<:Integer}, sig::Any)
    return single_op_wrapper_out_is_result(arith.remsi)
end

function generate_mlir(::Val{Base.sitofp}, rettype::Type{<:AbstractFloat}, sig::Any)
    return single_op_wrapper_with_result(arith.sitofp)
end

