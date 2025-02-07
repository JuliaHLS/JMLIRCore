# using MLIR
using LLVM: LLVM
using Core: PhiNode, GotoNode, GotoIfNot, SSAValue, Argument, ReturnNode, PiNode
using MLIR.IR
using MLIR
using MLIR.Dialects: arith, func, cf

include("mapping.jl")

# generic check is fop is registered as a math function
function is_math(fop)::Bool
    idx = reinterpret(Int32, fop) + 1
    return Core.Compiler.T_IFUNC[idx] == Core.Compiler.math_tfunc
end


# process comparator predicates
function cmpi_pred(predicate)
  function (ops...; location=Location())
    return arith.cmpi(ops...; result=IR.Type(Bool), predicate, location)
  end
end


# insert single operations
function single_op_wrapper(fop, target::Function)
    # if fop is a math operation, it needs to forward the return type
    if is_math(target)
        return (block::Block, args::Vector{Value}; result, location=Location()) ->
        push!(block, fop(args...; result, location))
    else
        return (block::Block, args::Vector{Value}; result, location=Location()) ->
        push!(block, fop(args...; location))
    end
end


## conversion to MLIR

# map Julia intrinsics to MLIR
function intrinsic_to_mlir(target_function)
    if target_function in keys(operations)                        # map operations
        return single_op_wrapper(operations[target_function], target_function)
    elseif target_function in keys(predicate)                    # operator mappings
        return single_op_wrapper(cmpi_pred(predicate[target_function]), target_function)
    elseif target_function in keys(custom_intrinsics)             # custom intrinsics
        return custom_intrinsics[target_function]
    end

    error("Intrinsic cannot be mapped to MLIR: $target_function. Please update 'mapping.jl' create a Pull Request")
end
