# using MLIR
using LLVM: LLVM
using Core: PhiNode, GotoNode, GotoIfNot, SSAValue, Argument, ReturnNode, PiNode
using MLIR.IR
using MLIR
using MLIR.Dialects: arith, func, cf

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


# process comparator predicates
function cmpi_pred(predicate)
  function (ops...; location=Location())
    return arith.cmpi(ops...; result=IR.Type(Bool), predicate, location)
  end
end

# ops = [
#     (:add_int, (arith.addi,)),
#     (:sub_int, (arith.subi,)),
# ]


#TODO: see if its possible to extract the math_tfunc
const calculation_intrinsics = Set([
    arith.addi,
    arith.subi,
    arith.muli,
])                                    

# generic check is fop is registered as a math function
function is_math(fop)::Bool
    idx = reinterpret(Int32, fop) + 1
    return CC.T_IFUNC[idx] == Core.Compiler.math_tfunc
end

# compare single operations
function single_op_wrapper(fop)
    # if fop is a math operation, it needs to forward the return type
    if is_math(fop)
        return (block::Block, args::Vector{Value}; result, location=Location()) ->
        push!(block, fop(args...; result, location))
    else
        return (block::Block, args::Vector{Value}; result, location=Location()) ->
        push!(block, fop(args...; location))
    end
end


# CALCULATION_INTRINSICS

# INTEGER INTRINSICS
const integer_intrinsics_to_mlir = Dict([
  Base.add_int => single_op_wrapper(arith.addi),
  Base.sub_int => single_op_wrapper(arith.subi),
  Base.sle_int => single_op_wrapper(cmpi_pred(Predicates.sle)),
  Base.slt_int => single_op_wrapper(cmpi_pred(Predicates.slt)),
  Base.mul_int => single_op_wrapper(arith.muli),
  Base.not_int => function (block, args; location=Location())
    arg = only(args)
    mT = IR.type(arg)
    T = IR.julia_type(mT)
    ones = IR.result(
      push!(block, arith.constant(; value=typemax(UInt64) % T, result=mT, location)),
    )
    return push!(block, arith.xori(arg, ones; location))
  end,

])

# OPERATOR INTRINSICS
const operator_intrinsics_to_mlir = Dict([
  Base.:(===) => single_op_wrapper(cmpi_pred(Predicates.eq)),
])



## conversion to MLIR

function intrinsic_to_mlir(target_function)
  # map the intrinsic

  # integer mappings
  if target_function in keys(integer_intrinsics_to_mlir)
    return integer_intrinsics_to_mlir[target_function]

  # operator mappings
  elseif target_function in keys(operator_intrinsics_to_mlir)
    return operator_intrinsics_to_mlir[target_function]

  end

  error("Intrinsic cannot be mapped to MLIR: $target_function")
end
