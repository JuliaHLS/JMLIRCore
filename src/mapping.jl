# Define the mapping from Julia IR -> MLIR Operations

# using MLIR
using LLVM: LLVM
using Core: PhiNode, GotoNode, GotoIfNot, SSAValue, Argument, ReturnNode, PiNode
using MLIR.IR
using MLIR
using MLIR.Dialects: arith, func, cf


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


# Julia Operations -> MLIR Operations
const operations = Dict([
  Base.add_int => arith.addi,
  Base.sub_int => arith.subi,
  Base.mul_int => arith.muli,

  Base.add_float => arith.addf, 
  Base.sub_float => arith.subf,
  Base.mul_float => arith.mulf,
  Base.div_float => arith.divf,
])


# Julia Predicates -> Predicate enumeration
const predicate = Dict([
    Base.sle_int => Predicates.sle,
    Base.slt_int => Predicates.slt,
    Base.:(===)  => Predicates.eq,
])


# Edge-case intrinsics -> MLIR (complicated cases)
const custom_intrinsics = Dict([
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

