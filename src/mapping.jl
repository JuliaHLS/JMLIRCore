# Define the mapping from Julia IR -> MLIR Operations
using MLIR.Dialects: arith, func, cf, memref, linalg, tosa


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
  Base.lshr_int  => arith.shrui,
  Base.trunc_int => arith.trunci,
  Base.checked_srem_int => arith.remsi,

  Base.add_float => arith.addf, 
  Base.sub_float => arith.subf,
  Base.mul_float => arith.mulf,
  Base.div_float => arith.divf,
  Base.sitofp    => arith.sitofp,
])


# Julia Predicates -> Predicate enumeration
const int_predicate = Dict([
    Base.sle_int => Predicates.sle,
    Base.slt_int => Predicates.slt,
    Base.:(===)  => Predicates.eq,
])

const float_predicate = Dict([
    Base.ne_float => Predicates.ne,
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

