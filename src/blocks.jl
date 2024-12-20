# using MLIR
using LLVM: LLVM
using Core: PhiNode, GotoNode, GotoIfNot, SSAValue, Argument, ReturnNode, PiNode
using MLIR.IR
using MLIR
using MLIR.Dialects: arith, func, cf


## Basic Block Preprocessing

"Generates a block argument for each phi node present in the block."
function prepare_block(ir, bb)
  b = Block()

  for sidx in bb.stmts
    stmt = ir.stmts[sidx]
    inst = stmt[:inst]
    inst isa Core.PhiNode || continue

    type = stmt[:type]
    IR.push_argument!(b, IR.Type(type))
  end

  return b
end


"Values to populate the Phi Node when jumping from `from` to `to`."
function collect_value_arguments(ir, from, to)
  to = ir.cfg.blocks[to]
  values = []
  for s in to.stmts
    stmt = ir.stmts[s]
    inst = stmt[:inst]
    inst isa Core.PhiNode || continue

    edge = findfirst(==(from), inst.edges)
    if isnothing(edge) # use dummy scalar val instead
      val = zero(stmt[:type])
      push!(values, val)
    else
      push!(values, inst.values[edge])
    end
  end
  return values
end


# get value
function get_value(x)::Value
  if x isa Core.SSAValue
    @assert isassigned(values, x.id) "value $x was not assigned"
    values[x.id]
  elseif x isa Core.Argument
    IR.argument(entry_block, x.n - 1)
  elseif x isa ScalarTypes 
    IR.result(push!(current_block, arith.constant(; value=x)))
  else
    error("could not use value $x inside MLIR")
  end
end



function preprocess_code_blocks(ir)
  @assert first(ir.argtypes) isa Core.Const

  # preprocess all blocks
  blocks = [prepare_block(ir, bb) for bb in ir.cfg.blocks]

  # preprocess first block
  entry_block = blocks[begin]


  # add argtypes
  for argtype in types.parameters
    IR.push_argument!(entry_block, IR.Type(argtype))
  end

  return entry_block, blocks
end
