# using MLIR
using LLVM: LLVM
using Core: PhiNode, GotoNode, GotoIfNot, SSAValue, Argument, ReturnNode, PiNode
using MLIR.IR
using MLIR
using MLIR.Dialects: arith, func, cf

include("common_types.jl")


# process PhiNode
function process_node(inst::PhiNode, context::Context, blocks::Blocks)
  context.values[context.sidx] = IR.argument(blocks.current_block, context.n_phi_nodes += 1)
end



# process PiNode
function process_node(inst::PiNode, context::Context, blocks::Blocks)
  context.values[context.sidx] = get_value(inst.val)
end



function process_node(inst::GotoNode, context::Context, blocks::Blocks)
  args = get_value.(collect_value_arguments(context.ir, blocks.block_id, inst.label))
  dest = blocks.blocks[inst.label]
  location = Location(string(context.line.file), context.line.line, 0)
  push!(blocks.current_block, cf.br(args; dest, location))
end


function process_node(inst::GotoIfNot, context::Context, blocks::Blocks)
  false_args = get_value.(collect_value_arguments(context.ir, blocks.block_id, inst.dest))
  cond = get_value(inst.cond)
  @assert length(blocks.bb.succs) == 2 # NOTE: We assume that length(bb.succs) == 2, this might be wrong
  other_dest = only(setdiff(blocks.bb.succs, inst.dest))
  true_args = get_value.(collect_value_arguments(context.ir, blocks.block_id, other_dest))
  other_dest = blocks.blocks[other_dest]
  dest = blocks.blocks[inst.dest]

  location = Location(string(context.line.file), context.line.line, 0)
  cond_br = cf.cond_br(
    cond,
    true_args,
    false_args;
    trueDest=other_dest,
    falseDest=dest,
    location,
  )
  push!(blocks.current_block, cond_br)
end


function process_node(inst::ReturnNode, context::Context, blocks::Blocks)
  location = Location(string(context.line.file), context.line.line, 0)
  push!(blocks.current_block, func.return_([get_value(inst.val)]; location))
end


function process_node(inst, context::Context, blocks::Blocks)
  error("unhandled ir $(inst)")
end
