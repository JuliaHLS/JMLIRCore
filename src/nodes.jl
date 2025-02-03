# using MLIR
using LLVM: LLVM
using Core: PhiNode, GotoNode, GotoIfNot, SSAValue, Argument, ReturnNode, PiNode
using MLIR.IR
using MLIR
using MLIR.Dialects: arith, func, cf

include("common_types.jl")

function postfix_nodes(context, blocks)
  # NOTE: CIRCT only allows 2 paths to merge, this needs to be fixed with the merge block insertion pass when writing the HLS tool
  
  # check if each block completes the route to the current block

  for (source_block, values) in context.phi_nodes_metadata 
    block = blocks.blocks[source_block]
    bb = context.ir.cfg.blocks[source_block]
    
    goto_exists = false

    for ctx_idx in bb.stmts
      tmp_inst = context.ir.stmts[ctx_idx][:inst]

      if tmp_inst isa GotoNode || tmp_inst isa GotoIfNot
        goto_exists = true
      end
    end


    # insert a goto to the current block with the required arguments
    if !goto_exists
      # change current block to the destination
      destination = blocks.current_block 
      blocks.current_block = block

      # expand values
      values = get_value.(values, context, blocks)

      # insert the conditional break with the expected parameters
      cond_br = cf.br(
        values, 
          dest=destination
        )
      push!(block, cond_br)

      # reset current block
      blocks.current_block = destination
    end


  end


end

# process PhiNode
function process_node(inst::PhiNode, context::Context, blocks::Blocks)
  
  # collect phi node metadata
  for (source_block_idx, value) in zip(inst.edges, inst.values)
    mapped_ref = get!(context.phi_nodes_metadata, source_block_idx, [])
    push!(mapped_ref, value)
  end


  context.values[context.sidx] = IR.argument(blocks.current_block, context.n_phi_nodes += 1)
end



# process PiNode
function process_node(inst::PiNode, context::Context, blocks::Blocks)
  context.values[context.sidx] = get_value(inst.val, context, blocks)
end



function process_node(inst::GotoNode, context::Context, blocks::Blocks)
  args = get_value.(collect_value_arguments(context.ir, blocks.block_id, inst.label), context, blocks)
  dest = blocks.blocks[inst.label]
  location = Location(string(context.line.file), context.line.line, 0)
  push!(blocks.current_block, cf.br(args; dest, location))
end


function process_node(inst::GotoIfNot, context::Context, blocks::Blocks)
  false_args = get_value.(collect_value_arguments(context.ir, blocks.block_id, inst.dest), Ref(context), Ref(blocks))
  cond = get_value(inst.cond, context, blocks)
  @assert length(blocks.bb.succs) == 2 # NOTE: We assume that length(bb.succs) == 2, this might be wrong
  other_dest = only(setdiff(blocks.bb.succs, inst.dest))
  true_args = get_value.(collect_value_arguments(context.ir, blocks.block_id, other_dest), context, blocks)
  other_dest = blocks.blocks[other_dest]
  dest = blocks.blocks[inst.dest]

  location = Location() #string(context.line.file), context.line.line, 0)
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
  # find the symbols tag where the return comes from (with column number)
  # location = Location(string(context.line.file), context.line.line, 0)

  # add to the block for debugging
  push!(blocks.current_block, func.return_([get_value(inst.val, context, blocks)]))
end


function process_node(inst, context::Context, blocks::Blocks)
  error("unhandled ir $(inst)")
end
