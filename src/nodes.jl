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
      val = get_value.(values, context, blocks)

      # insert the conditional break with the expected parameters
      cond_br = cf.br(
        val, 
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
    # collect arguments
    args::Vector{IR.Value} = get_value.(collect_value_arguments(context.ir, blocks.block_id, inst.label), context, blocks)

    # collect destination block (MLIR)
    dest = blocks.blocks[inst.label]

    location = Location()#Location(string(context.line.file), context.line.line, 0)

    # add unconditional break (goto)
    push!(blocks.current_block, cf.br(args; dest, location))
end

function process_node(inst::GotoIfNot, context::Context, blocks::Blocks)
    # collect arguments for the false route
    false_args::Vector{Value} = get_value.(collect_value_arguments(context.ir, blocks.block_id, inst.dest), Ref(context), Ref(blocks))

    # collect condition IR val
    cond = get_value(inst.cond, context, blocks)
    @assert length(blocks.bb.succs) == 2 # NOTE: We assume that length(bb.succs) == 2, this might be wrong

    # collect second destination
    other_dest = only(setdiff(blocks.bb.succs, inst.dest))

    true_args::Vector{Value} = get_value.(collect_value_arguments(context.ir, blocks.block_id, other_dest), context, blocks)

    other_dest = blocks.blocks[other_dest]
    dest = blocks.blocks[inst.dest]

    location = Location() #string(context.line.file), context.line.line, 0)

    # create conditional jump
    cond_br = cf.cond_br(
        cond,
        true_args,
        false_args;
        trueDest=other_dest,
        falseDest=dest,
        location,
    )

    # push into the block
    push!(blocks.current_block, cond_br)
end


function process_node(inst::ReturnNode, context::Context, blocks::Blocks)
    # find the symbols tag where the return comes from (with column number)
    # location = Location(string(context.line.file), context.line.line, 0)

    # add to the block for debugging
    push!(blocks.current_block, func.return_([get_value(inst.val, context, blocks)]))
end

function process_node(inst::Nothing, context::Context, blocks::Blocks)
    println("Info: Received empty node from Julia IR. Skipping...")
end

function process_node(inst::GlobalRef, context::Context, blocks::Blocks)
    error("Julia IR cannot process GlobalRef")
end

function process_node(inst, context::Context, blocks::Blocks)
    error("unhandled ir $(inst) of type $(typeof(inst))")
end
