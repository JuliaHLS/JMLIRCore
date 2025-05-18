include("nodes.jl")

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
    println("processing stmt: $s")
    stmt = ir.stmts[s]
    inst = stmt[:inst]

    if inst isa Core.PhiNode
        edge = findfirst(==(from), inst.edges)
        if isnothing(edge) # use dummy scalar val instead
          val = zero(stmt[:type])
          push!(values, val)
        else
          push!(values, inst.values[edge])
        end
    end
  end
  return values
end

function collect_value_arguments_ir(ir, from, to, dest_block::Block)
  to = ir.cfg.blocks[to]
  values= []

  seen_vals = Set([])

  println("to: $to")
  println("ir: $ir")

  for s in to.stmts
      println("s: $s")
      println("seen: $seen_vals")
    stmt = ir.stmts[s]
    inst = stmt[:inst]

    type = stmt[:type]
    # println("inst: $(inst.id) and $(typeof(stmt))")
    println("inst: $inst")
    if Meta.isexpr(inst, :invoke)
        push!(seen_vals, s)
    end
      println("seen: $seen_vals")

    if inst isa Core.PhiNode
        edge = findfirst(==(from), inst.edges)
                  # println("push arg: $s")
        # IR.push_argument!(dest_block, IR.Type(type))

        if isnothing(edge) # use dummy scalar val instead
          val = zero(stmt[:type])
          push!(values, val)
        else
          push!(values, inst.values[edge])
        end
    elseif Meta.isexpr(inst, :invoke) # forward externally accessed values
        for arg in inst.args[(begin+1):end]
            # if arg isa Core.SSAValue && !(arg.id in seen_vals)
            #     if !(arg in values)
            #       push!(values, arg)
            #       println("push arg: $s")

            #       IR.push_argument!(dest_block, IR.Type(type))
            #   end
            # elseif arg isa Core.Argument
            #     println("HERE2")
            #     if !(arg in values)
            #       push!(values, arg)
            #       println("push arg: $s")
            #       IR.push_argument!(dest_block, IR.Type(type))
            #   end
            # end
        end
    end
  end
  return values
end


# get value
function get_value(x, context::Context, blocks::Blocks)
    println("processing : $x")
    if x isa Core.SSAValue
        @assert isassigned(context.values, x.id) "value $x was not assigned"
        context.values[x.id]
    elseif x isa Core.Argument
        IR.argument(blocks.entry_block, x.n - 1)
    elseif x isa ScalarTypes 
        IR.result(push!(blocks.current_block, arith.constant(; value=x)))
    elseif x isa Tuple       # process all tuple types
        results::Vector{IR.Value} = []
        for init_val ∈ collect(x)
            ssa_res = IR.result(push!(blocks.current_block, arith.constant(; value=init_val)))
            push!(results, ssa_res)
        end
        results::Vector{IR.Value}
    else
        error("could not use value $x of type $(typeof(x)) inside MLIR. Please review ScalarTypes.")
    end
end

function get_value_ir(x, context::Context, blocks::Blocks)
    println("processing : $x")
    if x isa Core.SSAValue
        if !(x.id in context.values)
            println("is assigned :$(isassigned(context.values, x.id))") 
            return nothing
        end
        @assert isassigned(context.values, x.id) "value $x was not assigned"
        context.values[x.id]
    elseif x isa Core.Argument
        IR.argument(blocks.entry_block, x.n - 1)
    elseif x isa ScalarTypes 
        IR.result(push!(blocks.current_block, arith.constant(; value=x)))
    elseif x isa Tuple       # process all tuple types
        results::Vector{IR.Value} = []
        for init_val ∈ collect(x)
            ssa_res = IR.result(push!(blocks.current_block, arith.constant(; value=init_val)))
            push!(results, ssa_res)
        end
        results::Vector{IR.Value}
    else
        error("could not use value $x of type $(typeof(x)) inside MLIR. Please review ScalarTypes.")
    end
end





function preprocess_code_blocks(ir, types)
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


function process_blocks(blocks::Blocks, context::Context)
  
  # process each block into MLIR
  for (idx, (curr_block, bb)) in enumerate(zip(blocks.blocks, context.ir.cfg.blocks))
    blocks.block_id = idx
    blocks.current_block = curr_block
    blocks.bb = bb
    context.n_phi_nodes = 0
    context.phi_nodes_metadata = Dict{Int, Vector{Any}}()


    # process block statementiterate through block stmtss
    for context.sidx in blocks.bb.stmts
      context.stmt = context.ir.stmts[context.sidx]
      inst = context.stmt[:inst]

      line = context.line
      
      # TODO: find a proper way to track the source of statements in the linetable, as this breaks when returning built-in functions (i.e return a + b) and needs to be set without the +1
      # if (context.stmt[:line]+1 > length(context.ir.linetable))
      #   context.line = context.ir.linetable[context.stmt[:line]]
      # else
      #   # find the location of the statement in the linetable
      #   context.line = context.ir.linetable[context.stmt[:line]+1]
      # end

      # process struction
      if inst isa Expr
        # process expression
        process_expr(inst, context, blocks)
      else
        # process node
        process_node(inst, context, blocks)
      end
    end

    ### FIX NODES
    postfix_nodes(context, blocks)
  end
end
