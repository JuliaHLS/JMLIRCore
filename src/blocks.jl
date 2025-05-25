include("nodes.jl")
include("common_types.jl")

## Basic Block Preprocessing
using Base.Compiler: FixedNode
using FixedPointNumbers

using MLIR.Dialects: quant

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

# get value
function get_value(x, context::Context, blocks::Blocks)
    if x isa Core.SSAValue
        @assert isassigned(context.values, x.id) "value $x was not assigned"
        context.values[x.id]
    elseif x isa Core.Argument
        IR.argument(blocks.entry_block, x.n - 1)
    elseif x isa ScalarTypes 
        recast_x = recast_fixed(x)
        ssa_res = IR.result(push!(blocks.current_block, arith.constant(; value=recast_x)))
        println("SCALAR")

#         if typeof(x) <: Fixed
#             ssa_res = IR.result(push!(blocks.current_block, MLIR.Dialects.quant.scast(ssa_res; res=IR.Type(typeof(x)))))
#         end

        return ssa_res
    elseif x isa Tuple       # process all tuple types
        results::Vector{IR.Value} = []
        for init_val ∈ collect(x)
            recast_init_val = recast_fixed(init_val)
            ssa_res = IR.result(push!(blocks.current_block, arith.constant(; value=recast_init_val)))
            # ssa_res = IR.result(push!(blocks.current_block, arith.constant(; value=init_val)))
            # if typeof(init_val) <: Fixed
            #     ssa_res = IR.result(push!(blocks.current_block, MLIR.Dialects.quant.scast(ssa_res; res=IR.Type(typeof(init_val)))))
            # end
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
