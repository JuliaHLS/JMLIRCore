using LLVM: LLVM
using Core: PhiNode, GotoNode, GotoIfNot, SSAValue, Argument, ReturnNode, PiNode
using MLIR.IR
using MLIR
using MLIR.Dialects: arith, func, cf
using Core: Expr


const BrutusScalar = Union{Bool,Int64,Int32,Float32,Float64}


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




function code_mlir(f, args)
  # get value
  function getValue(x)::Value
    if x isa Core.SSAValue
      println("Processed SSA assigned")
      @assert isassigned(values, x.id) "value $x was not assigned"
      values[x.id]
    elseif x isa Core.Argument
      IR.argument(entry_block, x.n - 1)
    elseif x isa BrutusScalar
      IR.result(push!(current_block, arith.constant(; value=x)))
    else
      error("could not use value $x inside MLIR")
    end
  end

  # handle :call
  function processStatement(inst::Expr)
    if inst.head === :call
      println("Processing call expression: ", inst)
    elseif inst.head === :code_coverage_effect
      println("Processing code coverage effect: ", inst)
    end
    # handle
  end

  function processStatement(inst::ReturnNode)
    println("handling return node")
  end

  function processStatement(inst)
    println("Unknown type: ", typeof(inst))
  end




  # load context
  ctx = IR.Context()
  ir, ret = only(Core.Compiler.code_ircode(f, args))
  @assert first(ir.argtypes) isa Core.Const


  # get the values
  values = Vector{Value}(undef, length(ir.stmts))


  # load dialects
  for dialect in (:func, :cf)
    IR.register_dialect!(IR.DialectHandle(dialect))
  end
  IR.load_all_available_dialects()


  # gather basic blocks
  blocks = [prepare_block(ir, bb) for bb in ir.cfg.blocks]


  # enter block 1
  current_block = entry_block = blocks[begin]


  # add argtypes
  for argtype in types.parameters
    IR.push_argument!(entry_block, IR.Type(argtype))
  end


  # iterate through the basic blocks
  for (block_id, (b, bb)) in enumerate(zip(blocks, ir.cfg.blocks))
    current_block = b
    n_phi_nodes = 0


    # process block statementiterate through block stmtss
    for sidx in bb.stmts
      stmt = ir.stmts[sidx]
      inst = stmt[:inst]

      println("here")
      println("found statement with type: ", typeof(inst))
      processStatement(inst)
    end
  end


  func_name = nameof(f)

  region = Region()
  for b in blocks
    push!(region, b)
  end

  input_types = IR.Type[
    IR.type(IR.argument(entry_block, i)) for i in 1:IR.nargs(entry_block)
  ]
  result_types = [IR.Type(ret)]

  ftype = IR.FunctionType(input_types, result_types)
  op = IR.create_operation(
    "func.func",
    Location();
    attributes=[
      IR.NamedAttribute("sym_name", IR.Attribute(string(func_name))),
      IR.NamedAttribute("function_type", IR.Attribute(ftype)),
    ],
    owned_regions=Region[region],
    result_inference=false,
  )

  IR.verifyall(op)

  return op


end




println("Testing code")

function add(a, b)
  return a + b
end

f = add
types = Tuple{Int,Int}

println(code_mlir(f, types))
