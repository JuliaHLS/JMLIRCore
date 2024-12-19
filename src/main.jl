# using MLIR
using LLVM: LLVM
using Core: PhiNode, GotoNode, GotoIfNot, SSAValue, Argument, ReturnNode, PiNode
using MLIR.IR
using MLIR
using MLIR.Dialects: arith, func, cf


include("intrinsics.jl")


#### USEFUL FUNCTIONS

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


#### MAIN SCRIPT

function get_function_details(func)
  # Get the function name
  f = nameof(func)

  # Get the arguments
  method = first(methods(func))
  args = [arg for arg in method.sig.parameters[2:end]]

  return f, args
end

# Example function
function add(a, b)
  return a + b
end

function pow(x::F, n) where {F}
  p = one(F)
  for _ in 1:n
    p *= x
  end
  return p
end



f = pow


# f = esc(first(call.args))
# args = esc(
#   Expr(
#     :curly,
#     Tuple,
#     map(arg -> :($(Core.Typeof)($arg)), call.args[(begin+1):end])...,
#   ),
# )

types = Tuple{Int,Int}


const ScalarTypes = Union{Bool,Int64,Int32,Float32,Float64}


# load the basic context
ctx = IR.Context()
ir, ret = only(Core.Compiler.code_ircode(f, types))
@assert first(ir.argtypes) isa Core.Const

# values
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


# iterate through the basic blocks
for (block_id, (b, bb)) in enumerate(zip(blocks, ir.cfg.blocks))
  current_block = b
  n_phi_nodes = 0


  # process block statementiterate through block stmtss
  for sidx in bb.stmts
    stmt = ir.stmts[sidx]
    inst = stmt[:inst]
    # line = @static if VERSION <= v"1.11"
    line = ir.linetable[stmt[:line]+1]
    
    # if constant val
    if Meta.isexpr(inst, :call)
      val_type = stmt[:type]
      if !(val_type <: ScalarTypes)
        error("type $val_type is not supported")
      end
      out_type = IR.Type(val_type)

      called_func = first(inst.args)
      if called_func isa GlobalRef # TODO: should probably use something else here
        called_func = getproperty(called_func.mod, called_func.name)
      end

      fop! = intrinsic_to_mlir(called_func)
      args = get_value.(@view inst.args[(begin+1):end])

      location = Location(string(line.file), line.line, 0)
      res = IR.result(fop!(current_block, args; location))

      values[sidx] = res


    elseif inst isa PhiNode     # is a PhiNode
      values[sidx] = IR.argument(current_block, n_phi_nodes += 1)


    elseif inst isa PiNode      # is a PiNode
      values[sidx] = get_value(inst.val)


    elseif inst isa GotoNode    # is a GotoNode
      ReturnNode
      args = get_value.(collect_value_arguments(ir, block_id, inst.label))
      dest = blocks[inst.label]
      location = Location(string(line.file), line.line, 0)
      push!(current_block, cf.br(args; dest, location))


    elseif inst isa GotoIfNot   # is a GotoIfNot
      println("GotoIfNot")
      false_args = get_value.(collect_value_arguments(ir, block_id, inst.dest))
      cond = get_value(inst.cond)
      @assert length(bb.succs) == 2 # NOTE: We assume that length(bb.succs) == 2, this might be wrong
      other_dest = only(setdiff(bb.succs, inst.dest))
      true_args = get_value.(collect_value_arguments(ir, block_id, other_dest))
      other_dest = blocks[other_dest]
      dest = blocks[inst.dest]

      location = Location(string(line.file), line.line, 0)
      cond_br = cf.cond_br(
        cond,
        true_args,
        false_args;
        trueDest=other_dest,
        falseDest=dest,
        location,
      )
      push!(current_block, cond_br)


    elseif inst isa ReturnNode   # is a ReturnNode
      println("ReturnNode")
      location = Location(string(line.file), line.line, 0)
      push!(current_block, func.return_([get_value(inst.val)]; location))


    elseif Meta.isexpr(inst, :code_coverage_effect)
      # Skip
    else
      error("unhandled ir $(inst)")
    end
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

println("GOT RESULT: ", op)


