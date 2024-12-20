# using MLIR
using LLVM: LLVM
using Core: PhiNode, GotoNode, GotoIfNot, SSAValue, Argument, ReturnNode, PiNode
using MLIR.IR
using MLIR
using MLIR.Dialects: arith, func, cf

include("intrinsics.jl")
include("blocks.jl")
include("expressions.jl")


"Macro @code_mlir f(args...)"
macro code_mlir(call)
    @assert Meta.isexpr(call, :call) "only calls are supported"

    f = esc(first(call.args))
    args = esc(
        Expr(
            :curly,
            Tuple,
            map(arg -> :($(Core.Typeof)($arg)), call.args[(begin + 1):end])...,
        ),
    )

    quote
        code_mlir($f, $args)
    end
end


"Translate typed IR into MLIR"
function code_mlir(f, input_types)
  ### Setup the context ###
  
  # load the basic context
  ctx = IR.Context()

  # load dialects
  for dialect in (:func, :cf)
    IR.register_dialect!(IR.DialectHandle(dialect))
  end
  IR.load_all_available_dialects()

  
  ### Preprocess ###
  ir, ret = only(Core.Compiler.code_ircode(f, input_types))
  @assert first(ir.argtypes) isa Core.Const
  result_types = [IR.Type(ret)]

  # values
  values = Vector{Value}(undef, length(ir.stmts))

  # gather basic blocks
  entry_block, block_array = preprocess_code_blocks(ir, input_types)
  current_block = entry_block

  # set up context variables
  context = Context(
    ir,
    values,
    0,
    nothing,
    0,
    nothing,
  )

  blocks = Blocks(
    nothing,
    current_block,
    entry_block,
    block_array,
    nothing
  )


  ### Process into blocks ###
  process_blocks(blocks, context)

  region = Region()
  for b in blocks.blocks
    push!(region, b)
  end


  ### Format output ###
  input_types = IR.Type[
    IR.type(IR.argument(entry_block, i)) for i in 1:IR.nargs(entry_block)
  ]

  f_name = nameof(f)

  ftype = IR.FunctionType(input_types, result_types)
  op = IR.create_operation(
    "func.func",
    Location();
    attributes=[
      IR.NamedAttribute("sym_name", IR.Attribute(string(f_name))),
      IR.NamedAttribute("function_type", IR.Attribute(ftype)),
    ],
    owned_regions=Region[region],
    result_inference=false,
  )

  ### Verify validity of the MLIR generated ###
  IR.verifyall(op)


  ### return result ###
  return op 
end
