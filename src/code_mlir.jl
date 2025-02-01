# using MLIR
using LLVM: LLVM
using Core: PhiNode, GotoNode, GotoIfNot, SSAValue, Argument, ReturnNode, PiNode
using MLIR.IR
using MLIR
using MLIR.Dialects: arith, func, cf

include("intrinsics.jl")
include("blocks.jl")
include("expressions.jl")
include("MLIRInterpreter.jl")
include("llvm.jl")


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

    println("got args: ", args)

    quote
        code_mlir($f, $args)
    end
end


"Translate typed IR into MLIR"
function code_mlir(f, types)
  ### Setup the context ###
  
  println("args: ", types)
  if !IR._has_context()
      ctx = IR.Context
  end
  
  # load dialects
  for dialect in (:func, :cf)
    IR.register_dialect!(IR.DialectHandle(dialect))
  end
  IR.load_all_available_dialects()


  ### Initialise abstract interpreter ###
  interp = MLIRInterpreter()
  println("h2")
  
  ### Preprocess ###
  ir, ret = only(CC.code_ircode(f, types; interp=interp))
  @assert first(ir.argtypes) isa Core.Const
  result_types = [IR.Type(ret)]

  # values
  values = Vector{Value}(undef, length(ir.stmts))

  # gather basic blocks
  println("types: ", types, " with typeof: ", typeof(types))
  entry_block, block_array = preprocess_code_blocks(ir, types)
  current_block = entry_block

  # set up context variables
  context = Context(
    ir,
    values,
    0,
    Dict{Int, Vector{Any}}(),
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

  
  println("here")
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


  # extract function metadata
  f_name = nameof(f)
  ftype = IR.FunctionType(input_types, result_types)

  # create mlir operation (function call)
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


  println("verified")

  ### return result ###
  return op 
end
