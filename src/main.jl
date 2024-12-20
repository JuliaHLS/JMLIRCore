# using MLIR
using LLVM: LLVM
using Core: PhiNode, GotoNode, GotoIfNot, SSAValue, Argument, ReturnNode, PiNode
using MLIR.IR
using MLIR
using MLIR.Dialects: arith, func, cf


include("intrinsics.jl")
include("blocks.jl")
include("expressions.jl")


#### USEFUL FUNCTIONS

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
entry_block, block_array = preprocess_code_blocks(ir)
current_block = entry_block

context = Context(
  ir,
  values,
  0,
  nothing,
  nothing,
  nothing,
)

blocks = Blocks(
  nothing,
  current_block,
  block_array,
  nothing
)


# iterate through the basic blocks
# for (idx, (curr_block, bb)) in enumerate(zip(block_array, context.ir.cfg.blocks))

process_blocks(blocks, context)

func_name = "testFunc"

region = Region()
for b in blocks.blocks
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

println(op)


