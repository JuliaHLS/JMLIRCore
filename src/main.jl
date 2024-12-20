# using MLIR
using LLVM: LLVM
using Core: PhiNode, GotoNode, GotoIfNot, SSAValue, Argument, ReturnNode, PiNode
using MLIR.IR
using MLIR
using MLIR.Dialects: arith, func, cf

include("code_mlir.jl")


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


op = code_mlir(f, types)

println(op)


