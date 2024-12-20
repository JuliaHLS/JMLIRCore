# using MLIR
using LLVM: LLVM
using Core: PhiNode, GotoNode, GotoIfNot, SSAValue, Argument, ReturnNode, PiNode
using MLIR.IR
using MLIR
using MLIR.Dialects: arith, func, cf


function process_expr(inst::Expr, context::Context, blocks::Blocks)
  if Meta.isexpr(inst, :call)
    val_type = context.stmt[:type]
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

    location = Location(string(context.line.file), context.line.line, 0)
    res = IR.result(fop!(blocks.current_block, args; location))

    context.values[context.sidx] = res

  elseif Meta.isexpr(inst, :code_coverage_effect)
      # Skip
  else
    error("Unknown expr: $inst")
  end

end


