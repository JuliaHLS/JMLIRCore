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

    quote
        code_mlir($f, $args)
    end
end


"Translate typed IR into MLIR"
function code_mlir(f, input_types)
  ### Setup the context ###
  glob_op = "" 
  
  # load the basic context
  fptr = IR.context!(IR.Context()) do
      # load dialects
      for dialect in (:func, :cf)
        IR.register_dialect!(IR.DialectHandle(dialect))
      end
      IR.load_all_available_dialects()


      ### Initialise abstract interprete ###
      interp = MLIRInterpreter()
      
      ### Preprocess ###
      ir, ret = only(CC.code_ircode(f, input_types; interp=interp))
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

      MLIR.IR.Block

      ### Process into blocks ###
      process_blocks(blocks, context)

      region = Region()
      for b in blocks.blocks
        push!(region, b)
      end
      
      # push!(region, blocks.blocks[1])


      ### Format output ###
      input_types = IR.Type[
        IR.type(IR.argument(entry_block, i)) for i in 1:IR.nargs(entry_block)
      ]

      # println("Got input types: ", input_types)

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

      println(typeof(op))
              

      ### Verify validity of the MLIR generated ###
      IR.verifyall(op)
      glob_op = op

        mod = IR.Module(Location())
        body = IR.body(mod)
        push!(body, op)

        pm = IR.PassManager()
        opm = IR.OpPassManager(pm)

        MLIR.API.mlirRegisterAllPasses()
        MLIR.API.mlirRegisterAllLLVMTranslations(IR.context())
        MLIR.API.mlirRegisterConversionConvertToLLVMPass()

        IR.add_pipeline!(opm, "convert-arith-to-llvm,convert-func-to-llvm")

        IR.run!(pm, mod)
        println(mod)

        IR.enable_verifier!(pm, true)


        jit = MLIR.API.mlirExecutionEngineCreate(mod, 0, 0, C_NULL, false)
        MLIR.API.mlirExecutionEngineLookup(jit, "test4")
        # println("Created MLIR execution engine")
  end


  x, y = 3, 4

  # println("lowering ops for LLVM execution: ", typeof(fptr))
  # evaluate_llvm(lower_llvm(fptr))

  println("Calling execution engine: ", ccall(fptr, Int, (Int, Int), x, y))

  ### return result ###
  return glob_op 
end
