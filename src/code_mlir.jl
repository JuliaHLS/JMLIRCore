# using MLIR
include("intrinsics.jl")
include("blocks.jl")
include("expressions.jl")
include("MLIRInterpreter.jl")
include("julia/passes.jl")

"Macro @code_mlir f(args...)"
macro code_mlir(call)
    @assert Meta.isexpr(call, :call) "only calls are supported"

    f = esc(first(call.args))
    args = esc(
        Expr(
            :curly,
            Tuple,
            map(arg -> :($(Core.Typeof)($arg)), call.args[(begin+1):end])...,
        ),
    )


    # force get a new by default (if we call code_mlir via the macro)
    # ctx = IR.Context()

    quote
        code_mlir($f, $args)
    end
end


"Translate typed IR into MLIR"
function code_mlir(f, types; ctx = IR.context())
    # if !IR._has_context()
    #     ctx = IR.Context()
    # end

    # load dialects
    for dialect in (:func, :cf, :memref, :linalg, :tensor, :math)
        IR.register_dialect!(IR.DialectHandle(dialect); context=ctx)
    end

    IR.load_all_available_dialects(; context=ctx)

    IR.allow_unregistered_dialects!(true; context=ctx)

    ### Initialise abstract interpreter ###
    interp = MLIRInterpreter()
    Base.code_ircode

    ### Preprocess ###
    ir, ret = only(CC.code_ircode(f, types; interp=interp))
    @assert first(ir.argtypes) isa Core.Const

    result_types = [IR.Type(ret)]

    # values
    values = Vector{Value}(undef, length(ir.stmts))

    # gather basic blocks
    entry_block, block_array = preprocess_code_blocks(ir, types)
    current_block = entry_block

    # set up context variables
    context = Context(
        ir,
        values,
        0,
        Dict{Int,Vector{Any}}(),
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
            IR.NamedAttribute("sym_visibility", IR.Attribute(string("public")))
        ],
        owned_regions=Region[region],
        result_inference=false,
    )

    ### Verify validity of the MLIR generated ###
    IR.verifyall(op)

    GC.@preserve op ctx begin
        mod = IR.Module(Location())
        body = IR.body(mod)
        push!(body, op)

        println("Code gen produced the following MLIR: $mod")

        ### Lower from julia dialect ###
        run!(JuliaPasses.FixTensorSSA(), mod, ctx)
        run!(JuliaPasses.FixImplicitControlFlow(), mod, ctx)
        run!(JuliaPasses.LowerJuliaArith(), mod, ctx)
        run!(JuliaPasses.LowerJuliaMat(), mod, ctx)

        ### standard optimisation passes
        MLIR.API.mlirRegisterAllPasses()
        pm = IR.PassManager()
        opm = IR.OpPassManager(pm)

        IR.add_pipeline!(opm,
                         "canonicalize{region-simplify=disabled},\
                         cse,\
                         loop-invariant-code-motion,\
                         sroa,\
                         sccp,\
                         remove-dead-values,\
                         symbol-dce,\
                         fold-memref-alias-ops,\
                         control-flow-sink"
                        )
        
        println("Lowered and optimised MLIR: $mod")
    end

    ### return result ###
    return mod
end
