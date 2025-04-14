include("common_types.jl")

function process_expr(inst::Expr, context::Context, blocks::Blocks)
    if Meta.isexpr(inst, :call) || Meta.isexpr(inst, :invoke)
        val_type = context.stmt[:type]
        if !(val_type <: ScalarTypes)
          error("type $val_type is not supported")
        end

        # Check function
        called_func = first(inst.args)
        if called_func isa GlobalRef
          called_func = getproperty(called_func.mod, called_func.name)
        end

        # store type as IR.Type
        type = IR.Type(val_type)


        # extract metadata
        fop! = intrinsic_to_mlir(called_func)

        # filter out unwanted arguments
        extracted_args = filter(arg -> !(arg isa DataType || arg isa GlobalRef), inst.args[(begin+1):end])

        args = get_value.(extracted_args, context, blocks)


        # if inst.args[(begin+1):end][1] isa GlobalRef
        #     println("modifying to a vector")
        #     args = map(x -> [x], args)
        # end

        # TODO: investigate the feasibility of reintroducing location in Julia v1.12
        # location = Location(string(context.line.file), context.line.line, 0)
        res = IR.result(fop!(blocks.current_block, args; result=type::Union{Nothing,IR.Type}))

        context.values[context.sidx] = res
    elseif Meta.isexpr(inst, :code_coverage_effect)
        # Skip
    else
        error("Unknown expr: $inst")
    end

end


