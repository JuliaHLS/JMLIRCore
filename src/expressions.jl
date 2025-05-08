include("common_types.jl")
using LinearAlgebra

function process_expr(inst::Expr, context::Context, blocks::Blocks)
    if Meta.isexpr(inst, :call) || Meta.isexpr(inst, :invoke)
        val_type = context.stmt[:type]

        if !(val_type <: ScalarTypes || val_type isa Any)
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

        # TODO: investigate the feasibility of reintroducing location in Julia v1.12
        # location = Location(string(context.line.file), context.line.line, 0)
        res = IR.result(fop!(blocks.current_block, args; result=type::Union{Nothing,IR.Type}))

        context.values[context.sidx] = res
    elseif Meta.isexpr(inst, :code_coverage_effect)
        # Skip
    elseif Meta.isexpr(inst, :new)
        val_type = context.stmt[:type]

        # TODO: tidy once I know what types i want here
        if !(val_type <: ScalarTypes || val_type <: LinearAlgebra.Adjoint)
          error("type $val_type is not supported")
        end

        # extract metadata
        fop! = intrinsic_to_mlir(inst)

        extracted_args = filter(arg -> !(arg isa DataType || arg isa GlobalRef), inst.args[(begin+1):end])

        args = get_value.(extracted_args, context, blocks)

        # perform transpose
        type = IR.Type(val_type)

        res = IR.result(fop!(blocks.current_block, args; result=type::Union{Nothing,IR.Type}))

        context.values[context.sidx] = res
    else
        error("Unknown expr: $inst of type $(typeof(inst))")
    end

end


