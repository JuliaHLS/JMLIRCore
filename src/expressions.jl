include("common_types.jl")
using LinearAlgebra

function get_q_bits(arg)
    return last(arg.parameters)
end


# function get_q_info(args)::Union{Nothing, IR.NamedAttribute}
#     println("HERE########################################################")
#     arg_array = []
#     for arg in args
#         println("ARG: $arg")
#         push!(arg_array, typeof(arg) isa Real && typeof(arg) <: Fixed ? get_q_bits(arg) : -1)
#     end

#     if length(arg_array) == 0
#         return q_attr = IR.NamedAttribute("fixed_q", IR.DenseArrayAttribute(Int32.(arg_array)))
#     else
#         return nothing
#     end
# end

function process_expr(inst::Expr, context::Context, blocks::Blocks)
    if Meta.isexpr(inst, :call) || Meta.isexpr(inst, :invoke)
        val_type = context.stmt[:type]

        println("running: $inst with type $(typeof(inst))")
        if !(val_type <: ScalarTypes || val_type isa Any)
          error("type $val_type is not supported")
        end

        println("1")

        # Check function
        called_func = first(inst.args)
        if called_func isa GlobalRef
          called_func = getproperty(called_func.mod, called_func.name)
        end
        println("2")

        # store type as IR.Type
        println("val_type: $val_type")
        type = IR.Type(val_type)
        
        # extract metadata
        fop! = intrinsic_to_mlir(called_func)

        # filter out unwanted arguments
        extracted_args = filter(arg -> !(arg isa DataType || arg isa GlobalRef || arg isa QuoteNode), inst.args[(begin+1):end])

        println("PRocessing args: $extracted_args")

        println("processign extracting args: $extracted_args")
        println("val1: $extracted_args")
        println("processing block: $(blocks.current_block)")
        args = get_value.(extracted_args, context, blocks)

        # println("GOT ARGS: $args")
        println("Collected")

        # q_info = get_q_info(extracted_args)
        # println("Q: $q_info")

        # TODO: investigate the feasibility of reintroducing location in Julia v1.12
        # location = Location(string(context.line.file), context.line.line, 0)
        res = IR.result(fop!(blocks.current_block, args; result=type::Union{Nothing,IR.Type})) 
        println("Processing RES: $res")

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

        # if length(args) == 0
        #     args::Vector{Vector{Value}} = [[]]
        # end

        # perform transpose
        type = IR.Type(val_type)

        res = IR.result(fop!(blocks.current_block, args; result=type::Union{Nothing,IR.Type}))

        println("Processing RES: $res")

        context.values[context.sidx] = res
    else
        error("Unknown expr: $inst of type $(typeof(inst))")
    end

end


