# helpers for lowering passes



""" High-level Op lowering interface (External Use) """
module JuliaLowerOp

using MLIR.IR
using MLIR.API

using MLIR.IR
import MLIR.IR
using MLIR.API
using MLIR.Dialects: arith, tosa, tensor

export lower_op_to_mlir

""" Julia Helpers, do not use externally """
module _JuliaPassHelpers
    using MLIR.IR
    using MLIR.API

    using MLIR.IR
    import MLIR.IR
    using MLIR.API
    using MLIR.Dialects: arith, tosa, tensor

    export unroll_operation!
    export unroll_operation_mat!
    export array_unimplemented
    export lower_cmp!
    export collect_operands
    export collect_results

    # used for unrolling arbitrarily large intrinsics
    # Note: mutable to allow pass by reference for mutable types
    mutable struct reference_information
        operand_types::Vector{DataType}
        prev_op::IR.Operation
        prev_val::IR.Value
        block::IR.Block
        ret

        function reference_information(operand_types::Vector{DataType}, prev_op::IR.Operation, prev_val::IR.Value, block::IR.Block, ret)
            new(operand_types, prev_op, prev_val, block, ret)
        end
    end

    function update_reference_information(ref_info::reference_information, _prev_op::IR.Operation, _prev_val::IR.Value)
        ref_info.operand_types = ref_info.operand_types[2:end]
        ref_info.prev_op = _prev_op
        ref_info.prev_val = _prev_val
    end


    ### Generic Helpers for handling MLIR C-bindings ###

    @noinline function array_unimplemented(op)
        operands = collect_operands(op)
        types = IR.julia_type.((IR.type.(operands)))

        ret = IR.type.(collect_results(op))[1]

        prev_op = op
        prev_ref = operands[1]
        prev_val = operands[1]

        replaced = false

        for new_ref in operands[2:end]
            if types[1] <: AbstractArray && types[2] <: AbstractArray
                error("Unimplemented in MLIR pass: $op")
            end
        end
    end

    # function get_mat_dims(op)
        
    # end

    function collect_operands(op::IR.Operation)::Vector{IR.Value}
        operands::Vector{IR.Value} = []

        for i in 1:IR.noperands(op)
            push!(operands, IR.operand(op, i))
        end

        return operands 
    end

    function collect_results(op::IR.Operation)
        results = []

        for i ∈ 1:IR.nresults(op)
            push!(results, IR.result(op, i))
        end

        return results
    end

    function collect_attributes(op::IR.Operation)
        attributes = []

        for i ∈ 1:IR.nattrs(op)
            push!(attributes, IR.attr(op, i))
        end

        return attributes
    end



    # extract the underlying type, e.g Integer given an array of Integers
    function underlying_type(type::IR.Type)::Type
        if IR.istensor(type) == AbstractArray 
            if IR.julia_type(eltype(type)) <: Integer
                return Integer
            elseif IR.julia_type(eltype(type)) <: AbstractFloat
                return AbstractFloat
            else
                error("Unrecognized tensor type: $(IR.julia_type(eltype(type)))")
            end
        elseif IR.julia_type(type) <: Integer || IR.julia_type(type) <: AbstractFloat
            return IR.julia_type(type)
        else 
            error("Received unrecognized type when Lowering julia to MLIR, Type: $(type)")
        end
    end

    # TODO: only considering the types from one side!
    """ translate Julia predicates into arith predicate """
    function julia_predicate_to_arith(pred, type)::Int64
        # the offset between signed and unsigned types == 4
        if type <: Unsigned && Int(pred) >= 2
            return Int(pred) + 4
        else
            return Int(pred)
        end
    end

    # TODO: turn into proper dynamic dispatch
    """ lower julia cmp to arith cmp """
    function lower_cmp!(op::IR.Operation, block, replace_ops)
        operands = collect_operands(op)
        raw_types = IR.type.(operands)
        types = IR.julia_type.(raw_types)

        ret = IR.type.(collect_results(op))[1]

        attributes = collect_attributes(op)
        pred = IR.attr(op, "predicate")

        # TODO: add asserts
        if underlying_type(raw_types[1]) <: Integer && underlying_type(raw_types[2]) <: Integer 
            new_op = arith.cmpi(operands...; result=IR.Type(Bool), predicate=julia_predicate_to_arith(pred, types[1]))
        elseif underlying_type(raw_types[1]) <: AbstractFloat && underlying_type(raw_types[2]) <: AbstractFloat
            new_op = arith.cmpf(operands...; result=IR.Type(Bool), predicate=julia_predicate_to_arith(pred, types[1]))
        else
            error("Error: unable to translate julia.cmp with types: $types")
        end

        IR.insert_after!(block, op, new_op)
        push!(replace_ops, [op, new_op])
    end


    """ unroll n-way julia operation into a tree of operations """
    function unroll_operation!(op::IR.Operation, block, fn_int::Function, fn_float::Function, replace_ops)
        operands = collect_operands(op)
        types = IR.julia_type.(IR.type.(operands))

        ret = IR.type.(collect_results(op))[1]

        prev_op = op
        prev_ref = operands[1]
        prev_val = operands[1]

        replaced = false

        for new_ref in operands[2:end]
            if types[1] <: Integer && types[2] <: Integer
                new_op = fn_int(prev_val, new_ref; result=ret)
                IR.insert_after!(block, prev_op, new_op)
                prev_op = new_op
                prev_ref = new_ref
                prev_val = collect_results(prev_op)[1]

                replaced = true

            elseif types[1] <: AbstractFloat && types[2] <: AbstractFloat
                new_op = fn_float((prev_val), new_ref; result=ret)
                IR.insert_after!(block, prev_op, new_op)
                prev_op = new_op
                prev_ref = new_ref
                prev_val = collect_results(prev_op)[1]

                replaced = true
            end

        end

        if replaced
            push!(replace_ops, [op, prev_op])
        end
    end


    """ unroll n-way julia operation into a tree of operations """
    function unroll_operation!(op::IR.Operation, block, fn_sint::Function, fn_uint::Function, fn_float::Function, replace_ops)
        operands = collect_operands(op)
        types = IR.julia_type.(IR.type.(operands))

        ret = IR.type.(collect_results(op))[1]

        prev_info = reference_information(types, op, operands[1], block, ret)

        replaced = false

        for new_ref in operands[2:end]
            if convert_julia_op_to_mlir!(prev_info, fn_sint, Signed, new_ref)
                replaced = true 
            elseif convert_julia_op_to_mlir!(prev_info, fn_uint, Unsigned, new_ref)
                replaced = true 
            elseif convert_julia_op_to_mlir!(prev_info, fn_float, AbstractFloat, new_ref)
                replaced = true 
            end
        end

        if replaced
            push!(replace_ops, [op, prev_info.prev_op])
        end
    end


    # TODO: turn into proper dynamic dispatch
    """ unroll n-way julia matrix operation into a tree of operations """
    function unroll_operation_mat!(op::IR.Operation, block, fn, replace_ops)
        operands = collect_operands(op)
        types = IR.julia_type.((IR.type.(operands)))

        ret = IR.type.(collect_results(op))[1]

        prev_info = reference_information(types, op, operands[1], block, ret)

        replaced = false

        for new_ref in operands[2:end]
            if convert_julia_op_to_mlir!(prev_info, fn, AbstractArray, new_ref)
                replaced = true
            end
        end

        if replaced
            push!(replace_ops, [op, prev_info.prev_op])
        end
    end


    """ Convert generic julia op to mlir equivalent, return true if replaced """
    function convert_julia_op_to_mlir!(prev::reference_information, mlir_fn::Function, target_type::Type, new_ref)::Bool
        if prev.operand_types[1] <: target_type && prev.operand_types[2] <: target_type
            new_op = mlir_fn(prev.prev_val, new_ref; result=prev.ret)
            IR.insert_after!(prev.block, prev.prev_op, new_op)
            update_reference_information(prev, new_op, collect_results(new_op)[1])

            true
        else
            false
        end
    end

    """ Convert Abstract Array julia op to mlir equivalent, return true if replaced """
    function convert_julia_op_to_mlir!(prev::reference_information, mlir_fn::Function, target_type::Type{AbstractArray}, new_ref)::Bool
        if prev.operand_types[1] <: target_type && prev.operand_types[2] <: target_type
            if mlir_fn === tosa.matmul
                new_op = mlir_fn(prev.prev_val, new_ref, c=prev.ret)
            else
                new_op = mlir_fn(prev.prev_val, new_ref, output=prev.ret)
            end

            new_op = mlir_fn(prev.prev_val, new_ref; result=prev.ret)
            IR.insert_after!(prev.block, prev.prev_op, new_op)
            update_reference_information(prev, new_op, collect_results(new_op)[1])

            true
        else
            false
        end
    end

end

using ._JuliaPassHelpers

function lower_op_to_mlir(op_name::Val{Any}, block::IR.Block, op::IR.Operation)
    error("Lower operation not found for $op_name")
end

function lower_op_to_mlir(op_name::Val{:(julia_add)}, block::IR.Block, op::IR.Operation, replace_ops)
    unroll_operation!(op, block, arith.addi, arith.addf, replace_ops)
    unroll_operation_mat!(op, block, tosa.add, replace_ops)
end


function lower_op_to_mlir(op_name::Val{:(julia_sub)}, block::IR.Block, op::IR.Operation, replace_ops)
    unroll_operation!(op, block, arith.subi, arith.subf, replace_ops)
    unroll_operation_mat!(op, block, tosa.sub, replace_ops)
end


function lower_op_to_mlir(op_name::Val{:(julia_mul)}, block::IR.Block, op::IR.Operation, replace_ops)
    unroll_operation!(op, block, arith.muli, arith.mulf, replace_ops)
    unroll_operation_mat!(op, block, tosa.matmul, replace_ops)
end

function lower_op_to_mlir(op_name::Val{:(julia_div)}, block::IR.Block, op::IR.Operation, replace_ops)
    unroll_operation!(op, block, arith.divsi, arith.divui, arith.divf, replace_ops)
end

function lower_op_to_mlir(op_name::Val{:(julia_rem)}, block::IR.Block, op::IR.Operation, replace_ops)
    unroll_operation!(op, block, arith.remsi, arith.remui, arith.remf, replace_ops)
    array_unimplemented(op)
end

function lower_op_to_mlir(op_name::Val{:(julia_cmp)}, block::IR.Block, op::IR.Operation, replace_ops)
    lower_cmp!(op, block, replace_ops)
end

function lower_op_to_mlir(op_name::Val{:(julia_mat_inst)}, block::IR.Block, op::IR.Operation, replace_ops)
    operands = collect_operands(op)
    types = IR.julia_type.((IR.type.(operands)))

    ret = IR.type.(collect_results(op))[1]

    # reorder the input matrix based on the input dimensions
    println("ret is: $ret with dims $(size(ret))")
    new_operand_order::Vector{IR.Value} = []
    step_size = first(size(ret))
    for step in 1:step_size
        for outer_dim in 1:last(size(ret))
            push!(new_operand_order, operands[(step_size * (outer_dim - 1)) + step])
        end
    end

    println("Created new operands order $new_operand_order")

    if IR.istensor(ret)
        new_op = tensor.from_elements(new_operand_order, result=ret)
        IR.insert_after!(block, op, new_op)
    else
        error("Error: incorrect types used for julia.mat_inst")
    end

    push!(replace_ops, [op, new_op])
end

function lower_op_to_mlir(op_name::Val{:(julia_mat_adjoint)}, block::IR.Block, op::IR.Operation, replace_ops)
    # create permutation map
    target_array = IR.DenseElementsAttribute([1,0])
    ret = IR.type.(collect_results(op))[1]

    # create transpose op
    new_op = tosa.transpose(first(collect_operands(op)), IR.NamedAttribute("perms", target_array), output=ret)

    # insert into the program
    IR.insert_after!(block, op, new_op)
    push!(replace_ops, [op, new_op])

end

function lower_op_to_mlir(op_name::Val{:(julia_mat_getindex)}, block::IR.Block, op::IR.Operation, replace_ops)
    operands = collect_operands(op)
    types = IR.julia_type.((IR.type.(operands)))

    # naively cast to index TODO: only works for single requests
    sub_const = arith.constant(;value=1,result=IR.Type(Int))
    IR.insert_before!(block, op, sub_const)

    sub_op = arith.subi(operands[2], IR.result(sub_const))
    IR.insert_before!(block, op, sub_op)

    index_op = arith.index_cast(IR.result(sub_op); out=IR.IndexType())
    IR.insert_before!(block, op, index_op)

    operands[2] = IR.result(index_op)

    # create operation
    gather_dims = IR.DenseArrayAttribute([0])
    ret = IR.type.(collect_results(op))[1]

    # create output type
    # ret = IR.TensorType([1], ret)

    new_indices = operands[2:end]

    # fix index annotations
    if length(new_indices) == 1
        index_op = arith.constant(;value=0,result=IR.Type(Int))
        IR.insert_before!(block, op, index_op)

        idx_cast_op = arith.index_cast(IR.result(index_op, 1); out=IR.IndexType())
        IR.insert_before!(block, op, idx_cast_op)

        push!(new_indices, IR.result(idx_cast_op, 1))
        
        if first(size(operands[1])) == 1 && last(size(operands[1])) != 1
            new_indices = reverse(new_indices)
        end
    end


    # create transpose op
    new_op = tensor.extract(operands[1], new_indices; result=ret)

    # insert into the program
    IR.insert_after!(block, op, new_op)
    push!(replace_ops, [op, new_op])
end


function lower_op_to_mlir(op_name::Val{:(julia_mat_adjoint)}, block::IR.Block, op::IR.Operation, replace_ops)
    operands = collect_operands(op)
    types = IR.julia_type.((IR.type.(operands)))
    ret = IR.type.(collect_results(op))[1]

    scalar::Value = operands[1]
    dest::Value = operands[2]
    indices::Vector{Value} = operands[3:end]

    # convert input type to indextype
    new_indices::Vector{Value} = []

    sub_const = arith.constant(;value=1,result=IR.Type(Int))
    IR.insert_before!(block, op, sub_const)

    for index in indices
        # if type(index) != IR.IndexType()
            # IR.type!(scalar, IR.IndexType())
            # new_scalar = arith.constant(;result=IR.IndexType(), value=scalar)
            sub_op = arith.subi(index, IR.result(sub_const))
            IR.insert_before!(block, op, sub_op)

            index_op = arith.index_cast(IR.result(sub_op); out=IR.IndexType())
            IR.insert_before!(block, op, index_op)

            res = IR.result(index_op, 1)
            push!(new_indices, res)
        # else
            # push!(new_indices, index)
        # end
    end

    # deal with vector indices
    if length(new_indices) == 1
        index_op = arith.constant(;value=0,result=IR.Type(Int))
        IR.insert_before!(block, op, index_op)

        idx_cast_op = arith.index_cast(IR.result(index_op, 1); out=IR.IndexType())
        IR.insert_before!(block, op, idx_cast_op)

        push!(new_indices, IR.result(idx_cast_op, 1))
        
        if first(size(dest)) == 1 && last(size(dest)) != 1
            new_indices = reverse(new_indices)
        end
    end

    new_op = tensor.insert(scalar, dest, new_indices; result=ret)
    IR.insert_after!(block, op, new_op)

    push!(replace_ops, [op, new_op])

    # fix the rewrite issue from Julia's IR
    ctx = context(op)
    rewriter = API.mlirIRRewriterCreate(ctx)

    GC.@preserve rewriter begin
        API.mlirRewriterBaseReplaceAllUsesExcept(rewriter, dest, IR.result(new_op, 1), new_op)
    end

    API.mlirIRRewriterDestroy(rewriter)
end

end

