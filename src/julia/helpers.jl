# helpers for lowering passes



""" High-level Op lowering interface (External Use) """
module JuliaLowerOp

using MLIR.IR
using MLIR.API

using MLIR.IR
import MLIR.IR
using MLIR.API
using MLIR.Dialects: arith, tosa, tensor, math, scf

export lower_op_to_mlir

""" Julia Helpers, do not use externally """
module _JuliaPassHelpers
    using MLIR.IR
    using MLIR.API

    using MLIR.IR
    import MLIR.IR
    using MLIR.API
    using MLIR.Dialects: arith, tosa, tensor, math, scf

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

    # transform vector indices into index type, 0-based indexing and fill
    # missing indices to match dimensions required
    function transform_indices(block, op, indices::Vector{Value})::Vector{Value}
        # convert input type to indextype
        new_indices::Vector{Value} = []

        sub_const = arith.constant(;value=1,result=IR.Type(Int))
        IR.insert_before!(block, op, sub_const)

        for index in indices
            sub_op = arith.subi(index, IR.result(sub_const))
            IR.insert_before!(block, op, sub_op)

            index_op = arith.index_cast(IR.result(sub_op); out=IR.IndexType())
            IR.insert_before!(block, op, index_op)

            res = IR.result(index_op, 1)
            push!(new_indices, res)
        end

        # deal with vector indices
        rev = false
        new_indices = reverse(new_indices)
        for _ in 1:(3 - length(new_indices))
            index_op = arith.constant(;value=0,result=IR.Type(Int))
            IR.insert_before!(block, op, index_op)

            idx_cast_op = arith.index_cast(IR.result(index_op, 1); out=IR.IndexType())
            IR.insert_before!(block, op, idx_cast_op)

            push!(new_indices, IR.result(idx_cast_op, 1))
            
            rev = true
        end

        if rev
            new_indices = reverse(new_indices)
        end

        new_indices
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
                ret_type = IR.julia_type(eltype(IR.type.(collect_results(prev.prev_op))[1]))
                zero_dense_attr = IR.DenseElementsAttribute([convert(ret_type, 0)])

                a_zp = tosa.const_(output=IR.TensorType([1], IR.Type(ret_type)), value=zero_dense_attr)
                b_zp = tosa.const_(output=IR.TensorType([1], IR.Type(ret_type)), value=zero_dense_attr)
                IR.insert_before!(prev.block, prev.prev_op, a_zp)
                IR.insert_before!(prev.block, prev.prev_op, b_zp)
                new_op = mlir_fn(prev.prev_val, new_ref, c=prev.ret; a_zp=IR.result(a_zp), b_zp=IR.result(b_zp))
            else
                new_op = mlir_fn(prev.prev_val, new_ref; output=prev.ret)
            end

            # new_op = mlir_fn(prev.prev_val, new_ref; result=prev.ret)
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

function lower_op_to_mlir(op_name::Val{:(julia_pow)}, block::IR.Block, op::IR.Operation, replace_ops)
    # unroll_operation!(op, block, math.ipowi, math.powf, replace_ops)
    # array_unimplemented(op)
    operands = collect_operands(op)
    types = IR.julia_type.(IR.type.(operands))

    ret = IR.type.(collect_results(op))[1]

    prev_op = op
    prev_ref = operands[1]
    prev_val = operands[1]

    replaced = false

    for new_ref in operands[2:end]
        if types[1] <: Integer && types[2] <: Integer
            new_op = math.ipowi(prev_val, new_ref; result=ret)
            IR.insert_after!(block, prev_op, new_op)
            prev_op = new_op
            prev_ref = new_ref
            prev_val = collect_results(prev_op)[1]

            replaced = true

        elseif types[1] <: AbstractFloat && types[2] <: Integer
            ops = collect_operands(op)

            # create constant with 0 idx attribute with type 
            index_type = IR.Type(API.mlirIndexTypeGet(context()))
            zero_idx_attr = Attribute(API.mlirIntegerAttrGet(index_type, Int64(0)))
            one_idx_attr = Attribute(API.mlirIntegerAttrGet(index_type, Int64(1)))

            # create constants to iterate over
            c0 = arith.constant(; value=zero_idx_attr, result=index_type)
            c1 = arith.constant(; value=one_idx_attr, result=index_type)
            cst = arith.constant(; value=1.0, result=IR.Type(types[1]))

            # cast power to index type
            idx_cast_op = arith.index_cast((ops[2]); out=IR.IndexType())
            IR.insert_before!(block, op, idx_cast_op)

            # insert ops
            IR.insert_before!(block, op, c0)
            IR.insert_before!(block, op, c1)
            IR.insert_before!(block, op, cst)

            # internal Region
            region = IR.Region()
            loop_block = IR.Block()

            # add loop invariant
            v1 = IR.push_argument!(loop_block, index_type)
            v2 = IR.push_argument!(loop_block, IR.Type(types[1]))
            push!(region, loop_block)

            # create main scf loop
            main_loop = scf.for_(IR.result(c0), IR.result(idx_cast_op), IR.result(c1), [IR.result(cst)]; results= [IR.Type(types[1])], region=region)

            # new_op = ((prev_val), new_ref; result=ret)
            # println("Created main loop with operands: $(v)")

            IR.insert_after!(block, prev_op, main_loop)

            # insert items into the block
            mul = arith.mulf(ops[1], v2)
            push!(loop_block, mul)
            
            yield = scf.yield([IR.result(mul)])
            push!(loop_block, yield)

            # update references
            prev_op = main_loop
            prev_ref = IR.result(main_loop)
            prev_val = collect_results(prev_op)[1]

            replaced = true
        else
            error("Raising pow to with type arguments: $types is not supported yet")
        end

    end

    if replaced
        push!(replace_ops, [op, prev_op])
    end

end

function lower_op_to_mlir(op_name::Val{:(julia_cmp)}, block::IR.Block, op::IR.Operation, replace_ops)
    lower_cmp!(op, block, replace_ops)
end

function lower_op_to_mlir(op_name::Val{:(julia_mat_inst)}, block::IR.Block, op::IR.Operation, replace_ops)
    operands = collect_operands(op)
    types = IR.julia_type.((IR.type.(operands)))

    ret = IR.type.(collect_results(op))[1]

    # reorder the input matrix based on the input dimensions
    new_operand_order::Vector{IR.Value} = []

    dim1, dim2, dim3 = size(ret)

    reshaped_operands = reshape(operands, dim1, dim2, dim3)
    reshaped_operands = permutedims(reshaped_operands, (3, 2, 1))
    col_major_operands = vec(reshaped_operands)

    if IR.istensor(ret)
        new_op = tensor.from_elements(col_major_operands, result=ret)
        IR.insert_after!(block, op, new_op)
    else
        error("Error: incorrect types used for julia.mat_inst")
    end

    push!(replace_ops, [op, new_op])
end

function tosa.transpose(input1::Value, perms::Value; output::IR.Type, location=Location())
    _results = IR.Type[output,]
    _operands = Value[input1, perms]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = IR.NamedAttribute[]

    return IR.create_operation(
        "tosa.transpose",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

function lower_op_to_mlir(op_name::Val{:(julia_mat_adjoint)}, block::IR.Block, op::IR.Operation, replace_ops)
    # create permutation map
    ops = first(collect_operands(op))

    ret = IR.type.(collect_results(op))[1]

    IR.allow_unregistered_dialects!(true)

    target_transformation = collect(ntuple(i -> findfirst(==(size(ret)[i]), size(ops)), 3))
    target_transformation = target_transformation .- 1
    target_array = IR.NamedAttribute("perms", IR.DenseArrayAttribute(Int32.(target_transformation)::Vector{Int32}))

    # create the operation
    # perm_op = tosa.const_(output = IR.TensorType(size(target_transformation), IR.Type(Int32)), values=target_array)
    # IR.insert_before!(block, op, perm_op) 

    # create transpose op
    new_op = tosa.transpose(ops, target_array; output=ret)

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

    indices::Vector{Value} = operands[2:end]

    # create operation
    gather_dims = IR.DenseArrayAttribute([0])
    ret = IR.type.(collect_results(op))[1]

    # create transpose op
    new_op = tensor.extract(operands[1], _JuliaPassHelpers.transform_indices(block, op, indices); result=ret)

    # insert into the program
    IR.insert_after!(block, op, new_op)
    push!(replace_ops, [op, new_op])
end


function lower_op_to_mlir(op_name::Val{:(julia_mat_setindex)}, block::IR.Block, op::IR.Operation, replace_ops)
    operands = collect_operands(op)
    types = IR.julia_type.((IR.type.(operands)))
    ret = IR.type.(collect_results(op))[1]

    scalar::Value = operands[1]
    dest::Value = operands[2]
    indices::Vector{Value} = operands[3:end]

    new_op = tensor.insert(scalar, dest, _JuliaPassHelpers.transform_indices(block, op, indices); result=ret)
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

