using MLIR.IR
using MLIR.API

is_registered_operation(opname, ctx) = API.mlirContextIsRegisteredOperation(ctx, opname)

# helper to run an AbstractPass pass on a module
function run!(pass::IR.AbstractPass, mod::IR.Module, ctx)
    opname = IR.opname(pass)

    nameof_pass = string(nameof(typeof(pass)))

    pm = IR.PassManager()
    mlir_pass = IR.create_external_pass!(pm, pass, nameof_pass, nameof_pass, "", opname)

    println("Created external pass")

    if isempty(opname)
        IR.add_owned_pass!(pm, mlir_pass)
    else
        @assert is_registered_operation(opname, ctx) "$opname is not registered"
        opm = IR.OpPassManager(pm, opname)
        IR.add_owned_pass!(opm, mlir_pass)
    end

    status = API.mlirPassManagerRunOnOp(pm, IR.Operation(mod))
end

module JuliaPasses

using MLIR.IR
import MLIR.IR
using MLIR.API
using MLIR.Dialects: arith, tosa, tensor


location(operation) = Location(API.mlirOperationGetLocation(operation))
name(operation) = String(API.mlirOperationGetName(operation))
block(operation) = Block(API.mlirOperationGetBlock(operation), false)
parent_operation(operation) = Operation(API.mlirOperationGetParentOperation(operation), false)
dialect(operation) = first(split(name(operation), '.')) |> Symbol

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



struct LowerJuliaMat <: IR.AbstractPass end

IR.opname(::LowerJuliaMat) = "func.func"

function IR.pass_run(::LowerJuliaMat, func_op)
    println("Running LowerJuliaMat")
    
    replace_ops = []

    for region in IR.RegionIterator(func_op)
        for block in IR.BlockIterator(region)
            for op in IR.OperationIterator(block)
                println("processing: $(name(op))")
                if name(op) == "julia.mat_inst"
                    operands = collect_operands(op)
                    types = IR.julia_type.((IR.type.(operands)))

                    ret = IR.type.(collect_results(op))[1]

                    if IR.istensor(ret)
                        new_op = tensor.from_elements(operands,result=ret)
                        IR.insert_after!(block, op, new_op)
                    else
                        error("Error: incorrect types used for julia.mat_inst")
                    end

                    push!(replace_ops, [op, new_op])
                elseif name(op) == "julia.mat_adjoint"
                    # create permutation map
                    target_array = IR.DenseElementsAttribute([1,0])
                    ret = IR.type.(collect_results(op))[1]

                    # create transpose op
                    new_op = tosa.transpose(first(collect_operands(op)), IR.NamedAttribute("perms", target_array), output=ret)

                    # insert into the program
                    IR.insert_after!(block, op, new_op)
                    push!(replace_ops, [op, new_op])
                elseif name(op) == "julia.mat_getindex" # TODO: add support for returning arrays
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

                elseif name(op) == "julia.mat_setindex"
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
        end
    end

    for (original, replacement) in replace_ops
        rewriter = API.mlirIRRewriterCreateFromOp(original)

        GC.@preserve rewriter begin
            API.mlirRewriterBaseReplaceOpWithOperation(rewriter, original, replacement)
        end
        API.mlirIRRewriterDestroy(rewriter)
    end

end


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

struct LowerJuliaArith <: IR.AbstractPass end

IR.opname(::LowerJuliaArith) = "func.func"

function IR.pass_run(::LowerJuliaArith, func_op)
    println("Running LowerJuliaArith")
    replace_ops = []

    function unroll_operation!(op::IR.Operation, block, fn_int::Function, fn_float::Function)
        operands = collect_operands(op)
        types = IR.julia_type.(IR.type.(operands))

        prev_op = op
        prev_ref = operands[1]
        prev_val = operands[1]

        replaced = false

        for new_ref in operands[2:end]
            if types[1] <: Integer && types[2] <: Integer
                new_op = fn_int(prev_val, new_ref)
                IR.insert_after!(block, prev_op, new_op)
                prev_op = new_op
                prev_ref = new_ref
                prev_val = collect_results(prev_op)[1]

                replaced = true

            elseif types[1] <: AbstractFloat && types[2] <: AbstractFloat
                new_op = fn_float((prev_val), new_ref)
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

    function unroll_operation!(op::IR.Operation, block, fn_sint::Function, fn_uint::Function, fn_float::Function)
        operands = collect_operands(op)
        types = IR.julia_type.(IR.type.(operands))

        prev_op = op
        prev_ref = operands[1]
        prev_val = operands[1]

        replaced = false

        for new_ref in operands[2:end]
            if types[1] <: Signed && types[2] <: Signed
                new_op = fn_sint(prev_val, new_ref)
                IR.insert_after!(block, prev_op, new_op)
                prev_op = new_op
                prev_ref = new_ref
                prev_val = collect_results(prev_op)[1]
                replaced = true
            elseif types[1] <: Unsigned && types[2] <: Unsigned 
                new_op = fn_uint(prev_val, new_ref)
                IR.insert_after!(block, prev_op, new_op)
                prev_op = new_op
                prev_ref = new_ref
                prev_val = collect_results(prev_op)[1]
                replaced = true
            elseif types[1] <: AbstractFloat && types[2] <: AbstractFloat
                new_op = fn_float((prev_val), new_ref)
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



    # TODO: turn into proper dynamic dispatch
    function unroll_operation_mat!(op::IR.Operation, block, fn)
        operands = collect_operands(op)
        types = IR.julia_type.((IR.type.(operands)))

        ret = IR.type.(collect_results(op))[1]

        prev_op = op
        prev_ref = operands[1]
        prev_val = operands[1]

        replaced = false

        for new_ref in operands[2:end]
            if types[1] <: AbstractArray && types[2] <: AbstractArray
                if fn === tosa.matmul
                    new_op = fn(prev_val, new_ref, c=ret)
                else
                    new_op = fn(prev_val, new_ref, output=ret)
                end
                 
                IR.insert_after!(block, prev_op, new_op)
                prev_op = new_op
                prev_ref = new_ref
                prev_val = collect_results(prev_op)[1]

                replaced = true
            # else
            #     error("Error in LowerJuliaArith pass, unrecognized return signature $types")
            end
        end

        if replaced
            push!(replace_ops, [op, prev_op])
        end
    end

    function underlying_type(type::IR.Type)
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
    function translate_predicate(pred, type)::Int64
        # the offset between signed and unsigned types == 4
        if type <: Unsigned && Int(pred) >= 2
            return Int(pred) + 4
        else
            return Int(pred)
        end
    end


    # TODO: turn into proper dynamic dispatch
    function lower_cmp!(op::IR.Operation, block)
        operands = collect_operands(op)
        raw_types = IR.type.(operands)
        types = IR.julia_type.(raw_types)

        ret = IR.type.(collect_results(op))[1]

        attributes = collect_attributes(op)
        pred = IR.attr(op, "predicate")

        # TODO: add asserts
        if underlying_type(raw_types[1]) <: Integer && underlying_type(raw_types[2]) <: Integer 
            new_op = arith.cmpi(operands...; result=IR.Type(Bool), predicate=translate_predicate(pred, types[1]))
        elseif underlying_type(raw_types[1]) <: AbstractFloat && underlying_type(raw_types[2]) <: AbstractFloat
            new_op = arith.cmpf(operands...; result=IR.Type(Bool), predicate=translate_predicate(pred, types[1]))
        else
            error("Error: unable to translate julia.cmp with types: $types")
        end

        IR.insert_after!(block, op, new_op)
        push!(replace_ops, [op, new_op])
    end



    for region in IR.RegionIterator(func_op)
        for block in IR.BlockIterator(region)
            for op in IR.OperationIterator(block)
                println("Processing op: $(name(op))")
                if name(op) == "julia.add"
                    unroll_operation!(op, block, arith.addi, arith.addf)
                    unroll_operation_mat!(op, block, tosa.add)
                elseif name(op) == "julia.sub"
                    println("now processing the sub op")
                    unroll_operation!(op, block, arith.subi, arith.subf)
                    println("unrolled simple arith")
                    unroll_operation_mat!(op, block, tosa.sub)
                    println("unrolled simple tensor arith")
                elseif name(op) == "julia.mul"
                    unroll_operation!(op, block, arith.muli, arith.mulf)
                    unroll_operation_mat!(op, block, tosa.matmul)
                elseif name(op) == "julia.div"
                    # TODO: check there is no div equivalnet
                    unroll_operation!(op, block, arith.divf, arith.divf)
                elseif name(op) == "julia.rem"
                    unroll_operation!(op, block, arith.remsi, arith.remui, arith.remf)
                    array_unimplemented(op)
                    
                elseif name(op) == "julia.cmp"
                    lower_cmp!(op, block)
                end
            end
        end
    end

    for (original, replacement) in replace_ops
        rewriter = API.mlirIRRewriterCreateFromOp(original)

        GC.@preserve rewriter begin
            API.mlirRewriterBaseReplaceOpWithOperation(rewriter, original, replacement)
        end
        API.mlirIRRewriterDestroy(rewriter)
    end
end

end
