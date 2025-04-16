using MLIR.IR
using MLIR.API

is_registered_operation(opname, ctx) = API.mlirContextIsRegisteredOperation(ctx, opname)

# helper to run an AbstractPass pass on a module
function run!(pass::IR.AbstractPass, mod::IR.Module, ctx)
    opname = IR.opname(pass)

    nameof_pass = string(nameof(typeof(pass)))

    pm = IR.PassManager()
    mlir_pass = IR.create_external_pass!(pm, pass, nameof_pass, nameof_pass, "", opname)

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
using MLIR.Dialects: arith, tosa


location(operation) = Location(API.mlirOperationGetLocation(operation))
name(operation) = String(API.mlirOperationGetName(operation))
block(operation) = Block(API.mlirOperationGetBlock(operation), false)
parent_operation(operation) = Operation(API.mlirOperationGetParentOperation(operation), false)
dialect(operation) = first(split(name(operation), '.')) |> Symbol

function collect_operands(op::IR.Operation)
    operands = []

    for i in 1:IR.noperands(op)
        push!(operands, IR.operand(op, i))
    end

    return operands 
end

function collect_results(op::IR.Operation)
    results = []

    for i in 1:IR.nresults(op)
        push!(results, IR.result(op, i))
    end

    return results
end


struct LowerJuliaArith <: IR.AbstractPass end

IR.opname(::LowerJuliaArith) = "func.func"

function IR.pass_run(::LowerJuliaArith, func_op)
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

            # else
            #     error("Error in LowerJuliaArith pass, unrecognized return signature $types")
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
                new_op = fn(prev_val, new_ref, output=ret)
            # else
            #     error("Error in LowerJuliaArith pass, unrecognized return signature $types")

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



    for region in IR.RegionIterator(func_op)
        for block in IR.BlockIterator(region)
            for op in IR.OperationIterator(block)
                if name(op) == "julia.add"
                    unroll_operation!(op, block, arith.addi, arith.addf)
                    unroll_operation_mat!(op, block, tosa.add)
                elseif name(op) == "julia.sub"
                    unroll_operation!(op, block, arith.subi, arith.subf)
                    unroll_operation_mat!(op, block, tosa.sub)
                elseif name(op) == "julia.mul"
                    unroll_operation!(op, block, arith.muli, arith.mulf)
                    unroll_operation_mat!(op, block, tosa.matmul)
                elseif name(op) == "julia.div"
                    # TODO: check there is no div equivalnet
                    unroll_operation!(op, block, arith.divf, arith.divf)
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
