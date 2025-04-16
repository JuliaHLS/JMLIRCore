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
using MLIR.API
using MLIR.Dialects: arith

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

        for new_ref in operands[2:end]
            if types[1] <: Integer && types[2] <: Integer
                new_op = fn_int(prev_val, new_ref)
            elseif types[1] <: AbstractFloat && types[2] <: AbstractFloat
                new_op = fn_float((prev_val), new_ref)
            else
                error("Error in LowerJuliaArith pass, unrecognized return signature $types")
            end

            IR.insert_after!(block, prev_op, new_op)
            prev_op = new_op
            prev_ref = new_ref
            prev_val = collect_results(prev_op)[1]
        end

        push!(replace_ops, [op, prev_op])
    end



    for region in IR.RegionIterator(func_op)
        for block in IR.BlockIterator(region)
            for op in IR.OperationIterator(block)
                if name(op) == "julia.add"
                    unroll_operation!(op, block, arith.addi, arith.addf)
                elseif name(op) == "julia.sub"
                    unroll_operation!(op, block, arith.subi, arith.subf)
                elseif name(op) == "julia.mul"
                    unroll_operation!(op, block, arith.muli, arith.mulf)
                elseif name(op) == "julia.div"
                    # TODO: check there is no div equivalent
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
