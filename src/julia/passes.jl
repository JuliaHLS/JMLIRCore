using MLIR.IR
using MLIR.API

is_registered_operation(opname, ctx) = API.mlirContextIsRegisteredOperation(ctx, opname)

# helper to run an AbstractPass pass on a module
function run!(pass::IR.AbstractPass, mod::IR.Module, ctx)
    opname = IR.opname(pass)

    nameof_pass = string(nameof(typeof(pass)))

    pm = IR.PassManager()
    mlir_pass = IR.create_external_pass!(pm, pass, nameof_pass, nameof_pass, "", opname)

    GC.@preserve mlir_pass mod pass begin
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
end

module JuliaPasses

using MLIR.IR
import MLIR.IR
using MLIR.API
using MLIR.Dialects: arith, tosa, tensor

include("helpers.jl")

using .JuliaLowerOp

location(operation) = Location(API.mlirOperationGetLocation(operation))
name(operation) = String(API.mlirOperationGetName(operation))
block(operation) = Block(API.mlirOperationGetBlock(operation), false)
parent_operation(operation) = Operation(API.mlirOperationGetParentOperation(operation), false)
dialect(operation) = first(split(name(operation), '.')) |> Symbol




struct LowerJuliaMat <: IR.AbstractPass end

IR.opname(::LowerJuliaMat) = "func.func"

function IR.pass_run(::LowerJuliaMat, func_op)
    println("Running LowerJuliaMat")
    
    replace_ops = []

    for region in IR.RegionIterator(func_op)
        for block in IR.BlockIterator(region)
            for op in IR.OperationIterator(block)
                op_name = replace(name(op), "." => "_")

                if length(op_name) >= 5 && op_name[1:5] == "julia"
                    op_sym = Symbol(op_name)
                    lower_op_to_mlir(Val(op_sym), block, op, replace_ops)
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

struct LowerJuliaArith <: IR.AbstractPass end

IR.opname(::LowerJuliaArith) = "func.func"



function IR.pass_run(::LowerJuliaArith, func_op)
    println("Running LowerJuliaArith")
    replace_ops = []

    for region in IR.RegionIterator(func_op)
        for block in IR.BlockIterator(region)
            for op in IR.OperationIterator(block)
                op_name = replace(name(op), "." => "_")

                if length(op_name) >= 5 && op_name[1:5] == "julia"
                    op_sym = Symbol(op_name)
                    lower_op_to_mlir(Val(op_sym), block, op, replace_ops)
                end
            end
        end
    end

    for (original, replacement) in replace_ops
        println("Replacing $original with $replacement")
        rewriter = API.mlirIRRewriterCreateFromOp(original)

        GC.@preserve rewriter begin
            API.mlirRewriterBaseReplaceOpWithOperation(rewriter, original, replacement)
        end
        API.mlirIRRewriterDestroy(rewriter)
    end
end

end
