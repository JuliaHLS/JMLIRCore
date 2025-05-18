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
using MLIR.Dialects: arith, tosa, tensor, math, cf, bufferization

include("helpers.jl")

using .JuliaLowerOp

location(operation) = Location(API.mlirOperationGetLocation(operation))
name(operation) = String(API.mlirOperationGetName(operation))
block(operation) = Block(API.mlirOperationGetBlock(operation), false)
parent_operation(operation) = Operation(API.mlirOperationGetParentOperation(operation), false)
dialect(operation) = first(split(name(operation), '.')) |> Symbol


function rewrite_references(replace_ops)
  for (original, replacement) in replace_ops
        rewriter = API.mlirIRRewriterCreateFromOp(original)

        GC.@preserve rewriter begin
            API.mlirRewriterBaseReplaceOpWithOperation(rewriter, original, replacement)
        end
        API.mlirIRRewriterDestroy(rewriter)
    end
end

struct LowerJuliaMat <: IR.AbstractPass end

IR.opname(::LowerJuliaMat) = "func.func"

function IR.pass_run(::LowerJuliaMat, func_op)
    println("Running LowerJuliaMat")
    
    replace_ops = []

    for region in IR.RegionIterator(func_op)
        for block in IR.BlockIterator(region)
            for op in IR.OperationIterator(block)
                op_name = replace(name(op), "." => "_")

                # if op is in the julia dialect
                if JuliaFixSSA.check_name(op, "julia_mat")
                    op_sym = Symbol(op_name)
                    lower_op_to_mlir(Val(op_sym), block, op, replace_ops)
                end
            end
        end
    end

    rewrite_references(replace_ops)  
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

                if JuliaFixSSA.check_name(op, "julia")
                    # if op is in the julia dialect
                    if JuliaFixSSA.check_name(op, "julia_mat")
                        continue
                    end

                    op_sym = Symbol(op_name)
                    lower_op_to_mlir(Val(op_sym), block, op, replace_ops)
                end
            end
        end
    end

    rewrite_references(replace_ops)
end


struct FixTensorSSA <: IR.AbstractPass end

IR.opname(::FixTensorSSA) = "func.func"

function IR.pass_run(::FixTensorSSA, func_op)
    println("Running FixTensorSSA")
    
    dominating_blocks = JuliaFixSSA.collect_dominating_blocks(func_op)

    target_collection = JuliaFixSSA.collect_dominated_branches(dominating_blocks)
    
    # fix SSA on dominated branch
    for (dom_block, collection) in zip(dominating_blocks, target_collection)
        # fix SSA for the entry block

        new_ssa = JuliaFixSSA.fix_ssa_dominating_block!(last(dom_block), first(dom_block))

        # skip blocks with a single dominating block
        length(collection[2:end]) > 0 || continue

        if new_ssa != nothing
            # fix SSA for the dominated blocks
            for target_block in collection[2:end]
                JuliaFixSSA.fix_ssa_dominated_block!(last(dom_block), target_block, new_ssa)
            end
        end
    end
end

struct FixImplicitControlFlow <: IR.AbstractPass end

IR.opname(::FixImplicitControlFlow) = "func.func"

function IR.pass_run(::FixImplicitControlFlow, func_op)
    println("Running FixImplicitControlFlow")
    
    # collect implicit blocks
    implicit_blocks = JuliaFixSSA.collect_implicit_blocks(func_op)

    # fix implicit blocks
    JuliaFixSSA.fix_implicit_blocks!(implicit_blocks)
end

end
