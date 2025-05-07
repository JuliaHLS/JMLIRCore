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
                if length(op_name) >= 9 && op_name[1:5] == "julia.mat"
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

                # if op is in the julia dialect
                if length(op_name) >= 5 && op_name[1:5] == "julia"
                    if length(op_name) >= 9 && op_name[6:9] == "mat"
                        continue
                    end
                    if op_name == "julia_mul"
                    end

                    op_sym = Symbol(op_name)
                    lower_op_to_mlir(Val(op_sym), block, op, replace_ops)
                end
            end
        end
    end

    rewrite_references(replace_ops)
end


struct TensorToSSA <: IR.AbstractPass end

IR.opname(::TensorToSSA) = "func.func"

function check_name(op, target_name)::Bool
    op_name = name(op)
    return length(op_name) >= length(target_name) && op_name[1:length(target_name)] == target_name
end 

function get_child_blocks!(visited::Set, curr::Block, targets::Vector{IR.Block})
    if curr.block ∈ visited
        return
    end

    push!(visited, curr.block)
    push!(targets, curr)

    for op in IR.OperationIterator(curr)
        if check_name(op, "cf.br") || check_name(op, "cf.cond_br")
            
            for n_succ in 1:IR.nsuccessors(op)
                target_block = IR.successor(op, n_succ)

                if target_block ∉ visited
                    get_child_blocks!(visited, target_block, targets)
                end
            end
        end
    end
end

get_ret(op::IR.Operation) = IR.type.(JuliaLowerOp._JuliaPassHelpers.collect_results(op))[1]

function return_block(block::Block)::Bool
    for op in IR.OperationIterator(block)
        if check_name(op, "func.return")
            return true
        end
    end

    return false
end

function fix_block!(original_op::Operation, block::Block)
    # create memref type arg
    memref_type_with_dims = IR.MemRefType(eltype(get_ret(original_op)), [size(get_ret(original_op))...], IR.Attribute(0))

    IR.push_argument!(block, memref_type_with_dims) 

    new_ssa::IR.Value = IR.argument(block, IR.nargs(block))
    new_op = original_op

    # convert to tensor
    if !return_block(block)
        writable_cond = IR.UnitAttribute()
        restrict_cond = IR.UnitAttribute()
    else
        writable_cond = nothing
        writable_cond = IR.UnitAttribute()
        restrict_cond = IR.UnitAttribute()
    end

    new_op = bufferization.to_tensor(new_ssa; result=get_ret(original_op), restrict = restrict_cond, writable=writable_cond)
    
    first = IR.first_op(block)
    IR.insert_before!(block, first, new_op)

    new_ssa = IR.result(new_op)

    for op in IR.OperationIterator(block)
        operands = JuliaLowerOp._JuliaPassHelpers.collect_operands(op)

        for (i, present_operand) in enumerate(operands)
            if (present_operand) === IR.result(original_op) && !(check_name(op, "func.return"))
                IR.operand!(op, i, new_ssa)

                if IR.julia_type(get_ret(op)) <: AbstractArray && size(get_ret(op)) == size(get_ret(new_op))
                    new_ssa = IR.result(op)
                    new_op = op
                end
            elseif new_ssa == present_operand && !(check_name(op, "cf.br") || check_name(op, "cf.cond_br") ) # if SSA chain is found, progress

                if IR.julia_type(get_ret(op)) <: AbstractArray && size(get_ret(op)) == size(get_ret(new_op))
                    # convert back to a tensor
                    convert_to_memref_op = bufferization.to_memref(new_ssa; memref=memref_type_with_dims)
                    IR.insert_before!(block, op, convert_to_memref_op)

                    new_ssa = IR.result(convert_to_memref_op)
                    new_op = convert_to_memref_op
                end
            end
        end
        
        if check_name(op, "func.return")
            operands = JuliaLowerOp._JuliaPassHelpers.collect_operands(op)
            @assert length(operands) == 1

            if size(operands[1]) == size(new_ssa)
                new_operands = [new_ssa]
                API.mlirOperationSetOperands(op, length(new_operands), new_operands)
            end
        end

        if check_name(op, "cf.br") || check_name(op, "cf.cond_br")
            # fix the operands here to be in the right order
            convert_to_memref_op = bufferization.to_memref(new_ssa; memref=memref_type_with_dims)
            IR.insert_before!(block, op, convert_to_memref_op)

            new_ssa = IR.result(convert_to_memref_op)
            new_op = convert_to_memref_op

            if check_name(op, "cf.cond_br")
                halfway_length = Int32((length(operands) - 1) / 2)
                new_operands = [operands[1:(halfway_length+1)]...]
                push!(new_operands,new_ssa)
                push!(new_operands, operands[(halfway_length + 2):end]...)
                push!(new_operands, new_ssa)

                # new_operands = push!(operands, new_ssa)
                API.mlirOperationSetOperands(op, length(new_operands), new_operands)

                # fix attributes
                at = IR.attr(op, "operandSegmentSizes")

                cond = Int32(at[0])
                first_args = Int32(at[1])
                second_args = Int32(at[2])

                first_args += 1
                second_args += 1

                new_at = IR.DenseArrayAttribute(Int32.([cond, first_args, second_args])::Vector{Int32})
                # IR.rmattr!(op, "operandSegmentSizes")
                IR.attr!(op, "operandSegmentSizes", new_at)
            else
                new_operands = push!(operands, new_ssa)
                API.mlirOperationSetOperands(op, length(new_operands), new_operands)
            end
        end
    end
end

using MLIR

function cond_br(
    operands::Vector{Value};
    trueDest::Block,
    falseDest::Block,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[operands...]
    _owned_regions = Region[]
    _successors = Block[trueDest, falseDest]
    _attributes = IR.NamedAttribute[]
    push!(
        _attributes,
        MLIR.Dialects.operandsegmentsizes([1, length(trueDestOperands), length(falseDestOperands)]),
    )

    return IR.create_operation(
        "cf.cond_br",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end



function postfix_block!(original_op::Operation, block::Block)
    # IR.push_argument!(block, get_ret(original_op)) 
    new_ssa = IR.argument(block, IR.nargs(block))
    replace_ops = []

    for op in IR.OperationIterator(block)
        operands = JuliaLowerOp._JuliaPassHelpers.collect_operands(op)
        if check_name(op, "cf.br") || check_name(op, "cf.cond_br")
            succ = []
            for n_succ in 1:IR.nsuccessors(op)
                push!(succ, IR.successor(op, n_succ))
            end

            if check_name(op, "cf.br")
                new_op = cf.br(operands, dest=succ[1])
            elseif check_name(op, "cf.cond_br")
                len = length(succ)
                new_op = cond_br(operands; trueDest=succ[(len-1)], falseDest=last(succ))
            else
                error("Unrecognised cf.br detected $op")
            end

            IR.insert_before!(block, op, new_op)

            push!(replace_ops, [op, new_op])
        end
    end
end

function IR.pass_run(::TensorToSSA, func_op)
    println("Running TensorToSSA")

    # store dominating blocks
    dominating_blocks::Vector{Any} = []

    for region in IR.RegionIterator(func_op)
        for block in IR.BlockIterator(region)
            for op in IR.OperationIterator(block)
                op_name = name(op)

                if length(op_name) >= 14 && op_name[1:14] == "julia.mat_inst"
                    push!(dominating_blocks, [block, op])
                end
            end
        end
    end

    # collect branches
    target_collection = []
    for (dominator, dominating_op) in dominating_blocks 
        visited = Set()
        targets = Vector{IR.Block}()
        get_child_blocks!(visited, dominator, targets)

        push!(target_collection, targets)
    end

    # correct branch
    for (dom_block, collection) in zip(dominating_blocks, target_collection)
        # println("First: $((collection)[2:end])")
        for op in IR.OperationIterator(collection[1])
            operands = JuliaLowerOp._JuliaPassHelpers.collect_operands(op)

            if check_name(op, "cf.br") || check_name(op, "cf.cond_br")

                shape = [size(get_ret(last(dom_block)))...]
                memref_type_with_dims = IR.MemRefType(eltype(get_ret(last(dom_block))), shape, IR.Attribute(0))
                buff_op = bufferization.to_memref(IR.result(last(dom_block)); memref=memref_type_with_dims)
                IR.insert_before!(first(dom_block), op, buff_op)

                new_operands = push!(operands, IR.result(buff_op))
                API.mlirOperationSetOperands(op, length(new_operands), new_operands)
            end
        end

        for target_block in collection[2:end]
            fix_block!(last(dom_block), target_block)
            # postfix_block!(last(dom_block), target_block)
        end
    end

    # rewrite tree

end


end
