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

       


    for region in IR.RegionIterator(func_op)
        for block in IR.BlockIterator(region)
            for op in IR.OperationIterator(block)
                println("Processing op: $(name(op))")
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
