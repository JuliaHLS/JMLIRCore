using MLIR.IR
using MLIR.API

is_registered_operation(opname, ctx) = API.mlirContextIsRegisteredOperation(ctx, opname)


# Fix _pass_initialize in MLIR.jl (override, taken out of the source code to merge easily at a later date)
# function IR._pass_initialize(ctx, handle::IR.ExternalPassHandle)
#     try
#         handle.ctx = IR.Context(ctx)
#         API.Types.MlirLogicalResult(0)
#     catch
#         API.Types.MlirLogicalResult(1)
#     end
# end


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

location(operation) = Location(API.mlirOperationGetLocation(operation))
name(operation) = String(API.mlirOperationGetName(operation))
block(operation) = Block(API.mlirOperationGetBlock(operation), false)
parent_operation(operation) = Operation(API.mlirOperationGetParentOperation(operation), false)
dialect(operation) = first(split(name(operation), '.')) |> Symbol

function get_first_region(op::Operation)
    reg = iterate(RegionIterator(op))
    isnothing(reg) && return nothing
    first(reg)
end

function get_first_block(op::Operation)
    reg = get_first_region(op)
    isnothing(reg) && return nothing
    block = iterate(BlockIterator(reg))
    isnothing(block) && return nothing
    first(block)
end

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

    types = IR.Type.(results)

    return results, types
end


function get_type(attribute::Attribute)
    IR.Type(API.mlirAttributeGetType(attribute))
end
function type_value(attribute)
    @assert API.mlirAttributeIsAType(attribute) "attribute $(attribute) is not a type"
    IR.Type(API.mlirTypeAttrGetValue(attribute))
end
function bool_value(attribute)
    @assert API.mlirAttributeIsABool(attribute) "attribute $(attribute) is not a boolean"
    API.mlirBoolAttrGetValue(attribute)
end
function string_value(attribute)
    @assert API.mlirAttributeIsAString(attribute) "attribute $(attribute) is not a string attribute"
    String(API.mlirStringAttrGetValue(attribute))
end



struct LowerJuliaAdd <: IR.AbstractPass end

IR.opname(::LowerJuliaAdd) = "func.func"

function IR.pass_run(::LowerJuliaAdd, func_op)
    block = get_first_block(func_op)

    replace_ops = [] #Dict{IR.Operation, IR.Operation}()

    for op in IR.OperationIterator(block)
        if name(op) == "julia.add"
            operands = collect_operands(op)
            types = IR.julia_type.(IR.type.(operands))

            prev_op = op
            prev_ref = operands[1]

            for new_ref in operands[2:end]
                if types[1] <: Integer && types[2] <: Integer
                    new_op = arith.addi(prev_ref, new_ref)
                elseif types[1] <: AbstractFloat && types[2] <: AbstractFloat
                    new_op = arith.addf(prev_ref, new_ref)
                else
                    error("Error in LowerJuliaAdd pass, unrecognized return signature $types")
                end

                IR.insert_after!(block, prev_op, new_op)
                prev_op = new_op
                prev_ref = new_ref
            end

            push!(replace_ops, [op, prev_op])
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

