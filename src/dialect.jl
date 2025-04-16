module julia

using MLIR
using MLIR:get_type, julia_type
using MLIR.IR

# julia.add - arbitrary amount of input args
function add(
    operands::Value...; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[operands...]
    _owned_regions = Region[]
    _successors = Block[]
    # _attributes = IR.NamedAttribute[]
    !isnothing(result) && push!(_results, result)

    return IR.create_operation(
        "julia.add",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        # attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end


# julia.sub - arbitrary amount of input args
function sub(
    operands::Value...; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[operands...]
    _owned_regions = Region[]
    _successors = Block[]
    # _attributes = IR.NamedAttribute[]
    !isnothing(result) && push!(_results, result)

    return IR.create_operation(
        "julia.sub",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        # attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end



# julia.mul - arbitrary amount of input args
function mul(
    operands::Value...; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[operands...]
    _owned_regions = Region[]
    _successors = Block[]
    # _attributes = IR.NamedAttribute[]
    !isnothing(result) && push!(_results, result)

    return IR.create_operation(
        "julia.mul",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        # attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end



# julia.div - arbitrary amount of input args
function div(
    operands::Value...; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[operands...]
    _owned_regions = Region[]
    _successors = Block[]
    # _attributes = IR.NamedAttribute[]
    !isnothing(result) && push!(_results, result)

    return IR.create_operation(
        "julia.div",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        # attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end


end
