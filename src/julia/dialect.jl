module julia

using MLIR
using MLIR:get_type, julia_type
using MLIR.IR

# Generate Arithmetic Operators
for f in (:add, :sub, :mul, :div, :rem, :pow, :not_int)
    @eval function $f(
                      operands::Value...; result=nothing::Union{Nothing,IR.Type}, output=nothing::Union{Nothing,IR.Type}, quant=nothing::Union{Nothing, IR.NamedAttribute}, location=Location()
    )
        _results = IR.Type[]
        _operands = Value[operands...]
        _owned_regions = Region[]
        _successors = Block[]
        # _attributes = IR.NamedAttribute[]
        @assert !(!isnothing(result) && !isnothing(output))
        !isnothing(result) && push!(_results, result)
        !isnothing(output) && push!(_results, output)
        !isnothing(quant) && push!(_attributes, quant)

        return IR.create_operation(
            $(string("julia.", f)),
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

# Note: signed operations are handled in the associated lowering passes
# module cmp
module predicate
    const eq = 0
    const ne = 1
    const lt = 2
    const le = 3
    const gt = 4
    const ge = 5
end
# end

function cmp(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    predicate,
    fastmath=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = IR.NamedAttribute[MLIR.Dialects.namedattribute("predicate", predicate),]
    !isnothing(result) && push!(_results, result)
    !isnothing(fastmath) && push!(_attributes, namedattribute("fastmath", fastmath))

    return IR.create_operation(
        "julia.cmp",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

# Array initialiser
function mat_inst(elements::Array{Value}; result::IR.Type, location=Location())
    _results = IR.Type[result,]
    _operands = Value[elements...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = IR.NamedAttribute[]

    return IR.create_operation(
        "julia.mat_inst",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

# only called for vectors (done in-place by julia otherwise)
function mat_adjoint(input1::Value; output::IR.Type, location=Location())
    _results = IR.Type[output,]
    _operands = Value[input1]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = IR.NamedAttribute[]

    return IR.create_operation(
        "julia.mat_adjoint",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

function neg_int(input1::Value; output::IR.Type, location=Location())
    _results = IR.Type[output,]
    _operands = Value[input1]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = IR.NamedAttribute[]

    return IR.create_operation(
        "julia.neg_int",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

function mat_setindex(
    args::Vector{Value};
    result=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    # unpack the result
    dest::Value = args[1]
    scalar::Value = args[2]
    indices::Vector{Value} = args[3:end]

    _results = IR.Type[]

    if result == nothing
        dest_op = IR.type(dest)
        result = dest_op
    end

    _operands = Value[scalar, dest, indices...]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = IR.NamedAttribute[]
    !isnothing(result) && push!(_results, result)

    return IR.create_operation(
        "julia.mat_setindex",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=false,
    )
end

function mat_getindex(
    args::Vector{Value};
    result::IR.Type,
    # gather_dims, TODO: add support for dims
    unique=nothing,
    location=Location(),
)
    _results = IR.Type[result,]
    _operands = Value[args...]
    _owned_regions = Region[]
    _successors = Block[]
    # _attributes = NamedAttribute[namedattribute("gather_dims", gather_dims),]
    _attributes = IR.NamedAttribute[]
    !isnothing(unique) && push!(_attributes, namedattribute("unique", unique))

    return IR.create_operation(
        "julia.mat_getindex",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

end
