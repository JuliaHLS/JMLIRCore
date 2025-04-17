module julia

using MLIR
using MLIR:get_type, julia_type
using MLIR.IR

# Generate Arithmetic Operators
for f in (:add, :sub, :mul, :div)
    @eval function $f(
        operands::Value...; result=nothing::Union{Nothing,IR.Type}, output=nothing::Union{Nothing,IR.Type}, location=Location()
    )
        _results = IR.Type[]
        _operands = Value[operands...]
        _owned_regions = Region[]
        _successors = Block[]
        # _attributes = IR.NamedAttribute[]
        @assert !(!isnothing(result) && !isnothing(output))
        !isnothing(result) && push!(_results, result)
        !isnothing(output) && push!(_results, output)

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

end
