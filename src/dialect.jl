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

end
