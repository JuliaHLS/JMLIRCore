include("common_types.jl")
include("generate_mlir.jl")

""" Translation layer to move Julia intrinsics into mlir """
function intrinsic_to_mlir(target_function)
    md = MethodDetails(target_function)
    fop = generate_mlir(md)
    return fop
end
