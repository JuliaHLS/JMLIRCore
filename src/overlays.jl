# Overlayed MethodTable satisfying the requirements
# for the MLIR Translation Layer
module MLIROverlays
using StaticArrays  

""" Instantiate the binding to a custom, separate MethodTable """
Base.Experimental.@MethodTable MLIR_MT

""" Overlays for Core Operations, e.g +, -, ...) """
Base.Experimental.@overlay MLIR_MT +(a::MVector{N, T}, b::MVector{N, T}) where {N, T} = add_type(a,b)::MVector{N, T}

Base.Experimental.@overlay MLIR_MT -(a::MVector{N, T}, b::MVector{N, T}) where {N, T} = sub_type(a,b)::MVector{N, T}

Base.Experimental.@overlay MLIR_MT :*(a::MVector{N, T}, b::MVector{N, T}) where {N, T} = mul_type(a,b)::MVector{N, T}
end

