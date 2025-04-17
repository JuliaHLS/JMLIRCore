# Overlayed MethodTable satisfying the requirements
# for the MLIR Translation Layer
module MLIROverlays
using StaticArrays  

""" Instantiate the binding to a custom, separate MethodTable """
Base.Experimental.@MethodTable MLIR_MT

""" Overlays for Core Operations, e.g +, -, ...) """
# Note: these specific functions aim to resolve ambiguous type inference
# to help the typ inference passes correctly infer their target. This is
# not arbitrarily true for other functions
Base.Experimental.@overlay MLIR_MT Base.:+(a::MVector{N, T}, b::MVector{N, T}) where {N, T} = add_type(a,b)::MVector{N, T}

Base.Experimental.@overlay MLIR_MT Base.:-(a::MVector{N, T}, b::MVector{N, T}) where {N, T} = sub_type(a,b)::MVector{N, T}

Base.Experimental.@overlay MLIR_MT Base.:*(a::MVector{N, T}, b::MVector{N, T}) where {N, T} = mul_type(a,b)::MVector{N, T}

Base.Experimental.@overlay MLIR_MT MVector{N, T}(x::Tuple) where {N, T} = new_matrix(x)::MVector{N, T}
end

