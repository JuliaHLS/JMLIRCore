# Overlayed MethodTable satisfying the requirements
# for the MLIR Translation Layer
module MLIROverlays
using StaticArrays  

""" Instantiate the binding to a custom, separate MethodTable """
Base.Experimental.@MethodTable MLIR_MT
using LinearAlgebra

""" Overlays for Core Operations, e.g +, -, ...) """
# Note: these specific functions aim to resolve ambiguous type inference
# to help the typ inference passes correctly infer their target. This is
# not arbitrarily true for other functions
Base.Experimental.@overlay MLIR_MT Base.:+(a::MArray{S, T, M, N}, b::MArray{S, T, M, N}) where {S, T, M, N} = add_type(a,b)::MArray{S, T, M, N}

Base.Experimental.@overlay MLIR_MT Base.:-(a::MArray{S, T, M, N}, b::MArray{S, T, M, N}) where {S, T, M, N} = sub_type(a,b)::MArray{S, T, M, N}

Base.Experimental.@overlay MLIR_MT Base.:*(a::MArray{S, T, M, N}, b::MArray{S, T, M, N}) where {S, T, M, N} = mul_type(a,b)::MArray{S, T, M, N}

Base.Experimental.@overlay MLIR_MT MArray{S, T, M, N}(x::Tuple) where {S, T, M, N} = new_array(x)::MArray{S, T, M, N}

# Base.Experimental.@overlay MLIR_MT MVector{T, N}(x::Tuple) where {T, N} = MMatrix{T, N}(x)

# Base.Experimental.@overlay MLIR_MT LinearAlgebra.Adjoint{T, N}(x::Type{<:MVector{T, N}}) where {T, N} = new_array(x)::

end

