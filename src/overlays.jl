# Overlayed MethodTable satisfying the requirements
# for the MLIR Translation Layer
module MLIROverlays
using StaticArrays  

""" Instantiate the binding to a custom, separate MethodTable """
Base.Experimental.@MethodTable MLIR_MT
using LinearAlgebra
using FixedPointNumbers
import FixedPointNumbers: Fixed
include("force_inline.jl")

import Base: <<, >>

""" Overlays for Core Operations, e.g +, -, ...) """
# Note: these specific functions aim to resolve ambiguous type inference
# to help the typ inference passes correctly infer their target. This is
# not arbitrarily true for other functions
Base.Experimental.@overlay MLIR_MT Base.:+(a::MArray{S, T, M, N}, b::MArray{S, T, M, N}) where {S, T, M, N} = add_type(a,b)::MArray{S, T, M, N}

Base.Experimental.@overlay MLIR_MT Base.:-(a::MArray{S, T, M, N}, b::MArray{S, T, M, N}) where {S, T, M, N} = sub_type(a,b)::MArray{S, T, M, N}

Base.Experimental.@overlay MLIR_MT Base.:*(a::MArray{S, T, M, N}, b::MArray{S, T, M, N}) where {S, T, M, N} = mul_type(a,b)::MArray{S, T, M, N}

Base.Experimental.@overlay MLIR_MT MArray{S, T, M, N}(x::Tuple) where {S, T, M, N} = new_array(x)::MArray{S, T, M, N}

Base.Experimental.@overlay MLIR_MT Base.:<<(x::Fixed{T, F}, s::Integer) where {T, F} = left_shift(x)::Fixed{T, F}

Base.Experimental.@overlay MLIR_MT Base.:>>(x::Fixed{T, F}, s::Integer) where {T, F} = right_shift(x)::Fixed{T, F}
end

