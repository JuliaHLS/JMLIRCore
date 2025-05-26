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


# # Fixed Point overlays
# Base.Experimental.@overlay MLIR_MT @force_inline function Base.:*(x::Fixed{T,Q}, y::Fixed{T,Q})::Fixed{T,Q} where {T<:Integer,Q}
#     raw = reinterpret(T, x) * reinterpret(T, y)
#     raw2 = raw >> Q
#     return reinterpret(Fixed{T,Q},raw2)
# end

# Base.Experimental.@overlay MLIR_MT @force_inline function Base.:*(x::T, y::Fixed{T,Q})::Fixed{T,Q} where {T<:Integer,Q}
#     raw = (x << Q) * reinterpret(T, y)
#     raw2 = raw >> Q
#     return reinterpret(Fixed{T,Q},raw2)
# end

# Base.Experimental.@overlay MLIR_MT @force_inline function Base.:+(x::Fixed{T,Q}, y::Integer)::Fixed{T,Q} where {T<:Integer,Q}
#     raw = reinterpret(T, x) + (y << Q)
#     return reinterpret(Fixed{T,Q},raw)
# end

# Base.Experimental.@overlay MLIR_MT @force_inline function Base.:/(x::Fixed{T,Q}, y::Fixed{T,Q})::Fixed{T,Q} where {T<:Integer,Q}
#     raw = (reinterpret(T, x) << Q) / reinterpret(T, y)
#     return reinterpret(Fixed{T,Q},raw)
# end

# Base.Experimental.@overlay MLIR_MT @force_inline function Base.:^(x::Fixed{T,Q}, y::Integer)::Fixed{T,Q} where {T<:Integer,Q}
#     raw = reinterpret(T, x)
#     # raw2 = raw >> (Q * y - 1)
#     out = 1 << Q
#     for _ in 1:y
#         out *= raw
#         out  = out >> Q
#     end

#     return reinterpret(Fixed{T,Q},out)
# end


end

