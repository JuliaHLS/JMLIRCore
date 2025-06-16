using StaticArrays

using FixedPointNumbers
using MLIR
using MLIR.IR
using Test

function IR.Type(T::Core.Type{<:Fixed}; context::IR.Context=context())
    return IR.Type(MLIR.API.mlirIntegerTypeGet(context, sizeof(T) * 8))
end



@inline function cordic_gain(M::Integer)
    K = 1.0
    for i in 0:M-1
        K *= 1 / sqrt(1 + 2.0^(-2*i))
    end
    return K
end


# import Base: *
# @force_inline function Base.:*(x::Fixed{T,Q}, y::Fixed{T,Q})::Fixed{T,Q} where {T<:Integer,Q}
#     raw = reinterpret(T, x) * reinterpret(T, y)
#     raw2 = raw >> Q
#     # println("GOT RAW: $raw2")
#     return reinterpret(Fixed{T,Q},raw2)
# end

# @force_inline function Base.:*(x::T, y::Fixed{T,Q})::Fixed{T,Q} where {T<:Integer,Q}
#     raw = (x << Q) * reinterpret(T, y)
#     raw2 = raw >>> Q
#     return reinterpret(Fixed{T,Q},raw2)
# end

function cordic(theta::Fixed{T,N}, K::Fixed{T, N}) where {T, N}
    # arctangent lookup table 
    angles = @MMatrix [
       Fixed{T, N}(atan(1));
       Fixed{T,N}(atan(0.5));
       Fixed{T,N}(atan(0.25));
       Fixed{T,N}(atan(0.125));
       Fixed{T,N}(atan(0.0625));
       Fixed{T,N}(atan(0.03125));
       Fixed{T,N}(atan(0.015625));
       Fixed{T,N}(atan(0.0078125));
       Fixed{T,N}(atan(0.00390625));
       Fixed{T,N}(atan(0.001953125));
       Fixed{T,N}(atan(0.0009765625));
       Fixed{T,N}(atan(0.00048828125));
       Fixed{T,N}(atan(0.000244140625));
       Fixed{T,N}(atan(0.0001220703125));
       Fixed{T,N}(atan(6.103515625e-5));
       Fixed{T,N}(atan(3.0517578125e-5));
    ]

    x = K
    y = Fixed{T,N}(0)
    z = theta

    for i in one(T):T(length(angles))
        di::Fixed{T, N} = z ≥ Fixed{T, N}(0) ? Fixed{T, N}(1) : Fixed{T, N}(-1)
        shift::T = i - one(T)

        # CORDIC rotation updates
        x_new = x - (di * (y >> shift))
        y_new = y + (di * (x >> shift))
        z_new = z - di * angles[i]

        x, y, z = x_new, y_new, z_new
    end

    return (x,y)
end
