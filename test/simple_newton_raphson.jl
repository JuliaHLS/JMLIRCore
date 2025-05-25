using FixedPointNumbers
using MLIR
using MLIR.IR
using Test

function IR.Type(T::Core.Type{<:Fixed}; context::IR.Context=context())
    return IR.Type(MLIR.API.mlirIntegerTypeGet(context, sizeof(T) * 8))
end

import Base: *
@force_inline function Base.:*(x::Fixed{T,Q}, y::Fixed{T,Q})::Fixed{T,Q} where {T<:Integer,Q}
    # simply delegate to Base.* 
    # but the @inline on * will override any no-inline policy
    # Base.:*(x, y)
    # return (Base.:*(x * y) >> Q)::Fixed{T,Q}
    raw = reinterpret(T, x) * reinterpret(T, y)
    raw2 = raw >> Q
    return reinterpret(Fixed{T,Q},raw2)
end

f(x) = x^4 - 5x + 3
f_deriv(x) = 4x^3 - 5

function newton_raphson_1D(x0::Fixed{T,Q}, err_tol::Fixed{T,Q}, max_itr=10) where {T,Q}
    x_k = x0
    itr = 0
    err = typemax(Fixed{T, Q})
    
    while itr < max_itr
        # x_next = x_k - (f(x_k) / f_deriv(x_k))
        # err = x_k - x_next
        x_k = x_k * x_k

        # x_k = x_next
        itr += 1
    end

    return x_k
end
