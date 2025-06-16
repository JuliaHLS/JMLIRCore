using FixedPointNumbers
using MLIR
using MLIR.IR
using Test
using JMLIRCore


function IR.Type(T::Core.Type{<:Fixed}; context::IR.Context=context())
    return IR.Type(MLIR.API.mlirIntegerTypeGet(context, sizeof(T) * 8))
end

# Fixed Point overlays
@force_inline function Base.:*(x::Fixed{T,Q}, y::Fixed{T,Q})::Fixed{T,Q} where {T<:Integer,Q}
    raw = reinterpret(T, x) * reinterpret(T, y)
    raw2 = raw >> Q
    return reinterpret(Fixed{T,Q},raw2)
end

@force_inline function Base.:*(x::T, y::Fixed{T,Q})::Fixed{T,Q} where {T<:Integer,Q}
    raw = (x << Q) * reinterpret(T, y)
    raw2 = raw >> Q
    return reinterpret(Fixed{T,Q},raw2)
end

@force_inline function Base.:+(x::Fixed{T,Q}, y::Integer)::Fixed{T,Q} where {T<:Integer,Q}
    raw = reinterpret(T, x) + (y << Q)
    return reinterpret(Fixed{T,Q},raw)
end

@force_inline function Base.:-(x::Fixed{T,Q}, y::Integer)::Fixed{T,Q} where {T<:Integer,Q}
    raw = reinterpret(T, x) - (y << Q)
    return reinterpret(Fixed{T,Q},raw)
end


@inline function overlayed_div(x::Fixed{T,Q}, y::Fixed{T,Q})::Fixed{T,Q} where {T<:Integer,Q}
    raw = (reinterpret(T, x) << Q) / reinterpret(T, y)
    return reinterpret(Fixed{T,Q},raw)
end

@force_inline function Base.:^(x::Fixed{T,Q}, y::Integer)::Fixed{T,Q} where {T<:Integer,Q}
    raw = reinterpret(T, x)
    # raw2 = raw >> (Q * y - 1)
    out = 1 << Q
    for _ in 1:y
        out *= raw
        out  = out >> Q
    end

    return reinterpret(Fixed{T,Q},out)
end



# f(x) = x^4 - 5x + 3
f(x) = x^2 - 5x + 3
f_deriv(x) = 2x - 5

function newton_raphson_1D(x0, err_tol::Fixed{T,Q}, max_itr=10) where {T,Q}
    x_k = x0
    itr = 0
    
    while itr < max_itr
        # x_next = x_k - (f(x_k) / f_deriv(x_k))
        x_next = overlayed_div(x_k / x_k)
        # x_next = f_deriv(x_k)
        # x_next = x_k * x_k


        # if (x_next - x_k) < err_tol
        #    itr = max_itr 
        # end

        x_k = x_next
        itr += 1
    end

    return x_k
end
