using StaticArrays

@inline function cordic_gain(M::Integer)
    K = 1.0
    for i in 0:M-1
        K *= 1 / sqrt(1 + 2.0^(-2*i))
    end
    return K
end
import Base: >>, <<, *

function <<(x::Fixed{T,F}, s::Integer) where {T,F}
    raw = x.i << s
    # the `true` flag means “raw bits already in Q-F format”
    Fixed{T,F}(raw, true)
end

function >>(x::Fixed{T,F}, s::Integer) where {T,F}
   raw = x.i >> s
   Fixed{T,F}(raw, true)
end

@inline function Base.:*(x::Fixed{T,F}, y::Fixed{T,F})::Fixed{T, F} where {T,F}
    res = (x + y)::Fixed{T,F}
    return res::Fixed{T, F}
end

function cordic_int(theta::Fixed{T,N}, K::Fixed{T, N}, iterations::Int) where {T, N}
    # Local fixed‐point “1.0” in Q30
    # fp_one = (5) 

    # Local arctangent lookup table in Q30
    angles = @MMatrix [
       Fixed{T, N}(atan(1));  # atan(1)         * 2^30
       Fixed{T,N}(atan(0.5)); # 497837829;  # atan(0.5)       * 2^30
       Fixed{T,N}(atan(0.25)); # 262043836;  # atan(0.25)      * 2^30
       Fixed{T,N}(atan(0.125)); # 133525159;  # atan(0.125)     * 2^30
       Fixed{T,N}(atan(0.0625)); # 67021687;  # atan(0.0625)    * 2^30
       Fixed{T,N}(atan(0.03125)); # 33543516;  # atan(0.03125)   * 2^30
       Fixed{T,N}(atan(0.015625)); # 16775851;  # atan(0.015625)  * 2^30
       Fixed{T,N}(atan(0.0078125)); # 8388437;  # atan(0.0078125) * 2^30
       Fixed{T,N}(atan(0.00390625));# 4194287;  # atan(0.00390625)* 2^30
       Fixed{T,N}(atan(0.001953125));# 2097149;  # atan(0.001953125)*2^30
       Fixed{T,N}(atan(0.0009765625));# 1048575;  # atan(0.0009765625)*2^30
       Fixed{T,N}(atan(0.00048828125));# 524288;  # atan(0.00048828125)*2^30
       Fixed{T,N}(atan(0.000244140625));# 262144;  # atan(0.000244140625)*2^30
       Fixed{T,N}(atan(0.0001220703125));# 131072;  # atan(0.0001220703125)*2^30
       Fixed{T,N}(atan(6.103515625e-5));# 65536;  # atan(6.103515625e-5)*2^30
       Fixed{T,N}(atan(3.0517578125e-5));# 32768;  # atan(3.0517578125e-5)*2^30
    ]

    # Initialize vector
    # x = Fixed{T, N}(1.0) # 2^62# fp_one
    # x = Fixed{T, N}(cordic_gain(iterations))
    # x = K
    # y = Fixed{T,N}(0)
    z = theta

    for i in 1:iterations
        # direction: +1 if z>=0, else -1
        # one = 1
        di::Fixed{T, N} = z ≥ Fixed{T, N}(0) ? Fixed{T, N}(1) : Fixed{T, N}(-1)
        # shift::Int = i - one

        # CORDIC rotation updates
        # x_new::Fixed{T, N} = di
        # x_new = x - (di * (y >> shift))
        # println("got x_new $x_new")
        # y_new = y + (di * (x >> shift))
        # y_new = (x >> shift)
        # println(x_new)
        # z_new = z - di * angles[i]
        z_new = di * angles[i]::Fixed{T, N}

        # x, y, z = x_new, y_new, z_new
        # x = x_new
        # y = y_new
        z = z_new
    end

    return z
end
