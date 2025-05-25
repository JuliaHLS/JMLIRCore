using StaticArrays
# function conv2d_matmul_mmatrix(
#   A::MMatrix{M,N,T},
#   K::MMatrix{P,Q,T},
# ) where {M,N,P,Q,T}
#   outM = M - P + 1
#   outN = N - Q + 1
#   rows = P*Q
#   cols = outM*outN

#   # 1) Build the patch‐matrix S::rows×cols
#   S = MMatrix{rows, cols, T}(undef)
#   for j in 1:outN, i in 1:outM, v in 1:Q, u in 1:P
#     col = (j-1)*outM + i       # which output‐column
#     row = (v-1)*P + u          # which kernel‐row
#     S[row, col] = A[i+u-1, j+v-1]
#   end

#   # 2) Flatten the kernel into k::rows×1
#   k = MMatrix{rows, 1, T}(undef)
#   for v in 1:Q, u in 1:P
#     idx = (v-1)*P + u
#     k[idx, 1] = K[u, v]
#   end

#   # 3) One matrix‐multiply: (S' is cols×rows) * (rows×1) → cols×1
#   ycol = S' * k   # result is MMatrix{cols,1,T}

#   # 4) Reshape back into the outM×outN result
#   Y = MMatrix{outM, outN, T}(undef)
#   for j in 1:outN, i in 1:outM
#     idx = (j-1)*outM + i
#     Y[i, j] = ycol[idx, 1]
#   end

#   return Y
# end

# function conv2d_matmul(
#   A::MMatrix{3,3,Int64},
#   K::MMatrix{2,2,Int64},
# )
#   outM = 3 - 2 + 1   # 2
#   outN = 3 - 2 + 1   # 2
#   rows = 2 * 2        # 4
#   cols = outM * outN  # 4

#   # 1) zero-init S via @MMatrix and fill
#   S = MMatrix{rows, cols, Int64}( Tuple(zero(Int64) for _ in 1:9, _ in 1:64)...)
#   # S = @MMatrix [0 0 0; 0 0 0; 0 0 0]
#   for j in 1:outN, i in 1:outM, v in 1:3, u in 1:3
#     row = (v-1)*3 + u
#     col = (j-1)*outM + i
#     S[row, col] = A[i+u-1, j+v-1]
#   end

#   # 2) zero-init k via @MMatrix and fill
#   k = @MMatrix [ zero(Int64) for _ in 1:9, _ in 1:1 ]
#   for v in 1:3, u in 1:3
#     idx = (v-1)*3 + u
#     k[idx, 1] = K[u, v]
#   end

#   # 3) single mat-mul
#   ycol = S' * k   # MMatrix{64,1,Int64}

#   # 4) scatter back into an 8×8 result
#   Y = @MMatrix [ zero(Int64) for _ in 1:8, _ in 1:8]
#   for j in 1:outN, i in 1:outM
#     idx = (j-1)*outM + i
#     Y[i, j] = ycol[idx, 1]
#   end

#   return Y
# end

function conv2d_mat(
  A::MMatrix{3,3,Int64},
  K::MMatrix{2,2,Int64},
)
  # output dims
  outM = 3 - 2 + 1   # 2
  outN = 3 - 2 + 1   # 2

  # patch‐matrix dims
  rows = 2 * 2       # 4
  cols = outM * outN # 4

  # 1) Build S::4×4 via @MMatrix of zeros (manually unrolled)
  S = @MMatrix [
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    0 0 0 0
  ]
  for j in 1:outN, i in 1:outM, v in 1:2, u in 1:2
    row = (v-1)*2 + u          # 1..4
    col = (j-1)*outM + i       # 1..4
    S[row, col] = A[i+u-1, j+v-1]
  end

  # 2) Build k::4×1 via @MMatrix of zeros
  k = @MMatrix [
    0;
    0;
    0;
    0
  ]
  for v in 1:2, u in 1:2
    idx = (v-1)*2 + u
    k[idx, 1] = K[u, v]
  end

  # 3) One mat‐mul does all the inner‐products
  ycol = S' * k   # 4×1

  # 4) Scatter back into the 2×2 output via @MMatrix of zeros
  Y = @MMatrix [
    0 0;
    0 0
  ]
  for j in 1:outN, i in 1:outM
    idx = (j-1)*outM + i
    Y[i, j] = ycol[idx, 1]
  end

  return Y
end

function conv2d_mat_test(
  # A::MMatrix{3,3,Int64},
  # K::MMatrix{2,2,Int64},
)
    A = @MMatrix [
          1  2  3;
          4  5  6;
          7  8  9
       ]
    K = @MMatrix [
          1  0;
          0  1
       ]
  # output dims
  outM = 3 - 2 + 1   # 2
  outN = 3 - 2 + 1   # 2

  # patch‐matrix dims
  rows = 2 * 2       # 4
  cols = outM * outN # 4

  # 1) Build S::4×4 via @MMatrix of zeros (manually unrolled)
  S = @MMatrix [
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    0 0 0 0
  ]
  for j in 1:outN, i in 1:outM, v in 1:2, u in 1:2
    row = (v-1)*2 + u          # 1..4
    col = (j-1)*outM + i       # 1..4
    S[row, col] = A[i+u-1, j+v-1]
  end

  # 2) Build k::4×1 via @MMatrix of zeros
  k = @MMatrix [
    0;
    0;
    0;
    0
  ]
  for v in 1:2, u in 1:2
    idx = (v-1)*2 + u
    k[idx, 1] = K[u, v]
  end

  # 3) One mat‐mul does all the inner‐products
  ycol = S * k   # 4×1

  # 4) Scatter back into the 2×2 output via @MMatrix of zeros
  Y = @MMatrix [
    0 0;
    0 0
  ]
  for j in 1:outN, i in 1:outM
    idx = (j-1)*outM + i
    Y[i, j] = ycol[idx, 1]
  end

  return Y
  # return 1
end


# """
#     conv2d_matmul(A, K)

# 2D “valid” convolution of A::M×N by K::P×Q  
# using an im2col → matmul approach on StaticArrays.
# """
# function conv2d_matmul{M,N,P,Q,T}(
#       A::MMatrix{M,N,T}, K::MMatrix{P,Q,T}
#     ) where {M,N,P,Q,T}

#   # output size
#   outM = M - P + 1
#   outN = N - Q + 1

#   # 1) build a (P*Q)×(outM*outN) patch‐matrix S
#   #    each column is the flattened P×Q patch of A
#   S = @MMatrix [ A[i+u-1, j+v-1] for u in 1:P, v in 1:Q,
#                                   i in 1:outM, j in 1:outN ]

#   # 2) flatten the kernel to a (P*Q) vector
#   k = vec(K)   # SVector{P*Q,T}

#   # 3) multiply: gives an SVector{outM*outN}
#   y = S' * k

#   # 4) reshape back into an outM×outN MMatrix
#   return MMatrix{outM,outN,T}(y)
# end

# # — example —
# A = @MMatrix [ 1 2 3 4 5;
#                6 7 8 9 10;
#               11 12 13 14 15;
#               16 17 18 19 20;
#               21 22 23 24 25 ]

# K = @MMatrix [ 1 0 -1;
#                1 0 -1;
#                1 0 -1 ]

# Y_loop   = conv2d(A, K)          # your original loop version
# Y_matmul = conv2d_matmul(A, K)   # now via one matmul

# @assert Y_loop == Y_matmul      # sanity check!
# println("Result via matmul:\n", Y_matmul)

