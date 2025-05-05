using JMLIRCore
using Test
using StaticArrays


add_test(a, b) = a + b
sub_test(a, b) = a - b
mul_test(a, b) = a * b
div_test(a, b) = a / b
pow_test(a, b) = a ^ b
rem(A, B) = A % B

function create_mat()
    return @MMatrix [1 2 3 4 5; 6 7 8 9 0]
end

function modify_mat(idx1, idx2, val)
    a = @MMatrix [1 2 3 4 5; 6 7 8 9 0]
    a[idx1, idx2] = val
    return a
end

function set_and_get_mat(idx1, idx2, ret_idx1, ret_idx2, val)
    a = @MMatrix [1 2 3 4 5; 6 7 8 9 0]
    a[idx1, idx2] = val
    return a[ret_idx1, ret_idx2]
end

function add_mat_test()
    a = @MMatrix [1 2 3 4 5; 6 7 8 9 0]
    b = @MMatrix [10 20 30 40 0; 50 60 70 80 90]

    return a + b
end

function sub_mat_test()
    a = @MMatrix [1 2 3 4 5; 6 7 8 9 0]
    b = @MMatrix [10 20 30 40 0; 50 60 70 80 90]

    return a - b
end

function matmul_test()
    a = @MMatrix [1 2 3 4 5; 6 7 8 9 0]
    b = @MMatrix [10 20 30 40 0; 50 60 70 80 90]

    return a * b'
end


function create_mat_float()
    return @MMatrix [1.2 2.2 3.2 4.2 5.2; 6.2 7.2 8.2 9.2 0.2]
end

function modify_mat_float(idx1, idx2, val)
    a = @MMatrix [1.2 2.2 3.2 4.2 5.2; 6.2 7.2 8.2 9.2 0.2]
    a[idx1, idx2] = val
    return a
end

function set_and_get_mat_float(idx1, idx2, ret_idx1, ret_idx2, val)
    a = @MMatrix [1.2 2.2 3.2 4.2 5.2; 6.2 7.2 8.2 9.2 0.2]
    a[idx1, idx2] = val
    return a[ret_idx1, ret_idx2]
end

function add_mat_float_test()
    a = @MMatrix [1.2 2.2 3.2 4.2 5.2; 6.2 7.2 8.2 9.2 0.2]
    b = @MMatrix [10.2 20.2 30.2 40.2 0.2; 50.2 60.2 70.2 80.2 90.2]

    return a + b
end

function sub_mat_float_test()
    a = @MMatrix [1.2 2.2 3.2 4.2 5.2; 6.2 7.2 8.2 9.2 0.2]
    b = @MMatrix [10.2 20.2 30.2 40.2 0.2; 50.2 60.2 70.2 80.2 90.2]

    return a - b
end

function matmul_float_test()
    a = @MMatrix [1.2 2.2 3.2 4.2 5.2; 6.2 7.2 8.2 9.2 0.2]
    b = @MMatrix [10.2 20.2 30.2 40.2 0.2; 50.2 60.2 70.2 80.2 90.2]

    return a * b'
end

function transpose_test()
   a = @MMatrix [1 2 3 4 5; 6 7 8 9 0]

   return a'
end


## produces two basic blocks that merge at the same node
function multi_route_node(A, B)
	result = 0
	if A < B
	   result = A + B
	elseif A > B
	   result = A - B
	end
	
 	return result
end


@testset "JMLIRCore.jl" begin
   ## BASIC ARITHMETIC TESTS ##
   
   ### INTEGER ###

   #### ADD ####
   # simple add
   @test (@eval_mlir add_test(5, 10)) == (@eval add_test(5, 10))
   @test (@eval_mlir add_test(5, -10)) == (@eval add_test(5, -10))

   # testing equivalence of int32 and int64
   @test (@eval_mlir add_test(9223372036854775807, 10)) == (@eval add_test(9223372036854775807, 10))
   @test (@eval_mlir add_test(123456789, 12345678909876)) == (@eval add_test(123456789, 12345678909876))
   @test (@eval_mlir add_test(123456789, -12345678909876)) == (@eval add_test(123456789, -12345678909876))

   # only UInt
   @test (@eval_mlir add_test(UInt(5), UInt(10))) == (@eval add_test(UInt(5), UInt(10)))

   @test (reinterpret(UInt, @eval_mlir add_test(UInt(9223372036854775807), UInt(10)))) == (@eval add_test(UInt(9223372036854775807), UInt(10)))
   @test (@eval_mlir add_test(UInt(123456789), UInt(12345678909876))) == (@eval add_test(UInt(123456789), UInt(12345678909876)))

   # #### SUB ####
   # # simple add
   @test (@eval_mlir sub_test(5, 10)) == (@eval sub_test(5, 10))
   @test (@eval_mlir sub_test(5, -10)) == (@eval sub_test(5, -10))

   # # testing equivalence of int32 and int64
   @test (@eval_mlir sub_test(9223372036854775807, 10)) == (@eval sub_test(9223372036854775807, 10))
   @test (@eval_mlir sub_test(123456789, 12345678909876)) == (@eval sub_test(123456789, 12345678909876))
   @test (@eval_mlir sub_test(123456789, -12345678909876)) == (@eval sub_test(123456789, -12345678909876))


   # #### MUL ####
   # # simple add
   @test (@eval_mlir mul_test(5, 10)) == (@eval mul_test(5, 10))
   @test (@eval_mlir mul_test(5, -10)) == (@eval mul_test(5, -10))


   # #### DIV ####
   # # simple add
   @test (@eval_mlir div_test(5, 10)) == (@eval div_test(5, 10))
   @test (@eval_mlir div_test(5, -10)) == (@eval div_test(5, -10))

    
   # #### REM ####
   @test (@eval_mlir rem(5, 10)) == (@eval rem(5, 10))
   @test (@eval_mlir rem(UInt(5), UInt(10))) == (@eval rem(UInt(5), UInt(10)))

   # #### POW ####
   @test (@eval_mlir pow(5, 10)) == (@eval pow(5, 10))
   @test (@eval_mlir pow(UInt(5), UInt(10))) == (@eval pow(UInt(5), UInt(10)))



   ### CONTROL FLOW ###
   @test (@eval_mlir multi_route_node(5, 10)) == (@eval multi_route_node(5, 10)) 
   @test (@eval_mlir multi_route_node(10, 5)) == (@eval multi_route_node(10, 5)) 


   ### FLOAT ###

   #### ADD ####
   # simple add
   @test (@eval_mlir add_test(5.0, 10.0)) == (@eval add_test(5.0, 10.0))
   @test (@eval_mlir add_test(5.0, -10.0)) == (@eval add_test(5.0, -10.0))

   # testing equivalence of int32 and int64
   @test (@eval_mlir add_test(9223372036854775807.0, 10.0)) == (@eval add_test(9223372036854775807.0, 10.0))
   @test (@eval_mlir add_test(123456789.0, 12345678909876.0)) == (@eval add_test(123456789.0, 12345678909876.0))
   @test (@eval_mlir add_test(123456789.0, -12345678909876.0)) == (@eval add_test(123456789.0, -12345678909876.0))


   # #### SUB ####
   # # simple add
   @test (@eval_mlir sub_test(5.0, 10.0)) == (@eval sub_test(5.0, 10.0))
   @test (@eval_mlir sub_test(5.0, -10.0)) == (@eval sub_test(5.0, -10.0))

   # # testing equivalence of int32 and int64
   @test (@eval_mlir sub_test(9223372036854775807.0, 10.0)) == (@eval sub_test(9223372036854775807.0, 10.0))
   @test (@eval_mlir sub_test(123456789.0, 12345678909876.0)) == (@eval sub_test(123456789.0, 12345678909876.0))
   @test (@eval_mlir sub_test(123456789.0, -12345678909876.0)) == (@eval sub_test(123456789.0, -12345678909876.0))


   # #### MUL ####
   # # simple add
   @test (@eval_mlir mul_test(5.0, 10.0)) == (@eval mul_test(5.0, 10.0))
   @test (@eval_mlir mul_test(5.0, -10.0)) == (@eval mul_test(5.0, -10.0))


   # #### DIV ####
   # # simple add
   @test (@eval_mlir div_test(5.0, 10.0)) == (@eval div_test(5.0, 10.0))
   @test (@eval_mlir div_test(5.0, -10.0)) == (@eval div_test(5.0, -10.0))

    
   # #### REM ####
   @test (@eval_mlir rem(5.0, 10.0)) == (@eval rem(5.0, 10.0))

   # #### POW ####
   @test (@eval_mlir pow(5.0, 10.0)) == (@eval pow(5.0, 10.0))
   @test (@eval_mlir pow(5.0, -10.0)) == (@eval pow(5.0, -10.0))

   ### CONTROL FLOW ###
   # @test (@eval_mlir multi_route_node(5.0, 10.0)) == (@eval multi_route_node(5.0, 10.0)) 
   # @test (@eval_mlir multi_route_node(10.5, 5.2)) == (@eval multi_route_node(10.5, 5.2)) 


   ## MATRIX OPERATION TEST ##
   @test (@eval_mlir create_mat()) == (@eval create_mat())
   @test (@eval_mlir modify_mat(2, 3, 5)) == (@eval modify_mat(2, 3, 5))
   @test (@eval_mlir set_and_get_mat(2, 3, 2, 2, 5)) == (@eval set_and_get_mat(2, 3, 2, 2, 5))
   @test (@eval_mlir add_mat_test()) == (@eval add_mat_test())
   @test (@eval_mlir sub_mat_test()) == (@eval sub_mat_test())
   @test (@eval_mlir matmul_test()) == (@eval matmul_test())

   @test (@eval_mlir create_mat_float()) == (@eval create_mat_float())
   # @test (@eval_mlir modify_mat_float(2, 3, 5)) == (@eval modify_mat_float(2, 3, 5))
   # @test (@eval_mlir set_and_get_mat_float(2, 3, 2, 2, 5)) == (@eval set_and_get_mat_float(2, 3, 2, 2, 5))
   @test (@eval_mlir add_mat_float_test()) == (@eval add_mat_float_test())
   @test (@eval_mlir sub_mat_float_test()) == (@eval sub_mat_float_test())
   @test (@eval_mlir matmul_float_test()) == (@eval matmul_float_test())
   @test (@eval_mlir modify_mat_float(2, 3, 5.0)) == (@eval modify_mat_float(2, 3, 5.0))
   @test (@eval_mlir set_and_get_mat_float(2, 3, 2, 2, 5.0)) == (@eval set_and_get_mat_float(2, 3, 2, 2, 5.0))
   @test (@eval_mlir transpose_test()) == (@eval transpose_test())
end
