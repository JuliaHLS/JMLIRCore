using JMLIRCore
using Test


function add_test(a, b)
    return a + b
end


function sub_test(a, b)
    return a - b
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
   # @test (@eval_mlir add_test(UInt(5), UInt(10))) == (@eval add_test(UInt(5), UInt(10)))

   # @test (@eval_mlir add_test(UInt(9223372036854775807), UInt(10))) == (@eval add_test(UInt(9223372036854775807), UInt(10)))
   # @test (@eval_mlir add_test(UInt(123456789), UInt(12345678909876))) == (@eval add_test(UInt(123456789), UInt(12345678909876)))

   # #### SUB ####
   # # simple add
   @test (@eval_mlir sub_test(5, 10)) == (@eval sub_test(5, 10))
   @test (@eval_mlir sub_test(5, -10)) == (@eval sub_test(5, -10))

   # # testing equivalence of int32 and int64
   @test (@eval_mlir sub_test(9223372036854775807, 10)) == (@eval sub_test(9223372036854775807, 10))
   @test (@eval_mlir sub_test(123456789, 12345678909876)) == (@eval sub_test(123456789, 12345678909876))
   @test (@eval_mlir sub_test(123456789, -12345678909876)) == (@eval sub_test(123456789, -12345678909876))


   ### CONTROL FLOW ###
   @test (@eval_mlir multi_route_node(5, 10)) == (@eval multi_route_node(5, 10)) 
   @test (@eval_mlir multi_route_node(10, 5)) == (@eval multi_route_node(10, 5)) 

end
