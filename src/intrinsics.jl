include("common_types.jl")
include("mapping.jl")
# using MLIR

# generic check is fop is registered as a math function
function is_math(fop)::Bool
    idx = reinterpret(Int32, fop) + 1
    return Core.Compiler.T_IFUNC[idx][end] == Core.Compiler.math_tfunc
end


# generic check is fop is registered as a math function
function is_conversion(fop)::Bool
    idx = reinterpret(Int32, fop) + 1
    return Core.Compiler.T_IFUNC[idx][end] == Core.Compiler.conversion_tfunc
end


function is_shift(fop)::Bool
    idx = reinterpret(Int32, fop) + 1
    return Core.Compiler.T_IFUNC[idx][end] == Core.Compiler.shift_tfunc
end


function is_cmp(fop)::Bool
    idx = reinterpret(Int32, fop) + 1
    return Core.Compiler.T_IFUNC[idx][end] == Core.Compiler.cmp_tfunc
end



# process comparator predicates
function cmpi_pred(predicate, isFloat)
    if isFloat
        function (ops...; location=Location())
          return arith.cmpf(ops...; result=IR.Type(Bool), predicate, location)
        end
    else
        function (ops...; location=Location())
          return arith.cmpi(ops...; result=IR.Type(Bool), predicate, location)
        end
    end
end


function linalg_op()
    function (ops...; result, location=Location())
        println("adding linalg add with ops: ", ops)

        # affine_maps = []
        
        # for op in ops 
        #     push!(affine_maps, MLIR.API.mlirAffineMapAttrGet(op))
        # end
        # func @add(%arg0: tensor<5xi64>, %arg1: tensor<5xi64>) -> tensor<5xi64> {
        #   // Create an output tensor of the same shape.
        #   %init = linalg.init_tensor [5] : tensor<5xi64>
          
        #   // linalg.add takes two inputs and an output, with a region that performs the addition.
        #   %result = linalg.add(%arg0, %arg1, %init) {operand_segment_sizes = [1, 1, 1]} {
        #   ^bb0(%a: i64, %b: i64):         // Block arguments: scalar elements from each tensor.
        #     %sum = arith.addi %a, %b : i64  // Element-wise addition.
        #     linalg.yield %sum : i64         // Yield the computed sum.
        #   }
        #   return %result : tensor<5xi64>
        # }

        # return value_world = linalg.generic(ops...; result_tensors=[result])
        #
        
        # inner_block = let
        #     # unranked_tensor_type = MLIRType(T, ())
        #     args_size = length(ops)
            
        #     locs::Vector{Location} = []
        #     for _ in args_size
        #         push!(locs, Location())
        #     end

# #             ops_processed = collect(x -> x, ops)

# #             ops_type::Vector{IR.Type} = []
# #             # println("TYPES: ", ops_processed, " with type: ", typeof(ops_processed))



# #             for i in args_size 
# #                 push!(ops_type, type(ops[i]))
# #             end

        #     inner_block = Block(
        #         [IR.Type(Int64), IR.Type(Int64)],
        #         [Location(), Location()],
        #     )
        #     println("HERE1")
        #     lhs = IR.argument(inner_block, 1)
        #     rhs = IR.argument(inner_block, 2)
        #     println("here2", typeof(rhs))

        #     # lhs_v = IR.Value(lhs)
        #     # rhs_v = IR.Value(rhs)
        #     println("here3")
        #     # arith.addi(lhs, rhs)
        #     println("here4")

        #     fop! = single_op_wrapper(arith.addi)

        #     res = IR.result(fop!(inner_block, [lhs, rhs]; result=IR.Type(Int64)))

        #     println("HERERERE", IR.is_op_res(res))
            
        #     push!(inner_block, linalg.yield([res]))
        #     # out = get_result(maxop, 1)
        #     # push!(inner_block, stablehlo.return_(mlir_ctx, out; loc=@loc()))
        #     inner_block
        # end
        # println("CREATED INNER BLOCK")
        # region = IR.Region()
        # push!(region, inner_block)

        # println("Processing Region: ", region)

        # return linalg.add(ops...; result_tensors=[result], region=region, location)
        return tosa.add(ops...;output=result, location=location)
    end
end


# insert single operations
function single_op_wrapper(fop, target::Function)
    # if fop is a math operation, it needs to forward the return type
    if is_math(target)
        return (block::Block, args::Vector{Value}; result, location=Location()) ->
        push!(block, fop(args...; result, location))
    elseif is_shift(target)
        return (block::Block, args::Vector{Value}; result, location=Location()) ->
        push!(block, fop(args...; result, location))
    elseif is_conversion(target)
        return (block::Block, args::Vector{Value}; result, location=Location()) ->
        push!(block, fop(args...; out=result, location))       
    elseif is_cmp(target)
        return (block::Block, args::Vector{Value}; result, location=Location()) ->
        push!(block, fop(args...; location))       
   else
        return (block::Block, args::Vector{Value}; result, location=Location()) ->
        push!(block, fop(args...; location))
    end
end


function single_op_wrapper(fop)
    return (block::Block, args::Vector{Value}; result, location=Location()) ->
        push!(block, fop(args...; result, location))
end

## conversion to MLIR

# map Julia intrinsics to MLIR
function intrinsic_to_mlir(target_function)
    if target_function in keys(operations)                        # map operations
        return single_op_wrapper(operations[target_function], target_function)
    elseif target_function in keys(int_predicate)                     # operator mappings
        return single_op_wrapper(cmpi_pred(int_predicate[target_function], false), target_function)
    elseif target_function in keys(float_predicate)                     # operator mappings
        return single_op_wrapper(cmpi_pred(float_predicate[target_function], true), target_function)
    elseif target_function in keys(custom_intrinsics)             # custom intrinsics
        return custom_intrinsics[target_function]
    else
        println("adding linalg_op")
        fop = linalg_op()
        return (block::Block, args; result, location=Location()) ->
        push!(block, fop(args...; result=result, location))
    end

    # println("ismath: ", is_math(target_function))

    error("Intrinsic cannot be mapped to MLIR: $target_function. Please update 'mapping.jl' create a Pull Request")
end
