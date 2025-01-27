if occursin("-DEV", string(VERSION))
    const CC = Base
else
    const CC = Core.Compiler
end

# import Core.Compiler

# function inline_cost(ir::IRCode, params::OptimizationParams, cost_threshold::Int)
#     println("using custom cost")
#     return 1
# end

@noinline function test2(a, b)
    return a + b
end

function test(a, b)
    return test2(a, b)
end


"""
    NativeInterpreter <: AbstractInterpreter

This represents Julia's native type inference algorithm and the Julia-LLVM codegen backend.
"""
struct MLIRInterpreter <: CC.AbstractInterpreter
    # The world age we're working inside of
    world::UInt

    # method table to lookup for during inference on this world age
    method_table::CC.CachedMethodTable{CC.InternalMethodTable}

    # Cache of inference results for this particular interpreter
    inf_cache::Vector{CC.InferenceResult}
    codegen::IdDict{CC.CodeInstance,CC.CodeInfo}

    # Parameters for inference and optimization
    inf_params::CC.InferenceParams
    opt_params::CC.OptimizationParams
end




function MLIRInterpreter(world::UInt=CC.get_world_counter();
    inf_params::CC.InferenceParams=CC.InferenceParams(),
    opt_params::CC.OptimizationParams=CC.OptimizationParams())
    curr_max_world = CC.get_world_counter()
    # Sometimes the caller is lazy and passes typemax(UInt).
    # we cap it to the current world age for correctness
    if world == typemax(UInt)
        world = curr_max_world
    end
    # If they didn't pass typemax(UInt) but passed something more subtly
    # incorrect, fail out loudly.
    @assert world <= curr_max_world
    method_table = CC.CachedMethodTable(CC.InternalMethodTable(world))
    inf_cache = Vector{CC.InferenceResult}() # Initially empty cache
    codegen = IdDict{CC.CodeInstance,CC.CodeInfo}()
    return MLIRInterpreter(world, method_table, inf_cache, codegen, inf_params, opt_params)
end

# Quickly and easily satisfy the AbstractInterpreter API contract
CC.InferenceParams(interp::MLIRInterpreter) = interp.inf_params
CC.OptimizationParams(interp::MLIRInterpreter) = interp.opt_params
CC.get_inference_world(interp::MLIRInterpreter) = interp.world
CC.get_inference_cache(interp::MLIRInterpreter) = interp.inf_cache
CC.cache_owner(interp::MLIRInterpreter) = nothing


# get_interpreter(@nospecialize(job::CC.CompilerJob)) =
#     MLIRInterpreter(job.world; method_table=method_table(job),
#                    token=ci_cache_token(job), inf_params=inference_params(job),
#                    opt_params=optimization_params(job))
interp = MLIRInterpreter()


"""
    finish(interp::AbstractInterpreter, opt::OptimizationState,
           ir::IRCode, caller::InferenceResult)

Post-process information derived by Julia-level optimizations for later use.
In particular, this function determines the inlineability of the optimized code.
"""
function CC.finish(interp::CC.AbstractInterpreter, opt::CC.OptimizationState,
    ir::CC.IRCode, caller::CC.InferenceResult)
    println("using: ", typeof(interp))
    # (; src, linfo) = opt
    # (; def, specTypes) = linfo

    # force_noinline = is_declared_noinline(src)

    # # compute inlining and other related optimizations
    # result = caller.result
    # @assert !(result isa LimitedAccuracy)
    # result = widenslotwrapper(result)

    # opt.ir = ir

    # # determine and cache inlineability
    # if !force_noinline
    #     sig = unwrap_unionall(specTypes)
    #     if !(isa(sig, DataType) && sig.name === Tuple.name)
    #         force_noinline = true
    #     end
    #     if !is_declared_inline(src) && result === Bottom
    #         force_noinline = true
    #     end
    # end
    # if force_noinline
    #     set_inlineable!(src, false)
    # elseif isa(def, Method)
    #     if is_declared_inline(src) && isdispatchtuple(specTypes)
    #         # obey @inline declaration if a dispatch barrier would not help
    #         set_inlineable!(src, true)
    #     else
    #         # compute the cost (size) of inlining this code
    #         params = OptimizationParams(interp)
    #         cost_threshold = default = params.inline_cost_threshold
    #         if ⊑(optimizer_lattice(interp), result, Tuple) && !isconcretetype(widenconst(result))
    #             cost_threshold += params.inline_tupleret_bonus
    #         end
    #         # if the method is declared as `@inline`, increase the cost threshold 20x
    #         if is_declared_inline(src)
    #             cost_threshold += 19*default
    #         end
    #         # a few functions get special treatment
    #         if def.module === _topmod(def.module)
    #             name = def.name
    #             if name === :iterate || name === :unsafe_convert || name === :cconvert
    #                 cost_threshold += 4*default
    #             end
    #         end
    #         src.inlining_cost = inline_cost(ir, params, cost_threshold)
    #     end
    # end
    return nothing
end


println("starting job with type: ", typeof(interp))
result = CC.code_ircode(test, Tuple{Int,Int}; interp=interp)

println(result)
