include("compiler.jl")

"""
    MLIRInterpreter <: AbstractInterpreter

MLIRInterpreter implements custom behaviour for the MLIR hardware compilation pipeline
without modifying default behaviour
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



# default constructor
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

# Satisfy the AbstractInterpreter API contract
CC.InferenceParams(interp::MLIRInterpreter) = interp.inf_params
CC.OptimizationParams(interp::MLIRInterpreter) = interp.opt_params
CC.get_inference_world(interp::MLIRInterpreter) = interp.world
CC.get_inference_cache(interp::MLIRInterpreter) = interp.inf_cache
CC.cache_owner(interp::MLIRInterpreter) = nothing


"""
    finish(interp::AbstractInterpreter, opt::OptimizationState,
           ir::IRCode, caller::InferenceResult)

Override the default post-processing functionality to enforce method inlining at the Julia level
"""
function CC.finish(interp::CC.AbstractInterpreter, opt::CC.OptimizationState, ir::CC.IRCode, caller::CC.InferenceResult)
    (; src, linfo) = opt
    (; def, specTypes) = linfo

    # set default for no inlining
    force_noinline = false

    # compute inlining and other related optimizations
    result = caller.result
    @assert !(result isa CC.LimitedAccuracy)
    result = CC.widenslotwrapper(result)

    opt.ir = ir

    # determine edgecases and cache the inlineability
    sig = CC.unwrap_unionall(specTypes)
    if !(isa(sig, DataType) && sig.name === Tuple.name)
        force_noinline = true
    end
    if !CC.is_declared_inline(src) && result === CC.Bottom
        force_noinline = true
    end

    # determine if we inline the method
    if isa(def, Method)
        CC.set_inlineable!(src, !force_noinline)
    end

    return nothing
end


function test(a, b)
    if(a < 0)
        return 0
    else
        return a + b + test(a, b - 1)
    end
end

# example usage:
interp = MLIRInterpreter()
result = CC.code_ircode(test, Tuple{Int,Int}; interp=interp)
println(result)
