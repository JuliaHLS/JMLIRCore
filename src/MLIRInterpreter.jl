include("compiler.jl")

"""
    MLIRInterpreter <: AbstractInterpreter

MLIRInterpreter implements custom behaviour for the MLIR hardware compilation pipeline
without modifying default behaviour
"""
struct MLIRInterpreter <: Core.Compiler.AbstractInterpreter
    # The world age we're working inside of
    world::UInt

    # method table to lookup for during inference on this world age
    method_table::Core.Compiler.CachedMethodTable{Core.Compiler.InternalMethodTable}

    # Cache of inference results for this particular interpreter
    inf_cache::Vector{Core.Compiler.InferenceResult}
    codegen::IdDict{CC.CodeInstance,CC.CodeInfo}

    # Parameters for inference and optimization
    inf_params::Core.Compiler.InferenceParams
    opt_params::Core.Compiler.OptimizationParams
end



# default constructor
function MLIRInterpreter(world::UInt=CC.get_world_counter();
    inf_params::Core.Compiler.InferenceParams=Core.Compiler.InferenceParams(),
    opt_params::Core.Compiler.OptimizationParams=Core.Compiler.OptimizationParams())
    curr_max_world = CC.get_world_counter()

    # Sometimes the caller is lazy and passes typemax(UInt).
    # we cap it to the current world age for correctness
    if world == typemax(UInt)
        world = curr_max_world
    end

    # If they didn't pass typemax(UInt) but passed something more subtly
    # incorrect, fail out loudly.
    @assert world <= curr_max_world

    method_table = Core.Compiler.CachedMethodTable(Core.Compiler.InternalMethodTable(world))
    inf_cache = Vector{Core.Compiler.InferenceResult}() # Initially empty cache
    codegen = IdDict{CC.CodeInstance,CC.CodeInfo}()

    return MLIRInterpreter(world, method_table, inf_cache, codegen, inf_params, opt_params)
end

# Satisfy the AbstractInterpreter API contract
Core.Compiler.InferenceParams(interp::MLIRInterpreter) = interp.inf_params
Core.Compiler.OptimizationParams(interp::MLIRInterpreter) = interp.opt_params
Core.Compiler.get_inference_world(interp::MLIRInterpreter) = interp.world
Core.Compiler.get_inference_cache(interp::MLIRInterpreter) = interp.inf_cache
Core.Compiler.cache_owner(interp::MLIRInterpreter) = nothing


"""
    finish(interp::AbstractInterpreter, opt::OptimizationState,
           ir::IRCode, caller::InferenceResult)

Override the default post-processing functionality to enforce method inlining at the Julia level
"""
function Core.Compiler.finish(interp::Core.Compiler.AbstractInterpreter, opt::Core.Compiler.OptimizationState, ir::Core.Compiler.IRCode, caller::Core.Compiler.InferenceResult)
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

