include("compiler.jl")
include("overlays.jl")
include("intrinsics.jl")

import Core
using Core.Compiler

import .Core.Compiler: CallInfo

"""
    MLIRInterpreter <: AbstractInterpreter

MLIRInterpreter implements custom behaviour for the MLIR hardware compilation pipeline
without modifying default behaviour
"""
struct MLIRInterpreter <: Core.Compiler.AbstractInterpreter
    # The world age we're working inside of
    world::UInt

    # Cache of inference results for this particular interpreter
    inf_cache::Vector{Core.Compiler.InferenceResult}
    codegen::IdDict{CC.CodeInstance,CC.CodeInfo}

    # Parameters for inference and optimization
    inf_params::Core.Compiler.InferenceParams
    opt_params::Core.Compiler.OptimizationParams
end


""" Default Constructor """
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

    inf_cache = Vector{Core.Compiler.InferenceResult}() # Initially empty cache
    codegen = IdDict{CC.CodeInstance,CC.CodeInfo}()

    return MLIRInterpreter(world, inf_cache, codegen, inf_params, opt_params)
end


# Satisfy the AbstractInterpreter API contract
Core.Compiler.InferenceParams(interp::MLIRInterpreter) = interp.inf_params
Core.Compiler.OptimizationParams(interp::MLIRInterpreter) = interp.opt_params
Core.Compiler.get_inference_world(interp::MLIRInterpreter) = interp.world
Core.Compiler.get_inference_cache(interp::MLIRInterpreter) = interp.inf_cache
Core.Compiler.cache_owner(interp::MLIRInterpreter) = nothing

# Set custom method table bindings
Compiler.method_table(interp::MLIRInterpreter) = Compiler.OverlayMethodTable(Compiler.get_inference_world(interp), MLIROverlays.MLIR_MT)

"""
    Custom Inlining Mechanism

Implemented in the Abstract Interpreter for robustness
"""

struct NoinlineCallInfo <: CallInfo
    info::CallInfo  # wrapped call info
end

# add edges
Compiler.add_edges_impl(edges::Vector{Any}, info::NoinlineCallInfo) = Compiler.add_edges!(edges, info.info)
Compiler.nsplit_impl(info::NoinlineCallInfo) = Compiler.nsplit(info.info)
Compiler.getsplit_impl(info::NoinlineCallInfo, idx::Int) = Compiler.getsplit(info.info, idx)
Compiler.getresult_impl(info::NoinlineCallInfo, idx::Int) = Compiler.getresult(info.info, idx)



# TODO: can I simplify this, given that they are an intrinsic_type?
const NOINLINE_OPERATORS = Set([Base.:+, Base.:-, Base.:*, Base.:/, Base.:<, Base.:>, Base.:(==), Base.:≤, Base.:≥, Base.:≠, StaticArrays.construct_type, Base.setindex!, Base.getindex, Base.:(===)])
""" Tag abstract calls with NoinlineCallInfo when needed """
function Compiler.abstract_call(interp::MLIRInterpreter, arginfo::Compiler.ArgInfo, si::Compiler.StmtInfo, sv::Compiler.InferenceState, max_methods::Int)

    ret = @invoke Compiler.abstract_call(interp::Compiler.AbstractInterpreter, arginfo::Compiler.ArgInfo, si::Compiler.StmtInfo, sv::Compiler.InferenceState, max_methods::Int)

    return Compiler.Future{Compiler.CallMeta}(ret, interp, sv) do ret, interp, sv
        println("arginfo: $arginfo")
        if first(arginfo.argtypes) isa Core.Const && first(arginfo.argtypes).val in NOINLINE_OPERATORS
            (; rt, exct, effects, info) = ret
            return Compiler.CallMeta(rt, exct, effects, NoinlineCallInfo(info))
        end
        return ret
    end
end


""" Custom inlining policy """
function Compiler.src_inlining_policy(interp::MLIRInterpreter,
    @nospecialize(src), @nospecialize(info::CallInfo), stmt_flag::UInt32)

    # don't inline tagged items
    if isa(info, NoinlineCallInfo)
        return false
    end
    
    # else invoke default policy
    return @invoke Compiler.src_inlining_policy(interp::Compiler.AbstractInterpreter,
        src::Any, info::CallInfo, stmt_flag::UInt32)
end


function Core.Compiler.finish(interp::MLIRInterpreter, opt::Core.Compiler.OptimizationState, ir::Core.Compiler.IRCode, caller::Core.Compiler.InferenceResult)
    (; src, linfo) = opt
    (; def, specTypes) = linfo

    # set default for no inlining
    force_noinline = false

    # compute inlining and other related optimizations
    result = caller.result
    @assert !(result isa Core.Compiler.LimitedAccuracy)
    result = Core.Compiler.widenslotwrapper(result)

    opt.ir = ir

    # determine edgecases and cache the inlineability
    sig = CC.unwrap_unionall(specTypes)
    if !(isa(sig, DataType) && sig.name === Tuple.name)
        force_noinline = true
    end
    if !Core.Compiler.is_declared_inline(src) && result === Core.Compiler.Bottom
        force_noinline = true
    end

    # determine if we inline the method
    if isa(def, Method)
        Core.Compiler.set_inlineable!(src, !force_noinline)
    end

    return nothing
end

