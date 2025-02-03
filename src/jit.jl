# Integrate JMLIR backedn with an LLVM Orc-based JIT
include("compiler.jl")

using Core.Compiler

module TestRuntime
# dummy methods
signal_exception() = return
malloc(sz) = C_NULL
report_oom(sz) = return
report_exception(ex) = return
report_exception_name(ex) = return
report_exception_frame(idx, func, file, line) = return
end

# struct TestCompilerParams <: Core.Compiler.AbstractCompilerParams end
# GPUCompiler.runtime_module(::CompilerJob{<:Any,TestCompilerParams}) = TestRuntime


## JIT integration

using LLVM, LLVM.Interop

function absolute_symbol_materialization(name, ptr)
    address = LLVM.API.LLVMOrcJITTargetAddress(reinterpret(UInt, ptr))
    flags = LLVM.API.LLVMJITSymbolFlags(LLVM.API.LLVMJITSymbolGenericFlagsExported, 0)
    symbol = LLVM.API.LLVMJITEvaluatedSymbol(address, flags)
    gv = LLVM.API.LLVMOrcCSymbolMapPair(name, symbol)

    return LLVM.absolute_symbols(Ref(gv))
end

function define_absolute_symbol(jd, name)
    ptr = LLVM.find_symbol(name)
    if ptr !== C_NULL
        LLVM.define(jd, absolute_symbol_materialization(name, ptr))
        return true
    end
    return false
end

struct CompilerInstance
    jit::LLVM.LLJIT
    lctm::LLVM.LazyCallThroughManager
    ism::LLVM.IndirectStubsManager
end
const jit = Ref{CompilerInstance}()

function get_trampoline(job)
    compiler = jit[]
    lljit = compiler.jit
    lctm = compiler.lctm
    ism = compiler.ism

    # We could also use one dylib per job
    jd = JITDylib(lljit)

    entry_sym = String(gensym(:entry))
    target_sym = String(gensym(:target))
    flags = LLVM.API.LLVMJITSymbolFlags(
        LLVM.API.LLVMJITSymbolGenericFlagsCallable |
        LLVM.API.LLVMJITSymbolGenericFlagsExported, 0)
    entry = LLVM.API.LLVMOrcCSymbolAliasMapPair(
        mangle(lljit, entry_sym),
        LLVM.API.LLVMOrcCSymbolAliasMapEntry(
            mangle(lljit, target_sym), flags))

    mu = LLVM.reexports(lctm, ism, jd, Ref(entry))
    LLVM.define(jd, mu)

    # 2. Lookup address of entry symbol
    addr = lookup(lljit, entry_sym)

    # 3. add MU that will call back into the compiler
    sym = LLVM.API.LLVMOrcCSymbolFlagsMapPair(mangle(lljit, target_sym), flags)

    function materialize(mr)
        buf = JuliaContext() do ctx
            ir, meta = GPUCompiler.compile(:llvm, job; validate=false)

            # Rename entry to match target_sym
            LLVM.name!(meta.entry, target_sym)

            # So 1. serialize the module
            buf = convert(MemoryBuffer, ir)

            # 2. deserialize and wrap by a ThreadSafeModule
            ThreadSafeContext() do ts_ctx
                tsm = context!(context(ts_ctx)) do
                    mod = parse(LLVM.Module, buf)
                    ThreadSafeModule(mod)
                end

                il = LLVM.IRTransformLayer(lljit)
                LLVM.emit(il, mr, tsm)
            end
        end

        return nothing
    end

    function discard(jd, sym)
    end

    mu = LLVM.CustomMaterializationUnit(entry_sym, Ref(sym), materialize, discard)
    LLVM.define(jd, mu)
    return addr
end

@generated function deferred_codegen(f::F, ::Val{tt}, ::Val{world}) where {F,tt,world}
    # manual version of native_job because we have a function type
    source = methodinstance(F, Base.to_tuple_type(tt), world)
    target = NativeCompilerTarget(; jlruntime=true, llvm_always_inline=true)
    # XXX: do we actually require the Julia runtime?
    #      with jlruntime=false, we reach an unreachable.
    params = TestCompilerParams()
    config = CompilerConfig(target, params; kernel=false)
    job = CompilerJob(source, config, world)
    # XXX: invoking GPUCompiler from a generated function is not allowed!
    #      for things to work, we need to forward the correct world, at least.

    addr = get_trampoline(job)
    trampoline = pointer(addr)
    id = Base.reinterpret(Int, trampoline)

    quote
        ptr = ccall("extern deferred_codegen", llvmcall, Ptr{Cvoid}, (Ptr{Cvoid},), $trampoline)
        assume(ptr != C_NULL)
        return ptr
    end
end

@generated function abi_call(f::Ptr{Cvoid}, rt::Type{RT}, tt::Type{T}, func::F, args::Vararg{Any,N}) where {T,RT,F,N}
    argtt = tt.parameters[1]
    rettype = rt.parameters[1]
    argtypes = DataType[argtt.parameters...]

    argexprs = Union{Expr,Symbol}[]
    ccall_types = DataType[]

    before = :()
    after = :(ret)

    # Note this follows: emit_call_specfun_other
    JuliaContext() do ctx
        if !isghosttype(F) && !Core.Compiler.isconstType(F)
            isboxed = GPUCompiler.deserves_argbox(F)
            argexpr = :(func)
            if isboxed
                push!(ccall_types, Any)
            else
                et = convert(LLVMType, func)
                if isa(et, LLVM.SequentialType) # et->isAggregateType
                    push!(ccall_types, Ptr{F})
                    argexpr = Expr(:call, GlobalRef(Base, :Ref), argexpr)
                else
                    push!(ccall_types, F)
                end
            end
            push!(argexprs, argexpr)
        end

        T_jlvalue = LLVM.StructType(LLVMType[])
        T_prjlvalue = LLVM.PointerType(T_jlvalue, 10) #= AddressSpace::Tracked =#

        for (source_i, source_typ) in enumerate(argtypes)
            if GPUCompiler.isghosttype(source_typ) || Core.Compiler.isconstType(source_typ)
                continue
            end

            argexpr = :(args[$source_i])

            isboxed = GPUCompiler.deserves_argbox(source_typ)
            et = isboxed ? T_prjlvalue : convert(LLVMType, source_typ)

            if isboxed
                push!(ccall_types, Any)
            elseif isa(et, LLVM.SequentialType) # et->isAggregateType
                push!(ccall_types, Ptr{source_typ})
                argexpr = Expr(:call, GlobalRef(Base, :Ref), argexpr)
            else
                push!(ccall_types, source_typ)
            end
            push!(argexprs, argexpr)
        end

        if GPUCompiler.isghosttype(rettype) || Core.Compiler.isconstType(rettype)
            # Do nothing...
            # In theory we could set `rettype` to `T_void`, but ccall will do that for us
            # elseif jl_is_uniontype?
        elseif !GPUCompiler.deserves_retbox(rettype)
            rt = convert(LLVMType, rettype)
            if !isa(rt, LLVM.VoidType) && GPUCompiler.deserves_sret(rettype, rt)
                before = :(sret = Ref{$rettype}())
                pushfirst!(argexprs, :(sret))
                pushfirst!(ccall_types, Ptr{rettype})
                rettype = Nothing
                after = :(sret[])
            end
        else
            # rt = T_prjlvalue
        end
    end

    quote
        $before
        ret = ccall(f, $rettype, ($(ccall_types...),), $(argexprs...))
        $after
    end
end

@inline function call_delayed(f::F, args...) where {F}
    tt = Tuple{map(Core.Typeof, args)...}
    rt = Core.Compiler.return_type(f, tt)
    world = GPUCompiler.tls_world_age()
    ptr = deferred_codegen(f, Val(tt), Val(world))
    abi_call(ptr, rt, tt, f, args...)
end

optlevel = LLVM.API.LLVMCodeGenLevelDefault
tm = GPUCompiler.JITTargetMachine(optlevel=optlevel)
LLVM.asm_verbosity!(tm, true)

lljit = LLJIT(; tm)

jd_main = JITDylib(lljit)

prefix = LLVM.get_prefix(lljit)
dg = LLVM.CreateDynamicLibrarySearchGeneratorForProcess(prefix)
add!(jd_main, dg)
if Sys.iswindows() && Int === Int64
    # TODO can we check isGNU?
    define_absolute_symbol(jd_main, mangle(lljit, "___chkstk_ms"))
end

es = ExecutionSession(lljit)

lctm = LLVM.LocalLazyCallThroughManager(triple(lljit), es)
ism = LLVM.LocalIndirectStubsManager(triple(lljit))

jit[] = CompilerInstance(lljit, lctm, ism)
atexit() do
    ci = jit[]
    dispose(ci.ism)
    dispose(ci.lctm)
    dispose(ci.jit)
end



