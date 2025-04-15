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
        # TODO: add support for more operations
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

struct MethodDetails
    sym::Symbol
    sig::Any
    rettype::DataType
end

function clean_mangled_symbol(sym::Symbol)::Symbol
    cleaned = lstrip(String(sym), '#')
    return Symbol(cleaned)
end

function MethodDetails(fn::Core.CodeInstance)::MethodDetails
    return MethodDetails(clean_mangled_symbol(fn.def.def.name), fn.def.def.sig, fn.rettype)
end

function generate_mlir(op, rettype, sig)
    error("Error: no mlir translation defined for $op")
end


function generate_mlir(::Val{:+}, rettype::Type{<:MVector}, sig::Any)
    println("HIT TARGET for MVECTOR")
    function (ops...; result, location=Location())
        return tosa.add(ops...;output=result, location=location)
    end
end


function generate_mlir(::Val{:-}, rettype::Type{<:MVector}, sig::Any)
    println("HIT TARGET for MVECTOR")
    function (ops...; result, location=Location())
        return tosa.sub(ops...;output=result, location=location)
    end
end

function generate_mlir(md::MethodDetails)
    println("Looking for dispatch target: ", (Val(md.sym)), ", ", (Val(md.sig)), ", ", (md.rettype))
    return generate_mlir(Val(md.sym), (md.rettype), md.sig)
end

## conversion to MLIR
test = [:+, :-]

# map Julia intrinsics to MLIR
function intrinsic_to_mlir(target_function)
    println("target_function: ", target_function, " with type: ", typeof(target_function))
    if target_function in keys(operations)                        # map operations
        return single_op_wrapper(operations[target_function], target_function)
    elseif target_function in keys(int_predicate)                     # operator mappings
        return single_op_wrapper(cmpi_pred(int_predicate[target_function], false), target_function)
    elseif target_function in keys(float_predicate)                     # operator mappings
        return single_op_wrapper(cmpi_pred(float_predicate[target_function], true), target_function)
    elseif target_function in keys(custom_intrinsics)             # custom intrinsics
        return custom_intrinsics[target_function]
    else
        println("mi: ", target_function.def, " with type: ", typeof(target_function.def))
        println("mi_def: ", target_function.def.def, " with type: ", typeof(target_function.def.def))
        println("eq: ", target_function.def.def.name in test)
        println("name: ", target_function.def.def.name, " of type: ", typeof(target_function.def.def.name))
        println("sig: ", target_function.def.def.sig, " of type: ", typeof(target_function.def.def.sig))
        println("sig: ", target_function.rettype, " of type: ", typeof(target_function.rettype))
        # println("sig: ", target_function.sig, " of type: ", typeof(target_function.sig))
        println("eq2: ", target_function.def.def.name === :-)
        println("against: ", :-)
        println("cleaned: ", clean_mangled_symbol(target_function.def.def.name) == :-)
        # fop = linalg_op()

        md = MethodDetails(target_function)
        println(typeof(md.sym), ", ", typeof(md.sig), ", ", typeof(md.rettype))
        # generate_mlir(Val(md.sym), md.rettype, md.sig)
        fop = generate_mlir(md)

        return (block::Block, args; result, location=Location()) ->
        push!(block, fop(args...; result=result, location))
    end

    error("Intrinsic cannot be mapped to MLIR: $target_function. Please update 'mapping.jl' create a Pull Request")
end
