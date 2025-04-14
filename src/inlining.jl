function run_passes_ipo_safe(
    ci::CodeInfo,
    sv::OptimizationState,
    optimize_until = nothing,  # run all passes by default
)
    __stage__ = 0  # used by @pass
    # NOTE: The pass name MUST be unique for `optimize_until::String` to work
    @pass "convert"   ir = convert_to_ircode(ci, sv)
    @pass "slot2reg"  ir = slot2reg(ir, ci, sv)
    # TODO: Domsorting can produce an updated domtree - no need to recompute here
    @pass "compact 1" ir = compact!(ir)
    @pass "Inlining"  ir = ssa_inlining_pass!(ir, sv.inlining, ci.propagate_inbounds)
    # @timeit "verify 2" verify_ir(ir)
    @pass "compact 2" ir = compact!(ir)
    @pass "SROA"      ir = sroa_pass!(ir, sv.inlining)
    @pass "ADCE"      (ir, made_changes) = adce_pass!(ir, sv.inlining)
    if made_changes
        @pass "compact 3" ir = compact!(ir, true)
    end
    if is_asserts()
        @timeit "verify 3" begin
            verify_ir(ir, true, false, optimizer_lattice(sv.inlining.interp), sv.linfo)
            verify_linetable(ir.debuginfo, length(ir.stmts))
        end
    end
    @label __done__  # used by @pass
    return ir
end
