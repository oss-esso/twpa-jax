def test_pump_backend_imports():
    from twpa_solver.pump.backends import (
        SchurPartition,
        SchurReducedProblem,
        FastCoupledPreconditioner,
        analytic_jvp,
        assemble_schur_complements,
        back_substitute_full,
        build_partition,
        build_schur_problem,
        fd_jvp,
        jax_available,
        jvp_relative_error,
        pardiso_available,
        reduced_linear_apply,
        restrict,
    )

    assert SchurPartition is not None
    assert SchurReducedProblem is not None
    assert FastCoupledPreconditioner is not None
    assert callable(analytic_jvp)
    assert callable(assemble_schur_complements)
    assert callable(back_substitute_full)
    assert callable(build_partition)
    assert callable(build_schur_problem)
    assert callable(fd_jvp)
    assert callable(jax_available)
    assert callable(jvp_relative_error)
    assert callable(pardiso_available)
    assert callable(reduced_linear_apply)
    assert callable(restrict)
