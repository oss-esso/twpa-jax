def test_signal_imports():
    from twpa_solver.signal.gain import (
        GainResult,
        complex_to_pair,
        db10,
        db20,
        gain_db_from_s,
    )
    from twpa_solver.signal.io import (
        PumpSolution,
        infer_circuit_dir_from_pump_report,
        load_pump,
        write_outputs,
    )
    from twpa_solver.signal.gamma import (
        build_khat,
        compute_gamma_hat,
        load_dc_branch_flux,
        synthesize_real_from_positive_harmonics,
        write_gamma_hat_summary,
    )
    from twpa_solver.signal.floquet import (
        assemble_conversion_matrix,
        assemble_conversion_matrix_from_base,
        assemble_khat_conversion_base,
        sideband_list,
        solve_gain_one,
        solve_gain_one_schur,
        solve_linear_system,
        voltage_from_flux,
    )

    assert GainResult is not None
    assert PumpSolution is not None
    assert callable(complex_to_pair)
    assert callable(db10)
    assert callable(db20)
    assert callable(gain_db_from_s)
    assert callable(infer_circuit_dir_from_pump_report)
    assert callable(load_pump)
    assert callable(write_outputs)
    assert callable(build_khat)
    assert callable(compute_gamma_hat)
    assert callable(load_dc_branch_flux)
    assert callable(synthesize_real_from_positive_harmonics)
    assert callable(write_gamma_hat_summary)
    assert callable(assemble_conversion_matrix)
    assert callable(assemble_conversion_matrix_from_base)
    assert callable(assemble_khat_conversion_base)
    assert callable(sideband_list)
    assert callable(solve_gain_one)
    assert callable(solve_gain_one_schur)
    assert callable(solve_linear_system)
    assert callable(voltage_from_flux)
