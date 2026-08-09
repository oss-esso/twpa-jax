# Python File Catalogue

Scope: every Python file found by `rg --files -g '*.py'`; non-Python files are excluded.

The catalogue is split into two evidence-based groups:

- **Current / reworked stack:** `src/`, plus scripts, experiments, notebooks, and tests that import the current `twpa_solver` package.
- **Legacy / historical / auxiliary:** `twpa_solver_old/`, the parallel `twpa/` package, and Python files without a current-stack dependency.

This is an inventory classification, not a deletion recommendation. Functionality descriptions are based on module docstrings, imports, and top-level definitions.

Total Python files inventoried: **514**.
### experiments/


### notebooks/

#### Legacy/Historical/Auxiliary (1 files)

| File | Functionality / observed role | Top-level symbols |
|---|---|---|
| notebooks/09_production_100mm_demo.py | Module; defines jsonify, write_json, write_npz, print_section, safe_float, try_make_gain_plan | jsonify, write_json, write_npz, print_section, safe_float, try_make_gain_plan |
### scripts/



#### Legacy/Historical/Auxiliary (98 files)

| File | Functionality / observed role | Top-level symbols |
|---|---|---|

| scripts/build_harmonia_ethz_jtl_linear_dataset.py | Module; defines main | main |
| scripts/build_harmonia_jtl_linear_dataset.py | Build an ML-ready dataset from a Harmonia CircuitIR JTL linear campaign | main |
| scripts/build_harmonia_lumped_jpa_linear_dataset.py | Module; defines main | main |
| scripts/build_harmonia_rf_jtl_linear_dataset.py | Python support/entry-point module | - |
| scripts/build_jc_jpa_reflection_dataset.py | Build an ML-ready dataset from a JosephsonCircuits JPA reflection campaign | main |
| scripts/build_linear_sparams_dataset.py | Build an ML-ready dataset from a linear S-parameter campaign registry | main |

| scripts/build_standalone_coupler.py | Build an isolated four-port coupler from an existing IPM design | parse_args, main |


| scripts/debug_power_sweep_col3_p7_p8.py | Fine power-axis sweep between the last-converged cell (point 7) and the first stalled cell (point 8) of the fp=7.329 GHz column (outputs/measurement_match_debug | build_sweep_points, main, _print_row, write_summary, plot_coeff_rel |

| scripts/estimate_hb_resources.py | Estimate dense HB Jacobian storage without allocating numerical arrays. | estimate, main |
| scripts/evaluate_harmonia_ethz_jtl_linear_objective.py | Module; defines _as_str_list, _complex_s, _passivity_penalty, find_nearest_parameter_index, evaluate_dataset_against_target, main | _as_str_list, _complex_s, _passivity_penalty, find_nearest_parameter_index, evaluate_dataset_against_target, main |
| scripts/evaluate_harmonia_jtl_linear_objective.py | Module; defines main | main |
| scripts/evaluate_harmonia_lumped_jpa_linear_objective.py | Module; defines _as_str_list, _complex_s, find_nearest_parameter_index, evaluate_dataset_against_target, main | _as_str_list, _complex_s, find_nearest_parameter_index, evaluate_dataset_against_target, main |
| scripts/evaluate_harmonia_rf_jtl_linear_objective.py | Python support/entry-point module | - |
| scripts/evaluate_jc_jpa_reflection_objective.py | Evaluate the first JosephsonCircuits-backed one-port reflection objective | _decode_names, build_jc_reflection_objective_summary, print_human_summary, main |
| scripts/evaluate_linear_sparams_objective.py | Evaluate the first calibration objective on a linear S-parameter dataset | _decode_names, build_objective_summary, print_human_summary, main |
| scripts/export_bridge_dataset.py | Export a unified TWPA bridge dataset | RunStatus, SourceKind, SourceArtifact, BridgeDatasetConfig, StageResult, BridgeDatasetResult... |
| scripts/extract_dispersion.py | Extract dispersion diagnostics from a TWPA linear response | RunStatus, DispersionMethod, PhaseSign, ExtractDispersionConfig, StageResult, DispersionExtractionResult... |
| scripts/fit_measurements.py | Fit TWPA model parameters to measured gain / S-parameter data | RunStatus, FitMode, MeasurementFitConfig, StageResult, MeasurementFitResult, jsonify... |
| scripts/full_gain_map_100mm.py | Run a full 100 mm TWPA gain-map workflow | RunStatus, FullGainMap100mmConfig, GainMapPointResult, FullGainMap100mmResult, jsonify, nested_get... |
| scripts/full_pump_hb_100mm.py | Run the full 100 mm pump harmonic-balance workflow | RunStatus, FullPumpMode, FullPump100mmConfig, StageResult, WarmupResult, FullPump100mmResult... |
| scripts/gain_from_pumped_solution.py | Compute small-signal gain from a pumped TWPA solution | RunStatus, GainSolverMode, GainFromPumpConfig, StageResult, GainFromPumpResult, jsonify... |
| scripts/inspect_existing_ipm_pumpsweep_h5.py | Python support/entry-point module | - |
| scripts/inspect_wall_time_timing_schema.py | Python support/entry-point module | - |
| scripts/inventory_harmonia_workspace.py | Inventory both Harmonia Julia repos | JuliaFileRecord, read_text_safe, count_keyword_hits, extract_functions, extract_imports, extract_includes... |
| scripts/ipm_experiment.py | Module; defines IPMTWPAParams, db_to_power_ratio, db_to_amplitude_ratio, safe_db_power, jj_cap_from_plasma_frequency, josephson_critical_current, jj_series_impe | IPMTWPAParams, db_to_power_ratio, db_to_amplitude_ratio, safe_db_power, jj_cap_from_plasma_frequency, josephson_critical_current... |
| scripts/linear_100mm_baseline.py | Run a pump-off 100 mm linear baseline simulation | RunStatus, Linear100mmConfig, StageResult, Linear100mmResult, jsonify, run_stage... |
| scripts/make_corrected_wall_time_accounting.py | Python support/entry-point module | - |
| scripts/make_direct_linear_backend_report.py | Python support/entry-point module | - |
| scripts/make_final_closed_wall_time_accounting.py | Python support/entry-point module | - |
| scripts/make_run_report.py | Create a consolidated TWPA run report | RunStatus, RunReportConfig, SummaryRecord, MarkdownRecord, RunReportResult, jsonify... |
| scripts/make_wall_time_budget_coarse_report.py | Python support/entry-point module | - |
| scripts/make_wall_time_five_block_probe.py | Python support/entry-point module | - |
| scripts/make_wall_time_five_block_ws_matrix.py | Python support/entry-point module | - |
| scripts/map_jc_cache_integration_points.py | Python support/entry-point module | - |
| scripts/map_jc_cache_julia_boundaries.py | Python support/entry-point module | - |




| scripts/plot_jc_old_ipm_gain_map_with_status.py | Python support/entry-point module | - |

| scripts/plot_measurement_gain_vs_pumppower_signalfreq.py | Plot raw Themis transmission vs pump power and signal frequency, at a fixed pump-frequency file | main |
| scripts/plot_old_ipm_map_slide_set.py | Python support/entry-point module | - |


| scripts/plot_report_rf_squid_reproduction.py | Python support/entry-point module | - |
| scripts/plot_rf_jtl_colleague_regime.py | Python support/entry-point module | - |
| scripts/plot_rf_jtl_physics_maps.py | Python support/entry-point module | - |

| scripts/plot_spectrum_compare.py | Overlay signal-gain spectra for the 4 clean (Lj,Cg,scale) designs at one fixed, converged pump point (fp=7.8965517241 GHz), plus the Julia reference if availabl | load_case, load_measurement, main |
| scripts/plot_template_five_block_budget.py | Python support/entry-point module | - |
| scripts/plot_tiny_ipm_schemdraw.py | Draw the tiny IPM probe circuit as a real schematic using schemdraw | build_tiny_circuit, main |
| scripts/plot_tiny_ipm_skrf.py | Draw the tiny IPM probe circuit topology using scikit-rf's Circuit graph | build_tiny_circuit, main |
| scripts/plot_wall_time_budget_one_shot_vs_batch.py | Python support/entry-point module | - |
| scripts/plot_wall_time_budget_templates.py | Python support/entry-point module | - |
| scripts/pump_hb_scaling_study.py | Run a pump-HB scaling study over ladder size and pump drive | RunStatus, ScalingStudyConfig, ScalingRunResult, ScalingStudyResult, jsonify, nested_get... |
| scripts/pump_hb_small_ladder.py | Run a small-ladder pump harmonic-balance simulation | RunStatus, PumpSolverMode, NumericalBackend, PumpHBSmallLadderConfig, StageResult, PumpHBSmallLadderResult... |
| scripts/read_julia_run.py | Inspect a Julia/Harmonia simulation run folder from Python | _json_ready, build_summary, print_human, main |
| scripts/recompress_pump_solutions.py | Recompress existing pump_solution.npz files to float32 + DEFLATE | is_already_optimized, recompress, parse_args, main |
| scripts/register_julia_run.py | Register a Julia/Harmonia run folder in a CSV registry | main |
| scripts/resume_column_force_gain.py | Resume a gain-map frequency column past its convergence wall, forcing gain | _finite_X, march_column, write_column_csv, plot_column, main |
| scripts/run_calibration.py | Python support/entry-point module | - |
| scripts/run_harmonia_benchmark_suite.py | Module; defines BenchmarkCase, utc_timestamp, sha256_file, safe_git_commit, safe_git_status_short, collect_environment, select_cases, _shape_or_none and more | BenchmarkCase, utc_timestamp, sha256_file, safe_git_commit, safe_git_status_short, collect_environment... |
| scripts/run_harmonia_ethz_jtl_linear_campaign.py | Module; defines make_harmonia_ethz_jtl_linear_config, campaign_paths, compute_ethz_jtl_linear_metrics, run_campaign, print_human_summary, main | make_harmonia_ethz_jtl_linear_config, campaign_paths, compute_ethz_jtl_linear_metrics, run_campaign, print_human_summary, main |
| scripts/run_harmonia_jtl_linear_campaign.py | Run a tiny CircuitIR/Harmonia/JosephsonCircuits JTL linear campaign | make_harmonia_jtl_linear_config, compute_jtl_linear_metrics, _jtl_run_name, _result_from_status_path, run_campaign, print_human_summary... |
| scripts/run_harmonia_lumped_jpa_linear_campaign.py | Module; defines make_harmonia_lumped_jpa_linear_config, campaign_paths, compute_lumped_jpa_linear_metrics, run_campaign, print_human_summary, main | make_harmonia_lumped_jpa_linear_config, campaign_paths, compute_lumped_jpa_linear_metrics, run_campaign, print_human_summary, main |
| scripts/run_harmonia_rf_jtl_linear_campaign.py | Module; defines make_config, run_campaign | make_config, run_campaign |
| scripts/run_harmonia_tiny_nonlinear_campaign.py | Module; defines config, run_campaign | config, run_campaign |
| scripts/run_industrial_100mm.py | Python support/entry-point module | - |

| scripts/run_jc_cached_setup_workload.py | Python support/entry-point module | - |
| scripts/run_jc_jpa_reflection_campaign.py | Run a tiny JosephsonCircuits-backed JPA reflection campaign | make_jc_jpa_reflection_config, campaign_paths, compute_one_port_reflection_metrics, run_campaign, print_human_summary, main |
| scripts/run_julia_simulation.py | Launch a Harmonia/JosephsonCircuits Julia simulation from Python | _status_summary, main |
| scripts/run_linear_sparams_campaign.py | Run a tiny linear S-parameter campaign through the Julia/Harmonia engine | make_linear_sparams_config, campaign_paths, compute_2port_metrics, run_campaign, print_human_summary, main |
| scripts/run_linear_validation.py | Python support/entry-point module | - |
| scripts/run_pump_hb.py | Python support/entry-point module | - |
| scripts/run_resource_bounded.py | Run one command with conservative time, memory, and artifact-size limits. | _process_tree_pids, _working_set_bytes, _cpu_seconds, _directory_bytes, _tree_metrics, _terminate... |
| scripts/run_schema_smoke_campaign.py | Run a tiny schema-smoke Julia/Harmonia campaign from Python | make_schema_smoke_config, campaign_paths, run_campaign, print_human_summary, main |
| scripts/run_synthetic_benchmarks.py | Python support/entry-point module | - |
| scripts/run_validation_suite.py | Run the TWPA production validation suite | CheckStatus, CheckResult, _jsonify, run_check, check_imports, check_linear_solvers... |
| scripts/run_wall_time_budget_one_shot_vs_batch.py | Python support/entry-point module | - |




### tests/

#### Current/Reworked Stack (10 files)

| File | Functionality / observed role | Top-level symbols |
|---|---|---|
| tests/test_adaptive_continuation_fallback.py | Regression test for the adaptive-continuation fixed-step fallback | _build_problem, _settings, test_solve_continuation_lambda_start_resumes_span, test_adaptive_continuation_fallback_resumes_not_restarts |
| tests/test_advanced_continuation.py | Tests for the advanced intra-cell continuation methods added to the pump solver: the tangent (Euler) predictor, pseudo-transient continuation, and pseudo-arclen | _build_problem, _settings, _solver, test_tangent_predictor_beats_copy_near_the_branch, test_pseudo_transient_converges_from_zero, test_arclength_reaches_target_lambda... |
| tests/test_floquet_stability.py | Tests for twpa_solver.signal.stability: the Tier-1 Floquet stability proxy | _random_complex_sparse, test_estimate_sigma_min_matches_dense_svd, test_estimate_sigma_min_near_singular_matrix_is_small, test_estimate_sigma_min_is_deterministic_for_fixed_seed, test_local_minima_finds_dip_and_excludes_endpoints, test_local_minima_respects_k_limit... |
| tests/test_loss_model.py | Test module; defines test_frozen_coeffs_match_csv_refit, test_fit_quality_within_tolerance, test_dc_value_is_offset, test_pump_band_matches_old_flat_35db, test_ | test_frozen_coeffs_match_csv_refit, test_fit_quality_within_tolerance, test_dc_value_is_offset, test_pump_band_matches_old_flat_35db, test_attenuation_is_monotonic_increasing, test_scalar_returns_float_array_returns_array... |
| tests/test_plotting_imports.py | Import smoke tests for the plotting package. | test_plotting_modules_import |
| tests/test_plotting_spectrum_residuals.py | Test module; defines test_fit_error_uses_all_finite_signal_samples, test_symmetric_error_edges_use_quarter_db_bins | test_fit_error_uses_all_finite_signal_samples, test_symmetric_error_edges_use_quarter_db_bins |
| tests/test_plotting_status_map.py | Status-map classification tests for saved gain-map outputs. | test_status_label_uses_pump_and_gain_diagnostics, test_fit_metrics_preserve_solver_status_diagnostics |
| tests/test_predictors.py | Unit tests for the inter-cell state predictors (pure functions). | _state, test_copy_returns_independent_copy, test_axis_secant_exact_on_linear_field, test_axis_secant_degenerate_returns_none, test_corner_exact_on_bilinear_index_field, test_corner_missing_neighbour_returns_none... |
| tests/test_prune_map_solutions.py | Test module; defines _load_script, _build_chunked_map, _survivor_keys, test_gain_ranked_prefers_strong_then_falls_back, test_subpath_under_map_slices_after_map_ | _load_script, _build_chunked_map, _survivor_keys, test_gain_ranked_prefers_strong_then_falls_back, test_subpath_under_map_slices_after_map_name, test_keep_paths_disambiguate_colliding_local_indices... |
| tests/test_pump_solution_io.py | Test module; defines _write, test_solution_stored_float32, test_solution_is_compressed, test_roundtrip_loads_complex128_within_float32_tol | _write, test_solution_stored_float32, test_solution_is_compressed, test_roundtrip_loads_complex128_within_float32_tol |

#### Legacy/Historical/Auxiliary (126 files)

| File | Functionality / observed role | Top-level symbols |
|---|---|---|
| tests/conftest.py | Test module; defines pytest_addoption, pytest_configure, pytest_collection_modifyitems | pytest_addoption, pytest_configure, pytest_collection_modifyitems |
| tests/test_align_map.py | Tests for scripts/align_map_to_measurement.py | _ridge, _make_maps, test_recovers_known_shift_l2, test_recovers_zero_shift, test_huber_matches_l2_on_clean_data, test_huber_robust_to_outliers... |
| tests/test_all_backend_script_config.py | Test module; defines test_all_backend_script_exists_and_uses_canonical_builder, test_all_backend_script_default_backends_and_axes | test_all_backend_script_exists_and_uses_canonical_builder, test_all_backend_script_default_backends_and_axes |
| tests/test_backend_adapter_contract.py | Test module; defines test_julia_backend_adapter_contract_exists | test_julia_backend_adapter_contract_exists |
| tests/test_backend_comparison_reference_inventory.py | Test module; defines _write_rows, test_backend_comparison_tiny_fake_5x5 | _write_rows, test_backend_comparison_tiny_fake_5x5 |
| tests/test_backend_rows_incremental_resume.py | Test module; defines test_all_backend_script_has_incremental_resume_artifacts, test_all_backend_script_writes_required_backend_grids | test_all_backend_script_has_incremental_resume_artifacts, test_all_backend_script_writes_required_backend_grids |
| tests/test_backend_status_forbidden_placeholders.py | Test module; defines test_all_independent_backends_return_non_placeholder_status | test_all_independent_backends_return_non_placeholder_status |
| tests/test_builder_imports.py | Python support/entry-point module | - |
| tests/test_calibration_objectives.py | Test module; defines _matched_s, _mismatched_s, test_objective_zero_for_identical_sparameters, test_objective_penalizes_mismatch, test_dataset_evaluation_ranks_ | _matched_s, _mismatched_s, test_objective_zero_for_identical_sparameters, test_objective_penalizes_mismatch, test_dataset_evaluation_ranks_target_first, test_actual_linear_objective_pipeline_if_available |
| tests/test_campaign_batch_runner.py | Python support/entry-point module | - |
| tests/test_cli_and_compat.py | Test module; defines test_cli_parser_exposes_expected_commands, test_harmonics_selected_plan_wrapper_warns, test_selected_harmonic_one_node_wrapper_warns, test_ | test_cli_parser_exposes_expected_commands, test_harmonics_selected_plan_wrapper_warns, test_selected_harmonic_one_node_wrapper_warns, test_selected_harmonic_distributed_wrapper_warns, test_cell_local_block_jacobi_factory_builds_ready_preconditioner |
| tests/test_compression_native_path.py | Test module; defines _config, test_native_point_solver_auto_decision, test_native_finite_signal_compression_point_runs, test_native_wideband_compression_writes_ | _config, test_native_point_solver_auto_decision, test_native_finite_signal_compression_point_runs, test_native_wideband_compression_writes_gain_matrix |
| tests/test_continuation.py | Tests for twpa.solvers.continuation | _call_with_supported_kwargs, _get_attr_or_key, _as_mapping, ToySolverFailure, _make_param_grid, _extract_final_state... |
| tests/test_conversion_matrix.py | Tests for twpa.nonlinear.conversion | _call_with_supported_kwargs, _get_attr_or_key, _as_mapping, _validate_orders, _coeff_mapping, _manual_conversion_matrix... |
| tests/test_dataset_builder.py | Test module; defines test_extract_parameter_vector, test_extract_parameter_vector_rejects_missing_parameter, test_actual_linear_campaign_dataset_if_available | test_extract_parameter_vector, test_extract_parameter_vector_rejects_missing_parameter, test_actual_linear_campaign_dataset_if_available |
| tests/test_exact_old_ipm_backend_does_not_use_surrogate.py | Test module; defines test_all_backend_runner_does_not_reference_surrogate_topologies, test_python_point_backend_reports_no_surrogate_topology | test_all_backend_runner_does_not_reference_surrogate_topologies, test_python_point_backend_reports_no_surrogate_topology |
| tests/test_exp08_seed_adaptive_warmstart.py | Focused tests for the exp08 pump-solve speedup paths | _build_problem, _settings, problem, solver, test_linear_phasor_seed_solves_fundamental_block, test_linear_phasor_seed_direct_matches_gmres... |
| tests/test_exp10_gate.py | Tests for the exp10 warm-start map gate | _row, test_speedup_is_per_point_with_sparse_cold_spotcheck, test_sparse_nonconvergence_within_threshold_passes, test_too_many_nonconverged_fails, test_gain_drift_over_gate_fails, test_secant_guess_uniform_step_doubles_the_delta... |
| tests/test_exported_julia_json_schema.py | Test module; defines test_exported_julia_json_schema_tiny_fixture | test_exported_julia_json_schema_tiny_fixture |
| tests/test_fxjtwpa_node_order.py | Regression test for the fxjtwpa JC-seed node-order fix | _sorted_rank_perm, test_fxjtwpa_node_order_is_unsorted, test_fxjtwpa_permutation_drops_seed_residual |
| tests/test_harmonia_benchmark_suite.py | Test module; defines test_select_cases_prefers_modern_architecture, test_select_cases_can_include_legacy, test_actual_minimal_harmonia_benchmark_suite_if_availa | test_select_cases_prefers_modern_architecture, test_select_cases_can_include_legacy, test_actual_minimal_harmonia_benchmark_suite_if_available |
| tests/test_harmonia_benchmark_suite_batch_runner.py | Python support/entry-point module | - |
| tests/test_harmonia_benchmark_suite_jtl_hblinsolve_direct.py | Python support/entry-point module | - |
| tests/test_harmonia_benchmark_suite_jtl_rf_ethz_hblinsolve_direct.py | Python support/entry-point module | - |
| tests/test_harmonia_benchmark_suite_jtl_rf_hblinsolve_direct.py | Python support/entry-point module | - |
| tests/test_harmonia_coupler_topology_smoke.py | Test module; defines test_coupler | test_coupler |
| tests/test_harmonia_ethz_jtl_linear_campaign.py | Test module; defines test_make_harmonia_ethz_jtl_linear_config, test_campaign_paths, test_compute_metrics_on_existing_ethz_linear_if_available, test_actual_harm | test_make_harmonia_ethz_jtl_linear_config, test_campaign_paths, test_compute_metrics_on_existing_ethz_linear_if_available, test_actual_harmonia_ethz_jtl_linear_campaign_if_available |
| tests/test_harmonia_ethz_jtl_linear_dataset.py | Test module; defines test_extract_harmonia_ethz_jtl_linear_parameter_vector, test_actual_harmonia_ethz_jtl_linear_dataset_if_available | test_extract_harmonia_ethz_jtl_linear_parameter_vector, test_actual_harmonia_ethz_jtl_linear_dataset_if_available |
| tests/test_harmonia_ethz_jtl_linear_jc_smoke.py | Test module; defines test_actual_harmonia_ethz_jtl_linear_jc_smoke_if_available | test_actual_harmonia_ethz_jtl_linear_jc_smoke_if_available |
| tests/test_harmonia_ethz_jtl_linear_objective.py | Test module; defines test_find_nearest_parameter_index, test_actual_harmonia_ethz_jtl_linear_objective_if_available | test_find_nearest_parameter_index, test_actual_harmonia_ethz_jtl_linear_objective_if_available |
| tests/test_harmonia_ethz_jtl_topology_smoke.py | Test module; defines test_actual_harmonia_ethz_jtl_topology_smoke_if_available | test_actual_harmonia_ethz_jtl_topology_smoke_if_available |
| tests/test_harmonia_ipm_topology_smoke.py | Test module; defines test_ipm | test_ipm |
| tests/test_harmonia_jtl_linear_campaign.py | Test module; defines test_make_harmonia_jtl_linear_config, test_campaign_paths, test_compute_metrics_on_existing_harmonia_jtl_linear_if_available, test_actual_h | test_make_harmonia_jtl_linear_config, test_campaign_paths, test_compute_metrics_on_existing_harmonia_jtl_linear_if_available, test_actual_harmonia_jtl_linear_campaign_if_available |
| tests/test_harmonia_jtl_linear_campaign_batch_cli.py | Python support/entry-point module | - |
| tests/test_harmonia_jtl_linear_campaign_hblinsolve_direct.py | Python support/entry-point module | - |
| tests/test_harmonia_jtl_linear_dataset.py | Test module; defines test_extract_harmonia_jtl_linear_parameter_vector, test_actual_harmonia_jtl_linear_dataset_if_available | test_extract_harmonia_jtl_linear_parameter_vector, test_actual_harmonia_jtl_linear_dataset_if_available |
| tests/test_harmonia_jtl_linear_jc_smoke.py | Test module; defines test_actual_harmonia_jtl_linear_jc_smoke_if_available | test_actual_harmonia_jtl_linear_jc_smoke_if_available |
| tests/test_harmonia_jtl_linear_objective.py | Test module; defines test_target_sample_ranks_first | test_target_sample_ranks_first |
| tests/test_harmonia_jtl_topology_smoke.py | Test module; defines test_actual_harmonia_jtl_topology_smoke_if_available | test_actual_harmonia_jtl_topology_smoke_if_available |
| tests/test_harmonia_lumped_jpa_linear_campaign_dataset_objective.py | Test module; defines test_make_harmonia_lumped_jpa_linear_config, test_campaign_paths, test_extract_lumped_jpa_parameter_vector, test_find_nearest_parameter_ind | test_make_harmonia_lumped_jpa_linear_config, test_campaign_paths, test_extract_lumped_jpa_parameter_vector, test_find_nearest_parameter_index, test_actual_lumped_jpa_campaign_dataset_objective_if_available, test_compute_metrics_on_existing_lumped_jpa_linear_if_available |
| tests/test_harmonia_lumped_jpa_linear_jc_smoke.py | Test module; defines test_actual_harmonia_lumped_jpa_linear_jc_smoke_if_available | test_actual_harmonia_lumped_jpa_linear_jc_smoke_if_available |
| tests/test_harmonia_lumped_jpa_topology_smoke.py | Test module; defines test_actual_harmonia_lumped_jpa_topology_smoke_if_available | test_actual_harmonia_lumped_jpa_topology_smoke_if_available |
| tests/test_harmonia_rf_jtl_linear_campaign.py | Test module; defines test_rf_campaign | test_rf_campaign |
| tests/test_harmonia_rf_jtl_linear_dataset.py | Test module; defines test_rf_dataset | test_rf_dataset |
| tests/test_harmonia_rf_jtl_linear_jc_smoke.py | Test module; defines test_actual_harmonia_rf_jtl_linear_jc_smoke | test_actual_harmonia_rf_jtl_linear_jc_smoke |
| tests/test_harmonia_rf_jtl_linear_objective.py | Test module; defines test_rf_objective | test_rf_objective |
| tests/test_harmonia_rf_jtl_topology_smoke.py | Test module; defines test_actual_harmonia_rf_jtl_topology_smoke | test_actual_harmonia_rf_jtl_topology_smoke |
| tests/test_harmonia_tiny_nonlinear_campaign.py | Test module; defines test_nonlinear_campaign | test_nonlinear_campaign |
| tests/test_harmonia_tiny_nonlinear_hb_smoke.py | Test module; defines test_actual_harmonia_tiny_nonlinear_hb_smoke | test_actual_harmonia_tiny_nonlinear_hb_smoke |
| tests/test_harmonia_workspace_inventory.py | Test module; defines test_extract_functions_basic, test_keyword_hits_detect_josephson_hb, test_device_tags_detect_ipm_rfsquid, test_inspect_real_harmonia_jl_fil | test_extract_functions_basic, test_keyword_hits_detect_josephson_hb, test_device_tags_detect_ipm_rfsquid, test_inspect_real_harmonia_jl_file_if_available |
| tests/test_harmonics.py | Tests for twpa.core.harmonics | _call_with_supported_kwargs, _get_attr_or_key, _as_mapping, _make_harmonic_set, _extract_orders, _index_of... |
| tests/test_hb_element.py | Tests for twpa.nonlinear.hb_element | _call_with_supported_kwargs, _get_attr_or_key, _as_mapping, _validate_orders, _time_grid, _manual_synthesize... |
| tests/test_hb_fft.py | Tests for twpa.core.hb_fft | _call_with_supported_kwargs, _get_attr_or_key, _time_grid, _synthesize, _analyze, _round_trip... |
| tests/test_import_josephson_branches.py | Test module; defines test_josephson_branch_import_sets_incidence_and_ic | test_josephson_branch_import_sets_incidence_and_ic |
| tests/test_import_named_mutual_inductors.py | Test module; defines test_named_mutual_inductors_are_assembled | test_named_mutual_inductors_are_assembled |
| tests/test_import_old_ipm_json_tiny_fixture.py | Test module; defines write_tiny_export, test_import_tiny_fixture_node_labels_and_counts | write_tiny_export, test_import_tiny_fixture_node_labels_and_counts |
| tests/test_imported_old_ipm_conversion_minimal.py | Test module; defines test_imported_old_ipm_conversion_minimal_tiny_fixture | test_imported_old_ipm_conversion_minimal_tiny_fixture |
| tests/test_imported_old_ipm_pump_residual_shape.py | Test module; defines test_imported_old_ipm_pump_residual_shape_tiny_fixture | test_imported_old_ipm_pump_residual_shape_tiny_fixture |
| tests/test_imported_old_ipm_pump_source_injection.py | Test module; defines test_imported_old_ipm_pump_source_uses_port4_when_present | test_imported_old_ipm_pump_source_uses_port4_when_present |
| tests/test_imported_old_ipm_residual_jacobian.py | Test module; defines _tiny_residual, test_imported_old_ipm_sparse_jacobian_shape, test_imported_old_ipm_residual_jvp_matches_finite_difference | _tiny_residual, test_imported_old_ipm_sparse_jacobian_shape, test_imported_old_ipm_residual_jvp_matches_finite_difference |
| tests/test_imported_old_ipm_zero_source_residual.py | Test module; defines test_imported_old_ipm_zero_source_zero_flux_residual | test_imported_old_ipm_zero_source_zero_flux_residual |
| tests/test_jc_cached_setup_workload.py | Python support/entry-point module | - |
| tests/test_jc_jpa_reflection_campaign.py | Test module; defines test_make_jc_jpa_reflection_config, test_campaign_paths, test_compute_metrics_on_existing_jc_smoke_if_available, test_actual_jc_jpa_reflect | test_make_jc_jpa_reflection_config, test_campaign_paths, test_compute_metrics_on_existing_jc_smoke_if_available, test_actual_jc_jpa_reflection_campaign_if_available |
| tests/test_jc_jpa_reflection_dataset.py | Test module; defines test_extract_jc_jpa_parameter_vector, test_actual_jc_jpa_reflection_dataset_if_available | test_extract_jc_jpa_parameter_vector, test_actual_jc_jpa_reflection_dataset_if_available |
| tests/test_jc_jpa_reflection_objective.py | Test module; defines test_one_port_reflection_objective_zero_for_identical_curves, test_one_port_reflection_objective_penalizes_different_curves, test_jc_reflec | test_one_port_reflection_objective_zero_for_identical_curves, test_one_port_reflection_objective_penalizes_different_curves, test_jc_reflection_dataset_objective_ranks_target_first, test_actual_jc_reflection_objective_pipeline_if_available |
| tests/test_jc_jpa_smoke_reader.py | Test module; defines test_actual_jc_jpa_reflection_smoke_if_available | test_actual_jc_jpa_reflection_smoke_if_available |
| tests/test_julia_batch_runner.py | Python support/entry-point module | - |
| tests/test_julia_batch_runner_cache_telemetry.py | Python support/entry-point module | - |
| tests/test_julia_bridge_reader.py | Test module; defines _write_json, test_read_status_json_pass_schema_smoke, test_pass_status_rejects_nonfinite_residual, test_pass_status_rejects_failure_reason, | _write_json, test_read_status_json_pass_schema_smoke, test_pass_status_rejects_nonfinite_residual, test_pass_status_rejects_failure_reason, test_load_actual_schema_smoke_if_available |
| tests/test_julia_map_backend_selection.py | Test module; defines test_julia_map_backend_selection_lists_required_backends | test_julia_map_backend_selection_lists_required_backends |
| tests/test_julia_runner.py | Test module; defines _write_status, test_build_julia_command, test_runner_returns_cached_status_without_julia, test_actual_schema_smoke_launch_if_available | _write_status, test_build_julia_command, test_runner_returns_cached_status_without_julia, test_actual_schema_smoke_launch_if_available |
| tests/test_kinetic_inductance.py | Tests for twpa.nonlinear.kinetic_inductance | _call_with_supported_kwargs, _get_attr_or_key, _as_mapping, _make_params, _extract_L0, _extract_I_star... |
| tests/test_ladder_mna.py | Tests for twpa.linear.ladder_mna | _call_with_supported_kwargs, _get_attr_or_key, _as_mapping, _make_ladder_params, _broadcast_series_l, _broadcast_shunt_c... |
| tests/test_linear_scattering_smoke.py | Python support/entry-point module | - |
| tests/test_linear_sparams_campaign.py | Test module; defines test_make_linear_sparams_config, test_campaign_paths, test_actual_linear_sparams_campaign_if_available, test_compute_2port_metrics_on_exist | test_make_linear_sparams_config, test_campaign_paths, test_actual_linear_sparams_campaign_if_available, test_compute_2port_metrics_on_existing_linear_run_if_available |
| tests/test_native_workflow_cleanup.py | Test module; defines test_native_gain_map_workflow_exports_compact_cube, test_recovery_gain_residual_uses_nominal_target_and_factories, test_calibration_diagnos | test_native_gain_map_workflow_exports_compact_cube, test_recovery_gain_residual_uses_nominal_target_and_factories, test_calibration_diagnostics_report_parameter_correlation, test_run_report_excludes_generated_reports_and_active_output, test_bridge_manifest_deduplicates_and_compacts_json |
| tests/test_old_ipm_full_export_import.py | Test module; defines test_full_old_ipm_export_import_assembly | test_full_old_ipm_export_import_assembly |
| tests/test_old_ipm_scipy_backend_point_not_placeholder.py | Test module; defines test_old_ipm_scipy_backend_point_not_placeholder | test_old_ipm_scipy_backend_point_not_placeholder |
| tests/test_old_power_convention_from_exported_json.py | Test module; defines test_old_power_convention_from_exported_json | test_old_power_convention_from_exported_json |
| tests/test_one_node.py | Tests for twpa.nonlinear.one_node | _call_with_supported_kwargs, _get_attr_or_key, _as_mapping, _validate_orders, _time_grid, _manual_synthesize... |
| tests/test_params.py | Tests for twpa.core.params | _has, _get_any, _call_with_supported_kwargs, _as_mapping, _get_attr_or_key, _make_line_params... |
| tests/test_production_stack_smoke.py | Smoke tests for the production TWPA simulator stack | pytest_addoption, pytest_configure, pytest_collection_modifyitems, assert_json_serializable, tiny_nonlinear_params, tiny_uniform_layout... |
| tests/test_pump_backend_imports.py | Python support/entry-point module | - |
| tests/test_pump_basis.py | Tests for the pump-mode basis policy layer (experiments/pump_basis.py) | test_positive_odd_modes_matches_jc_jtwpa_list, test_parse_explicit_modes, test_dense_real_preserves_legacy_behavior, test_positive_odd_jc_uses_mode_count, test_positive_phasor_explicit_requires_modes, test_auto_jc_reads_nmodulationharmonics... |
| tests/test_pump_core_imports.py | Python support/entry-point module | - |
| tests/test_pump_solvers_schur.py | Tests for the Schur-reduced / matrix-free pump-solver backends | _toy_problem, _settings, test_schur_algebra_matches_full_linear_solve, test_schur_pump_matches_full_pump, test_assembled_and_matrixfree_linear_apply_agree, test_analytic_jvp_matches_fd_full... |
| tests/test_python_backend_point_result_schema.py | Test module; defines test_python_backend_point_schema_for_non_least_squares_backend | test_python_backend_point_schema_for_non_least_squares_backend |
| tests/test_resource_safety.py | Test module; defines test_dense_pump_guard_refuses_large_reference_problem_before_allocation, test_dense_gain_guard_refuses_large_reference_problem_before_alloc | test_dense_pump_guard_refuses_large_reference_problem_before_allocation, test_dense_gain_guard_refuses_large_reference_problem_before_allocation, test_dense_reference_resource_estimator_matches_unknown_formula, test_dense_finite_signal_guard_refuses_large_reference_problem_before_allocation, test_hb_dispatcher_runs_matrix_free_newton_krylov, test_tiny_pump_newton_krylov_matches_dense_reference... |
| tests/test_rf_networks.py | Tests for twpa.linear.rf_networks | _call_with_supported_kwargs, _get_any, _as_abcd, _as_s, _identity_abcd, _series_abcd... |
| tests/test_run_exported_julia_circuit_smoke.py | Test module; defines test_run_exported_julia_circuit_smoke | test_run_exported_julia_circuit_smoke |
| tests/test_run_gain_map_cli.py | CLI defaults for the gain-map runner. | test_inproc_fail_fast_is_opt_in, test_inproc_fail_fast_flag_enables_fast_failure, test_all_intra_cell_continuation_methods_are_selectable, test_solve_deadline_alias_matches_canonical_flag, test_column_arclength_recovery_is_opt_in, test_column_arclength_has_separate_trace_deadline... |
| tests/test_run_gain_map_spectrum_offsets.py | Test module; defines test_default_spectrum_offsets_include_headline_signal | test_default_spectrum_offsets_include_headline_signal |
| tests/test_run_registry.py | Test module; defines _write_json, _make_run_dir, test_register_run_dir_writes_csv, test_registry_summary_counts_status_and_type, test_registry_csv_has_expected_ | _write_json, _make_run_dir, test_register_run_dir_writes_csv, test_registry_summary_counts_status_and_type, test_registry_csv_has_expected_columns |
| tests/test_schema_smoke_campaign.py | Test module; defines test_make_schema_smoke_config_is_deterministic, test_campaign_paths, test_actual_schema_smoke_campaign_if_available | test_make_schema_smoke_config_is_deterministic, test_campaign_paths, test_actual_schema_smoke_campaign_if_available |
| tests/test_scipy_backend_jacobian_metadata.py | Test module; defines test_scipy_backend_result_reports_residual_reduction, test_backend_adapter_parses_jacobian_metadata | test_scipy_backend_result_reports_residual_reduction, test_backend_adapter_parses_jacobian_metadata |
| tests/test_signal_imports.py | Python support/entry-point module | - |
| tests/test_simulation_schema.py | Test module; defines test_validate_status_payload_accepts_pass_with_null_residual, test_validate_status_payload_rejects_pass_with_failure_reason, test_validate_ | test_validate_status_payload_accepts_pass_with_null_residual, test_validate_status_payload_rejects_pass_with_failure_reason, test_validate_status_payload_rejects_pass_without_solver_success, test_validate_status_payload_rejects_nonfinite_residual, test_assert_json_serializable_catches_set, test_classify_status... |
| tests/test_small_distributed_hb.py | Tests for twpa.nonlinear.distributed_hb | _call_with_supported_kwargs, _get_attr_or_key, _as_mapping, _validate_orders, _broadcast_series_l, _broadcast_shunt_c... |
| tests/test_solver_imports.py | Python support/entry-point module | - |
| tests/test_topology_artifacts.py | Test module; defines test_load_actual_jtl_topology_artifact_if_available | test_load_actual_jtl_topology_artifact_if_available |
| tests/test_traversal.py | Tests for the map traversal ordering and neighbour selection (pure logic) | _grid, test_grid_dims, test_column_order_is_column_major_ascending_power, test_all_strategies_visit_every_cell_once, test_backbone_solves_lowest_power_row_first, test_backbone_center_out_starts_at_middle_frequency... |
| tests/test_twpa_aft_hb_residual.py | Test module; defines test_zero_amplitude_zero_source_residual_is_zero, test_projection_recovers_cosine_coefficient | test_zero_amplitude_zero_source_residual_is_zero, test_projection_recovers_cosine_coefficient |
| tests/test_twpa_continuation.py | Test module; defines test_continuation_reaches_parameter_path, test_snake_grid_indices | test_continuation_reaches_parameter_path, test_snake_grid_indices |
| tests/test_twpa_conversion_matrix.py | Test module; defines test_zero_pump_signal_s21_matches_linear_solver | test_zero_pump_signal_s21_matches_linear_solver |
| tests/test_twpa_conversion_small_pump_limit.py | Test module; defines _pump_solution_with_cos_amplitude, test_small_pump_even_josephson_sideband_conversion_tends_to_zero | _pump_solution_with_cos_amplitude, test_small_pump_even_josephson_sideband_conversion_tends_to_zero |
| tests/test_twpa_conversion_zero_pump_consistency.py | Test module; defines test_zero_pump_conversion_admittance_is_sideband_block_diagonal, test_zero_pump_signal_s21_matches_linear_solver_with_sidebands | test_zero_pump_conversion_admittance_is_sideband_block_diagonal, test_zero_pump_signal_s21_matches_linear_solver_with_sidebands |
| tests/test_twpa_coupler_model_taxonomy.py | Test module; defines test_coupler_model_taxonomy_contains_required_labels, test_topology_coupler_models_use_taxonomy | test_coupler_model_taxonomy_contains_required_labels, test_topology_coupler_models_use_taxonomy |
| tests/test_twpa_directional_coupler_block.py | Test module; defines test_directional_coupler_current_matches_manual_calculation | test_directional_coupler_current_matches_manual_calculation |
| tests/test_twpa_graph.py | Test module; defines test_incidence_matrix_branch_flux_consistency | test_incidence_matrix_branch_flux_consistency |
| tests/test_twpa_ipm_physical_coupler_topology.py | Test module; defines test_physical_coupler_ipm_assembles_and_solves_linear_sparams, test_physical_coupler_tiny_pump_and_conversion_smoke | test_physical_coupler_ipm_assembles_and_solves_linear_sparams, test_physical_coupler_tiny_pump_and_conversion_smoke |
| tests/test_twpa_ipm_topology.py | Test module; defines test_ipm_topology_assembles_matrices_and_ports, test_tiny_ipm_smoke_solve_returns_status_metadata | test_ipm_topology_assembles_matrices_and_ports, test_tiny_ipm_smoke_solve_returns_status_metadata |
| tests/test_twpa_jax_aft_residual.py | Test module; defines test_jax_aft_residual_matches_numpy, test_jax_jvp_matches_finite_difference_on_twpa_residual, test_jax_dense_newton_solves_tiny_twpa_zero_s | test_jax_aft_residual_matches_numpy, test_jax_jvp_matches_finite_difference_on_twpa_residual, test_jax_dense_newton_solves_tiny_twpa_zero_source |
| tests/test_twpa_jax_solvers.py | Test module; defines test_jax_dense_newton_solves_tiny_problem | test_jax_dense_newton_solves_tiny_problem |
| tests/test_twpa_linear_solver.py | Test module; defines test_passive_linearized_ladder_returns_finite_reciprocal_sparameters | test_passive_linearized_ladder_returns_finite_reciprocal_sparameters |
| tests/test_twpa_map_row_schema.py | Test module; defines test_parity_row_schema_contains_required_fields | test_parity_row_schema_contains_required_fields |
| tests/test_twpa_nonlinearities.py | Test module; defines test_josephson_derivative_matches_finite_difference | test_josephson_derivative_matches_finite_difference |
| tests/test_twpa_old_julia_parity_config.py | Test module; defines test_old_constants_surrogate_topology_records_warning_metadata, test_deprecated_old_julia_parity_name_is_marked_as_surrogate | test_old_constants_surrogate_topology_records_warning_metadata, test_deprecated_old_julia_parity_name_is_marked_as_surrogate |
| tests/test_twpa_old_julia_power_convention.py | Test module; defines test_old_julia_offset_sets_source_power_and_peak_current | test_old_julia_offset_sets_source_power_and_peak_current |
| tests/test_twpa_parity_comparison_alignment.py | Test module; defines test_parity_comparison_aligns_by_frequency_and_external_power | test_parity_comparison_aligns_by_frequency_and_external_power |
| tests/test_twpa_preconditioner.py | Test module; defines test_linear_passive_preconditioner_solves_same_tiny_residual | test_linear_passive_preconditioner_solves_same_tiny_residual |
| tests/test_twpa_pump_status_propagation.py | Test module; defines test_nonconverged_pump_status_masks_gain_row | test_nonconverged_pump_status_masks_gain_row |
| tests/test_twpa_scipy_solvers.py | Test module; defines test_scipy_least_squares_solves_toy_nonlinear_problem, test_scipy_root_solves_toy_problem | test_scipy_least_squares_solves_toy_nonlinear_problem, test_scipy_root_solves_toy_problem |
| tests/test_twpa_solver_readiness_classification.py | Test module; defines test_solver_readiness_has_single_allowed_class_per_solver | test_solver_readiness_has_single_allowed_class_per_solver |
| tests/test_twpa_solvers.py | Test module; defines test_arclength_traces_toy_fold_scaffold, test_shooting_for_stationary_periodic_state, test_anderson_accelerates_simple_fixed_point, test_de | test_arclength_traces_toy_fold_scaffold, test_shooting_for_stationary_periodic_state, test_anderson_accelerates_simple_fixed_point, test_deflation_clusters_multistart_solutions, test_mor_placeholder_roundtrip_and_two_tone_grid |
| tests/test_twpa_units.py | Test module; defines test_dbm_current_roundtrip, test_wave_normalization_matched_load_has_no_reflection | test_dbm_current_roundtrip, test_wave_normalization_matched_load_has_no_reflection |
| tests/test_units.py | Tests for twpa.core.units | _has, _get_any, _call_or_scale, _call_inverse_or_scale, _ghz_to_hz, _mhz_to_hz... |
### twpa/

#### Legacy/Historical/Auxiliary (68 files)

| File | Functionality / observed role | Top-level symbols |
|---|---|---|
| twpa/__init__.py | twpa ==== JAX-backed simulation toolkit for superconducting traveling-wave parametric amplifiers | __getattr__, __dir__ |
| twpa/accelerators/__init__.py | Optional accelerator capability detection without import-time GPU dependencies. | - |
| twpa/accelerators/backend.py | Capability reporting for optional WSL2 accelerator experiments. | AcceleratorCapabilities, detect_accelerator_capabilities |
| twpa/calibration/__init__.py | Package initializer/export module | - |
| twpa/calibration/dataset_objectives.py | Module; defines select_target_index, evaluate_two_port_dataset, evaluate_harmonia_jtl_linear_dataset | select_target_index, evaluate_two_port_dataset, evaluate_harmonia_jtl_linear_dataset |
| twpa/calibration/objectives.py | Calibration objectives for simulation-vs-target comparison | SParameterObjectiveWeights, SParameterObjectiveResult, OnePortReflectionObjectiveWeights, OnePortReflectionObjectiveResult, _mse_abs, _mse_real... |
| twpa/cli.py | Module; defines build_parser, _run_command, main | build_parser, _run_command, main |
| twpa/core/__init__.py | twpa.core ========= Core data structures and numerical utilities for the JAX-backed TWPA simulator | __getattr__, __dir__ |
| twpa/core/disorder.py | twpa.core.disorder ================== Fabrication-disorder and parameter-variation utilities | DistributionKind, CorrelationKernel, _check_nonnegative, _check_positive, _jsonify, RandomFieldConfig... |
| twpa/core/frequency_plan.py | twpa.core.frequency_plan ======================== Frequency bookkeeping for harmonic-balance and conversion-matrix simulation | ToneRole, FrequencyPlanKind, _check_positive, _as_1d_float_array, _unique_labels, _jsonify... |
| twpa/core/harmonics.py | twpa.core.harmonics =================== Harmonic-balance coefficient bookkeeping | _as_1d_array, _as_complex_array, _check_positive_int, _check_odd, complex_to_real_vector, real_vector_to_complex... |
| twpa/core/hb_fft.py | twpa.core.hb_fft ================ Harmonic-balance nonlinear projection utilities | _as_complex_array, _as_float_array, _check_positive, _broadcast_param_to_samples, ProjectionMode, HBProjectionConfig... |
| twpa/core/layout.py | twpa.core.layout ================ Vectorized layout representation for long TWPA transmission lines | _as_1d_array, _broadcast_to_n, _require_same_length, _require_nonnegative_array, _require_positive_array, _safe_json_value... |
| twpa/core/numerics.py | Numerical runtime configuration helpers. | enable_x64 |
| twpa/core/params.py | twpa.core.params ================ Immutable parameter and configuration containers for the TWPA simulator | BasicLineParams, CellParams, TWPAParams, OperatingPoint, josephson_inductance, line_params_from_z0_vp... |
| twpa/core/units.py | twpa.core.units =============== Unit constants and RF power/voltage/current conversion utilities | PhysicalConstants, asarray_si, _safe_positive, hz, ghz, mhz... |
| twpa/inference/__init__.py | twpa.inference ============== Inference and parameter-recovery tools for the JAX-backed TWPA simulator | __getattr__, __dir__ |
| twpa/inference/fitting.py | twpa.inference.fitting ====================== Parameter-fitting orchestration for TWPA inference workflows | FitStatus, FitOptimizerMethod, FitLossKind, FitConfig, FitEvaluation, FitIterationRecord... |
| twpa/inference/priors.py | twpa.inference.priors ===================== Prior distributions and parameter-vector helpers for TWPA inference | PriorKind, ParameterTransform, ParameterPrior, ParameterSample, PriorSet, make_scale_prior... |
| twpa/inference/recovery.py | twpa.inference.recovery ======================= End-to-end synthetic parameter-recovery experiments for TWPA simulations | RecoveryStatus, RecoveryDatasetMode, RecoveryToleranceConfig, RecoveryExperimentConfig, RecoveryTrialResult, RecoveryExperimentResult... |
| twpa/inference/synthetic.py | twpa.inference.synthetic ======================== Synthetic measurement generation for TWPA inference and recovery studies | SyntheticMeasurementKind, SyntheticNoiseConfig, SyntheticSParameterDataset, SyntheticGainDataset, SyntheticCombinedDataset, apply_parameter_scales_to_layout... |
| twpa/io/__init__.py | twpa.io ======= Input/output utilities for the JAX-backed TWPA simulator | __getattr__, __dir__ |
| twpa/io/campaigns.py | Reusable helpers for Julia/Harmonia simulation campaigns. | campaign_paths, compute_two_port_run_metrics, register_completed_run, compute_one_port_run_metrics, run_parameter_campaign |
| twpa/io/checkpoints.py | twpa.io.checkpoints =================== Checkpoint save/load utilities for TWPA simulations and calibration workflows | CheckpointKind, CheckpointCompression, CheckpointMetadata, Checkpoint, _json_dumps, _hash_file... |
| twpa/io/dataset_builder.py | Dataset builders for Julia/Harmonia simulation campaigns | BuiltDataset, _as_float, read_resolved_config, extract_parameter_vector, _require_same_frequency, build_linear_sparams_dataset... |
| twpa/io/hdf5_utils.py | Small HDF5 decoding helpers shared by readers and smoke tests. | decode_h5_scalar, decode_h5_string |
| twpa/io/julia_batch_runner.py | Python support/entry-point module | - |
| twpa/io/julia_bridge.py | Reader utilities for Julia/Harmonia simulation outputs | JuliaSimulationStatus, JuliaSimulationData, _optional_path, read_status_json, _decode_h5_attr, read_simulation_h5... |
| twpa/io/julia_runner.py | Python launcher for Harmonia/JosephsonCircuits Julia simulations | JuliaEnginePaths, JuliaRunResult, build_julia_command, _write_text, run_harmonia_simulation |
| twpa/io/measurement.py | twpa.io.measurement =================== Measurement dataset loaders and normalizers for TWPA calibration workflows | MeasurementKind, MeasurementFileFormat, MeasurementLoadConfig, SParameterMeasurement, GainMeasurement, _resolve_format... |
| twpa/io/netlist.py | twpa.io.netlist =============== Netlist import/export helpers for TWPA layouts | NetlistFormat, ShuntPlacement, SeriesBranchModel, ResonatorExportMode, NetlistExportConfig, NetlistExportResult... |
| twpa/io/reports.py | twpa.io.reports =============== Report-generation helpers for TWPA simulation, calibration, and inference runs | ReportFormat, ReportStatus, ReportArtifact, RunReport, RunTimer, now_iso... |
| twpa/io/run_registry.py | Run registry for Julia/Harmonia simulation outputs | RegisteredRun, utc_now_iso, _read_json_if_exists, _clean_optional_float, registered_run_from_dir, _csv_value... |
| twpa/io/simulation_schema.py | Shared simulation schema utilities | SimulationStatus, SimulationSchemaError, TwoPortMetrics, optional_float, optional_int, assert_json_serializable... |
| twpa/io/topology_artifacts.py | Module; defines decode_h5_scalar, read_json_dataset, TopologyArtifact, load_topology_artifact, require_topology_counts | decode_h5_scalar, read_json_dataset, TopologyArtifact, load_topology_artifact, require_topology_counts |
| twpa/linear/__init__.py | twpa.linear =========== Pump-off linear microwave simulation tools for the JAX-backed TWPA simulator | __getattr__, __dir__ |
| twpa/linear/cascade.py | twpa.linear.cascade =================== Long-line ABCD cascading utilities for vectorized TWPA layouts | CascadeStrategy, CascadeConfig, CascadeResult, _as_frequency_array, _check_cell_abcd_shape, _check_line_abcd_shape... |
| twpa/linear/cells.py | twpa.linear.cells ================= Cell-level linear RF models for vectorized TWPA layouts | CellModelKind, CellModelConfig, _as_frequency_array, _as_cell_array, _broadcast_cell_frequency, _zeros_cell_frequency... |
| twpa/linear/coarsening.py | twpa.linear.coarsening ====================== Effective-cell and supercell coarsening utilities | CoarseningMethod, CoarseningConfig, CoarseningResult, _require_divisible, _group_reduce, _safe_ratio... |
| twpa/linear/dispersion.py | twpa.linear.dispersion ====================== Dispersion extraction and phase-matching diagnostics for TWPA layouts | DispersionExtractionMethod, StopbandMetric, DispersionConfig, _as_frequency_array, _check_abcd_batch, _moving_average_1d... |
| twpa/linear/ladder_mna.py | twpa.linear.ladder_mna ====================== Dense modified/nodal analysis for linear lumped TWPA ladders | LadderDiscretization, PortSolveMode, LadderMNAConfig, _as_frequency_array, _as_complex_matrix, _eye2_batch... |
| twpa/linear/rf_networks.py | twpa.linear.rf_networks ======================= Low-level RF two-port network utilities | _as_frequency_array, _as_complex, _as_real, _broadcast_to_frequency, _check_abcd_shape, _check_s_shape... |
| twpa/nonlinear/__init__.py | twpa.nonlinear ============== Nonlinear harmonic-balance and gain tools for KI-TWPA simulation. | __getattr__, __dir__ |
| twpa/nonlinear/conversion.py | twpa.nonlinear.conversion ========================= Frequency-conversion matrix utilities for pumped TWPA simulations | ConversionMatrixNormalization, ConversionStatus, ConversionTone, DP4WMToneSet, ConversionMatrixConfig, ConversionColumnResult... |
| twpa/nonlinear/distributed_hb.py | twpa.nonlinear.distributed_hb ============================= Dense distributed harmonic-balance residual for nonlinear TWPA ladders | DistributedHBSourceKind, DistributedHBTerminationKind, DistributedHBConfig, DistributedHBState, DistributedHBResidual, DistributedHBSolveResult... |
| twpa/nonlinear/finite_signal_hb.py | twpa.nonlinear.finite_signal_hb =============================== Finite-signal harmonic-balance utilities for pumped KI-TWPA simulations | FiniteSignalStatus, FiniteSignalSweepKind, CompressionMetric, SignalDriveConfig, FiniteSignalHBConfig, FiniteSignalObservable... |
| twpa/nonlinear/gain.py | twpa.nonlinear.gain =================== Small-signal gain and conversion utilities for pumped TWPA simulations | GainQuantity, GainStatus, GainInputKind, complex_abs_db, power_abs_db, matched_power_gain_from_voltage_gain... |
| twpa/nonlinear/hb_element.py | twpa.nonlinear.hb_element ========================= Element-level harmonic-balance residuals | KineticInductanceHBElement, _selected_orders, flux_coeffs, voltage_coeffs, residual, voltage_jacobian... |
| twpa/nonlinear/kinetic_inductance.py | twpa.nonlinear.kinetic_inductance ================================= Kinetic-inductance and generic weak-current chi(3) nonlinear-inductor models | _check_positive, _check_nonnegative, _as_float_array, _as_complex_array, _jsonify, KineticInductanceModel... |
| twpa/nonlinear/linearization.py | twpa.nonlinear.linearization ============================ Small-signal linearization around pumped nonlinear HB states | LinearizationBackend, SmallSignalSourceKind, SmallSignalLinearizationConfig, SmallSignalState, SmallSignalSource, SmallSignalSolveResult... |
| twpa/nonlinear/one_node.py | twpa.nonlinear.one_node ======================= One-node harmonic-balance validation circuits | OneNodeCircuitKind, OneNodeHBConfig, OneNodeHBState, OneNodeHBResidual, OneNodeHBSolveResult, _as_complex_1d... |
| twpa/nonlinear/pump_hb_ladder.py | twpa.nonlinear.pump_hb_ladder ============================= Production-facing pump-only harmonic-balance driver for nonlinear TWPA ladders | PumpDriveKind, PumpContinuationKind, PumpHBStatus, dbm_to_watt, watt_to_dbm, norton_current_rms_from_available_power... |
| twpa/nonlinear/supercell_hb.py | twpa.nonlinear.supercell_hb =========================== Supercell and periodic-surrogate harmonic-balance utilities for KI-TWPA lines | SupercellBoundaryKind, SupercellConstructionMethod, SupercellExtractionConfig, SupercellSurrogateConfig, SupercellSurrogateResult, SupercellPumpHBConfig... |
| twpa/plotting/__init__.py | twpa.plotting ============= Plotting utilities for TWPA simulation diagnostics and gain maps | __getattr__, __dir__ |
| twpa/plotting/diagnostics.py | twpa.plotting.diagnostics ========================= Matplotlib diagnostic plots for TWPA simulation and inference workflows | PlotConfig, _plt, _asarray, _maybe_attr, _frequency_scale, _length_scale... |
| twpa/plotting/gain_maps.py | twpa.plotting.gain_maps ======================= Matplotlib plotting helpers for TWPA gain sweeps, gain maps, operating maps, and compression sweeps | GainMapPlotConfig, _plt, _asarray, _maybe_attr, _frequency_scale, _new_fig_ax... |
| twpa/solvers/__init__.py | twpa.solvers ============ Numerical solvers for the JAX-backed TWPA simulator | __getattr__, __dir__ |
| twpa/solvers/block_banded.py | twpa.solvers.block_banded ========================= Block-banded matrix containers and operations for structured TWPA solvers | BlockBandedStorage, BlockBandedSolveMethod, BlockBandedConfig, BlockBandedMatrix, build_block_banded_from_dense, build_block_tridiagonal_from_dense... |
| twpa/solvers/continuation.py | twpa.solvers.continuation ========================= Continuation / homotopy drivers for nonlinear HB solves | ContinuationStatus, StepStatus, ContinuationScheduleKind, ContinuationSolverConfig, ContinuationStepReport, ContinuationResult... |
| twpa/solvers/hb_solver.py | twpa.solvers.hb_solver ====================== Dense Newton solver and shared harmonic-balance solver reports | SolverStatus, LinearSolveMethod, NormKind, DenseNewtonConfig, NewtonIterationRecord, HBSolverReport... |
| twpa/solvers/linear_solvers.py | twpa.solvers.linear_solvers =========================== Shared linear-solver wrappers for the TWPA harmonic-balance stack | LinearSolveStatus, LinearSolverMethod, PreconditionerProtocol, LinearOperator, IterativeLinearSolveConfig, IterativeLinearSolveResult... |
| twpa/solvers/newton_krylov.py | twpa.solvers.newton_krylov ========================== Matrix-free Newton-Krylov solver for large harmonic-balance systems | JacobianVectorProductMethod, NewtonKrylovStatus, NewtonKrylovConfig, NewtonKrylovIterationRecord, NewtonKrylovResult, _norm... |
| twpa/solvers/preconditioners.py | twpa.solvers.preconditioners ============================ Preconditioners for structured TWPA harmonic-balance linear systems | PreconditionerKind, PreconditionerStatus, PreconditionerConfig, Preconditioner, identity_preconditioner, diagonal_preconditioner_from_diagonal... |
| twpa/workflows/__init__.py | twpa.workflows ============== Production workflows for TWPA simulation, calibration, and benchmarks. | __getattr__, __dir__ |
| twpa/workflows/calibration.py | twpa.workflows.calibration ========================== Parameter-extraction and calibration workflow for KI-TWPA simulations | _jsonify, write_json, write_npz, ParameterTransform, CalibrationParameterSpec, CalibrationVectorSpec... |
| twpa/workflows/gain_map.py | Package-native pump-HB plus small-signal gain-map orchestration. | NativeGainMapResult, solve_native_gain_map, export_native_gain_map_artifacts |
| twpa/workflows/industrial_100mm.py | twpa.workflows.industrial_100mm =============================== Production-facing workflow for a 100 mm / 20,000-cell KI-TWPA simulator | IndustrialStageStatus, IndustrialRunMode, _jsonify, write_json, write_npz, IndustrialLayoutSpec... |
| twpa/workflows/synthetic_benchmarks.py | twpa.workflows.synthetic_benchmarks =================================== Synthetic benchmark suite for the JAX-backed TWPA simulator | _jsonify, write_json, write_npz, BenchmarkStatus, SyntheticLayoutKind, SyntheticBenchmarkStage... |
### twpa_solver_old/

#### Legacy/Historical/Auxiliary (49 files)

| File | Functionality / observed role | Top-level symbols |
|---|---|---|
| twpa_solver_old/__init__.py | First-principles modular TWPA solver package. | - |
| twpa_solver_old/experiments/__init__.py | Executable experiments for the new TWPA solver. | - |
| twpa_solver_old/experiments/benchmark_solvers.py | Reduced solver comparison benchmark. | main |
| twpa_solver_old/experiments/compare_backend_5x5_to_jc_reference.py | Compare backend-substitution 5x5 runs to the stored JosephsonCircuits reference. | main, _classify, _missing_summary, _read_rows, _lookup, _float_or_none... |
| twpa_solver_old/experiments/compare_python_to_old_julia_map.py | Compare Python old-Julia parity maps against old Julia reference maps. | main, compare_rows, _summary, _best_point, _nanmax_abs, _write_summary... |
| twpa_solver_old/experiments/plot_all_backend_outputs.py | Plot canonical all-backend old-IPM map outputs. | main, _read_rows, _float_or_nan, _grid, _plot_heatmap, _plot_status_counts... |
| twpa_solver_old/experiments/plot_ipm_gain_map.py | Plot IPM gain-map artifacts. | plot_gain_map, _heatmap, main |
| twpa_solver_old/experiments/run_exported_julia_circuit_map.py | Run smoke map calculations from an exported Harmonia old-IPM circuit JSON. | main, _run_row, _write_rows, _write_grid, _write_status_grid, _write_report... |
| twpa_solver_old/experiments/run_ipm_25x25_gain_map.py | Run an IPM pump/conversion gain map with convergence artifacts. | main, _solve, _write_outputs, _apply_acceptance_tolerance, _parse_bool, _is_old_julia_parity... |
| twpa_solver_old/experiments/scaling_benchmark.py | Scaling benchmark for reduced-marker and physical-coupler IPM topologies. | main, _run_benchmark, _run_case, _write_rows, _write_summary, _plot... |
| twpa_solver_old/experiments/solve_old_ipm_backend_point.py | Single-point backend adapter for the canonical Julia old-IPM map runner. | main, _solve_independent_backend, _solve_scipy_least_squares, _complete_backend_schema, _base_metadata, _write_residual_history... |
| twpa_solver_old/experiments/solver_readiness.py | Generate solver production-readiness classification. | SolverReadiness, readiness_rows, write_readiness_report, main |
| twpa_solver_old/experiments/validate_conversion.py | Generate conversion S-parameter validation artifacts. | main, _validation_rows, _conversion_for_amplitude, _josephson_derivative_error, _row |
| twpa_solver_old/importers/__init__.py | Import external circuit netlists into twpa_solver models. | - |
| twpa_solver_old/importers/julia_circuit_json.py | Importer for Harmonia/JosephsonCircuits-style exported circuit JSON. | ImportedJuliaCircuit, _LinearInductor, import_julia_circuit_json, _collect_node_labels, _node_sort_key, _idx... |
| twpa_solver_old/model/__init__.py | Circuit model assembly primitives. | - |
| twpa_solver_old/model/blocks.py | Reusable topology blocks. | CircuitBlock, PortBlock, DirectionalCouplerBlock, DirectionalCouplerMarkerBlock, LinearCapacitorBlock, LinearInductorBlock... |
| twpa_solver_old/model/graph.py | Graph helpers for node-flux circuit models. | Branch, incidence_matrix, branch_fluxes |
| twpa_solver_old/model/ipm.py | IPM JTWPA topology assembled from reusable blocks. | IPMConfig, build_ipm_jtwpa, build_ipm_jtwpa_reduced_marker, build_ipm_jtwpa_physical_coupler, old_julia_parity_config, build_ipm_jtwpa_old_constants_compact_surrogate... |
| twpa_solver_old/model/nonlinearities.py | Explicit nonlinear branch laws independent of solver code. | Nonlinearity, JosephsonNonlinearity, RFSQUIDNonlinearity, KineticInductanceNonlinearity |
| twpa_solver_old/model/ports.py | Port definitions and wave normalization helpers. | Port, voltage_current_to_waves |
| twpa_solver_old/model/topology.py | Topology assembly for modular nonlinear node-flux circuits. | CircuitModel, CircuitBuilder, coupled_inductor_stiffness, coupled_inductor_branch_current |
| twpa_solver_old/model/units.py | SI constants and RF source conversion helpers. | PhysicalConstants, dbm_to_watts, watts_to_dbm, dbm_to_norton_current_rms, dbm_to_current_peak, dbm_to_old_julia_peak_current... |
| twpa_solver_old/residuals/__init__.py | Residual builders for nonlinear TWPA solves. | - |
| twpa_solver_old/residuals/aft_hb.py | Fourier pseudo-spectral pump-only harmonic-balance residual. | PumpAFTConfig, PumpAFTResidual |
| twpa_solver_old/residuals/conversion.py | Linearized pumped conversion-matrix residual assembly. | ConversionResult, build_conversion_sparameters, _fourier_coefficients, _sideband_port_admittance, _db |
| twpa_solver_old/residuals/jax_aft_hb.py | JAX-native AFT/HB residual for fixed-size TWPA models. | JaxPumpAFTResidual |
| twpa_solver_old/residuals/linear.py | Linear frequency-domain admittance and S-parameter solves. | LinearSParameters, nodal_voltage_admittance, port_admittance, y_to_s, solve_linear_sparameters |
| twpa_solver_old/residuals/scaling.py | Residual scaling helpers. | safe_scale |
| twpa_solver_old/residuals/time_domain.py | Time-domain residual helpers for validation solvers. | second_order_state_rhs |
| twpa_solver_old/residuals/two_tone.py | Two-tone HB frequency-grid scaffold for compression/intermodulation. | TwoToneIndex, make_two_tone_grid |
| twpa_solver_old/solvers/__init__.py | Nonlinear solver wrappers. | - |
| twpa_solver_old/solvers/anderson.py | Basic Anderson acceleration for fixed-point iterations. | anderson_accelerate |
| twpa_solver_old/solvers/arclength.py | Pseudo-arclength continuation scaffold. | ArclengthPoint, trace_scalar_branch |
| twpa_solver_old/solvers/base.py | Shared solver result containers. | SolverResult, classify_residual, result_from_residual |
| twpa_solver_old/solvers/continuation.py | Parameter continuation utilities. | continue_parameter, snake_grid_indices |
| twpa_solver_old/solvers/deflation.py | Multistart and branch clustering hooks. | BranchCluster, cluster_solutions, run_multistart |
| twpa_solver_old/solvers/jax_dense_newton.py | Dense JAX Newton solver for small residual systems. | solve_jax_dense_newton |
| twpa_solver_old/solvers/jax_newton_krylov.py | Matrix-free Newton-Krylov scaffold using JAX JVPs and SciPy GMRES. | solve_jax_newton_krylov |
| twpa_solver_old/solvers/mor.py | Model-order-reduction hooks for later conversion sweeps. | ReducedModelPlaceholder, build_full_order_conversion_system, sample_transfer_function, fit_reduced_model, evaluate_reduced_model |
| twpa_solver_old/solvers/preconditioners.py | Preconditioner builders for TWPA nonlinear residuals. | build_linear_passive_preconditioner, build_linear_passive_operator_matrix |
| twpa_solver_old/solvers/pseudo_transient.py | Pseudo-transient continuation globalizer. | solve_pseudo_transient |
| twpa_solver_old/solvers/scipy_least_squares.py | SciPy least-squares nonlinear solver baseline. | solve_least_squares |
| twpa_solver_old/solvers/scipy_root.py | SciPy root and Newton-Krylov wrappers. | solve_root, solve_newton_krylov |
| twpa_solver_old/solvers/shooting.py | Small-system periodic shooting validation solver. | solve_periodic_shooting |
| twpa_solver_old/sparams/__init__.py | S-parameter utilities. | - |
| twpa_solver_old/sparams/conversion_matrix.py | Public conversion-matrix S-parameter API. | - |
| twpa_solver_old/sparams/gain.py | Gain conversion helpers. | power_gain_db |
| twpa_solver_old/sparams/waves.py | Wave normalization wrappers. | - |
### root/

#### Legacy/Historical/Auxiliary (1 files)

| File | Functionality / observed role | Top-level symbols |
|---|---|---|
| sitecustomize.py | Repository-local Python startup hooks. | - |

## Confirmed architecture notes

- `src/twpa_solver/` is the current package for circuit construction, pump solving, signal/Floquet analysis, I/O, and plotting.
- `experiments/` contains chronological research and probe scripts; some directly use the current package while others preserve older investigation paths.
- `twpa_solver_old/` is a separate legacy solver stack with its own model, residual, solver, importer, and S-parameter modules.
- `tests/` spans both current and legacy behavior; its classification above follows imports.
- `scripts/` is a mixed operational layer: dataset builders, campaigns, plotting, reports, validation, and diagnostics.

## Classification caveat

The group assignment is based on import evidence and repository location. A script with no direct solver import may still be invoked by another current workflow; those indirect relationships should be checked before cleanup.
