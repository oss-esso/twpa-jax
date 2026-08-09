# Harmonia Design Candidate Map

Purpose: decide what to promote first into stable Harmonia/JosephsonCircuits templates.

## Recommended implementation order

1. `harmonia_jtl_topology_smoke`: use `add_JTL!` to generate a tiny JTL netlist; validate element counts and metadata only.
2. `harmonia_jtl_linear_jc_smoke`: same tiny JTL netlist, passed to JosephsonCircuits linearized S if compatible.
3. `harmonia_rf_jtl_topology_smoke`: use `add_RF_JTL!` to generate a tiny RF-SQUID/JTL netlist; validate topology.
4. `harmonia_coupler_topology_smoke`: use `generate_and_append_coupler!` with tiny/cheap settings; validate coupled mesh metadata.
5. `harmonia_ipm_topology_smoke`: only after coupler and JTL smoke tests pass, because `make_IPM` has a large positional signature.
6. Mine exploratory `Harmonia` scripts for realistic IPM/rf-SQUID/JTWPA parameters and fitting logic.

## Candidate files

### `Harmonia/IPM_old_new_scattering.jl`

- class: `OBSOLETE_OR_REFERENCE`
- tags: JTWPA, IPM, DIRECTIONAL_COUPLER, RPM
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: default, display, push!, run_harmonia_ideal, run_harmonia_scattered, run_legacy_ideal

### `Harmonia/JJ Circuit_old_new_scatter.jl`

- class: `OBSOLETE_OR_REFERENCE`
- tags: JTWPA, DIRECTIONAL_COUPLER, RPM, NOISE_QE_CM
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: plot!, push!

### `Harmonia/Old_JTL_ethz_PM.jl`

- class: `OBSOLETE_OR_REFERENCE`
- tags: JTWPA, DIRECTIONAL_COUPLER, RPM, NOISE_QE_CM
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: plot!, push!

### `Harmonia/broadcast_oldIPM.jl`

- class: `OBSOLETE_OR_REFERENCE`
- tags: JTWPA, IPM, DIRECTIONAL_COUPLER, RPM
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: display, get_max_gain_fast, push!

### `Harmonia/50JJ_bench_test.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, RPM
- action: Extract minimal solver call pattern and add a tiny schema-preserving smoke.
- functions: add_JTL_element!, display, hbsolve, push!

### `Harmonia/50JJ_bench_test_Npump_modulation.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, RPM
- action: Extract minimal solver call pattern and add a tiny schema-preserving smoke.
- functions: add_JTL_element!, display, hbsolve, push!

### `Harmonia/50JJ_bench_test_ite.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, RPM
- action: Extract minimal solver call pattern and add a tiny schema-preserving smoke.
- functions: add_JTL_element!, hbsolve, push!

### `Harmonia/50JJ_bench_test_pumpharmonia.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, RPM
- action: Extract minimal solver call pattern and add a tiny schema-preserving smoke.
- functions: add_JTL_element!, display, hbsolve, push!

### `Harmonia/Antoine_dev\Bandpass_3_Chebyschev_PCN.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: none
- action: Extract minimal solver call pattern and add a tiny schema-preserving smoke.
- functions: push!

### `Harmonia/Antoine_dev\Bandpass_filters_design_from_PCN.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: none
- action: Extract minimal solver call pattern and add a tiny schema-preserving smoke.
- functions: push!

### `Harmonia/Antoine_dev\Bandpass_filters_order_1_design_from_PCN.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: RPM
- action: Extract minimal solver call pattern and add a tiny schema-preserving smoke.
- functions: push!

### `Harmonia/Antoine_dev\Bandpass_filters_order_3_design_from_PCN.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: none
- action: Extract minimal solver call pattern and add a tiny schema-preserving smoke.
- functions: push!

### `Harmonia/Antoine_dev\Bandpass_filters_order_5_design_from_PCN.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: none
- action: Extract minimal solver call pattern and add a tiny schema-preserving smoke.
- functions: push!

### `Harmonia/Antoine_dev\Bandpass_order_3_from_PNC_with_PI_networks.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: none
- action: Extract minimal solver call pattern and add a tiny schema-preserving smoke.
- functions: plot!, push!

### `Harmonia/Antoine_dev\Directional_coupler_dev.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: DIRECTIONAL_COUPLER, FITTING
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: plot!, push!

### `Harmonia/Antoine_dev\FI_driven_3WM_RFJTWPA_paper.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, RF_SQUID, DIRECTIONAL_COUPLER, NOISE_QE_CM
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: build_circuit, entry, plot!, push!

### `Harmonia/Antoine_dev\Flux_bias_I_pump_4WM_RFJTWPA.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, RF_SQUID, DIRECTIONAL_COUPLER, NOISE_QE_CM
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: build_circuit, entry, plot!, push!

### `Harmonia/Antoine_dev\Flux_driven_3WM_JTWPA.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, NOISE_QE_CM
- action: Use as calibration-reference logic; port concepts into twpa_jax objectives/campaigns.
- functions: build_circuit, entry, plot!, push!

### `Harmonia/Antoine_dev\Flux_driven_3WM_RFJTWPA.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, RF_SQUID, DIRECTIONAL_COUPLER, NOISE_QE_CM
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: build_circuit, entry, plot!, push!

### `Harmonia/Antoine_dev\Flux_driven_3WM_SQTWPA.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, NOISE_QE_CM
- action: Use as calibration-reference logic; port concepts into twpa_jax objectives/campaigns.
- functions: build_circuit, entry, plot!, push!

### `Harmonia/Antoine_dev\Flux_driven_4WM_RFJTWPA.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, RF_SQUID, DIRECTIONAL_COUPLER, NOISE_QE_CM
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: build_circuit, entry, plot!, push!

### `Harmonia/Antoine_dev\IMJPA_fluxpumped_N_RF_SQUID.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: RF_SQUID, JPA
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: push!

### `Harmonia/Antoine_dev\IMPA_Naman.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: DIRECTIONAL_COUPLER, JPA
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: build_circuit, entry, push!

### `Harmonia/Antoine_dev\IMPA_Naman_TS0812.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: DIRECTIONAL_COUPLER, JPA
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: build_circuit, entry, push!

### `Harmonia/Antoine_dev\IMPA_Naman_TS0812PM.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: DIRECTIONAL_COUPLER, JPA
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: build_circuit, entry, push!

### `Harmonia/Antoine_dev\IMPA_Naman_TS0812_RDSQ.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: DIRECTIONAL_COUPLER, JPA
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: build_circuit, entry, push!

### `Harmonia/Antoine_dev\IMPA_Naman_TS0813_l4.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: DIRECTIONAL_COUPLER, JPA
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: build_circuit, entry, push!

### `Harmonia/Antoine_dev\IMPA_Naman_TS0813_l4s.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: DIRECTIONAL_COUPLER, JPA
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: build_circuit, entry, push!

### `Harmonia/Antoine_dev\IMPA_Naman_TS0912_Distrib.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: DIRECTIONAL_COUPLER, JPA
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: build_circuit, entry, push!

### `Harmonia/Antoine_dev\IMPA_Naman_TS0912_Distrib_matched1.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: DIRECTIONAL_COUPLER, JPA, NOISE_QE_CM
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: build_circuit, entry, push!

### `Harmonia/Antoine_dev\IMPA_Naman_TS0912_Distrib_matched2.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: DIRECTIONAL_COUPLER, JPA, NOISE_QE_CM
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: build_circuit, entry, push!

### `Harmonia/Antoine_dev\IMPA_Naman_TS0912_Distrib_matched3.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: DIRECTIONAL_COUPLER, JPA, NOISE_QE_CM
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: build_circuit, entry, push!

### `Harmonia/Antoine_dev\IMPA_Naman_TS1812_Matched_l4_l2.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: DIRECTIONAL_COUPLER, JPA
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: build_circuit, entry, push!

### `Harmonia/Antoine_dev\Investigation_lambda4.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: RPM, FITTING
- action: Use as calibration-reference logic; port concepts into twpa_jax objectives/campaigns.
- functions: push!

### `Harmonia/Antoine_dev\JPA_fluxpumped_DC_SQUID.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JPA
- action: Extract minimal solver call pattern and add a tiny schema-preserving smoke.
- functions: none

### `Harmonia/Antoine_dev\JPA_fluxpumped_DC_SQUID_from_PCN.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JPA
- action: Extract minimal solver call pattern and add a tiny schema-preserving smoke.
- functions: none

### `Harmonia/Antoine_dev\JPA_fluxpumped_N_RF_SQUID.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: RF_SQUID, JPA
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: print, push!

### `Harmonia/Antoine_dev\JPA_fluxpumped_N_RF_SQUID_Kaufman.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: RF_SQUID, JPA
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: print, push!

### `Harmonia/Antoine_dev\JPA_fluxpumped_N_RF_SQUID_Kaufman_with_PCN.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: RF_SQUID, JPA
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: print, push!

### `Harmonia/Antoine_dev\JPA_fluxpumped_N_RF_SQUID_LC_IMPA.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: RF_SQUID, JPA
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: push!

### `Harmonia/Antoine_dev\JPA_fluxpumped_RF_SQUID.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: RF_SQUID, JPA
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: none

### `Harmonia/Antoine_dev\JPA_fluxpumped_RF_SQUID_from_PCN.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: RF_SQUID, JPA
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: none

### `Harmonia/Antoine_dev\Negative_resistance_IMPA.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JPA
- action: Extract minimal solver call pattern and add a tiny schema-preserving smoke.
- functions: push!

### `Harmonia/Antoine_dev\Negative_resistance_PA.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JPA
- action: Extract minimal solver call pattern and add a tiny schema-preserving smoke.
- functions: push!

### `Harmonia/Antoine_dev\Qubit_simulation.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: FITTING
- action: Use as calibration-reference logic; port concepts into twpa_jax objectives/campaigns.
- functions: push!

### `Harmonia/Antoine_dev\RFJTL_3WM_DC.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, RF_SQUID
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: push!

### `Harmonia/Antoine_dev\RFJTL_3WM_DC_flux.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, RF_SQUID
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: push!

### `Harmonia/Antoine_dev\RFJTL_4WM_0DC.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, RF_SQUID
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: push!

### `Harmonia/Antoine_dev\Reflectionless_filters.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: none
- action: Extract minimal solver call pattern and add a tiny schema-preserving smoke.
- functions: push!

### `Harmonia/Antoine_dev\TWPA_PM_2DCs_RFsquid.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: RF_SQUID, DIRECTIONAL_COUPLER, FITTING, NOISE_QE_CM
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: Random.seed!, display, include, plot!, pythonplot, savefig

### `Harmonia/Coupler_Tests.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: DIRECTIONAL_COUPLER, RPM
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: plot!, push!

### `Harmonia/IPM_JTWPA.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, IPM, DIRECTIONAL_COUPLER, RPM
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: none

### `Harmonia/IPM_rf_squid.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, IPM, RF_SQUID, DIRECTIONAL_COUPLER, NOISE_QE_CM
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: push!

### `Harmonia/IPM_rf_squid_Cg.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, IPM, RF_SQUID, DIRECTIONAL_COUPLER, NOISE_QE_CM
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: push!

### `Harmonia/IPM_rf_squid_Lwp.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, IPM, RF_SQUID, DIRECTIONAL_COUPLER, NOISE_QE_CM
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: push!

### `Harmonia/Interferometer.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: DIRECTIONAL_COUPLER, RPM
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: plot!, push!

### `Harmonia/JPA_2pumps.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JPA
- action: Extract minimal solver call pattern and add a tiny schema-preserving smoke.
- functions: none

### `Harmonia/JPA_fluxpumped.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JPA
- action: Extract minimal solver call pattern and add a tiny schema-preserving smoke.
- functions: none

### `Harmonia/JPA_standard.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JPA
- action: Use as calibration-reference logic; port concepts into twpa_jax objectives/campaigns.
- functions: contourf, estimate_peak, estimate_spectrum

### `Harmonia/JTL_ethz_PM.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, DIRECTIONAL_COUPLER, RPM, NOISE_QE_CM
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: add_JJ!, add_JTL_element!, plot!, push!

### `Harmonia/JTL_ethz_PM_scan.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, RF_SQUID, DIRECTIONAL_COUPLER, RPM
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: display, elseif, plot!, push!

### `Harmonia/JTL_ethz_noPM.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, RPM, NOISE_QE_CM
- action: Extract minimal solver call pattern and add a tiny schema-preserving smoke.
- functions: plot!, push!, rpm.linearized.QE

### `Harmonia/JTL_ethz_noPM_JJTapered.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, RPM, NOISE_QE_CM
- action: Extract minimal solver call pattern and add a tiny schema-preserving smoke.
- functions: elseif, get_taper_factor, makeJJ, makePort, makeTL, plot!, push!, rpm.linearized.QE

### `Harmonia/JTL_ethz_noPM_manTapered_experimental.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, RPM, MEASUREMENT
- action: Extract minimal solver call pattern and add a tiny schema-preserving smoke.
- functions: C_exp, TWPA_circuit_simulation, get_Cg, make_JJ_element, make_JJ_taper_quadratic, make_TL_section, make_port, make_series_line_element, push!

### `Harmonia/JTL_ethz_rf_PM_scan.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, RF_SQUID, DIRECTIONAL_COUPLER, RPM, NOISE_QE_CM
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: display, plot!, push!

### `Harmonia/JTWPA_floquet.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, FLOQUET, NOISE_QE_CM
- action: Extract minimal solver call pattern and add a tiny schema-preserving smoke.
- functions: floquet.linearized.QE, plot!, push!

### `Harmonia/JTWPA_floquet_losses.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, FLOQUET, FITTING, NOISE_QE_CM
- action: Use as calibration-reference logic; port concepts into twpa_jax objectives/campaigns.
- functions: plot!, push!

### `Harmonia/JTWPA_fluxpumped.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, NOISE_QE_CM
- action: Use as calibration-reference logic; port concepts into twpa_jax objectives/campaigns.
- functions: build_circuit, entry, plot!, push!, sol.linearized.QE

### `Harmonia/JTWPA_standard.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, RPM, NOISE_QE_CM
- action: Extract minimal solver call pattern and add a tiny schema-preserving smoke.
- functions: add_JJ!, add_JTL_RPM_element!, add_JTL_element!, plot!, push!, rpm.linearized.QE

### `Harmonia/RF_squid.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, RF_SQUID, NOISE_QE_CM
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: add_one_period_24!, push!

### `Harmonia/RF_squid_test_ws_num.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, RF_SQUID, NOISE_QE_CM
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: add_one_period_24!, display, push!

### `Harmonia/broadcast_test_iter.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, RPM
- action: Extract minimal solver call pattern and add a tiny schema-preserving smoke.
- functions: add_JTL_element!, display, get_gain_at_Ip_and_Iter, hbsolve, push!

### `Harmonia/broadcasting.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, IPM, DIRECTIONAL_COUPLER, RPM
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: display, get_rpm_and_gain

### `Harmonia/core\User_scripts\Marius\TWPA_PM_GainLengthInvest.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: none
- action: Use as calibration-reference logic; port concepts into twpa_jax objectives/campaigns.
- functions: Random.seed!, display, include, plot!, push!, pythonplot, save_heatmap_to_csv, savefig

### `Harmonia/core\User_scripts\Marius\TWPA_PM_GainLengthInvest_TutorialVersion.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: none
- action: Use as calibration-reference logic; port concepts into twpa_jax objectives/campaigns.
- functions: display, include, pythonplot, save_heatmap_to_csv, savefig

### `Harmonia/core\User_scripts\Marius\TWPA_PM_JJarrays.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: none
- action: Use as calibration-reference logic; port concepts into twpa_jax objectives/campaigns.
- functions: display, include, plot!

### `Harmonia/core\User_scripts\Marius\TWPA_PM_nDCs.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: RF_SQUID, DIRECTIONAL_COUPLER, FITTING, NOISE_QE_CM
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: Random.seed!, display, include, plot!, pythonplot, savefig

### `Harmonia/core\User_scripts\Marius\TWPA_PM_nDCs_scattering.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: RF_SQUID, FITTING
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: Random.seed!, display, include, plot!, pythonplot

### `Harmonia/core\User_scripts\Marius\TWPA_PM_saturation_sims.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: FITTING
- action: Use as calibration-reference logic; port concepts into twpa_jax objectives/campaigns.
- functions: Random.seed!, display, include, plot!, pythonplot

### `Harmonia/core\User_scripts\Marius\gain_sim_vs_data_plots.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, FITTING, NOISE_QE_CM
- action: Use as calibration-reference logic; port concepts into twpa_jax objectives/campaigns.
- functions: CSV.write, cut_data, display, include, plot!, savefig

### `Harmonia/core\circuit_simulations_script.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: DIRECTIONAL_COUPLER, RPM
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: include, plot!, push!

### `Harmonia/core\fit_simulation_to_data.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, FITTING
- action: Use as calibration-reference logic; port concepts into twpa_jax objectives/campaigns.
- functions: include, simulation_function

### `Harmonia/core\fit_simulation_to_data_tutorial.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, FITTING
- action: Use as calibration-reference logic; port concepts into twpa_jax objectives/campaigns.
- functions: include, simulation_function

### `Harmonia/core\modules\CPW_Theory.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: DIRECTIONAL_COUPLER, FITTING
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: cost_function_2, cost_function_3, edgeCoupledCPW, ellipticalIntegral, ellipticalIntegralPrime, estimateEdgeCoupledDirectionalCoupler, getEdgeCoupledDirectionalCouplerParameters, return

### `Harmonia/core\modules\circuit_simulators.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: RF_SQUID, DIRECTIONAL_COUPLER, RPM, FITTING, NOISE_QE_CM
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: add_element, calculate_DC_coupling, error, hex, make_DC, make_DC_circuit, make_DC_section, make_DC_simple, make_JJ_element, make_RF_squid_element, make_TL_section, make_circuit_interferometer

### `Harmonia/core\modules\cost_functions.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: FITTING
- action: Use as calibration-reference logic; port concepts into twpa_jax objectives/campaigns.
- functions: bounds_regularization_cost, convex_sum_cost, linear_norm_cost, negative_values_cost, return, spectral_diff_cost, sum

### `Harmonia/core\modules\io_utils.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: none
- action: Inspect manually before promotion.
- functions: CSV.write, get_data_frame_columns, import_data_from_CSV, import_data_from_Sonnet, load_heatmap_from_csv, plot_data_frame_columns, rename!

### `Harmonia/core\modules\optimization_utils.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: FITTING
- action: Use as calibration-reference logic; port concepts into twpa_jax objectives/campaigns.
- functions: cfun, display, get_magnitude_Db, get_phase_from_complex, include, optimize_BBO, optimize_LBFGS, optimize_NelderMead, push!, rescale_parameters, unscale_parameters

### `Harmonia/core\modules\preprocessing_utils.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: none
- action: Extract minimal solver call pattern and add a tiny schema-preserving smoke.
- functions: downsample_data, extract_pump_frequency, filter_data, rescale_data, return, straighten_data

### `Harmonia/directional_coupler.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: DIRECTIONAL_COUPLER, RPM
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: plot!, print, push!

### `Harmonia/read_2D_results.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, IPM, RPM
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: display, h5open

### `Harmonia/read_bench_data.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, IPM
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: h5open

### `Harmonia/read_saved_circuit.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: RPM
- action: Extract minimal solver call pattern and add a tiny schema-preserving smoke.
- functions: display, h5open

### `Harmonia/replot_2D_rf_SQUID.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: RF_SQUID
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: display, h5open

### `Harmonia/rf_SQUID_2D_plot.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, RF_SQUID
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: add_one_period_24!, display, push!

### `Harmonia/simple_interferometer.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: DIRECTIONAL_COUPLER, RPM
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: plot!, push!

### `Harmonia.jl/scripts\inspect_harmonia_api.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JTWPA, IPM, DIRECTIONAL_COUPLER, RPM
- action: Mine as design recipe; promote only after reproducing with tiny smoke tests.
- functions: exit, inspect_symbol, main, mkpath, open, safe_methods_text, safe_which_text, write_section

### `Harmonia.jl/scripts\run_simulation.jl`

- class: `REUSABLE_PROTOTYPE`
- tags: JPA
- action: Extract minimal solver call pattern and add a tiny schema-preserving smoke.
- functions: abcd_to_s, abs, attrs, complex_to_real_imag_arrays, config_hash, error, exit, finite_complex_array, frequency_axis_hz, get_nested, h5open, haskey

### `Harmonia.jl/src\CPW_Theory.jl`

- class: `STABLE_PACKAGE_CODE`
- tags: DIRECTIONAL_COUPLER, FITTING
- action: Keep in Harmonia.jl; wrap with typed config/tests/status schema.
- functions: cost_function_2, cost_function_3, edgeCoupledCPW, ellipticalIntegral, ellipticalIntegralPrime, estimateEdgeCoupledDirectionalCoupler, getEdgeCoupledDirectionalCouplerParameters, return

### `Harmonia.jl/src\Harmonia.jl`

- class: `STABLE_PACKAGE_CODE`
- tags: JTWPA, IPM, DIRECTIONAL_COUPLER, RPM
- action: Keep in Harmonia.jl; wrap with typed config/tests/status schema.
- functions: add_waveguide, hello, include, push!

### `Harmonia.jl/src\IPM.jl`

- class: `STABLE_PACKAGE_CODE`
- tags: JTWPA, IPM, DIRECTIONAL_COUPLER
- action: Keep in Harmonia.jl; wrap with typed config/tests/status schema.
- functions: add_JTL_element!, make_IPM, push!

### `Harmonia.jl/src\JTWPA_standard_block.jl`

- class: `STABLE_PACKAGE_CODE`
- tags: JTWPA, RPM
- action: Keep in Harmonia.jl; wrap with typed config/tests/status schema.
- functions: add_JJ!, add_JTL_RPM_element!, add_RPM_element!, push!

### `Harmonia.jl/src\Save_data.jl`

- class: `STABLE_PACKAGE_CODE`
- tags: RPM, NOISE_QE_CM
- action: Keep in Harmonia.jl; wrap with typed config/tests/status schema.
- functions: Serialization.serialize, h5open, mkpath, open, save_hdf5

### `Harmonia.jl/src\Transmission_line_block.jl`

- class: `STABLE_PACKAGE_CODE`
- tags: JTWPA, RF_SQUID
- action: Keep in Harmonia.jl; wrap with typed config/tests/status schema.
- functions: add_JJ!, add_JTL!, add_JTL_element!, add_RF_JTL!, add_RF_JTL_element!, add_TL!, add_TL_element!, push!

### `Harmonia.jl/src\directional_coupler_block.jl`

- class: `STABLE_PACKAGE_CODE`
- tags: DIRECTIONAL_COUPLER
- action: Keep in Harmonia.jl; wrap with typed config/tests/status schema.
- functions: add_TL_element!, add_coupling!, add_directional_coupler!, add_edge_coupled_directional_coupler!, calculate_discrete_params, generate_and_append_coupler!, push!

