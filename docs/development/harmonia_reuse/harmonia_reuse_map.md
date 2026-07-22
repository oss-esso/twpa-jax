# Harmonia Reuse Map

Read-only archaeology for future Julia-engine/Python-calibration architecture.

Decision: `Harmonia.jl` is stable Julia package target. `Harmonia` is exploratory reference. `twpa_jax` owns orchestration, readers, datasets, calibration, Bayesian optimization, SBI, ML, and reports. `JosephsonCircuits.jl` remains authoritative HB backend.

## Legend

Flags: `JC` calls/imports JosephsonCircuits; `Ckt` builds circuits; `Save` writes data; `Fit` has fitting/calibration logic. `Y` confirmed, `N` absent in inspected source, `?` uncertain. Family rows enumerate files with same role to keep map readable.

## Harmonia.jl package target

| file | class | physical/computational role | JC | Ckt | Save | Fit | action |
|---|---|---|---|---|---|---|---|
| `Harmonia.jl/src/Harmonia.jl` | `STABLE_PACKAGE_CODE` | Package entry point; includes and exports blocks | N | N | N | N | Promote as package root; remove placeholder exports later |
| `Harmonia.jl/src/Transmission_line_block.jl` | `STABLE_PACKAGE_CODE` | TL, JTL, RF-JTL and JJ netlist appenders | N | Y | N | N | Promote |
| `Harmonia.jl/src/JTWPA_standard_block.jl` | `STABLE_PACKAGE_CODE` | RPM resonator and JTL-RPM block appenders | N | Y | N | N | Promote |
| `Harmonia.jl/src/CPW_Theory.jl` | `STABLE_PACKAGE_CODE` | Edge-coupled CPW formulas and geometry optimization | N | N | N | Y | Promote |
| `Harmonia.jl/src/directional_coupler_block.jl` | `STABLE_PACKAGE_CODE` | Convert CPW design to discrete coupler and append netlist | N | Y | N | Y | Promote |
| `Harmonia.jl/src/IPM.jl` | `STABLE_PACKAGE_CODE` | Build interferometric phase-matching topology from blocks | N | Y | N | N | Promote |
| `Harmonia.jl/src/Save_data.jl` | `REUSABLE_PROTOTYPE` | Write HDF5, JSON metadata and serialized Julia objects | N | N | Y | N | Promote after schema redesign; do not expose `.jls` as Python contract |
| `Harmonia.jl/test/runtests.jl` | `REUSABLE_PROTOTYPE` | Minimal package smoke tests | N | Y | N | N | Expand |
| `Harmonia.jl/test/Project.toml` | `STABLE_PACKAGE_CODE` | Test environment metadata | N | N | N | N | Keep |

## Exploratory reusable core

| file | class | physical/computational role | JC | Ckt | Save | Fit | action |
|---|---|---|---|---|---|---|---|
| `Harmonia/core/modules/circuit_simulators.jl` | `REUSABLE_PROTOTYPE` | Broad TWPA/IPM/RF-SQUID builders; update lumped parameters; run `hbsolve` | Y | Y | N | N | Copy concepts into package APIs first |
| `Harmonia/core/modules/CPW_Theory.jl` | `OBSOLETE` | Earlier copy of CPW/coupler theory | N | N | N | Y | Ignore; package copy supersedes it |
| `Harmonia/core/modules/cost_functions.jl` | `REUSABLE_PROTOTYPE` | Norm, spectral and regularization objectives | N | N | N | Y | Python calibration reference only |
| `Harmonia/core/modules/optimization_utils.jl` | `REUSABLE_PROTOTYPE` | Optim/BlackBoxOptim wrappers and simulation objective loop | Y | N | N | Y | Python calibration reference only |
| `Harmonia/core/modules/preprocessing_utils.jl` | `REUSABLE_PROTOTYPE` | Filter, straighten, rescale and downsample measured traces | N | N | N | Y | Python preprocessing reference only |
| `Harmonia/core/modules/postprocessing.jl` | `REUSABLE_PROTOTYPE` | FFT frequency/value extraction helpers | N | N | N | N | Copy conceptually where engine output needs derived metrics |
| `Harmonia/core/modules/io_utils.jl` | `REUSABLE_PROTOTYPE` | CSV/Sonnet import and heatmap CSV persistence | N | N | Y | N | Python reader reference only |
| `Harmonia/core/fit_simulation_to_data.jl` | `REUSABLE_PROTOTYPE` | End-to-end TWPA fit driver | Y | Y | N | Y | Reference only; orchestration belongs in Python |
| `Harmonia/core/fit_simulation_to_data_tutorial.jl` | `REUSABLE_PROTOTYPE` | Tutorial duplicate of fit driver | Y | Y | N | Y | Reference only |
| `Harmonia/core/circuit_simulations_script.jl` | `REUSABLE_PROTOTYPE` | Manual HB simulation example | Y | Y | N | N | Reference for first CLI |
| `Harmonia/core/target_data/convert_mat2csv.py` | `REUSABLE_PROTOTYPE` | Convert measurement MAT data to CSV | N | N | Y | N | Move concept to `twpa_jax` ingestion |

## Marius campaign scripts

| file | class | physical/computational role | JC | Ckt | Save | Fit | action |
|---|---|---|---|---|---|---|---|
| `Harmonia/core/User_scripts/Marius/TWPA_PM_nDCs.jl` | `ONE_OFF_EXPERIMENT` | Sweep DC count for phase-matched TWPA gain | Y | Y | Y | N | Reference campaign shape |
| `Harmonia/core/User_scripts/Marius/TWPA_PM_nDCs_scattering.jl` | `ONE_OFF_EXPERIMENT` | Sweep DC count and scattering | Y | Y | Y | N | Reference campaign shape |
| `Harmonia/core/User_scripts/Marius/TWPA_PM_saturation_sims.jl` | `ONE_OFF_EXPERIMENT` | Saturation/pump-power sweep | Y | Y | Y | N | Reference campaign shape |
| `Harmonia/core/User_scripts/Marius/TWPA_PM_JJarrays.jl` | `ONE_OFF_EXPERIMENT` | JJ-array-count sweep | Y | Y | ? | N | Reference campaign shape |
| `Harmonia/core/User_scripts/Marius/TWPA_PM_GainLengthInvest.jl` | `ONE_OFF_EXPERIMENT` | Gain-versus-length sweep | Y | Y | Y | N | Reference campaign shape |
| `Harmonia/core/User_scripts/Marius/TWPA_PM_GainLengthInvest_TutorialVersion.jl` | `ONE_OFF_EXPERIMENT` | Short tutorial version of length sweep | Y | Y | Y | N | Reference only |
| `Harmonia/core/User_scripts/Marius/gain_sim_vs_data_plots.jl` | `REUSABLE_PROTOTYPE` | Compare measured and simulated gain traces | Y | Y | Y | Y | Copy plotting/report concepts into Python |

## Antoine_dev archaeology

All rows below are exploratory scripts, not package-quality. Direct `JosephsonCircuits.jl` use and inline constants dominate.

| files | class | physical/computational role | JC | Ckt | Save | Fit | action |
|---|---|---|---|---|---|---|---|
| `Directional_coupler_dev.jl` | `REUSABLE_PROTOTYPE` | Directional-coupler derivation/prototype | Y | Y | N | Y | Reference package coupler blocks |
| `TWPA_PM_2DCs_RFsquid.jl` | `REUSABLE_PROTOTYPE` | RF-SQUID phase-matched TWPA with two couplers | Y | Y | N | N | Reference for later topology |
| `Flux_driven_3WM_JTWPA.jl`, `Flux_driven_3WM_RFJTWPA.jl`, `Flux_driven_3WM_SQTWPA.jl`, `Flux_driven_4WM_RFJTWPA.jl`, `Flux_bias_I_pump_4WM_RFJTWPA.jl`, `FI_driven_3WM_RFJTWPA_paper.jl` | `REUSABLE_PROTOTYPE` | Flux/current-driven TWPA variants, 3WM/4WM | Y | Y | N | N | Reference only after baseline CLI |
| `RFJTL_3WM_DC.jl`, `RFJTL_3WM_DC_flux.jl`, `RFJTL_4WM_0DC.jl` | `REUSABLE_PROTOTYPE` | RF-JTL variants with DC/flux and mixing mode switches | Y | Y | N | N | Reference only |
| `JPA_fluxpumped_DC_SQUID.jl`, `JPA_fluxpumped_DC_SQUID_from_PCN.jl`, `JPA_fluxpumped_RF_SQUID.jl`, `JPA_fluxpumped_RF_SQUID_from_PCN.jl`, `JPA_fluxpumped_N_RF_SQUID.jl`, `JPA_fluxpumped_N_RF_SQUID_Kaufman.jl`, `JPA_fluxpumped_N_RF_SQUID_Kaufman_with_PCN.jl`, `JPA_fluxpumped_N_RF_SQUID_LC_IMPA.jl`, `IMJPA_fluxpumped_N_RF_SQUID.jl` | `ONE_OFF_EXPERIMENT` | Flux-pumped JPA/IMPA circuit variants | Y | Y | N | N | Reference only |
| `IMPA_Naman.jl`, `IMPA_Naman_TS0812.jl`, `IMPA_Naman_TS0812PM.jl`, `IMPA_Naman_TS0812_RDSQ.jl`, `IMPA_Naman_TS0813_l4.jl`, `IMPA_Naman_TS0813_l4s.jl`, `IMPA_Naman_TS0912_Distrib.jl`, `IMPA_Naman_TS0912_Distrib_matched1.jl`, `IMPA_Naman_TS0912_Distrib_matched2.jl`, `IMPA_Naman_TS0912_Distrib_matched3.jl`, `IMPA_Naman_TS1812_Matched_l4_l2.jl` | `ONE_OFF_EXPERIMENT` | IMPA matching and distributed resonator investigations | Y | Y | N | N | Reference only |
| `Bandpass_3_Chebyschev_PCN.jl`, `Bandpass_filters_design_from_PCN.jl`, `Bandpass_filters_order_1_design_from_PCN.jl`, `Bandpass_filters_order_3_design_from_PCN.jl`, `Bandpass_filters_order_5_design_from_PCN.jl`, `Bandpass_order_3_from_PNC_with_PI_networks.jl`, `Cauer_expansion_prototypes.py`, `Reflectionless_filters.jl` | `REUSABLE_PROTOTYPE` | Prototype filter synthesis and HB validation | Y | Y | N | Y | Copy concepts only if filters enter roadmap |
| `Negative_resistance_PA.jl`, `Negative_resistance_IMPA.jl`, `Investigation_lambda4.jl`, `Qubit_simulation.jl` | `ONE_OFF_EXPERIMENT` | Device-specific nonlinear investigations | Y | Y | N | N | Reference only |
| `Distributed_JPA_calculations.ipynb` | `UNKNOWN` | Notebook; not deeply inspected | ? | ? | ? | ? | Reference only |

## Harmonia top-level scripts

| files | class | physical/computational role | JC | Ckt | Save | Fit | action |
|---|---|---|---|---|---|---|---|
| `IPM_JTWPA.jl`, `IPM_rf_squid.jl`, `IPM_rf_squid_Cg.jl`, `IPM_rf_squid_Lwp.jl`, `IPM_old_new_scattering.jl`, `broadcasting.jl`, `broadcast_oldIPM.jl`, `broadcast_test_iter.jl`, `Interferometer.jl`, `simple_interferometer.jl` | `REUSABLE_PROTOTYPE` | IPM/interferometer topology and scattering variants | Y | Y | mixed | N | Reference package `IPM.jl`; extract missing variants selectively |
| `directional_coupler.jl`, `Coupler_Tests.jl` | `REUSABLE_PROTOTYPE` | Coupler construction and tests | Y | Y | N | Y | Reference package coupler tests |
| `JTWPA_standard.jl`, `JTWPA_fluxpumped.jl`, `JTWPA_floquet.jl`, `JTWPA_floquet_losses.jl` | `REUSABLE_PROTOTYPE` | Standard, flux-pumped and Floquet JTWPA simulations | Y | Y | mixed | N | Reference simulation options |
| `JTL_ethz_noPM.jl`, `JTL_ethz_noPM_JJTapered.jl`, `JTL_ethz_noPM_manTapered_experimental.jl`, `JTL_ethz_PM.jl`, `JTL_ethz_PM_scan.jl`, `JTL_ethz_rf_PM_scan.jl`, `Old_JTL_ethz_PM.jl` | `ONE_OFF_EXPERIMENT` | ETHZ JTL topology and taper/phase-matching scans | Y | Y | mixed | N | Reference only; `Old_*` is obsolete |
| `RF_squid.jl`, `RF_squid_test_ws_num.jl`, `rf_SQUID_2D_plot.jl`, `replot_2D_rf_SQUID.jl` | `ONE_OFF_EXPERIMENT` | RF-SQUID simulation, benchmark and plotting | Y | Y | mixed | N | Reference campaign/schema examples |
| `JPA_standard.jl`, `JPA_fluxpumped.jl`, `JPA_2pumps.jl` | `ONE_OFF_EXPERIMENT` | JPA HB examples | Y | Y | N | N | Reference only |
| `50JJ_bench_test.jl`, `50JJ_bench_test_ite.jl`, `50JJ_bench_test_Npump_modulation.jl`, `50JJ_bench_test_pumpharmonia.jl` | `ONE_OFF_EXPERIMENT` | HB runtime scaling benchmarks | Y | Y | mixed | N | Keep as benchmark references |
| `JJ Circuit_old_new_scatter.jl` | `OBSOLETE` | Old/new JJ circuit scattering comparison | Y | Y | N | N | Reference only |
| `read_2D_results.jl`, `read_bench_data.jl`, `read_saved_circuit.jl` | `REUSABLE_PROTOTYPE` | Load HDF5/JLS/JLD2 outputs and plot/read results | N | N | N | N | Reference Python HDF5 reader compatibility |
| `test.jl`, `testHarmonia.jl` | `OBSOLETE` | Tiny ad-hoc smoke scripts | mixed | mixed | N | N | Ignore |

## Saved exploratory artifacts

| files | class | role | action |
|---|---|---|---|
| `Harmonia/IPM_JTWPA/2D_HeatingPlot/2026-04-22/*/PumpSweep.h5`, `PumpSweep.jls` | `ONE_OFF_EXPERIMENT` | Two IPM pump sweep result snapshots | Reference schema/examples only |
| `Harmonia/50JJ/Benchmark_SimTime/2026-04-17/23-43-22_Nmodulationharmonics-scaling-benchmark/Nmod_sweep.h5`, `Nmod_sweep.jls`, `Manifest.toml` | `ONE_OFF_EXPERIMENT` | Runtime benchmark snapshot and environment | Keep as benchmark record |
| `Harmonia/JTWPA_PumpSweep_Data.jld2` | `ONE_OFF_EXPERIMENT` | Legacy pump sweep result | Reference only |

## Uncertainty

Classification means reuse readiness, not scientific validity. `STABLE_PACKAGE_CODE` marks code already organized as package source, not proof of production hardening. Binary `.h5`, `.jls`, `.jld2`, and notebook internals were inventoried but not deeply decoded. No simulation ran.
