"""Import smoke tests for the plotting package."""


def test_plotting_modules_import() -> None:
    import twpa_solver.plotting.candidates
    import twpa_solver.plotting.data
    import twpa_solver.plotting.maps
    import twpa_solver.plotting.metrics
    import twpa_solver.plotting.spectrum
    import twpa_solver.plotting.style

    assert twpa_solver.plotting.data is not None
