def test_builder_imports():
    import twpa_solver.builders.ipm as ipm
    import twpa_solver.builders.scattered as scattered
    import twpa_solver.builders.jc_doc as jc_doc

    assert ipm is not None
    assert scattered is not None
    assert jc_doc is not None
