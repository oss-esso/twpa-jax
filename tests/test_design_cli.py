from pathlib import Path
from tempfile import TemporaryDirectory

from twpa_solver.core.circuit import load_circuit
from twpa_solver.design.__main__ import main


def test_generic_cli_writes_loadable_artifacts():
    with TemporaryDirectory(dir=Path.cwd()) as directory:
        outdir = Path(directory) / "compiled"
        main(["--design", "designs/uniform_jtwpa.yaml", "--outdir", str(outdir),
              "--write-matrices"])
        loaded = load_circuit(outdir)
        assert loaded.Ic.size == 10
        assert (outdir / "elements.csv").exists()
        assert (outdir / "design_resolved.json").exists()
