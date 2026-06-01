from __future__ import annotations

import argparse
import importlib
from collections.abc import Sequence


COMMAND_MODULES = {
    "linear": "scripts.linear_100mm_baseline",
    "dispersion": "scripts.extract_dispersion",
    "pump": "scripts.pump_hb_small_ladder",
    "gain": "scripts.gain_from_pumped_solution",
    "compression": "scripts.compression_sweep",
    "calibrate": "scripts.fit_measurements",
    "report": "scripts.make_run_report",
    "validate": "scripts.run_validation_suite",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="twpa",
        description="TWPA package CLI wrapper over supported script entry points.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="twpa 0.1.0",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    for name, module_name in COMMAND_MODULES.items():
        subparser = subparsers.add_parser(
            name,
            help=f"Run {module_name}.main(...)",
        )
        subparser.add_argument(
            "args",
            nargs=argparse.REMAINDER,
            help="Arguments forwarded to the underlying script.",
        )

    return parser


def _run_command(command: str, forwarded_args: Sequence[str]) -> int:
    module = importlib.import_module(COMMAND_MODULES[command])
    main = getattr(module, "main", None)
    if not callable(main):
        raise RuntimeError(f"{COMMAND_MODULES[command]} does not expose callable main(argv)")

    argv = list(forwarded_args)
    if argv and argv[0] == "--":
        argv = argv[1:]
    result = main(argv)
    return 0 if result is None else int(result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command is None:
        parser.print_help()
        return 0
    return _run_command(args.command, args.args)


if __name__ == "__main__":
    raise SystemExit(main())
