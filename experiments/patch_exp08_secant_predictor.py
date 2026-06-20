from pathlib import Path
import re

path = Path("experiments/exp08_full_ipm_pump_solve.py")
text = path.read_text(encoding="utf-8")

backup = path.with_suffix(".py.before_secant_predictor")
if not backup.exists():
    backup.write_text(text, encoding="utf-8")

# Add setting field.
old = """    preconditioner: str
    compute_time_residual: bool
    verbose: bool"""
new = """    preconditioner: str
    compute_time_residual: bool
    verbose: bool
    continuation_predictor: str"""
if old not in text:
    raise SystemExit("Could not find NewtonKrylovSettings fields.")
text = text.replace(old, new)

# Replace solve_continuation.
pattern = r"""    def solve_continuation\(
        self,
        problem: FullIPMPumpProblem,
        continuation_steps: int,
    \) -> tuple\[np\.ndarray, list\[StepReport\]\]:
.*?
        return X, reports
"""

replacement = """    def solve_continuation(
        self,
        problem: FullIPMPumpProblem,
        continuation_steps: int,
    ) -> tuple[np.ndarray, list[StepReport]]:
        reports: list[StepReport] = []

        lambdas = np.linspace(1.0 / continuation_steps, 1.0, continuation_steps)

        X_prevprev: np.ndarray | None = None
        X_prev = problem.zeros()
        lam_prevprev: float | None = None
        lam_prev: float | None = None

        for lam_raw in lambdas:
            lam = float(lam_raw)

            X_guess = X_prev

            if (
                self.settings.continuation_predictor == "secant"
                and X_prevprev is not None
                and lam_prev is not None
                and lam_prevprev is not None
                and abs(lam_prev - lam_prevprev) > 0.0
            ):
                beta = (lam - lam_prev) / (lam_prev - lam_prevprev)
                X_guess = X_prev + beta * (X_prev - X_prevprev)

            print(f"\\n=== continuation lambda={lam:.6f} ===")
            X_new, report = self.solve_one(problem, X_guess, lam)
            reports.append(report)

            status = "VALID_CONVERGED" if report.converged else "FAIL"
            msg = (
                f"step_status={status} "
                f"coeff_rel={report.coeff_rel:.3e} "
                f"newton={report.newton_iterations} "
                f"gmres_total={report.gmres_iterations_total} "
                f"factor_s={report.factor_runtime_s:.3f} "
                f"runtime_s={report.runtime_s:.3f} "
                f"reason={report.failure_reason}"
            )
            if report.time_rel is not None:
                msg = msg.replace(
                    f"newton={report.newton_iterations}",
                    f"time_rel={report.time_rel:.3e} newton={report.newton_iterations}",
                )
            print(msg)

            if not report.converged:
                return X_new, reports

            X_prevprev = X_prev
            lam_prevprev = lam_prev
            X_prev = X_new
            lam_prev = lam

        return X_prev, reports
"""

new_text, n = re.subn(pattern, replacement, text, flags=re.S)
if n != 1:
    raise SystemExit(f"Could not replace solve_continuation; replacements={n}")
text = new_text

# Add CLI flag.
old = """    p.add_argument("--continuation-steps", type=int, default=20)"""
new = """    p.add_argument("--continuation-steps", type=int, default=20)
    p.add_argument("--continuation-predictor", choices=["none", "secant"], default="none")"""
if old not in text:
    raise SystemExit("Could not find continuation-steps argument.")
text = text.replace(old, new)

# Add to settings construction.
old = """        compute_time_residual=not args.skip_time_residual,
        verbose=not args.quiet,
    )"""
new = """        compute_time_residual=not args.skip_time_residual,
        verbose=not args.quiet,
        continuation_predictor=args.continuation_predictor,
    )"""
if old not in text:
    raise SystemExit("Could not find settings construction.")
text = text.replace(old, new)

# Print setting.
old = """    print(f"continuation_steps={args.continuation_steps}")
    print(f"preconditioner={args.preconditioner}")"""
new = """    print(f"continuation_steps={args.continuation_steps}")
    print(f"continuation_predictor={args.continuation_predictor}")
    print(f"preconditioner={args.preconditioner}")"""
if old in text:
    text = text.replace(old, new)

# Add metadata.
old = """        "continuation_steps": args.continuation_steps,
        "real_unknowns": 2 * args.harmonics * ipm.C.shape[0],"""
new = """        "continuation_steps": args.continuation_steps,
        "continuation_predictor": args.continuation_predictor,
        "real_unknowns": 2 * args.harmonics * ipm.C.shape[0],"""
if old in text:
    text = text.replace(old, new)

path.write_text(text, encoding="utf-8")
print("PATCH_OK secant continuation predictor applied")
print(f"backup={backup}")
