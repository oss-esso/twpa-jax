"""Build the plot set for the fold_plan.md Milestones A-D validation campaign
(scripts/validate_fold_plan_ad.py). Reads its CSV/JSON outputs, writes PNGs.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAMPAIGN_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("D:/tmp/fold_ad_campaign")
OUT_DIR = CAMPAIGN_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load(name: str) -> list[dict]:
    with (CAMPAIGN_DIR / name).open() as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            if k == "segment_id":
                r[k] = int(v)
            elif v not in (None, ""):
                try:
                    r[k] = float(v)
                except ValueError:
                    pass
    return rows


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=130)
    plt.close(fig)
    print("wrote", OUT_DIR / name)


legacy = load("branch_79ghz_legacy_ds02.csv")
adaptive = load("branch_79ghz_adaptive_ds02.csv")
adaptive01 = load("branch_79ghz_adaptive_ds01.csv")
t70 = load("branch_70ghz_adaptive.csv")

summary = json.loads((CAMPAIGN_DIR / "summary.json").read_text())

# --- Plot 1: mu(s), 7.9 GHz adaptive ---
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot([r["s"] for r in adaptive], [r["mu"] for r in adaptive], "-", lw=1.2, color="#2563eb")
fold_mu = summary["adaptive_fold_refined_mu"]
if fold_mu is not None:
    ax.axhline(fold_mu, color="#dc2626", ls="--", lw=1, label=f"refined fold mu={fold_mu:.4f}")
ax.set_xlabel("arclength s (cumulative step_size)")
ax.set_ylabel("mu")
ax.set_title("7.9 GHz: mu(s) -- adaptive step control")
ax.legend()
save(fig, "01_mu_vs_s_79ghz.png")

# --- Plot 3: t_mu(s) ---
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot([r["s"] for r in adaptive], [r["t_mu"] for r in adaptive], "-", lw=1.2, color="#059669")
ax.axhline(0.0, color="#111827", lw=0.7)
ax.set_xlabel("arclength s")
ax.set_ylabel("t_mu (tangent lambda-component)")
ax.set_title("7.9 GHz: t_mu(s) -- zero crossing at the fold")
save(fig, "03_tmu_vs_s_79ghz.png")

# --- Plot 4: ds(s) ---
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot([r["s"] for r in adaptive], [r["step_size"] for r in adaptive], "-", lw=1.0, color="#7c3aed")
ax.set_yscale("log")
ax.set_xlabel("arclength s")
ax.set_ylabel("ds (log scale)")
ax.set_title("7.9 GHz: adaptive ds(s)")
save(fig, "04_ds_vs_s_79ghz.png")

# --- Plot 5: Newton iterations(s) ---
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot([r["s"] for r in adaptive], [r["used_newton"] for r in adaptive], "-", lw=1.0, color="#ea580c")
ax.set_xlabel("arclength s")
ax.set_ylabel("Newton iterations (accepted step)")
ax.set_title("7.9 GHz: Newton effort(s)")
save(fig, "05_newton_vs_s_79ghz.png")

# --- Plot 6: tangent angle(s) ---
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot([r["s"] for r in adaptive], [r["theta_deg"] for r in adaptive], "-", lw=1.0, color="#0891b2")
ax.set_xlabel("arclength s")
ax.set_ylabel("tangent angle theta (deg)")
ax.set_title("7.9 GHz: curvature theta(s)")
save(fig, "06_theta_vs_s_79ghz.png")

# --- Combined C-comparison: legacy vs adaptive ds(s) ---
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot([r["s"] for r in legacy], [r["step_size"] for r in legacy], "-", lw=1.0,
        color="#9ca3af", label="legacy step_control")
ax.plot([r["s"] for r in adaptive], [r["step_size"] for r in adaptive], "-", lw=1.2,
        color="#7c3aed", label="adaptive step_control")
ax.set_yscale("log")
ax.set_xlabel("arclength s")
ax.set_ylabel("ds (log scale)")
ax.set_title("7.9 GHz: legacy vs adaptive ds(s)")
ax.legend()
save(fig, "04b_ds_legacy_vs_adaptive_79ghz.png")

# --- D5 bar chart: min eigenvalue before/at/after ---
d5 = summary["d5_rows"]
fig, ax = plt.subplots(figsize=(6, 4.5))
labels = [r["label"] for r in d5]
vals = [r["min_eigenvalue"] for r in d5]
colors = ["#2563eb" if v < 0 else "#dc2626" for v in vals]
ax.bar(labels, vals, color=colors)
ax.axhline(0.0, color="#111827", lw=0.7)
ax.set_ylabel("Jacobian min eigenvalue")
ax.set_title("7.9 GHz: Jacobian eigenvalue crosses zero at the fold")
save(fig, "07_jacobian_eig_before_at_after_79ghz.png")

# --- Plot 8: 7.0 GHz mu/ds/state_scale/t_mu panels ---
fig, axes = plt.subplots(4, 1, figsize=(7, 10), sharex=True)
axes[0].plot([r["s"] for r in t70], [r["mu"] for r in t70], color="#2563eb")
axes[0].set_ylabel("mu")
axes[1].plot([r["s"] for r in t70], [r["step_size"] for r in t70], color="#7c3aed")
axes[1].set_yscale("log")
axes[1].set_ylabel("ds (log)")
axes[2].plot([r["s"] for r in t70], [r["state_scale"] for r in t70], color="#059669")
axes[2].set_ylabel("state_scale")
axes[3].plot([r["s"] for r in t70], [r["t_mu"] for r in t70], color="#ea580c")
axes[3].axhline(0.0, color="#111827", lw=0.7)
axes[3].set_ylabel("t_mu")
axes[3].set_xlabel("arclength s")
fig.suptitle("7.0 GHz regression: mu, ds, state_scale, t_mu(s)")
save(fig, "08_70ghz_panels.png")

# --- Overview: mu(s) with fold markers for all three 7.9GHz runs ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot([r["s"] for r in legacy], [r["mu"] for r in legacy], color="#9ca3af", lw=1, label="legacy ds=0.02")
ax.plot([r["s"] for r in adaptive], [r["mu"] for r in adaptive], color="#7c3aed", lw=1.2, label="adaptive ds=0.02")
ax.plot([r["s"] for r in adaptive01], [r["mu"] for r in adaptive01], color="#059669", lw=1, label="adaptive ds=0.01")
ax.axhline(summary["adaptive_fold_refined_mu"], color="#dc2626", ls="--", lw=0.8,
           label=f"ds=0.02 fold mu={summary['adaptive_fold_refined_mu']:.4f}")
ax.axhline(summary["adaptive_ds01_fold_refined_mu"], color="#f59e0b", ls="--", lw=0.8,
           label=f"ds=0.01 fold mu={summary['adaptive_ds01_fold_refined_mu']:.4f}")
ax.set_xlabel("arclength s")
ax.set_ylabel("mu")
ax.set_title("7.9 GHz: fold location vs step size (D4) -- two distinct features")
ax.legend(fontsize=8)
save(fig, "09_d4_ds_sensitivity_79ghz.png")

print("all plots written to", OUT_DIR)
