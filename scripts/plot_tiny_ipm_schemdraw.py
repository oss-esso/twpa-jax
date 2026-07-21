"""Draw the tiny IPM probe circuit as a real schematic using schemdraw.

Same tiny circuit as scratchpad tiny_ipm_probe.py / plot_tiny_ipm_skrf.py
(built via the real exp07 builder functions): array_length=3, num_rows=2,
arrays_per_dc=1. Node x-position along each rail follows the same convention
(sorted node id -> rail order); the Josephson element is drawn as a plain
inductor labeled "Lj" (not the schemdraw Josephson-junction symbol) per
request -- this is a linearized topology sketch, not a JJ schematic.

mutual_inductor_k elements are not directly stampable as a single two-term
component here and are only annotated (dashed line + "k" label), not drawn
as a real transformer symbol.

Usage:
    python scripts/plot_tiny_ipm_schemdraw.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import schemdraw  # noqa: E402
import schemdraw.elements as elm  # noqa: E402
import exp07_python_ipm_design_builder as ipm  # noqa: E402


def build_tiny_circuit():
    params = ipm.IPMParams(
        array_length=3,
        num_rows=2,
        arrays_per_dc=1,
        len1=2, len2=2, len3=2, len4=2,
        length_of_long_TL=2,
        length_of_short_TL=2,
        coupler_section_length=2,
        cached_coupler_length_um=20.0,
        cell_length_um=10.0,
    )
    coupler = ipm.make_coupler_discrete(params, "cached")
    circuit, ends = ipm.make_ipm(params, coupler)
    return circuit, ends, params


def main() -> int:
    elements, ends, params = build_tiny_circuit()

    top_nodes, bot_nodes = [], []
    for e in elements:
        for n in (e.n1, e.n2):
            if isinstance(n, int) and n != 0:
                if n < params.start_node_bot and n not in top_nodes:
                    top_nodes.append(n)
                elif n >= params.start_node_bot and n not in bot_nodes:
                    bot_nodes.append(n)
    top_x = {n: i * 1.5 for i, n in enumerate(sorted(top_nodes))}
    bot_x = {n: i * 1.5 for i, n in enumerate(sorted(bot_nodes))}

    Y_TOP, Y_BOT = 3.0, -3.0
    Y_GND_TOP, Y_GND_BOT = 1.0, -1.0

    def xy(node: int) -> tuple[float, float]:
        if node in top_x:
            return (top_x[node], Y_TOP)
        return (bot_x[node], Y_BOT)

    d = schemdraw.Drawing()
    d.config(unit=1, fontsize=9)

    skipped_mutual = []

    for e in elements:
        if e.kind == "port":
            x, y = xy(e.n1)
            yofs = 0.7 if y > 0 else -0.7
            d += (elm.Line().at((x, y)).to((x, y + yofs)))
            d += (elm.Dot().at((x, y + yofs)))
            d += (elm.Label(label=f"P{e.value}").at((x, y + yofs * 1.15)))
            continue

        if e.kind == "mutual_inductor_k":
            skipped_mutual.append(e.name)
            continue

        if e.kind == "coupling_capacitor":
            x1, y1 = xy(e.n1)
            x2, y2 = xy(e.n2)
            d += (elm.Capacitor().at((x1, y1)).to((x2, y2))
                  .label("Cc", loc="center", fontsize=6).linestyle("--"))
            continue

        n1, n2 = e.n1, e.n2
        if n1 == 0 or n2 == 0:
            rail_node = n2 if n1 == 0 else n1
            x, y_rail = xy(rail_node)
            y_gnd = Y_GND_TOP if y_rail > 0 else Y_GND_BOT
            label = "Cg" if e.kind == "capacitor" else "Cj"
            comp = elm.Capacitor if e.kind == "capacitor" else elm.Capacitor2
            d += (comp().at((x, y_rail)).to((x, y_gnd)).label(label, fontsize=6))
            d += elm.Ground().at((x, y_gnd))
            continue

        x1, y1 = xy(n1)
        x2, y2 = xy(n2)
        if e.kind == "josephson_inductor":
            d += (elm.Inductor().at((x1, y1)).to((x2, y2)).label("Lj", fontsize=6))
        elif e.kind == "linear_inductor":
            d += (elm.Inductor().at((x1, y1)).to((x2, y2)).label("L", fontsize=6))
        elif e.kind == "capacitor":
            d += (elm.Capacitor().at((x1, y1)).to((x2, y2)).label("Cj", fontsize=6))
        elif e.kind == "resistor":
            d += (elm.Resistor().at((x1, y1)).to((x2, y2)).label(f"R={e.value:.3g}", fontsize=6))
        else:
            raise ValueError(f"unhandled element kind: {e.kind}")

    print(f"elements: {len(elements)}  (skipped mutual_inductor_k: {skipped_mutual})")

    out = ROOT / "outputs" / "tiny_ipm_probe_schemdraw.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    d.save(str(out), transparent=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
