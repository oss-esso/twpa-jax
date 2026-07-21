"""Draw the tiny IPM probe circuit topology using scikit-rf's Circuit graph.

Same tiny circuit as the scratchpad tiny_ipm_probe.py (built via the real
exp07 builder functions), but the nonlinear Josephson element is represented
as a plain linear inductor labeled "Lj" and the netlist is handed to
skrf.circuit.Circuit for its networkx graph (skrf.Circuit.graph). plot_graph()
itself uses spring_layout, which scrambles a ladder into a hairball, so this
draws the same bipartite (element-node <-> net-node) graph manually with a
fixed rail layout: top rail y=+1, bottom rail y=-1, ground y=0, x = node
order along each rail (same convention as the scratchpad's hand-drawn
version). Averaging each element's neighboring net positions to place its
own node happens to reproduce a ladder/stub schematic for free (series
elements land mid-rail, shunt caps land mid-way down to the ground rail).

mutual_inductor_k elements are not stampable as a simple 2-port lumped
network in this graph view and are skipped (printed, not drawn).

Usage:
    python scripts/plot_tiny_ipm_skrf.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import networkx as nx  # noqa: E402
import skrf as rf  # noqa: E402
import exp07_python_ipm_design_builder as ipm  # noqa: E402

KIND_STYLE = {
    "josephson_inductor": ("tab:red", "s"),
    "linear_inductor": ("black", "s"),
    "capacitor": ("tab:blue", "o"),
    "coupling_capacitor": ("tab:green", "o"),
    "resistor": ("tab:purple", "s"),
    "port": ("tab:red", ">"),
    "ground": ("grey", "o"),
}


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
    return circuit, ends


def main() -> int:
    params_start_node_bot = ipm.IPMParams().start_node_bot
    elements, ends = build_tiny_circuit()

    freq = rf.Frequency(7.5, 7.5, 1, unit="GHz")
    media = rf.media.DefinedGammaZ0(frequency=freq, z0=50.0)

    nodes: dict[int, list[tuple]] = {}
    kind_of: dict[str, str] = {}

    def add(node: int, net_port: tuple) -> None:
        nodes.setdefault(node, []).append(net_port)

    ground = rf.circuit.Circuit.Ground(freq, name="gnd", z0=50.0)
    ground_used = False
    skipped_mutual = []

    for i, e in enumerate(elements, 1):
        if e.kind == "port":
            port_net = rf.circuit.Circuit.Port(freq, name=f"P{e.value}", z0=50.0)
            add(e.n1, (port_net, 0))
            kind_of[port_net.name] = "port"
            continue

        if e.kind == "mutual_inductor_k":
            skipped_mutual.append(e.name)
            continue

        if e.kind in ("josephson_inductor", "linear_inductor"):
            label = "Lj" if e.kind == "josephson_inductor" else "L"
            net = media.inductor(e.value, name=f"{label}{i}")
        elif e.kind in ("capacitor", "coupling_capacitor"):
            label = "Cc" if e.kind == "coupling_capacitor" else "C"
            net = media.capacitor(e.value, name=f"{label}{i}")
        elif e.kind == "resistor":
            net = media.resistor(e.value, name=f"R{i}")
        else:
            raise ValueError(f"unhandled element kind: {e.kind}")
        kind_of[net.name] = e.kind

        n1, n2 = e.n1, e.n2
        if n1 == 0:
            if not ground_used:
                add(0, (ground, 0))
                ground_used = True
                kind_of["gnd"] = "ground"
            add(0, (net, 0))
            add(n2, (net, 1))
        elif n2 == 0:
            if not ground_used:
                add(0, (ground, 0))
                ground_used = True
                kind_of["gnd"] = "ground"
            add(0, (net, 1))
            add(n1, (net, 0))
        else:
            add(n1, (net, 0))
            add(n2, (net, 1))

    connections = list(nodes.values())
    circuit = rf.circuit.Circuit(connections)

    print(f"elements: {len(elements)}  (skipped mutual_inductor_k: {skipped_mutual})")
    print(f"nets: {len(connections)}")

    # -- fixed ladder-rail layout, same convention as scratchpad tiny_ipm_probe.py --
    node_order = list(nodes.keys())  # index k <-> connections[k] <-> graph node 'Xk'
    top_nodes = sorted(n for n in node_order if n != 0 and n < params_start_node_bot)
    bot_nodes = sorted(n for n in node_order if n != 0 and n >= params_start_node_bot)
    top_x = {n: i for i, n in enumerate(top_nodes)}
    bot_x = {n: i for i, n in enumerate(bot_nodes)}

    pos: dict[str, tuple[float, float]] = {}
    for k, node_id in enumerate(node_order):
        xk = f"X{k}"
        if node_id == 0:
            pos[xk] = (5.0, 0.0)
        elif node_id in top_x:
            pos[xk] = (top_x[node_id], 1.0)
        else:
            pos[xk] = (bot_x[node_id], -1.0)

    G = circuit.graph()
    for name in kind_of:
        neighbor_positions = [pos[nb] for nb in G.neighbors(name) if nb in pos]
        x = sum(p[0] for p in neighbor_positions) / len(neighbor_positions)
        y = sum(p[1] for p in neighbor_positions) / len(neighbor_positions)
        if len(neighbor_positions) == 1:  # port stub: push out past the rail
            y += 0.3 if y >= 0 else -0.3
        pos[name] = (x, y)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(20, 7))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="grey", width=1.0)
    for kind, (color, shape) in KIND_STYLE.items():
        names = [n for n, k in kind_of.items() if k == kind]
        if not names:
            continue
        nx.draw_networkx_nodes(G, pos, nodelist=names, ax=ax, node_color=color,
                                node_shape=shape, node_size=120)
    labels = {n: n for n in kind_of}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=6,
                             verticalalignment="bottom")
    ax.set_title("tiny IPM probe topology (scikit-rf Circuit graph, ladder-rail layout, "
                 "Lj as linear inductor)")
    ax.axis("off")
    fig.tight_layout()
    out = ROOT / "outputs" / "tiny_ipm_probe_skrf.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
