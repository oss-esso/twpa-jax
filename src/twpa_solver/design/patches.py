"""Exact-target design patch operations."""

from __future__ import annotations

from typing import Any

from twpa_solver.builders.blocks import BuildContext
from twpa_solver.builders.registry import BLOCK_BUILDERS
from twpa_solver.design.errors import DesignSchemaError


def apply_patches(ctx: BuildContext, patches: list[dict[str, Any]]) -> None:
    for index, patch in enumerate(patches):
        action = patch.get("action")
        path = f"patches[{index}]"
        if action == "add":
            cfg = dict(patch)
            cfg.setdefault("name", f"patch_{index}")
            cfg.setdefault("kind", "capacitor")
            nodes = cfg.get("nodes")
            if not isinstance(nodes, (list, tuple)) or len(nodes) != 2:
                raise DesignSchemaError(f"{path}.nodes: expected two endpoints")
            cfg["nodes"] = [ctx.named_nodes.get(node, node) for node in nodes]
            BLOCK_BUILDERS["raw_element"](ctx, cfg, path)
            continue
        target = patch.get("target")
        if not isinstance(target, str):
            raise DesignSchemaError(f"{path}.target: expected an exact path")
        name = ctx.named_elements.get(target, target)
        matches = [element for element in ctx.circuit if element.name == name]
        if len(matches) != 1:
            raise DesignSchemaError(
                f"{path}.target {target!r}: expected one match, found {len(matches)}")
        if action == "set":
            if patch.get("field", "value") != "value":
                raise DesignSchemaError(f"{path}: only field 'value' is supported")
            if "value" not in patch:
                raise DesignSchemaError(f"{path}.value: required for set")
            matches[0].value = patch["value"]
        elif action == "remove":
            ctx.circuit.remove(matches[0])
        else:
            raise DesignSchemaError(f"{path}.action: unknown action {action!r}")
