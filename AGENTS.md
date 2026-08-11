# Project working notes

- Dependency freedom: free Python packages may be installed when they materially
  improve the implementation or validation. The preinstalled environment is not
  a hard dependency limit; verify compatibility and record the package used.

## Branch and artifact policy

- `main` is the clean, installable package branch.
- `dev` is the development and cross-machine synchronization branch; it may
  contain experiments, generated artifacts, and incomplete work.
- Keep ignore rules compatible, but do not rely on `.gitignore` to separate
  tracked files. Ignored files are not synchronized; tracked files merge.
- Do not merge `dev` wholesale into `main`. Promote production work with
  focused commits and `git cherry-pick`, or use a manually curated merge.
- Keep generated outputs under ignored `outputs/` or another disposable path;
  source design YAML belongs under `designs/`.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
