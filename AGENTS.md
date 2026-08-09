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
