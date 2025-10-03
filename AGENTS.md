# EveNet_NERSC Agent Handbook

Welcome! This file captures the house rules for contributing to the EveNet project. It applies to the entire repository unless a deeper `AGENTS.md` overrides any point.

## Project orientation
- **Domains**
  - `evenet/`: Lightning + Ray training engine, model components, shared utilities.
  - `preprocessing/`: Dataset preparation CLIs that emit parquet shards and metadata.
  - `share/`: Example YAML configs and reusable option fragments; keep examples runnable.
  - `docs/`: User-facing guides—update when behavior or CLI flags change.
  - `downstreams/`: Reference analyses consuming EveNet outputs; treat as examples, not production.
  - `Docker/`, `NERSC/`: Environment recipes for containers and HPC deployments.
- Keep configuration-driven design. Prefer adding toggles or YAML options instead of hard-coding paths.

## Coding conventions
- Follow **PEP 8** with Black-like formatting (4-space indents, double quotes where practical, trailing commas in multi-line literals).
- Use type hints for new public functions and dataclasses; annotate return types explicitly when feasible.
- Log through the existing utilities (`evenet.utilities.logger`) rather than printing directly.
- Keep module-level imports at top-level; do not wrap imports in try/except blocks to silence missing dependencies.
- When modifying Lightning modules, maintain parity between training and prediction paths (e.g., keep tensor shapes and normalization consistent).
- For CLI scripts, provide clear `argparse` help strings and default values mirroring the documentation.

## Documentation expectations
- Update the relevant Markdown guide in `docs/` whenever you introduce a new option, flag, or workflow step.
- When adding configuration templates, ensure they have concise inline comments and fit the documented schema.
- Surface breaking changes or new user actions in `README.md` or the appropriate tutorial.

## Testing & validation
- Prefer lightweight smoke tests due to heavy dependencies: e.g., run targeted unit tests or a short training dry-run using the sample configs in `share/`.
- Validate preprocessing changes by running the CLI on a tiny sample dataset (`--limit` style flags) when available.
- If you touch Ray or Lightning integration, ensure distributed launch still works by running on at least two local workers where possible.

## Git & PR workflow
- Keep commits focused. Reference affected subsystems in commit messages (e.g., `preprocessing: ...`).
- Document behavioral changes and new configuration fields in your PR description. Include reproduction or testing steps.
- Before opening a PR, ensure generated artifacts (checkpoints, large datasets) are excluded via `.gitignore`.

Thanks for helping keep EveNet healthy and welcoming!
