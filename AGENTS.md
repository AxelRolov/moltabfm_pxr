# Repository Guidelines

## Project Structure & Module Organization
`evaluation/` contains scoring code for activity and structure submissions. `validation/` holds format and chemistry checks for CSV and ZIP submissions. `notebooks/` contains the tutorial workflows and also serves as the integration test surface in CI. `inputs/` stores reference protein assets, while `outputs/` contains example or generated submissions and should only be committed when they are intentional examples. Root metadata lives in `pyproject.toml`; `environment.yaml` remains as a legacy Conda setup.

## Build, Test, and Development Commands
```bash
uv sync
uv sync --group test
uv run jupyter lab
uv run pytest -n=auto --nbmake --nbmake-timeout=1200 --maxfail=0 --disable-warnings notebooks/
```
`uv sync` installs runtime dependencies. Add `--group test` when you need notebook test tooling. Launch JupyterLab for interactive tutorial work, and run the `pytest` command before opening a PR. Use `environment.yaml` only if you specifically need the older Conda flow.

## Coding Style & Naming Conventions
Target Python 3.11+ and follow the existing PEP 8-style conventions: 4-space indentation, `snake_case` for functions and modules, `UPPER_CASE` for constants, and type hints on public helpers. Prefer `pathlib.Path` for filesystem paths. Keep imports grouped as standard library, third-party, and local. No formatter or linter is configured in-repo, so match surrounding style and keep notebook cells deterministic and rerunnable from top to bottom.

## Testing Guidelines
CI executes notebooks on macOS and Ubuntu with Python 3.12 using `pytest`, `nbmake`, and `pytest-xdist`. Add new tutorials under `notebooks/` with descriptive `snake_case` filenames. If you change scoring or validation logic, also smoke-test against sample files in `outputs/` so submission formats still match challenge expectations.

## Commit & Pull Request Guidelines
Recent history uses short, imperative, lowercase commit subjects such as `add ...`, `migrate ...`, and `replace ...`. Keep commits focused and explain the reason when touching evaluation logic or generated outputs. PRs should state which track is affected (`activity` or `structure`), list the notebooks or modules changed, note any external data or OST dependency assumptions, and call out intentional artifacts committed under `outputs/`. Include screenshots only when notebook visualizations changed materially.
