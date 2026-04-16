# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `src/somehand/`. Keep domain models in `src/somehand/domain`, orchestration in `src/somehand/application`, adapters and I/O in `src/somehand/infrastructure`, CLI entrypoints in `src/somehand/cli`, and runtime/viewer logic in `src/somehand/runtime`. Retargeting presets live under `configs/retargeting/{left,right,bihand}`. Tests are in `tests/` and usually mirror the feature they cover, for example `tests/test_cli.py` or `tests/test_urdf_converter.py`. Utility scripts live in `scripts/`. Treat `assets/` and `recordings/` as local/generated data; large runtime assets are fetched, not committed. Vendor code stays in `third_party/`.

## Build, Test, and Development Commands
- `git submodule update --init --recursive` — sync SDK submodules before first run.
- `pip install -e ".[dev]"` — install the package plus `pytest` and `ruff`.
- `python scripts/setup/download_assets.py --only mjcf mediapipe` — fetch required runtime assets.
- `somehand --help` — inspect the CLI surface.
- `pytest -q` — run the full test suite.
- `pytest tests/test_cli.py -q` — run a focused regression pass for one area.
- `ruff check src tests` — lint Python sources and tests.

## Coding Style & Naming Conventions
Target Python 3.10+ and follow existing PEP 8 style: 4-space indentation, `snake_case` for modules/functions, `PascalCase` for classes and dataclasses, and explicit type hints on public helpers. Keep modules small and layered; prefer extending existing `domain / application / infrastructure / runtime` boundaries over adding cross-layer shortcuts. Match the current terse docstring style and avoid adding dependencies without a strong reason.

## Testing Guidelines
Use `pytest`. Name new tests `tests/test_<feature>.py` and test functions `test_<behavior>()`. Add regression coverage with each bug fix or refactor, especially for CLI parsing, config loading, and runtime adapters. Hardware-dependent flows should be mocked so tests stay deterministic in CI and local development.

## Documentation Guidelines
Keep `README.md` as a landing page only: project introduction, setup pointers, and quick runnable examples. Detailed user and maintainer documentation belongs under `docs/`. Documentation changes must update `docs/en` and `docs/zh` together with mirrored filenames and equivalent meaning. Documentation must match the current code, CLI help, default paths, checked-in configs, and verified behavior; do not document planned or unsupported capabilities as available.

## Commit & Pull Request Guidelines
Recent history uses short, intent-first subjects such as `Keep live mocap recordings on a fixed cadence`. Follow that with Lore-style trailers when useful: `Constraint:`, `Rejected:`, `Confidence:`, `Scope-risk:`, `Directive:`, `Tested:`, and `Not-tested:`. PRs should summarize why the change exists, list touched configs/assets, include executed test commands, and attach screenshots or short clips for viewer/UI changes.

## Assets & Configuration Tips
Do not commit generated MJCF files, downloaded models, or sample recordings into `assets/` or `recordings/`. Fetch or refresh them through `scripts/setup/download_assets.py`, and keep external asset locations aligned with `src/somehand/external_assets.py`.
