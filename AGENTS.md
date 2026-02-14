# Repository Guidelines

## Project Structure & Module Organization
- `src/`: Rust application code (UI, rendering, data loading).
- `python/`: dataset/export utilities (`export_stviz.py`, `export_video_cv2.py`, `mock_spatial_dataset.py`).
- `assets/`: shaders and icons.
- `data/`: sample datasets (if present).
- `docs/`: technical documentation (`technical_documentation.md`, `technical_documentation.pdf`).
- `scripts/`: packaging helpers for Windows/macOS/Linux.
- `dist/`: packaged release artifacts (tracked in repo).
- `output/`: runtime exports (screenshots, recordings, videos).
- `target/`: Cargo build artifacts (do not commit).
- `.stviz_venv/`: auto-created Python venv for conversion (do not commit).

## Build, Test, and Development Commands
- `cargo run` — build and run the app in debug mode.
- `cargo run --release` — optimized build for performance testing.
- `cargo build --release` — produce a release binary without running.
- `cargo fmt` — format Rust code with rustfmt (if installed).
- `cargo test` — run tests if/when added.
- `python python/export_stviz.py --input file.h5ad --output file.stviz` — convert datasets manually.
- `powershell -ExecutionPolicy Bypass -File scripts\package_windows.ps1` — Windows packaging.
- `cargo install cargo-bundle` — install macOS packaging helper (one-time).
- `./scripts/package_mac.sh` — macOS packaging.
- `./scripts/package_ubuntu.sh` — Ubuntu packaging.

## Coding Style & Naming Conventions
- Rust: follow rustfmt defaults (4-space indent, trailing commas).
- Naming: `snake_case` for functions/vars, `UpperCamelCase` for types, `SCREAMING_SNAKE_CASE` for constants.
- UI strings: keep labels concise and consistent (e.g., "Open .stviz", "Apply gene").

## Testing Guidelines
- No automated tests currently. Use `cargo test` if/when tests are added.
- For UI changes, run the app and validate interactions (timeline, filters, gene coloring).
- If adding tests, prefer module-level tests in the relevant `src/*.rs` file.

## Commit & Pull Request Guidelines
- Git history is minimal (single "first commit"), so no established commit format yet.
- Use short, imperative summaries (e.g., "Add keyframe gene coloring").
- PRs should include: a brief description, UI screenshots for visual changes, and notes on manual test coverage.

## Configuration Notes
- The exporter requires Python with `anndata`, `h5py`, `numpy`, `pandas`, `scipy`.
- The exporter will create `.stviz_venv/` in the repo as needed.
- The app prefers an active virtualenv/conda interpreter if detected.

## Licensing Notes
- Current license: MIT (see `LICENSE`).
- If changing licensing terms, update `LICENSE` and any references in `README.md`, and avoid mixing incompatible terms.
- "Academic/personal free, commercial paid" is a custom, non-OSI license; treat it as source-available.
