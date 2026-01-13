# Repository Guidelines

## Project Structure & Module Organization
- `src/`: Rust application code (UI, rendering, data loading).
- `python/export_stviz.py`: .h5ad → .stviz exporter.
- `assets/`, `data/`: static resources and sample data (if present).
- `output/`: runtime output folder created by the app (screenshots, recordings, exports).
- `target/`: Cargo build artifacts (do not commit).

## Build, Test, and Development Commands
- `cargo run` — build and run the app in debug mode.
- `cargo run --release` — optimized build for performance testing.
- `cargo build --release` — produce a release binary without running.
- `cargo fmt` — format Rust code with rustfmt (if installed).
- `python python/export_stviz.py --input file.h5ad --output file.stviz` — convert datasets manually.

## Coding Style & Naming Conventions
- Rust: follow rustfmt defaults (4-space indent, trailing commas).
- Naming: `snake_case` for functions/vars, `UpperCamelCase` for types, `SCREAMING_SNAKE_CASE` for constants.
- UI strings: keep labels concise and consistent (e.g., “Open .stviz”, “Apply gene”).

## Testing Guidelines
- No automated tests currently. Use `cargo test` if/when tests are added.
- For UI changes, run the app and validate interactions (timeline, filters, gene coloring).
- If adding tests, prefer module-level tests in the relevant `src/*.rs` file.

## Commit & Pull Request Guidelines
- Git history is minimal (single “first commit”), so no established commit format yet.
- Use short, imperative summaries (e.g., “Add keyframe gene coloring”).
- PRs should include: a brief description, UI screenshots for visual changes, and notes on manual test coverage.

## Configuration Notes
- The exporter requires Python with `anndata`, `h5py`, `numpy`, `pandas`, `scipy`.
- The app prefers an active virtualenv/conda interpreter if detected.
