## Third-Party Notices

This project is licensed under MIT (`LICENSE`).

The application may invoke or install third-party tools at runtime. You are responsible for complying with each third-party license when redistributing binaries.

### FFmpeg and video codecs

- `stviz-animate` can call an external `ffmpeg` binary for MP4 encoding.
- Packaging scripts skip bundling `ffmpeg` by default.
- If you choose to bundle `ffmpeg`, you must verify and satisfy the license obligations of the exact binary build you ship.
- Builds that include `libx264` are commonly GPL-governed and may require GPL compliance for that bundled binary distribution.
- H.264/x264 may also involve patent licensing considerations in some jurisdictions.

### Python converter dependencies

The app installs converter dependencies from `python/requirements.txt` on first run. These packages are downloaded from PyPI and licensed by their respective authors:

- `anndata`
- `h5py`
- `numpy`
- `pandas`
- `scipy`
- `opencv-python-headless`

Before redistribution, review and preserve required notices for the exact versions you ship.

### Rust crate dependencies

This project depends on Rust crates listed in `Cargo.toml` / `Cargo.lock`, each under its own license terms.

Recommended release process:

1. Generate a dependency license report (for example with `cargo-license`).
2. Include third-party license texts in release artifacts.
3. Keep this notice file in packaged distributions.
