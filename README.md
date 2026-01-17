# stviz-animate

GPU-accelerated viewer for large single-cell / spatial transcriptomics point clouds.
Animate transitions between coordinate spaces (spatial, UMAP, tSNE, PCA, ...), apply filters and color modes, and export presentation-ready videos.

## Highlights
- Fast GPU rendering for large datasets.
- Timeline keyframes with easing, color paths, and playback modes.
- Advanced timeline canvas for fine-grained sequencing and filtering (beta).
- Loop video export with quality presets (Current viewport, 1080p, 4K).
- Mock dataset generator for quick demos. The mock data generation needs some more work to look more tissue - like and potentially have other presets, but works for now.
- Built-in .h5ad -> .stviz converter with automatic Python venv setup.

## Quick start

## Running release zips

Windows:
1) Download the `stviz-animate-windows.zip` from GitHub Releases.
2) Extract it.
3) Run `stviz-animate.exe`.

NB: There will be a trusted developer warning message, which you can ignore.

Linux (Ubuntu):
1) Download the `stviz-animate-ubuntu.tar.gz` from GitHub Releases.
2) Extract it:
   ```bash
   tar -xzf stviz-animate-ubuntu.tar.gz
   ```
3) Run:
   ```bash
   ./stviz-animate/stviz-animate
   ```

### Windows

Manual install

1) Install Rust (stable) and Visual Studio Build Tools.
2) Run:

```powershell
cargo run --release
```

### macOS
1) Install Rust (stable).
2) Run:

```bash
cargo run --release
```

Note: without an Apple Developer ID, distributed macOS builds must be opened via right-click -> Open the first time.

### Linux (Ubuntu/Debian)
1) Install Rust (stable) and GTK dev libs for file dialogs.
2) Run:

```bash
sudo apt-get update
sudo apt-get install -y libgtk-3-dev
cargo run --release
```

## Data conversion (.h5ad -> .stviz)
- Drag a `.h5ad` file into the app or click the conversion box to pick a file.
- The converter creates a private Python venv in `.stviz_venv` as needed.

Manual conversion:

```bash
python -m pip install -U anndata h5py numpy pandas scipy
python python/export_stviz.py --input your_data.h5ad --output your_data.stviz
```

Requires Python 3.8+.

## Export
- Screenshot: saves a PNG to `output/`.
- Loop export: writes an MP4 to `output/`, with configurable fps, duration, and quality.
- ffmpeg is preferred. If missing, OpenCV is used as a fallback (installed into `.stviz_venv` on first use).

## Packaging
Scripts to produce portable bundles:

Windows:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\package_windows.ps1
```

macOS (requires cargo-bundle):
```bash
cargo install cargo-bundle
./scripts/package_mac.sh
```

Ubuntu:
```bash
./scripts/package_ubuntu.sh
```

## Docs
See `docs/stviz-animate_documentation.md` for a full user guide and workflow details.

## File format (.stviz)
Single file format:

```
[u64 json_len]
[json bytes]
[padding to 16-byte boundary]
[raw binary blocks] (offsets in JSON metadata)
```

The exporter aligns blocks to 4 bytes; the viewer memory-maps the file for fast access.
