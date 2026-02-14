# stviz-animate

GPU-accelerated viewer for large single-cell / spatial transcriptomics point clouds.
Animate transitions between coordinate spaces (spatial, UMAP, tSNE, PCA, ...), apply filters and color modes, and export presentation-ready videos.

## Highlights
- Fast GPU rendering for large datasets.
- Timeline keyframes with easing, color paths, and playback modes.
- Advanced timeline canvas for fine-grained sequencing and filtering (not fully fleshed out).
- Loop video export with quality presets (e.g. 1080p, 4K).
- Mock dataset generator for quick demos. The mock data generation needs some more work to look more like real tissue and potentially have other presets, but works for now.
- Built-in .h5ad -> .stviz converter (runs in an app-managed user-data venv).

## Quick start

## Running release zips

Windows:
1) Download the `stviz-animate-windows.zip` from GitHub Releases.
2) Extract it.
3) Run `stviz-animate.exe`.

Ignore the trusted developer warning message if it shows up.

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

### Building it on Windows

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
- Writable runtime data lives in your user-data directory:
  - Windows: `%APPDATA%/stviz-animate/`
  - macOS: `~/Library/Application Support/stviz-animate/`
  - Linux: `${XDG_DATA_HOME:-~/.local/share}/stviz-animate/`
- The app creates and uses a managed venv at `<user-data>/.stviz_venv` for conversion and OpenCV fallback.
- A system Python install is still required so the app can create that managed venv.
- On first conversion/export-fallback run, dependencies are silently installed from `python/requirements.txt` (internet required).

Manual conversion:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r python/requirements.txt
python python/export_stviz.py --input your_data.h5ad --output your_data.stviz
```

Requires Python 3.8+ (Python 3.10+ recommended).

## Export
- Screenshot: saves a PNG to `<user-data>/output/screenshots/` at 4K (3840x2160).
- Loop export: writes frames under `<user-data>/output/exports/` and MP4 output under `<user-data>/output/`.
- Video encoding presets: Standard (CRF 23), High (CRF 18), Ultra (CRF 14), all using H.264 yuv420p for Windows compatibility.
- ffmpeg is preferred. If missing, OpenCV fallback is auto-installed in the managed converter venv on first use.

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

Packaging includes Python converter scripts. Converter dependencies are installed on first use (internet required).

## Docs
See `docs/technical_documentation.md` for architecture and workflow details.

## File format (.stviz)
Single file format:

```
[8-byte magic "STVIZ\0\0\0"]
[u32 version]
[u64 json_len]
[json bytes]
[padding to 16-byte boundary]
[raw binary blocks] (offsets in JSON metadata)
```

The exporter aligns blocks to 4 bytes; the viewer memory-maps the file and rejects unsupported versions.
