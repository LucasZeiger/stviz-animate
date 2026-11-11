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
- Drag a `.h5ad` file into the app or click the conversion box to pick a file. You can also start the app with a file: `stviz-animate your_data.h5ad`.
- Progress is shown in the conversion log, and **Cancel conversion** stops it.
- Converted files are written to `<user-data>/output/<name>.stviz`. Converting the same `.h5ad` again replaces its file; a different `.h5ad` with the same name gets a numbered file.
- Under **Expression source** you can take gene expression from `X`, `raw` or a layer, and use a `var` column for gene names.
- Writable runtime data lives in your user-data directory:
  - Windows: `%APPDATA%/stviz-animate/`
  - macOS: `~/Library/Application Support/stviz-animate/`
  - Linux: `${XDG_DATA_HOME:-~/.local/share}/stviz-animate/`
- The app creates and uses a managed venv at `<user-data>/.stviz_venv` for conversion and OpenCV fallback.
- A system Python install is still required so the app can create that managed venv.
- On first conversion/export-fallback run, dependencies are installed from `python/requirements.txt` (internet required). Later conversions work offline.

What gets converted:
- Every `.obsm` entry with at least two numeric columns becomes a space (the first two dimensions are used). Cells with missing coordinates are not drawn.
- Categorical, string and boolean `.obs` columns become categorical fields; numeric columns become continuous fields. Integer columns with up to 50 distinct values (e.g. cluster ids) are available as both.
- Per-cell identifier columns such as barcodes are skipped.
- The expression matrix is streamed, so memory use stays low even for large files.

Manual conversion:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r python/requirements.txt
python python/export_stviz.py --input your_data.h5ad --output your_data.stviz --include-expr
```

Optional flags: `--layer NAME` or `--use-raw` to choose the expression matrix, and `--gene-symbols COLUMN` to take gene names from a `var` column.

Requires Python 3.9+ (Python 3.10+ recommended).

## Workspaces and settings
- **Save workspace** stores the timeline (including the advanced timeline layout), colors, filters, view and export settings in a `.workspace.json` file.
- A workspace finds its dataset at the saved path or, failing that, next to the workspace file, so the two can be moved or shared together.
- Appearance, the last used folder and recent datasets are remembered between launches in `<user-data>/settings.json`.

## Export
- Screenshot: saves a PNG to `<user-data>/output/screenshots/` at 4K (3840x2160).
- Loop export: writes frames under `<user-data>/output/exports/` and MP4 output under `<user-data>/output/`.
- Video encoding presets: Standard (CRF 23), High (CRF 18), Ultra (CRF 14), all using H.264 yuv420p for Windows compatibility.
- `ffmpeg` is preferred if available on the user system. If missing, OpenCV fallback is auto-installed in the managed converter venv on first use.
- If `ffmpeg` is missing, the Output panel offers an opt-in install via the system package manager (winget/choco, brew, or apt/dnf/pacman/zypper). It is off by default; internet access and possibly admin rights are required.
- Encoding, the OpenCV fallback and the ffmpeg install run in the background, so the app stays responsive and **Cancel export** stops them.

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

Packaging includes Python converter scripts and `THIRD_PARTY_NOTICES.md`.
Converter dependencies are installed on first use (internet required).
`ffmpeg` is not bundled by default in packaging scripts.
- Windows: pass `-IncludeFfmpeg` to `scripts/package_windows.ps1` to include local ffmpeg binaries.
- Ubuntu: set `INCLUDE_FFMPEG=1` when running `scripts/package_ubuntu.sh` to include local ffmpeg binaries.

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
