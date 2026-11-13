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

Download the latest build for your system from [GitHub Releases](https://github.com/LucasZeiger/stviz-animate/releases/latest).

## Running release zips

Windows (10 and 11):
1) Download the `stviz-animate-windows.zip` from GitHub Releases.
2) Right-click the zip, choose **Extract All...**, and extract it to a normal folder such as `Documents\stviz-animate`. Don't run it from inside the zip: the converter scripts in the `python` folder next to the `.exe` are needed.
3) Run `stviz-animate.exe`. The build is not signed, so Windows SmartScreen may say "Windows protected your PC": click **More info**, then **Run anyway**.
4) To convert `.h5ad` files, install Python 3.9 or newer from [python.org](https://www.python.org/downloads/windows/) and tick **Add python.exe to PATH** in the installer. The Microsoft Store version of Python does not work with the converter. The first conversion then sets up its own Python environment, which needs internet once and takes a few minutes.
5) For video export, install ffmpeg once, e.g. in a terminal: `winget install Gyan.FFmpeg` (or use **Install ffmpeg now** in the Output panel). Without it, the app falls back to a slower OpenCV encoder.

Tip: to open `.stviz` files by double-click, right-click one, choose **Open with** > **Choose another app**, and pick `stviz-animate.exe`.

macOS (Apple Silicon):
1) Download `stviz-animate-mac.zip` from GitHub Releases and unzip it.
2) Right-click `stviz-animate.app` and choose **Open** the first time (the build is not signed).

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

1) Install Rust (stable) from [rustup.rs](https://rustup.rs) and the Visual Studio Build Tools with the **Desktop development with C++** workload.
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

## First steps
- With no dataset loaded, the view shows a welcome screen: **Open a dataset...**, **Try demo data** (a 50,000-cell synthetic spatial dataset; needs Python), your recent datasets and a three-step guide.
- **Open data...** and **Shortcuts (?)** sit in the top bar. Press `?` or `F1` for the shortcut list: Space plays and pauses, the arrow keys step through frames, Esc leaves fullscreen, drag pans and scroll zooms.
- Most controls explain themselves when you hover over them.

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
- **Labels**: under *Show labels on view and exports* you can add a title, a legend for the current colouring, the space name and a scale bar. They are previewed in the viewport and drawn into screenshots and videos at any resolution. The scale bar appears on spaces with "spatial" in their name while the view rests on that space; set how many µm (or another unit) one coordinate is. Label settings are saved in workspaces.
- Screenshot: saves a PNG to `<user-data>/output/screenshots/` at 4K (3840x2160).
- Loop export: writes frames under `<user-data>/output/exports/` and MP4 output under `<user-data>/output/`.
- Video encoding presets: Standard (CRF 23), High (CRF 18), Ultra (CRF 14), all using H.264 yuv420p for Windows compatibility.
- `ffmpeg` is preferred if available on the user system. If missing, OpenCV fallback is auto-installed in the managed converter venv on first use.
- If `ffmpeg` is missing, the Output panel offers an opt-in install via the system package manager (winget/choco, brew, or apt/dnf/pacman/zypper). It is off by default; internet access and possibly admin rights are required.
- Encoding, the OpenCV fallback and the ffmpeg install run in the background, so the app stays responsive and **Cancel export** stops them.

## Troubleshooting
- **"Python not found"**: install Python 3.9+ from python.org (on Windows, tick **Add python.exe to PATH**), then restart the app. On Windows the app also finds Python through the `py` launcher.
- **Conversion fails while installing dependencies**: the first conversion needs internet. Behind a proxy or firewall, install the packages yourself: `python -m pip install -r python/requirements.txt` and start the app from that environment (it prefers an active venv or conda env).
- **Windows: "path too long" errors during setup**: enable long paths (Windows Settings > System > For developers > *Enable long paths*, or the `LongPathsEnabled` policy) and convert again.
- **Windows: "Could not replace ... it is open in another program"**: the `.stviz` is open in another stviz-animate window. Close it there and convert again.
- **Windows: conversion or export is blocked or very slow**: some antivirus tools scan every file the converter writes. Adding the user-data folder (`%APPDATA%\stviz-animate`) to the exclusions helps.
- **Video export uses OpenCV instead of ffmpeg**: install ffmpeg (`winget install Gyan.FFmpeg` on Windows, `brew install ffmpeg` on macOS, `sudo apt install ffmpeg` on Ubuntu). The app finds winget and Chocolatey installs without a restart.
- **Start over**: close the app and delete the user-data folder (see *Data conversion* for its location). Converted files, logs, settings and the managed Python environment are all inside it.

## Making a release
1) Set `version` in `Cargo.toml` (e.g. `1.1.0`) and merge it to `main`.
2) Tag and push: `git tag v1.1.0 && git push origin v1.1.0`.
3) The **Release** workflow builds Windows, macOS (Apple Silicon) and Ubuntu bundles and publishes them as a GitHub Release.

You can also run the workflow by hand from the Actions tab to get the bundles as build artifacts without publishing a release.

## Packaging
Scripts to produce portable bundles locally (written to `dist/`, which is not committed):

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
