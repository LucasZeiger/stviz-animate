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

Windows:
1) Download the `stviz-animate-windows.zip` from GitHub Releases.
2) Extract it (right-click > **Extract All...**).
3) Run `stviz-animate.exe`. If Windows says "Windows protected your PC", click **More info** > **Run anyway** (the build is not signed).

To convert `.h5ad` files, Python 3.9 or newer needs to be installed ([python.org](https://www.python.org/downloads/)). On Windows, if it isn't, the app shows an **Install Python** button that installs it for you. Everything else is set up automatically the first time.

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
- **Undo**, **Redo**, **Open data...** and **Shortcuts (?)** sit in the top bar. Undo (Ctrl/Cmd+Z) and redo (Ctrl/Cmd+Shift+Z or Ctrl+Y) cover colours, filters, keyframes, labels, point settings, lasso selections and saved scenes; panning and zooming are left alone. Press `?` or `F1` for the shortcut list: Space plays and pauses, the arrow keys step through frames, Esc leaves fullscreen, drag pans and scroll zooms. Hover a cell to see its details, click a legend entry to highlight a group, and Shift + drag to lasso cells and export them as CSV from the Selection section.
- Most controls explain themselves when you hover over them.

## Data conversion (.h5ad -> .stviz)
- Drag a `.h5ad` file into the app (or click the drop box to pick one). It is converted and opened automatically.
- The first conversion sets up the converter in the background, which needs internet once and takes a few minutes. After that, conversions work offline.
- Converted files are kept in the app's data folder, so the same dataset opens quickly next time (see **Recent datasets**).
- Optional: under **Expression source** you can take gene expression from `raw` or a layer instead of `X`.

Converting from the command line, and what the converter does with each part of an `.h5ad`, is described in the [technical documentation](docs/technical_documentation.md#data-conversion-pipeline).

## Workspaces and settings
- **Save workspace** stores the timeline (including the advanced timeline layout), colors, filters, view and export settings in a `.workspace.json` file.
- A workspace finds its dataset at the saved path or, failing that, next to the workspace file, so the two can be moved or shared together.
- **Saved scenes** stores the same view (timeline, colours and palettes, filters, camera, labels, playback and export settings) inside the `.stviz` file under a name. Type a name and press **Save scene**; **Load** brings it back and **Delete** removes it. A file can hold any number of scenes, so sending one `.stviz` is enough for someone else to reproduce a view or video. Saving a scene doesn't rewrite the data, files with scenes still open in older versions of the app, and converting the same `.h5ad` again keeps them.
- Appearance, the last used folder and recent datasets are remembered between launches.

## Export
- **Labels**: under *Show labels on view and exports* you can add a title, a legend for the current colouring, the space name and a scale bar. They are previewed in the viewport and drawn into screenshots and videos at any resolution. The scale bar appears on spaces with "spatial" in their name while the view rests on that space; set how many µm (or another unit) one coordinate is. Label settings are saved in workspaces.
- **Screenshot** saves a 4K PNG and **Export loop video** an MP4, both in the app's output folder (shown after each export).
- Videos work out of the box. If ffmpeg is installed it is used for faster, smaller files; otherwise a built-in fallback encoder is set up automatically. Encoding runs in the background and **Cancel export** stops it.

## Troubleshooting
If something doesn't work, the [technical documentation](docs/technical_documentation.md#troubleshooting) lists common problems and fixes.

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
