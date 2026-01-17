# stviz-animate Documentation

## Overview
stviz-animate is a desktop viewer for spatial transcriptomics timelines. It loads `.stviz` datasets, animates transitions across embeddings, supports multiple color modes and filters, and exports loops for presentations. The UI is optimized for large point clouds with GPU rendering.

## Install and run

### Windows
```powershell
cargo run --release
```

Release zip:
1) Download `stviz-animate-windows.zip` from GitHub Releases.
2) Extract it.
3) Run `stviz-animate.exe`.

### macOS
```bash
cargo run --release
```

Note: distributed macOS builds without an Apple Developer ID must be opened via right-click -> Open on first launch.

### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y libgtk-3-dev
cargo run --release
```

Release archive:
1) Download `stviz-animate-ubuntu.tar.gz` from GitHub Releases.
2) Extract:
   ```bash
   tar -xzf stviz-animate-ubuntu.tar.gz
   ```
3) Run:
   ```bash
   ./stviz-animate/stviz-animate
   ```

## Data import and conversion
- Drag a `.h5ad` into the conversion box to convert it to `.stviz`.
- Drag a `.stviz` file to load it directly.
- Conversion output goes to `output/`.
- Large datasets (>1GB) can take a minute or two to convert.
- The first conversion may be slow because the Python venv is created and dependencies are installed.

Manual conversion:
```bash
python python/export_stviz.py --input your_data.h5ad --output your_data.stviz
```

Requires Python 3.8+.

## Mock dataset
- Use "Generate mock dataset" to create a test dataset (default 300k cells).
- Mock datasets include spatial, PCA, UMAP, and tSNE spaces.
- Sample grid is enabled by default for mock datasets.

## Timeline and keyframes
- Keyframes define transitions between spaces and color modes.
- Add/remove keys and distribute timing evenly.
- Use easing to control transition dynamics.
- Reset scaling targets a specific key.
- Spacebar toggles play/pause without jumping time.

## Advanced timeline
- Open the Advanced timeline to arrange cards on a canvas.
- Cards are draggable and connectable; right-click a card for actions.
- Cards support per-card filters, color mode, and space selection.
- Grid mode snaps cards to a fixed grid size.

## Color and filtering
- Categorical, continuous, and gene color modes are supported.
- Filters apply automatically when toggled; no extra apply button.
- Category selectors expand for large lists and include All/None controls.

## Sample grid
- Sample grid renders a spatial grid grouped by a categorical field.
- Enable/disable in the left panel and choose the grouping field.
- Sample grid respects filters if enabled.

## Export
- Screenshot saves a PNG to `output/`.
- Loop export creates an MP4 with configurable fps, duration, and quality.
- Quality presets: Current viewport, Full HD (1920x1080), 4K (3840x2160).
- A Cancel export button stops export and cleans temporary frames.
- ffmpeg is preferred; OpenCV fallback is available if ffmpeg is missing.

## Output locations
- Conversions: `output/*.stviz`
- Mock datasets: `output/mock_spatial_*.h5ad` (temporary)
- Logs: `output/convert_log_*.txt`
- Screenshots: `output/*_screenshot_*.png`
- Videos: `output/*.mp4`

## Performance tips
- Large datasets benefit from GPU drivers and native OS rendering.
- WSL often uses CPU rendering; native Windows/Linux typically use GPU.
- Reduce point radius or enable fast render for very large datasets.

## Troubleshooting
- If ffmpeg is missing, the app will fall back to OpenCV for video export.
- If Python is not found, install Python 3.8+ and ensure it is on PATH.
- If conversion fails, open the Conversion log and copy it for diagnostics.
