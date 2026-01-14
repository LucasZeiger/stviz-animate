# stviz-animate Documentation

## Overview
stviz-animate is a desktop viewer for spatial transcriptomics timelines. It loads .stviz datasets, animates transitions across embeddings, and exports loops for presentations. The UI is optimized for large point clouds, with timeline keyframes, gene coloring, and recording tools.

## Quick start
1. Run the app:
   - `cargo run --release`
2. Load data:
   - Drop a .h5ad file in the conversion box, or open a .stviz file directly.
3. Explore:
   - Pick spaces and color modes, then add keyframes and press Play.

## Data import and conversion
- Drop or pick a .h5ad file to convert it to .stviz.
- The conversion log is collapsible and shows friendly progress updates.
- Output files are saved in `output/`.
- Large datasets (>1GB) can take a minute or two to convert.
- Use "Generate .stviz file only - don't load" when you only want the file.

## Mock dataset
- Use "Generate mock dataset" to create a test .h5ad with 10k to 10M cells (default 500k).
- The generator creates clustered, patterned UMAP, tSNE, PCA, and spatial distributions.
- Old mock datasets and logs are cleaned on startup and exit to avoid clutter.

## Timeline and keyframes
- The timeline shows keyframes and playback position.
- Add keyframes to append them to the end; keyframes are evenly spaced by default.
- "Space evenly" redistributes times across the full range.
- Use "Keyframes per column" to control the keyframe list layout.

## Rendering and view
- UI scale presets (with a fine-tune field) are always available in the left panel.
- Themes include Dark, Light, Slate, and Matrix.
- Point radius defaults to 0.5px with a slider clamped to 0.5-2.0.
- Stats can be shown in the top-right overlay.

## Export
- Record loops or frame sequences from the timeline.
- The app writes output under `output/` (frames, videos, screenshots).
- ffmpeg is required to assemble videos from frames.

## Output locations
- Conversions: `output/*.stviz`
- Mock datasets: `output/mock_spatial_*.h5ad` (temporary)
- Logs: `output/convert_log_*.txt`
- Screenshots: `output/*_screenshot_*.png`
- Videos: `output/*.mp4`

## Troubleshooting
- If GPU rendering is not used on Linux, install proper GPU drivers and Vulkan.
- WSL environments typically run on CPU; native Linux with drivers should use GPU.
