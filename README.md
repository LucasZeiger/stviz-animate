# stviz-animate

GPU-accelerated viewer for large single-cell / spatial transcriptomics point clouds.
Transitions the same cells between coordinate spaces (spatial, UMAP, tSNE, PCA, ...).

## 1) Export from .h5ad to .stviz

From the repo root:

```bash
python -m pip install -U anndata h5py numpy pandas scipy
python python/export_stviz.py --input your_data.h5ad --output your_data.stviz
```

Optional: include expression matrix for later gene coloring (can be big):

```bash
python python/export_stviz.py --input your_data.h5ad --output your_data.stviz --include-expr
```

## 2) Build & run

Windows (recommended for NVIDIA dev)

Install Rust (stable) and a C++ build toolchain (Visual Studio Build Tools).
Then:

```bash
cargo run --release
```

Linux

Install Rust and GTK dev libs (needed for file dialog via rfd):

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y libgtk-3-dev
cargo run --release
```

## 3) Controls

Mouse drag: pan

Mouse wheel: zoom

Left panel: choose "from" and "to" spaces, color mode, playback, filters

Screenshot: saves a PNG

Record: saves PNG frames; you can convert to video with ffmpeg, e.g.

```bash
ffmpeg -framerate 30 -i frames/frame_%06d.png -c:v libx264 -pix_fmt yuv420p out.mp4
```

## File format (.stviz)

Single file:

```
[u64 json_len]

[json bytes]

[padding to 16-byte boundary]

[raw binary blocks], with offsets given in the JSON metadata.
```

The exporter aligns blocks to 4 bytes; the viewer memory-maps and zero-copies where possible.
