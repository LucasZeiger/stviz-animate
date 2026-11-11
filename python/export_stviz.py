#!/usr/bin/env python3
"""Convert an AnnData ``.h5ad`` file into the ``.stviz`` format used by stviz-animate.

The converter is written to stay within a small memory budget:

* Only ``obs``, the first two columns of each ``obsm`` entry, the ``*_colors``
  palettes and the gene names are read into memory.
* The expression matrix is streamed from the HDF5 file in chunks and written
  straight into the output file as CSC, whatever its on-disk layout (CSR, CSC
  or dense).
* The output is written to ``<name>.partial`` and renamed into place at the
  end, so an existing ``.stviz`` is never left half-written.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd

STVIZ_MAGIC = b"STVIZ\x00\x00\x00"
STVIZ_VERSION = 1
U32_MAX = 0xFFFFFFFF

# Non-zero entries (sparse) or matrix cells (dense) processed per chunk.
CHUNK_ENTRIES = 8_000_000

# Integer columns with at most this many distinct values are also exported as
# categorical, so cluster ids stored as numbers can be used for colors/filters.
MAX_INT_CATEGORIES = 50
# String columns with more categories than this are never exported.
MAX_CATEGORIES = 100_000

NAMED_COLORS = {
    "black": "#000000",
    "white": "#ffffff",
    "red": "#ff0000",
    "green": "#008000",
    "lime": "#00ff00",
    "blue": "#0000ff",
    "yellow": "#ffff00",
    "cyan": "#00ffff",
    "aqua": "#00ffff",
    "magenta": "#ff00ff",
    "fuchsia": "#ff00ff",
    "orange": "#ffa500",
    "purple": "#800080",
    "brown": "#a52a2a",
    "pink": "#ffc0cb",
    "gray": "#808080",
    "grey": "#808080",
    "lightgray": "#d3d3d3",
    "lightgrey": "#d3d3d3",
    "darkgray": "#a9a9a9",
    "darkgrey": "#a9a9a9",
    "silver": "#c0c0c0",
    "olive": "#808000",
    "navy": "#000080",
    "teal": "#008080",
    "maroon": "#800000",
    "gold": "#ffd700",
    "salmon": "#fa8072",
    "tan": "#d2b48c",
    "violet": "#ee82ee",
    "indigo": "#4b0082",
    "turquoise": "#40e0d0",
    "khaki": "#f0e68c",
    "coral": "#ff7f50",
    "crimson": "#dc143c",
    "orchid": "#da70d6",
    "plum": "#dda0dd",
    "beige": "#f5f5dc",
    "lightblue": "#add8e6",
    "lightgreen": "#90ee90",
    "darkblue": "#00008b",
    "darkgreen": "#006400",
    "darkred": "#8b0000",
    "darkorange": "#ff8c00",
    "skyblue": "#87ceeb",
    "steelblue": "#4682b4",
    "firebrick": "#b22222",
    "forestgreen": "#228b22",
    "royalblue": "#4169e1",
    "slategray": "#708090",
    "slategrey": "#708090",
}


class ConvertError(RuntimeError):
    """A problem with the input that the user can act on."""


def log(msg: str) -> None:
    print(msg, flush=True)


def _align_to(n: int, a: int) -> int:
    r = n % a
    return n if r == 0 else (n + (a - r))


def parse_color(value) -> Optional[int]:
    """Parse a matplotlib-style color into packed RGBA8 (r | g<<8 | b<<16 | a<<24).

    Accepts ``#RGB``, ``#RGBA``, ``#RRGGBB``, ``#RRGGBBAA``, common color names
    and RGB(A) float tuples. Returns ``None`` when the value is not understood.
    """
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if isinstance(value, str):
        s = value.strip().lower()
        s = NAMED_COLORS.get(s.replace(" ", ""), s)
        if not s.startswith("#"):
            return None
        s = s[1:]
        if len(s) in (3, 4):
            s = "".join(ch * 2 for ch in s)
        if len(s) == 6:
            s += "ff"
        if len(s) != 8:
            return None
        try:
            r, g, b, a = (int(s[i : i + 2], 16) for i in (0, 2, 4, 6))
        except ValueError:
            return None
        return r | (g << 8) | (b << 16) | (a << 24)
    try:
        parts = [float(x) for x in value]
    except (TypeError, ValueError):
        return None
    if len(parts) not in (3, 4) or not all(np.isfinite(parts)):
        return None
    if len(parts) == 3:
        parts.append(1.0)
    r, g, b, a = (int(round(min(max(p, 0.0), 1.0) * 255)) for p in parts)
    return r | (g << 8) | (b << 16) | (a << 24)


def _read_elem(node):
    try:
        from anndata.io import read_elem  # anndata >= 0.11
    except ImportError:  # pragma: no cover - depends on the installed anndata
        from anndata.experimental import read_elem
    return read_elem(node)


def _make_unique(names: List[str]) -> List[str]:
    """Suffix repeated names (``GENE``, ``GENE-1``, ...) so lookups are unambiguous."""
    seen: Dict[str, int] = {}
    taken = set(names)
    out = []
    for name in names:
        count = seen.get(name, 0)
        seen[name] = count + 1
        if count == 0:
            out.append(name)
            continue
        candidate = f"{name}-{count}"
        while candidate in taken:
            count += 1
            candidate = f"{name}-{count}"
        seen[name] = count + 1
        taken.add(candidate)
        out.append(candidate)
    return out


# ---------------------------------------------------------------------------
# Spaces
# ---------------------------------------------------------------------------


def _clean_space(name: str, arr: np.ndarray) -> Optional[Tuple[np.ndarray, List[float]]]:
    """Return float32 xy with non-finite rows set to NaN, plus a finite bbox."""
    xy = np.array(arr, dtype=np.float32, copy=True)
    bad = ~np.isfinite(xy).all(axis=1)
    n_bad = int(bad.sum())
    if n_bad == xy.shape[0] and xy.shape[0] > 0:
        log(f"Skipping space '{name}': no finite coordinates.")
        return None
    if n_bad:
        xy[bad] = np.nan
        log(f"Space '{name}': {n_bad} cells have missing coordinates and will not be drawn.")
    if xy.shape[0] == 0:
        bbox = [0.0, 0.0, 1.0, 1.0]
    else:
        good = xy[~bad]
        bbox = [
            float(good[:, 0].min()),
            float(good[:, 1].min()),
            float(good[:, 0].max()),
            float(good[:, 1].max()),
        ]
    return xy, bbox


def _find_obs_space(obs_df: pd.DataFrame):
    candidates = [
        ("spatial_xy", "spatial_x", "spatial_y"),
        ("xy", "x", "y"),
        ("centroid_xy", "centroid_x", "centroid_y"),
        ("xy", "x_centroid", "y_centroid"),
    ]
    for name, xcol, ycol in candidates:
        if xcol in obs_df.columns and ycol in obs_df.columns:
            if pd.api.types.is_numeric_dtype(obs_df[xcol]) and pd.api.types.is_numeric_dtype(obs_df[ycol]):
                x = obs_df[xcol].to_numpy(dtype=np.float32)
                y = obs_df[ycol].to_numpy(dtype=np.float32)
                return name, np.stack([x, y], axis=1)
    return None


def _first_two_columns(name: str, values, n: int) -> Optional[np.ndarray]:
    """Take the first two dimensions of an obsm entry, or None if unusable."""
    if isinstance(values, h5py.Dataset):
        if values.ndim != 2 or values.shape[0] != n or values.shape[1] < 2:
            return None
        if values.dtype.kind not in "fiub":
            return None
        k = values.shape[1]
        arr = values[:, :2]
    elif isinstance(values, pd.DataFrame):
        numeric = values.select_dtypes(include="number")
        if numeric.shape[0] != n or numeric.shape[1] < 2:
            return None
        k = numeric.shape[1]
        arr = numeric.iloc[:, :2].to_numpy()
    else:
        try:
            arr = np.asarray(values)
        except Exception:
            return None
        if arr.ndim != 2 or arr.shape[0] != n or arr.shape[1] < 2 or arr.dtype.kind not in "fiub":
            return None
        k = arr.shape[1]
        arr = arr[:, :2]
    if k > 2:
        log(f"Space '{name}' has {k} dimensions; using the first two.")
    return arr


# ---------------------------------------------------------------------------
# Obs columns
# ---------------------------------------------------------------------------


def _categorical_block(series: pd.Series) -> Tuple[np.ndarray, List[str]]:
    cat = series.astype("category")
    codes = cat.cat.codes.to_numpy(dtype=np.int64)
    categories = [str(x) for x in cat.cat.categories]
    # pandas uses -1 for missing values; give those a real category.
    missing = codes < 0
    if missing.any():
        label = "<missing>"
        while label in categories:
            label += "_"
        codes = codes.copy()
        codes[missing] = len(categories)
        categories.append(label)
    return codes.astype(np.uint32), categories


def _palette_for(col: str, colors, n_categories: int) -> Optional[List[int]]:
    if colors is None:
        return None
    if isinstance(colors, (str, bytes)):
        raw = [colors]
    else:
        try:
            raw = list(np.atleast_1d(colors)) if isinstance(colors, np.ndarray) else list(colors)
        except TypeError:
            return None
    parsed = [parse_color(c) for c in raw[:n_categories]]
    if any(p is None for p in parsed):
        log(f"Column '{col}': palette has colors that could not be read; using the app palette instead.")
        return None
    grey = parse_color("#808080")
    parsed.extend([grey] * (n_categories - len(parsed)))
    return parsed


def collect_obs(obs_df: pd.DataFrame, palettes: Dict[str, object]):
    """Decide how each obs column is exported.

    Returns a list of ``(meta_without_offset, array)`` tuples.
    """
    n = len(obs_df)
    out = []
    skipped = []
    for col in obs_df.columns:
        s = obs_df[col]
        name = str(col)
        palette_src = palettes.get(name)
        dtype = s.dtype
        is_bool = pd.api.types.is_bool_dtype(dtype)
        is_numeric = pd.api.types.is_numeric_dtype(dtype) and not is_bool
        is_cat = isinstance(dtype, pd.CategoricalDtype)
        is_text = dtype == object or pd.api.types.is_string_dtype(dtype)

        as_categorical = is_bool or is_cat or is_text or palette_src is not None
        as_continuous = is_numeric and palette_src is None

        if is_numeric and palette_src is None and pd.api.types.is_integer_dtype(dtype):
            # Cluster ids are often stored as plain integers. Export those both
            # ways so they work for categorical colors/filters and gradients.
            n_unique = int(s.nunique(dropna=True))
            if 0 < n_unique <= MAX_INT_CATEGORIES:
                as_categorical = True

        if not as_categorical and not as_continuous:
            skipped.append(f"{name} ({dtype})")
            continue

        if as_categorical:
            n_unique = len(dtype.categories) if is_cat else int(s.nunique(dropna=True))
            # Per-cell identifiers (barcodes, cell ids) are not useful as colors
            # and would bloat the file with one category per cell.
            looks_like_id = n_unique > max(50, n // 2)
            if (is_text or is_cat) and (looks_like_id or n_unique > MAX_CATEGORIES):
                skipped.append(f"{name} ({n_unique} distinct values)")
            else:
                codes, categories = _categorical_block(s)
                out.append(
                    (
                        {
                            "kind": "categorical",
                            "name": name,
                            "categories": categories,
                            "palette_rgba8": _palette_for(name, palette_src, len(categories)),
                        },
                        codes,
                    )
                )

        if as_continuous:
            vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float32, na_value=np.nan)
            vals = np.where(np.isfinite(vals), vals, np.float32(np.nan)).astype(np.float32)
            out.append(({"kind": "continuous", "name": name}, vals))

    if skipped:
        log("Skipped obs columns: " + ", ".join(skipped))
    return out


# ---------------------------------------------------------------------------
# Expression (streamed to CSC)
# ---------------------------------------------------------------------------


class Progress:
    def __init__(self, label: str, total: int):
        self.label = label
        self.total = max(int(total), 1)
        self.last = -1

    def update(self, done: int) -> None:
        pct = min(100, int(done * 100 / self.total))
        if pct >= self.last + 5 or pct == 100 and self.last != 100:
            self.last = pct
            log(f"{self.label}: {pct}%")


def _sparse_format(node) -> Optional[str]:
    enc = node.attrs.get("encoding-type", None)
    if isinstance(enc, bytes):
        enc = enc.decode()
    if enc in ("csr_matrix", "csc_matrix"):
        return enc[:3]
    legacy = node.attrs.get("h5sparse_format", None)
    if isinstance(legacy, bytes):
        legacy = legacy.decode()
    return legacy if legacy in ("csr", "csc") else None


def _sparse_shape(node) -> Tuple[int, int]:
    shape = node.attrs.get("shape", None)
    if shape is None:
        shape = node.attrs.get("h5sparse_shape", None)
    if shape is None:
        raise ConvertError("Expression matrix has no shape information.")
    return int(shape[0]), int(shape[1])


class ExprPlan:
    """Counts entries per gene up front, then fills the CSC arrays in place."""

    def __init__(self, node, n_obs: int):
        self.node = node
        if isinstance(node, h5py.Group):
            self.fmt = _sparse_format(node)
            if self.fmt is None:
                raise ConvertError("Unsupported expression matrix encoding.")
            n_rows, self.n_genes = _sparse_shape(node)
        elif isinstance(node, h5py.Dataset) and node.ndim == 2:
            self.fmt = "dense"
            n_rows, self.n_genes = int(node.shape[0]), int(node.shape[1])
        else:
            raise ConvertError("Unsupported expression matrix encoding.")
        if n_rows != n_obs:
            raise ConvertError(f"Expression matrix has {n_rows} rows but obs has {n_obs} cells.")
        self.n_obs = n_obs
        self.indptr = self._count()
        self.nnz = int(self.indptr[-1])
        if self.nnz > U32_MAX:
            raise ConvertError(
                f"Expression matrix has {self.nnz:,} non-zero values; the .stviz format supports at most "
                f"{U32_MAX:,}. Convert without gene expression or subset the genes first."
            )

    # -- pass 1 ------------------------------------------------------------
    def _count(self) -> np.ndarray:
        g = self.n_genes
        counts = np.zeros(g, dtype=np.int64)
        if self.fmt == "csc":
            indptr = self.node["indptr"][:].astype(np.int64)
            if len(indptr) != g + 1:
                raise ConvertError("Expression matrix column pointers do not match the gene count.")
            return indptr
        if self.fmt == "csr":
            indices = self.node["indices"]
            total = int(indices.shape[0])
            prog = Progress("Scanning expression", total)
            for a in range(0, total, CHUNK_ENTRIES):
                chunk = indices[a : a + CHUNK_ENTRIES]
                c = np.bincount(chunk, minlength=g)
                if len(c) != g:
                    raise ConvertError("Expression matrix has gene indices outside the gene list.")
                counts += c
                prog.update(a + len(chunk))
        else:
            rows = self._dense_rows_per_chunk()
            prog = Progress("Scanning expression", self.n_obs)
            for r0 in range(0, self.n_obs, rows):
                chunk = np.nan_to_num(self.node[r0 : r0 + rows])
                counts += np.count_nonzero(chunk, axis=0)
                prog.update(r0 + chunk.shape[0])
        indptr = np.zeros(g + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        return indptr

    def _dense_rows_per_chunk(self) -> int:
        return max(1, CHUNK_ENTRIES // max(self.n_genes, 1))

    # -- pass 2 ------------------------------------------------------------
    def fill(self, out_indices: np.ndarray, out_data: np.ndarray) -> None:
        if self.nnz == 0:
            return
        if self.fmt == "csc":
            self._fill_csc(out_indices, out_data)
        elif self.fmt == "csr":
            self._fill_csr(out_indices, out_data)
        else:
            self._fill_dense(out_indices, out_data)

    def _fill_csc(self, out_indices, out_data) -> None:
        indices = self.node["indices"]
        data = self.node["data"]
        prog = Progress("Writing expression", self.nnz)
        for a in range(0, self.nnz, CHUNK_ENTRIES):
            b = min(a + CHUNK_ENTRIES, self.nnz)
            out_indices[a:b] = indices[a:b]
            out_data[a:b] = data[a:b]
            prog.update(b)

    def _scatter(self, cols, rows, vals, cursor, out_indices, out_data) -> None:
        """Place entries (any order) at the next free slots of their columns."""
        order = np.argsort(cols, kind="stable")
        cols_sorted = cols[order]
        c = np.bincount(cols, minlength=self.n_genes)
        starts = np.cumsum(c) - c
        pos = cursor[cols_sorted] + (np.arange(len(cols_sorted), dtype=np.int64) - starts[cols_sorted])
        out_indices[pos] = rows[order]
        out_data[pos] = vals[order]
        cursor += c

    def _fill_csr(self, out_indices, out_data) -> None:
        row_ptr = self.node["indptr"][:].astype(np.int64)
        indices = self.node["indices"]
        data = self.node["data"]
        cursor = self.indptr[:-1].copy()
        prog = Progress("Writing expression", self.nnz)
        r0 = 0
        while r0 < self.n_obs:
            r1 = int(np.searchsorted(row_ptr, row_ptr[r0] + CHUNK_ENTRIES, side="right")) - 1
            r1 = min(max(r1, r0 + 1), self.n_obs)
            a, b = int(row_ptr[r0]), int(row_ptr[r1])
            if b > a:
                cols = indices[a:b].astype(np.int64)
                vals = data[a:b]
                rows = np.repeat(np.arange(r0, r1, dtype=np.uint32), np.diff(row_ptr[r0 : r1 + 1]))
                self._scatter(cols, rows, vals, cursor, out_indices, out_data)
            prog.update(b)
            r0 = r1

    def _fill_dense(self, out_indices, out_data) -> None:
        rows_per = self._dense_rows_per_chunk()
        cursor = self.indptr[:-1].copy()
        prog = Progress("Writing expression", self.n_obs)
        for r0 in range(0, self.n_obs, rows_per):
            chunk = np.nan_to_num(self.node[r0 : r0 + rows_per])
            rows_local, cols = np.nonzero(chunk)
            if len(cols):
                vals = chunk[rows_local, cols]
                rows = (rows_local + r0).astype(np.uint32)
                self._scatter(cols.astype(np.int64), rows, vals, cursor, out_indices, out_data)
            prog.update(r0 + chunk.shape[0])


def _expr_node(f: h5py.File, layer: Optional[str], use_raw: bool):
    """Return (matrix node, var group) for the requested expression source."""
    if use_raw:
        if "raw" not in f or "X" not in f["raw"]:
            raise ConvertError("This file has no .raw expression matrix.")
        return f["raw"]["X"], f["raw"]["var"]
    if layer:
        if "layers" not in f or layer not in f["layers"]:
            available = ", ".join(f["layers"].keys()) if "layers" in f else "none"
            raise ConvertError(f"Layer '{layer}' not found. Available layers: {available}.")
        return f["layers"][layer], f["var"]
    if "X" not in f:
        raise ConvertError("This file has no expression matrix (X).")
    return f["X"], f["var"]


def _gene_names(var_group, gene_symbols: Optional[str], n_genes: int) -> List[str]:
    var = _read_elem(var_group)
    if gene_symbols:
        if gene_symbols not in var.columns:
            raise ConvertError(
                f"Gene symbol column '{gene_symbols}' not found in var. "
                f"Available columns: {', '.join(map(str, var.columns)) or 'none'}."
            )
        names = [str(x) for x in var[gene_symbols].tolist()]
    else:
        names = [str(x) for x in var.index.tolist()]
    if len(names) != n_genes:
        raise ConvertError(f"var has {len(names)} genes but the expression matrix has {n_genes} columns.")
    return _make_unique(names)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class Layout:
    """Assigns 4-byte aligned offsets (relative to the data section)."""

    def __init__(self):
        self.size = 0
        self.arrays: List[Tuple[int, np.ndarray]] = []

    def reserve(self, nbytes: int) -> int:
        off = _align_to(self.size, 4)
        self.size = off + nbytes
        return off

    def add(self, arr: np.ndarray) -> Tuple[int, int]:
        arr = np.ascontiguousarray(arr)
        off = self.reserve(arr.nbytes)
        self.arrays.append((off, arr))
        return off, arr.nbytes


def export_stviz(
    input_path: Path,
    output_path: Path,
    include_expr: bool,
    layer: Optional[str] = None,
    use_raw: bool = False,
    gene_symbols: Optional[str] = None,
    source: Optional[str] = None,
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise ConvertError(f"Input file not found: {input_path}")
    try:
        f = h5py.File(input_path, "r")
    except OSError as e:
        raise ConvertError(f"Could not open {input_path.name} as an .h5ad (HDF5) file: {e}") from e

    with f:
        if "obs" not in f:
            raise ConvertError("This file has no obs table; is it an AnnData .h5ad file?")
        log("Reading cell annotations...")
        obs_df: pd.DataFrame = _read_elem(f["obs"])
        n = len(obs_df)
        if n > U32_MAX:
            raise ConvertError(f"Too many cells ({n:,}); the .stviz format supports at most {U32_MAX:,}.")

        layout = Layout()

        # Spaces: every obsm entry with at least two numeric columns.
        log("Reading embeddings...")
        space_meta = []
        obsm = f["obsm"] if "obsm" in f else {}
        for key in obsm.keys():
            node = obsm[key]
            try:
                values = node if isinstance(node, h5py.Dataset) else _read_elem(node)
            except Exception as e:  # unreadable entry: skip it, keep going
                log(f"Skipping obsm['{key}']: {e}")
                continue
            arr = _first_two_columns(str(key), values, n)
            if arr is None:
                continue
            cleaned = _clean_space(str(key), arr)
            if cleaned is None:
                continue
            xy, bbox = cleaned
            off, nbytes = layout.add(xy)
            space_meta.append({"name": str(key), "dims": 2, "offset": off, "len_bytes": nbytes, "bbox": bbox})

        if not space_meta:
            obs_space = _find_obs_space(obs_df)
            cleaned = _clean_space(obs_space[0], obs_space[1]) if obs_space else None
            if cleaned is None:
                raise ConvertError(
                    "No 2D spaces found in .obsm and no suitable obs columns found "
                    "(expected e.g. 'spatial', 'X_umap' in obsm, or obs columns like "
                    "'centroid_x/centroid_y', 'spatial_x/spatial_y', 'x/y')."
                )
            xy, bbox = cleaned
            off, nbytes = layout.add(xy)
            space_meta.append({"name": obs_space[0], "dims": 2, "offset": off, "len_bytes": nbytes, "bbox": bbox})

        # Scanpy palettes live in uns["<column>_colors"].
        palettes: Dict[str, object] = {}
        if "uns" in f:
            for key in f["uns"].keys():
                if key.endswith("_colors"):
                    try:
                        palettes[key[: -len("_colors")]] = _read_elem(f["uns"][key])
                    except Exception as e:
                        log(f"Ignoring palette uns['{key}']: {e}")

        obs_meta = []
        for meta, arr in collect_obs(obs_df, palettes):
            off, nbytes = layout.add(arr)
            entry = {"kind": meta["kind"], "name": meta["name"], "offset": off, "len_bytes": nbytes}
            if meta["kind"] == "categorical":
                entry["categories"] = meta["categories"]
                entry["palette_rgba8"] = meta["palette_rgba8"]
            obs_meta.append(entry)

        expr_meta = None
        plan: Optional[ExprPlan] = None
        indices_off = data_off = 0
        if include_expr:
            node, var_group = _expr_node(f, layer, use_raw)
            plan = ExprPlan(node, n)
            var_names = _gene_names(var_group, gene_symbols, plan.n_genes)
            if plan.fmt == "dense":
                log("Expression matrix is dense; only non-zero values are stored.")
            indptr_off, indptr_bytes = layout.add(plan.indptr.astype(np.uint32))
            indices_off = layout.reserve(plan.nnz * 4)
            data_off = layout.reserve(plan.nnz * 4)
            expr_meta = {
                "kind": "csc",
                "n_genes": plan.n_genes,
                "nnz": plan.nnz,
                "var_names": var_names,
                "indptr_offset": indptr_off,
                "indptr_len_bytes": indptr_bytes,
                "indices_offset": indices_off,
                "indices_len_bytes": plan.nnz * 4,
                "data_offset": data_off,
                "data_len_bytes": plan.nnz * 4,
            }

        meta = {
            "version": STVIZ_VERSION,
            "n_points": n,
            # Where this file came from, so the app can tell conversions apart.
            "source": source if source is not None else str(input_path.resolve()),
            "spaces": space_meta,
            "obs": obs_meta,
            "expr": expr_meta,
        }
        meta_bytes = json.dumps(meta, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
        header = STVIZ_MAGIC + struct.pack("<I", STVIZ_VERSION) + struct.pack("<Q", len(meta_bytes))
        data_start = _align_to(len(header) + len(meta_bytes), 16)
        total_size = data_start + layout.size

        output_path.parent.mkdir(parents=True, exist_ok=True)
        partial = output_path.with_name(output_path.name + ".partial")
        log("Writing .stviz file...")
        try:
            with open(partial, "wb") as out:
                out.write(header)
                out.write(meta_bytes)
                for off, arr in layout.arrays:
                    out.seek(data_start + off)
                    out.write(memoryview(arr).cast("B"))
                out.truncate(total_size)
            if plan is not None and plan.nnz > 0:
                out_indices = np.memmap(partial, dtype=np.uint32, mode="r+", offset=data_start + indices_off, shape=(plan.nnz,))
                out_data = np.memmap(partial, dtype=np.float32, mode="r+", offset=data_start + data_off, shape=(plan.nnz,))
                try:
                    plan.fill(out_indices, out_data)
                    out_indices.flush()
                    out_data.flush()
                finally:
                    del out_indices, out_data
            os.replace(partial, output_path)
        except BaseException:
            try:
                partial.unlink()
            except OSError:
                pass
            raise

    log(f"Wrote: {output_path} ({total_size / 1e6:.2f} MB)")
    log(f"Cells: {n}, Spaces: {len(space_meta)}, Obs: {len(obs_meta)}, Expr: {'yes' if include_expr else 'no'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--include-expr", action="store_true", help="Export expression as CSC for gene coloring (can be large).")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--layer", default=None, help="Use adata.layers[LAYER] instead of adata.X for expression.")
    src.add_argument("--use-raw", action="store_true", help="Use adata.raw.X (and raw gene names) for expression.")
    ap.add_argument("--gene-symbols", default=None, help="var column holding gene names (default: var index).")
    ap.add_argument("--source", default=None, help="Label stored in the file to identify its origin (default: input path).")
    args = ap.parse_args()
    try:
        export_stviz(
            args.input, args.output, args.include_expr, args.layer, args.use_raw, args.gene_symbols, args.source
        )
    except ConvertError as e:
        print(f"Error: {e}", file=sys.stderr, flush=True)
        return 1
    except MemoryError:
        print("Error: ran out of memory while converting. Try converting without gene expression.", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
