#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def _align_to(n: int, a: int) -> int:
    r = n % a
    return n if r == 0 else (n + (a - r))


def _hex_to_rgba8_u32(s: str) -> int:
    s = s.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) != 6:
        return 0xFFFFFFFF
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    a = 255
    return (r) | (g << 8) | (b << 16) | (a << 24)


def _find_obs_space(obs_df: pd.DataFrame):
    candidates = [
        ("spatial_xy", "spatial_x", "spatial_y"),
        ("xy", "x", "y"),
        ("centroid_xy", "centroid_x", "centroid_y"),
    ]
    for name, xcol, ycol in candidates:
        if xcol in obs_df.columns and ycol in obs_df.columns:
            if pd.api.types.is_numeric_dtype(obs_df[xcol]) and pd.api.types.is_numeric_dtype(obs_df[ycol]):
                x = obs_df[xcol].to_numpy(dtype=np.float32, copy=False)
                y = obs_df[ycol].to_numpy(dtype=np.float32, copy=False)
                arr = np.stack([x, y], axis=1)
                return name, arr
    return None


def export_stviz(input_path: Path, output_path: Path, include_expr: bool) -> None:
    adata = ad.read_h5ad(str(input_path))

    n = int(adata.n_obs)

    obs_df: pd.DataFrame = adata.obs

    # Collect spaces: all obsm entries with 2 columns (2D)
    spaces = []
    for k in adata.obsm.keys():
        arr = np.asarray(adata.obsm[k])
        if arr.ndim == 2 and arr.shape[0] == n and arr.shape[1] == 2:
            spaces.append((k, arr.astype(np.float32, copy=False)))

    if len(spaces) == 0:
        obs_space = _find_obs_space(obs_df)
        if obs_space is not None:
            name, arr = obs_space
            spaces.append((name, arr.astype(np.float32, copy=False)))
        else:
            raise RuntimeError(
                "No 2D spaces found in .obsm and no suitable obs columns found "
                "(expected e.g. 'spatial', 'X_umap' in obsm, or obs columns like "
                "'centroid_x/centroid_y', 'spatial_x/spatial_y', 'x/y')."
            )

    # Collect obs: categorical + continuous
    obs_meta = []
    obs_blocks = []
    data_parts = []

    def add_bytes(b: bytes, align: int = 4) -> int:
        nonlocal data_parts
        # pad current length
        cur = sum(len(x) for x in data_parts)
        new_cur = _align_to(cur, align)
        if new_cur != cur:
            data_parts.append(b"\x00" * (new_cur - cur))
        off = sum(len(x) for x in data_parts)
        data_parts.append(b)
        return off

    # Spaces first
    space_meta = []
    for name, arr in spaces:
        bbox = [
            float(np.min(arr[:, 0])),
            float(np.min(arr[:, 1])),
            float(np.max(arr[:, 0])),
            float(np.max(arr[:, 1])),
        ]
        raw = arr.tobytes(order="C")
        off = add_bytes(raw, 4)
        space_meta.append(
            {
                "name": name,
                "dims": 2,
                "offset": int(off),
                "len_bytes": int(len(raw)),
                "bbox": bbox,
            }
        )

    # Obs columns
    for col in obs_df.columns:
        s = obs_df[col]
        uns_key = f"{col}_colors"
        has_palette = hasattr(adata, "uns") and uns_key in adata.uns
        is_cat = isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object
        # categorical (or numeric with an explicit palette)
        if is_cat or has_palette:
            cat = s.astype("category")
            codes = cat.cat.codes.to_numpy(dtype=np.int32, copy=False)
            codes_u32 = codes.astype(np.uint32, copy=False)

            raw = codes_u32.tobytes(order="C")
            off = add_bytes(raw, 4)

            categories = [str(x) for x in list(cat.cat.categories)]

            # Optional scanpy palette in .uns: "<col>_colors"
            palette_rgba8 = None
            if has_palette:
                cols = list(adata.uns[uns_key])
                palette_rgba8 = [_hex_to_rgba8_u32(str(x)) for x in cols]

            obs_meta.append(
                {
                    "kind": "categorical",
                    "name": str(col),
                    "offset": int(off),
                    "len_bytes": int(len(raw)),
                    "categories": categories,
                    "palette_rgba8": palette_rgba8,
                }
            )
        # continuous numeric
        elif pd.api.types.is_numeric_dtype(s):
            vals = s.to_numpy(dtype=np.float32, copy=False)
            raw = vals.tobytes(order="C")
            off = add_bytes(raw, 4)
            obs_meta.append(
                {
                    "kind": "continuous",
                    "name": str(col),
                    "offset": int(off),
                    "len_bytes": int(len(raw)),
                }
            )

    expr_meta = None
    if include_expr:
        # Export X as CSC for fast gene lookup: gene -> (cell indices, values)
        X = adata.X
        if sp.issparse(X):
            Xcsc = X.tocsc(copy=False)
        else:
            Xcsc = sp.csc_matrix(np.asarray(X), dtype=np.float32)

        Xcsc.sort_indices()
        n_genes = int(Xcsc.shape[1])
        nnz = int(Xcsc.nnz)

        indptr = Xcsc.indptr.astype(np.uint32, copy=False)
        indices = Xcsc.indices.astype(np.uint32, copy=False)
        data = Xcsc.data.astype(np.float32, copy=False)

        var_names = [str(x) for x in list(adata.var_names)]

        indptr_raw = indptr.tobytes(order="C")
        indices_raw = indices.tobytes(order="C")
        data_raw = data.tobytes(order="C")

        indptr_off = add_bytes(indptr_raw, 4)
        indices_off = add_bytes(indices_raw, 4)
        data_off = add_bytes(data_raw, 4)

        expr_meta = {
            "kind": "csc",
            "n_genes": n_genes,
            "nnz": nnz,
            "var_names": var_names,
            "indptr_offset": int(indptr_off),
            "indptr_len_bytes": int(len(indptr_raw)),
            "indices_offset": int(indices_off),
            "indices_len_bytes": int(len(indices_raw)),
            "data_offset": int(data_off),
            "data_len_bytes": int(len(data_raw)),
        }

    meta = {
        "version": 1,
        "n_points": n,
        "spaces": space_meta,
        "obs": obs_meta,
        "expr": expr_meta,
    }

    meta_bytes = json.dumps(meta, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    # header: u64 json_len
    out = bytearray()
    out += struct.pack("<Q", len(meta_bytes))
    out += meta_bytes
    # pad to 16-byte boundary
    out_len = len(out)
    out_pad = _align_to(out_len, 16) - out_len
    if out_pad:
        out += b"\x00" * out_pad

    out += b"".join(data_parts)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(out)
    print(f"Wrote: {output_path} ({len(out)/1e6:.2f} MB)")
    print(f"Spaces: {len(space_meta)}, Obs: {len(obs_meta)}, Expr: {'yes' if include_expr else 'no'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--include-expr", action="store_true", help="Export X as CSC for gene coloring (can be large).")
    args = ap.parse_args()
    export_stviz(args.input, args.output, args.include_expr)


if __name__ == "__main__":
    main()
