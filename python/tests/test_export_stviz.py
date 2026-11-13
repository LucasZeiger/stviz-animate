"""Round-trip tests for export_stviz.py.

Run with ``python python/tests/test_export_stviz.py`` or ``pytest python/tests``.
"""
from __future__ import annotations

import json
import struct
import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export_stviz as ex  # noqa: E402


def read_stviz(path: Path):
    raw = path.read_bytes()
    assert raw[:8] == ex.STVIZ_MAGIC
    version, json_len = struct.unpack("<IQ", raw[8:20])
    assert version == ex.STVIZ_VERSION
    meta = json.loads(raw[20 : 20 + json_len])
    data_start = ex._align_to(20 + json_len, 16)

    def block(off, nbytes, dtype):
        return np.frombuffer(raw, dtype=dtype, count=nbytes // 4, offset=data_start + off)

    return meta, block, len(raw) - data_start


def make_adata(n=200, g=30, seed=0):
    rng = np.random.default_rng(seed)
    dense = rng.poisson(0.3, size=(n, g)).astype(np.float32)
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(rng.choice(["a", "b", "c"], size=n)),
            "cluster_int": rng.integers(0, 5, size=n),
            "is_tumor": rng.random(n) > 0.5,
            "barcode": [f"CELL{i:05d}" for i in range(n)],
            "score": rng.random(n).astype(np.float64),
            "n_counts": dense.sum(axis=1).astype(np.int64) + np.arange(n) * 100,
        },
        index=[f"c{i}" for i in range(n)],
    )
    obs.loc[obs.index[3], "score"] = np.nan
    var = pd.DataFrame({"symbol": [f"G{i % 25}" for i in range(g)]}, index=[f"ENS{i}" for i in range(g)])
    adata = ad.AnnData(X=sp.csr_matrix(dense), obs=obs, var=var)
    spatial = rng.normal(size=(n, 2)) * 100
    spatial[7] = np.nan
    adata.obsm["spatial"] = spatial
    adata.obsm["X_umap"] = rng.normal(size=(n, 2)).astype(np.float32)
    adata.obsm["X_pca"] = rng.normal(size=(n, 10)).astype(np.float32)
    adata.uns["cell_type_colors"] = ["#ff0000", "red", "#0f08"]
    adata.layers["counts"] = sp.csc_matrix(dense * 2)
    return adata, dense


def convert(adata, **kwargs):
    tmp = Path(tempfile.mkdtemp(prefix="stviz_test_"))
    src = tmp / "in.h5ad"
    out = tmp / "out.stviz"
    adata.write_h5ad(src)
    ex.export_stviz(src, out, kwargs.pop("include_expr", True), **kwargs)
    assert not out.with_name(out.name + ".partial").exists()
    return out


def dense_from(meta, block, n):
    e = meta["expr"]
    indptr = block(e["indptr_offset"], e["indptr_len_bytes"], np.uint32)
    indices = block(e["indices_offset"], e["indices_len_bytes"], np.uint32)
    data = block(e["data_offset"], e["data_len_bytes"], np.float32)
    assert indptr[-1] == e["nnz"] == len(indices) == len(data)
    return sp.csc_matrix((data, indices, indptr), shape=(n, e["n_genes"])).toarray()


def test_roundtrip_csr():
    adata, dense = make_adata()
    n = adata.n_obs
    meta, block, data_size = read_stviz(convert(adata))
    assert meta["n_points"] == n

    spaces = {s["name"]: s for s in meta["spaces"]}
    assert set(spaces) == {"spatial", "X_umap", "X_pca"}
    for s in spaces.values():
        assert s["len_bytes"] == n * 8
        assert all(np.isfinite(s["bbox"]))
        assert s["offset"] + s["len_bytes"] <= data_size
    xy = block(spaces["spatial"]["offset"], n * 8, np.float32).reshape(n, 2)
    assert np.isnan(xy[7]).all() and np.isfinite(np.delete(xy, 7, axis=0)).all()
    pca = block(spaces["X_pca"]["offset"], n * 8, np.float32).reshape(n, 2)
    np.testing.assert_allclose(pca, adata.obsm["X_pca"][:, :2])

    cats = {o["name"]: o for o in meta["obs"] if o["kind"] == "categorical"}
    conts = {o["name"]: o for o in meta["obs"] if o["kind"] == "continuous"}
    # Per-cell ids are dropped; small integer columns and booleans become categorical.
    assert set(cats) == {"cell_type", "cluster_int", "is_tumor"}
    assert set(conts) == {"cluster_int", "score", "n_counts"}
    assert cats["is_tumor"]["categories"] == ["False", "True"]
    assert cats["cluster_int"]["categories"] == ["0", "1", "2", "3", "4"]
    codes = block(cats["cluster_int"]["offset"], n * 4, np.uint32)
    np.testing.assert_array_equal(codes, adata.obs["cluster_int"].to_numpy())
    assert cats["cell_type"]["palette_rgba8"] == [0xFF0000FF, 0xFF0000FF, 0x8800FF00]
    score = block(conts["score"]["offset"], n * 4, np.float32)
    assert np.isnan(score[3])

    assert meta["expr"]["var_names"] == list(adata.var_names)
    np.testing.assert_array_equal(dense_from(meta, block, n), dense)


def test_layer_raw_and_gene_symbols():
    adata, dense = make_adata()
    n = adata.n_obs
    meta, block, _ = read_stviz(convert(adata, layer="counts", gene_symbols="symbol"))
    np.testing.assert_array_equal(dense_from(meta, block, n), dense * 2)
    names = meta["expr"]["var_names"]
    assert len(set(names)) == len(names) and names[0] == "G0" and names[25] == "G0-1"

    adata.raw = adata
    adata = adata[:, :10].copy()
    meta, block, _ = read_stviz(convert(adata, use_raw=True))
    assert meta["expr"]["n_genes"] == dense.shape[1]
    np.testing.assert_array_equal(dense_from(meta, block, n), dense)


def test_dense_matrix_and_small_chunks():
    adata, dense = make_adata()
    old = ex.CHUNK_ENTRIES
    ex.CHUNK_ENTRIES = 97  # force many chunks through every code path
    try:
        meta, block, _ = read_stviz(convert(adata))
        np.testing.assert_array_equal(dense_from(meta, block, adata.n_obs), dense)
        meta, block, _ = read_stviz(convert(adata, layer="counts"))
        np.testing.assert_array_equal(dense_from(meta, block, adata.n_obs), dense * 2)
        adata.X = dense
        meta, block, _ = read_stviz(convert(adata))
        np.testing.assert_array_equal(dense_from(meta, block, adata.n_obs), dense)
    finally:
        ex.CHUNK_ENTRIES = old


def test_no_expr_and_obs_fallback_space():
    adata, _ = make_adata()
    for key in list(adata.obsm.keys()):
        del adata.obsm[key]
    adata.obs["x"] = np.arange(adata.n_obs, dtype=np.float64)
    adata.obs["y"] = np.arange(adata.n_obs, dtype=np.float64) * 2
    meta, _, _ = read_stviz(convert(adata, include_expr=False))
    assert meta["expr"] is None
    assert [s["name"] for s in meta["spaces"]] == ["xy"]


def test_errors_are_clear_and_leave_no_partial_file():
    adata, _ = make_adata()
    for key in list(adata.obsm.keys()):
        del adata.obsm[key]
    tmp = Path(tempfile.mkdtemp(prefix="stviz_test_"))
    src, out = tmp / "in.h5ad", tmp / "out.stviz"
    adata.write_h5ad(src)
    out.write_bytes(b"previous")
    for kwargs in ({}, {"layer": "nope"}):
        try:
            ex.export_stviz(src, out, True, **kwargs)
        except ex.ConvertError:
            pass
        else:
            raise AssertionError("expected ConvertError")
    assert out.read_bytes() == b"previous"
    assert list(tmp.glob("*.partial")) == []


def test_replacing_a_file_that_is_in_use_gives_a_clear_error():
    """On Windows a memory-mapped file (an open .stviz in the app) cannot be replaced."""
    if sys.platform != "win32":
        return
    import mmap

    adata, _ = make_adata()
    tmp = Path(tempfile.mkdtemp(prefix="stviz_test_"))
    src, out = tmp / "in.h5ad", tmp / "out.stviz"
    adata.write_h5ad(src)
    ex.export_stviz(src, out, False)
    with open(out, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ):
        try:
            ex.export_stviz(src, out, False)
        except ex.ConvertError as e:
            assert "open in another program" in str(e)
        else:
            raise AssertionError("expected ConvertError")
    assert list(tmp.glob("*.partial")) == []


def test_unicode_paths():
    adata, _ = make_adata()
    tmp = Path(tempfile.mkdtemp(prefix="stviz_tëst_ü_"))
    src, out = tmp / "zellen_ä.h5ad", tmp / "ausgabe_ö.stviz"
    adata.write_h5ad(src)
    ex.export_stviz(src, out, True)
    meta, _, _ = read_stviz(out)
    assert meta["n_points"] == adata.n_obs


def test_parse_color():
    assert ex.parse_color("#fff") == 0xFFFFFFFF
    assert ex.parse_color("Grey") == 0xFF808080
    assert ex.parse_color((1.0, 0.0, 0.0)) == 0xFF0000FF
    assert ex.parse_color("not-a-color") is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
