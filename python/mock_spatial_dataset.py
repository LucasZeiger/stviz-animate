#!/usr/bin/env python3
# Creates an AnnData "spatial transcriptomics-like" dataset:
# - rows = cells
# - .obs contains embedding_1 .. embedding_n (float32)
# - includes spatial coordinates (obsm["spatial"]) and extra 2D spaces for transitions
# - optional sparse gene counts (X)

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def _pick_2d(emb: np.ndarray, start: int, rng: np.random.Generator) -> np.ndarray:
    if emb.shape[1] >= start + 2:
        return emb[:, start : start + 2].astype(np.float32, copy=False)
    return rng.standard_normal(size=(emb.shape[0], 2)).astype(np.float32)


def make_mock_spatial(
    n_cells: int = 10_000,
    n_embeddings: int = 64,
    n_genes: int = 2_000,
    expr_density: float = 0.01,
    seed: int = 0,
    make_expression: bool = True,
) -> ad.AnnData:
    rng = np.random.default_rng(seed)

    # ---- obs (cell metadata) ----
    obs = pd.DataFrame(index=[f"cell_{i:05d}" for i in range(n_cells)])

    # Random scalar "embeddings": embedding_1 ... embedding_n
    emb = rng.standard_normal(size=(n_cells, n_embeddings)).astype(np.float32)
    for j in range(n_embeddings):
        obs[f"embedding_{j + 1}"] = emb[:, j]

    # Useful categorical fields
    obs["sample"] = pd.Categorical(rng.integers(1, 5, size=n_cells).astype(np.int16))
    obs["cluster"] = pd.Categorical(rng.integers(0, 20, size=n_cells).astype(np.int16))

    # A continuous field for color testing
    obs["score"] = rng.normal(size=n_cells).astype(np.float32)

    # ---- spatial coordinates ----
    spatial = rng.uniform(0, 1_000, size=(n_cells, 2)).astype(np.float32)
    obs["x"] = spatial[:, 0]
    obs["y"] = spatial[:, 1]

    # ---- expression matrix X ----
    if make_expression:
        n_nnz = int(n_cells * n_genes * expr_density)
        rows = rng.integers(0, n_cells, size=n_nnz, dtype=np.int32)
        cols = rng.integers(0, n_genes, size=n_nnz, dtype=np.int32)
        data = rng.poisson(lam=1.0, size=n_nnz).astype(np.int16)
        data[data == 0] = 1
        X = sp.coo_matrix((data, (rows, cols)), shape=(n_cells, n_genes)).tocsr()
    else:
        X = sp.csr_matrix((n_cells, n_genes), dtype=np.float32)

    var = pd.DataFrame(index=[f"gene_{g:05d}" for g in range(n_genes)])

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = spatial

    # Extra 2D spaces for transitions
    adata.obsm["X_umap"] = _pick_2d(emb, 0, rng)
    adata.obsm["X_tsne"] = _pick_2d(emb, 2, rng)
    adata.obsm["X_pca"] = _pick_2d(emb, 4, rng)

    # Optional: store parameters for reproducibility
    adata.uns["mock_params"] = dict(
        n_cells=n_cells,
        n_embeddings=n_embeddings,
        n_genes=n_genes,
        expr_density=expr_density,
        seed=seed,
        make_expression=make_expression,
    )
    return adata


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("mock_spatial_10k.h5ad"))
    ap.add_argument("--cells", type=int, default=10_000)
    ap.add_argument("--embeddings", type=int, default=64)
    ap.add_argument("--genes", type=int, default=2_000)
    ap.add_argument("--density", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-expr", action="store_true", help="Skip expression matrix.")
    args = ap.parse_args()

    adata = make_mock_spatial(
        n_cells=args.cells,
        n_embeddings=args.embeddings,
        n_genes=args.genes,
        expr_density=args.density,
        seed=args.seed,
        make_expression=not args.no_expr,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.out)
    print(f"Wrote: {args.out}")
    print(adata)


if __name__ == "__main__":
    main()
