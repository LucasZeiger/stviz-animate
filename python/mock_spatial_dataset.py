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


def _cluster_centers(
    n_clusters: int,
    rng: np.random.Generator,
    bounds: float,
    pattern: str,
) -> np.ndarray:
    center = np.array([bounds * 0.5, bounds * 0.5], dtype=np.float32)
    jitter = bounds * 0.02
    if pattern == "ring":
        angles = np.linspace(0.0, 2.0 * np.pi, n_clusters, endpoint=False)
        radius = rng.uniform(bounds * 0.25, bounds * 0.42)
        centers = np.stack([np.cos(angles), np.sin(angles)], axis=1) * radius
        centers = centers + center
    elif pattern == "spiral":
        angles = np.linspace(0.0, 4.0 * np.pi, n_clusters)
        radii = np.linspace(bounds * 0.08, bounds * 0.45, n_clusters)
        centers = np.stack([np.cos(angles), np.sin(angles)], axis=1) * radii[:, None]
        centers = centers + center
    elif pattern == "wave_grid":
        side = int(np.ceil(np.sqrt(n_clusters)))
        xs = np.linspace(bounds * 0.12, bounds * 0.88, side)
        ys = np.linspace(bounds * 0.12, bounds * 0.88, side)
        grid = np.array([(x, y) for y in ys for x in xs], dtype=np.float32)[:n_clusters]
        amp = bounds * 0.08
        freq = rng.uniform(0.006, 0.015)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        grid[:, 1] += amp * np.sin(grid[:, 0] * freq + phase)
        centers = grid
    else:
        xs = np.linspace(-1.0, 1.0, n_clusters)
        amp = bounds * 0.18
        centers = np.stack(
            [
                center[0] + xs * bounds * 0.42,
                center[1] + np.sin(xs * np.pi * 2.0) * amp,
            ],
            axis=1,
        )
    centers = centers + rng.normal(scale=jitter, size=centers.shape)
    return centers.astype(np.float32)


def _assign_clusters(
    n_cells: int,
    n_clusters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    weights = rng.random(n_clusters)
    weights /= weights.sum()
    return rng.choice(n_clusters, size=n_cells, p=weights)


def _make_clustered_coords(
    n_cells: int,
    rng: np.random.Generator,
    bounds: float,
    pattern: str,
    n_clusters: int,
    noise_min: float = 0.015,
    noise_max: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    centers = _cluster_centers(n_clusters, rng, bounds, pattern)
    assignments = _assign_clusters(n_cells, n_clusters, rng)
    scales = rng.uniform(bounds * noise_min, bounds * noise_max, size=n_clusters)
    coords = centers[assignments] + rng.normal(
        scale=scales[assignments, None], size=(n_cells, 2)
    )
    return coords.astype(np.float32), assignments


def _swirl(points: np.ndarray, strength: float) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    r = np.sqrt(x * x + y * y)
    ang = np.arctan2(y, x) + strength * r
    return np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1).astype(np.float32)


def _petal(points: np.ndarray, petals: int, amp: float, phase: float) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    r = np.sqrt(x * x + y * y)
    ang = np.arctan2(y, x)
    r = r * (1.0 + amp * np.sin(ang * petals + phase))
    return np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1).astype(np.float32)


def _wave(points: np.ndarray, amp: float, freq: float, phase: float) -> np.ndarray:
    out = points.copy()
    out[:, 1] += amp * np.sin(out[:, 0] * freq + phase)
    return out


def _scale_center(points: np.ndarray, target_max: float) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    max_abs = np.max(np.abs(centered))
    if max_abs < 1e-6:
        return centered.astype(np.float32)
    return (centered / max_abs * target_max).astype(np.float32)


def _ripple(points: np.ndarray, amp: float, freq: float, phase: float) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    r = np.sqrt(x * x + y * y)
    ripple = amp * np.sin(r * freq + phase)
    return np.stack([x + ripple * 0.6, y + ripple], axis=1).astype(np.float32)


def _cluster_labels_from_points(
    points: np.ndarray,
    n_clusters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    theta = rng.uniform(0.0, 2.0 * np.pi)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated = points @ rot.T
    kx = int(np.ceil(np.sqrt(n_clusters)))
    ky = int(np.ceil(n_clusters / kx))
    xs = rotated[:, 0]
    ys = rotated[:, 1]
    x_edges = np.quantile(xs, np.linspace(0.0, 1.0, kx + 1))
    y_edges = np.quantile(ys, np.linspace(0.0, 1.0, ky + 1))
    ix = np.searchsorted(x_edges, xs, side="right") - 1
    iy = np.searchsorted(y_edges, ys, side="right") - 1
    ix = np.clip(ix, 0, kx - 1)
    iy = np.clip(iy, 0, ky - 1)
    labels = (iy * kx + ix).astype(np.int16)
    if kx * ky > n_clusters:
        labels = np.minimum(labels, n_clusters - 1)
    return labels


def make_mock_spatial(
    n_cells: int = 500_000,
    n_embeddings: int = 10,
    n_genes: int = 5,
    expr_density: float = 0.2,
    seed: int | None = None,
    make_expression: bool = True,
) -> ad.AnnData:
    if n_cells < 10_000 or n_cells > 10_000_000:
        raise ValueError("n_cells must be between 10,000 and 10,000,000.")
    if seed is None:
        seed = int(np.random.SeedSequence().entropy)
    rng = np.random.default_rng(seed)
    bounds = 1_000.0

    # ---- obs (cell metadata) ----
    obs = pd.DataFrame(index=[f"cell_{i:05d}" for i in range(n_cells)])

    pattern_pool = ["spiral", "ring", "wave_grid", "ribbon"]
    rng.shuffle(pattern_pool)
    umap_pattern, tsne_pattern, pca_pattern = pattern_pool[:3]

    n_samples = 4
    sample_weights = rng.dirichlet(np.ones(n_samples))
    sample_sizes = rng.multinomial(n_cells, sample_weights)
    grid_side = int(np.ceil(np.sqrt(n_samples)))
    grid = np.linspace(-1.0, 1.0, grid_side)
    offsets = np.array([(x, y) for y in grid for x in grid], dtype=np.float32)[:n_samples]
    center = np.array([bounds * 0.5, bounds * 0.5], dtype=np.float32)
    spacing = bounds * rng.uniform(0.12, 0.18)
    centers = center + offsets * spacing
    centers += rng.normal(scale=bounds * 0.01, size=centers.shape)
    rng.shuffle(centers)
    sample_patterns = pattern_pool.copy()
    rng.shuffle(sample_patterns)
    sample_warps = ["swirl", "petal", "wave", "ripple"]
    rng.shuffle(sample_warps)

    spatial_base = np.zeros((n_cells, 2), dtype=np.float32)
    sample_labels = np.zeros(n_cells, dtype=np.int16)
    cursor = 0
    for s in range(n_samples):
        count = int(sample_sizes[s])
        if count <= 0:
            continue
        pattern = sample_patterns[s % len(sample_patterns)]
        warp = sample_warps[s % len(sample_warps)]
        sample_clusters = int(rng.integers(6, 12))
        coords, _ = _make_clustered_coords(
            count,
            rng,
            1.0,
            pattern,
            sample_clusters,
            noise_min=0.12,
            noise_max=0.22,
        )
        span = bounds * rng.uniform(0.16, 0.25)
        coords = _scale_center(coords, span)
        if warp == "swirl":
            coords = _swirl(coords, strength=rng.uniform(0.6, 1.1))
        elif warp == "petal":
            coords = _petal(
                coords,
                petals=int(rng.integers(4, 8)),
                amp=rng.uniform(0.15, 0.3),
                phase=rng.uniform(0.0, 2.0 * np.pi),
            )
        elif warp == "wave":
            coords = _wave(
                coords,
                amp=rng.uniform(0.18, 0.32) * span,
                freq=rng.uniform(0.02, 0.05),
                phase=rng.uniform(0.0, 2.0 * np.pi),
            )
        else:
            coords = _ripple(
                coords,
                amp=rng.uniform(0.18, 0.3) * span,
                freq=rng.uniform(0.02, 0.05),
                phase=rng.uniform(0.0, 2.0 * np.pi),
            )
        coords += centers[s]
        coords += rng.normal(scale=bounds * 0.004, size=coords.shape)
        coords = np.clip(coords, 0.0, bounds)

        spatial_base[cursor : cursor + count] = coords
        sample_labels[cursor : cursor + count] = s
        cursor += count

    if cursor < n_cells:
        remaining = n_cells - cursor
        spatial_base[cursor:] = rng.uniform(0.0, bounds, size=(remaining, 2))
        sample_labels[cursor:] = rng.integers(0, n_samples, size=remaining).astype(np.int16)

    spatial_clusters = int(rng.integers(8, 14))
    assignments = _cluster_labels_from_points(spatial_base, spatial_clusters, rng)

    # Useful categorical fields
    cluster_counts = [5, 8, 11, 15, 20]
    obs["sample"] = pd.Categorical(sample_labels.astype(np.int16))
    for count in cluster_counts:
        obs[f"cluster_{count:02d}"] = pd.Categorical(
            _cluster_labels_from_points(spatial_base, count, rng)
        )

    # A continuous field for color testing
    score = (
        (spatial_base[:, 0] - spatial_base[:, 0].mean())
        / (spatial_base[:, 0].std() + 1e-6)
    ).astype(np.float32)
    obs["score"] = score

    # ---- spatial coordinates ----
    spatial = spatial_base.copy()
    wave_amp = bounds * rng.uniform(0.05, 0.1)
    wave_freq = rng.uniform(0.004, 0.012)
    wave_phase = rng.uniform(0.0, 2.0 * np.pi)
    spatial = _wave(spatial, wave_amp, wave_freq, wave_phase)
    spatial[:, 0] += wave_amp * 0.6 * np.cos(spatial[:, 1] * wave_freq * 0.8 + wave_phase * 0.7)
    spatial = np.clip(spatial, 0.0, bounds)
    obs["x"] = spatial[:, 0]
    obs["y"] = spatial[:, 1]
    spatial_norm = spatial / bounds

    latent = _scale_center(spatial_base, 1.0)
    theta = rng.uniform(0.0, 2.0 * np.pi)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    latent = latent @ rot.T

    umap = _swirl(latent, strength=rng.uniform(0.8, 1.3))
    umap = _wave(
        umap,
        amp=rng.uniform(0.25, 0.45),
        freq=rng.uniform(3.0, 6.0),
        phase=rng.uniform(0.0, 2.0 * np.pi),
    )
    extra_umap, _ = _make_clustered_coords(
        n_cells,
        rng,
        1.0,
        umap_pattern,
        int(rng.integers(7, 13)),
        noise_min=0.08,
        noise_max=0.2,
    )
    umap = umap * 0.7 + extra_umap * 0.3
    umap += rng.normal(scale=0.08, size=umap.shape)

    tsne = _petal(
        latent,
        petals=int(rng.integers(5, 9)),
        amp=rng.uniform(0.2, 0.35),
        phase=rng.uniform(0.0, 2.0 * np.pi),
    )
    tsne = _wave(
        tsne,
        amp=rng.uniform(0.2, 0.35),
        freq=rng.uniform(2.0, 4.0),
        phase=rng.uniform(0.0, 2.0 * np.pi),
    )
    extra_tsne, _ = _make_clustered_coords(
        n_cells,
        rng,
        1.0,
        tsne_pattern,
        int(rng.integers(10, 18)),
        noise_min=0.1,
        noise_max=0.22,
    )
    tsne = tsne * 0.6 + extra_tsne * 0.4
    tsne += rng.normal(scale=0.1, size=tsne.shape)

    pca = latent.copy()
    pca[:, 0] *= rng.uniform(1.8, 2.4)
    pca[:, 1] *= rng.uniform(0.5, 0.8)
    pca[:, 0] += pca[:, 1] * rng.uniform(0.25, 0.55)
    pca = _wave(
        pca,
        amp=rng.uniform(0.15, 0.25),
        freq=rng.uniform(2.5, 4.5),
        phase=rng.uniform(0.0, 2.0 * np.pi),
    )
    extra_pca, _ = _make_clustered_coords(
        n_cells,
        rng,
        1.0,
        pca_pattern,
        int(rng.integers(6, 12)),
        noise_min=0.1,
        noise_max=0.2,
    )
    pca = pca * 0.75 + extra_pca * 0.25
    pca += rng.normal(scale=0.05, size=pca.shape)

    umap = _scale_center(umap, 10.0)
    tsne = _scale_center(tsne, 25.0)
    pca = _scale_center(pca, 6.0)

    radial = np.sqrt(
        (spatial_norm[:, 0] - 0.5) ** 2 + (spatial_norm[:, 1] - 0.5) ** 2
    )
    radial = radial / radial.max()
    score_norm = (score - score.min()) / (score.max() - score.min() + 1e-6)

    features = [
        umap[:, 0],
        umap[:, 1],
        tsne[:, 0],
        tsne[:, 1],
        pca[:, 0],
        pca[:, 1],
        spatial_norm[:, 0],
        spatial_norm[:, 1],
        radial.astype(np.float32),
        score_norm.astype(np.float32),
    ]

    emb = np.zeros((n_cells, n_embeddings), dtype=np.float32)
    for j in range(min(n_embeddings, len(features))):
        emb[:, j] = features[j].astype(np.float32, copy=False)
    if n_embeddings > len(features):
        extra = rng.normal(scale=0.2, size=(n_cells, n_embeddings - len(features)))
        basis = np.column_stack(features).astype(np.float32)
        mix = rng.normal(scale=0.1, size=(basis.shape[1], extra.shape[1]))
        emb[:, len(features) :] = (basis @ mix + extra).astype(np.float32)
    for j in range(n_embeddings):
        obs[f"embedding_{j + 1}"] = emb[:, j]

    # ---- expression matrix X ----
    if make_expression:
        n_clusters = int(assignments.max()) + 1
        weights = np.full((n_genes, n_clusters), 0.15, dtype=np.float32)
        pref_a = rng.integers(0, n_clusters, size=n_genes)
        pref_b = rng.integers(0, n_clusters, size=n_genes)
        weights[np.arange(n_genes), pref_a] = rng.uniform(0.7, 1.0, size=n_genes)
        weights[np.arange(n_genes), pref_b] = rng.uniform(0.4, 0.8, size=n_genes)

        gene_max = rng.integers(4, 8, size=n_genes, dtype=np.int16)
        wave_freq = rng.uniform(3.0, 7.0)
        wave_phase = rng.uniform(0.0, 2.0 * np.pi)
        wave = 0.5 + 0.5 * np.sin(
            spatial_norm[:, 0] * wave_freq * 2.0 * np.pi + wave_phase
        ) * np.cos(
            spatial_norm[:, 1] * wave_freq * 2.0 * np.pi + wave_phase * 0.7
        )

        cluster_strength = weights[:, assignments].T
        cluster_strength = (cluster_strength - 0.15) / 0.85
        cluster_strength = np.clip(cluster_strength, 0.0, 1.0)

        base_fields = [
            spatial_norm[:, 0],
            spatial_norm[:, 1],
            1.0 - radial,
            wave,
            cluster_strength[:, 0] if n_genes > 0 else wave,
        ]
        if n_genes > len(base_fields):
            extra = np.column_stack(
                [
                    score_norm,
                    (spatial_norm[:, 0] * spatial_norm[:, 1]),
                    wave * (1.0 - radial),
                ]
            )
            while len(base_fields) < n_genes:
                idx = len(base_fields) - 5
                base_fields.append(extra[:, idx % extra.shape[1]])

        values = np.zeros((n_cells, n_genes), dtype=np.float32)
        for g in range(n_genes):
            field = base_fields[g]
            if g < cluster_strength.shape[1]:
                field = 0.7 * field + 0.3 * cluster_strength[:, g]
            value = np.clip(field, 0.0, 1.0) * gene_max[g]
            values[:, g] = value

        mask = rng.random(size=values.shape) < expr_density
        sparse_vals = np.round(values * mask).astype(np.uint8)
        rows, cols = np.nonzero(sparse_vals)
        data = sparse_vals[rows, cols].astype(np.uint8)
        X = sp.coo_matrix((data, (rows, cols)), shape=(n_cells, n_genes)).tocsr()
    else:
        X = sp.csr_matrix((n_cells, n_genes), dtype=np.float32)

    var = pd.DataFrame(index=[f"gene_{g:05d}" for g in range(n_genes)])

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = spatial

    # Extra 2D spaces for transitions
    adata.obsm["X_umap"] = umap
    adata.obsm["X_tsne"] = tsne
    adata.obsm["X_pca"] = pca

    # Optional: store parameters for reproducibility
    adata.uns["mock_params"] = dict(
        n_cells=n_cells,
        n_embeddings=n_embeddings,
        n_genes=n_genes,
        expr_density=expr_density,
        seed=seed,
        make_expression=make_expression,
        cluster_counts=cluster_counts,
        sample_patterns=sample_patterns,
        sample_warps=sample_warps,
        umap_pattern=umap_pattern,
        tsne_pattern=tsne_pattern,
        pca_pattern=pca_pattern,
        spatial_clusters=spatial_clusters,
        expr_patterns="sparse_cluster_wave",
    )
    return adata


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("mock_spatial_500k.h5ad"))
    ap.add_argument("--cells", type=int, default=500_000)
    ap.add_argument("--embeddings", type=int, default=10)
    ap.add_argument("--genes", type=int, default=5)
    ap.add_argument("--density", type=float, default=0.2)
    ap.add_argument("--seed", type=int)
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
