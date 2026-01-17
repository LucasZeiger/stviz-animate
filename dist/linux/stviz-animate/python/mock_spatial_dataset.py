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
    elif pattern == "double_spiral":
        angles = np.linspace(0.0, 3.5 * np.pi, n_clusters)
        radii = np.linspace(bounds * 0.1, bounds * 0.48, n_clusters)
        centers = np.stack([np.cos(angles), np.sin(angles)], axis=1) * radii[:, None]
        flip = np.where(np.arange(n_clusters) % 2 == 0, 1.0, -1.0)
        centers[:, 0] *= flip
        centers = centers + center
    elif pattern == "arc":
        angles = np.linspace(-0.7 * np.pi, 0.7 * np.pi, n_clusters)
        radius = rng.uniform(bounds * 0.25, bounds * 0.42)
        centers = np.stack([np.cos(angles), np.sin(angles)], axis=1) * radius
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
    elif pattern == "constellation":
        centers = rng.uniform(bounds * 0.12, bounds * 0.88, size=(n_clusters, 2))
        pull = np.array([bounds * 0.5, bounds * 0.5], dtype=np.float32)
        centers = centers * 0.8 + pull * 0.2
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


def _shear(points: np.ndarray, sx: float, sy: float) -> np.ndarray:
    matrix = np.array([[1.0, sx], [sy, 1.0]], dtype=np.float32)
    return (points @ matrix.T).astype(np.float32)


def _lens(points: np.ndarray, strength: float) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    r2 = x * x + y * y
    scale = 1.0 + strength * np.exp(-r2 * 0.8)
    return np.stack([x * scale, y * scale], axis=1).astype(np.float32)


def _apply_gradient(points: np.ndarray, rng: np.random.Generator, bounds: float) -> np.ndarray:
    center = points.mean(axis=0, keepdims=True)
    direction = rng.normal(size=2)
    direction /= np.linalg.norm(direction) + 1e-6
    proj = (points - center) @ direction
    denom = np.max(np.abs(proj)) + 1e-6
    strength = rng.uniform(0.08, 0.18) * bounds
    drift = (proj / denom)[:, None] * direction[None, :] * strength
    return (points + drift).astype(np.float32)


def _apply_anisotropic(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    center = points.mean(axis=0, keepdims=True)
    theta = rng.uniform(0.0, 2.0 * np.pi)
    rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32
    )
    scale = np.diag(rng.uniform(0.6, 1.6, size=2).astype(np.float32))
    mat = rot.T @ scale @ rot
    return ((points - center) @ mat.T + center).astype(np.float32)


def _apply_voids(points: np.ndarray, rng: np.random.Generator, bounds: float) -> np.ndarray:
    out = points.copy()
    n_voids = int(rng.integers(1, 4))
    for _ in range(n_voids):
        center = rng.uniform(bounds * 0.2, bounds * 0.8, size=2)
        radius = rng.uniform(bounds * 0.06, bounds * 0.18)
        dx = out[:, 0] - center[0]
        dy = out[:, 1] - center[1]
        r = np.sqrt(dx * dx + dy * dy)
        mask = r < radius
        if not np.any(mask):
            continue
        jitter = rng.uniform(radius * 0.02, radius * 0.08, size=mask.sum())
        target = radius + jitter
        scale = target / (r[mask] + 1e-6)
        out[mask, 0] = center[0] + dx[mask] * scale
        out[mask, 1] = center[1] + dy[mask] * scale
    return out.astype(np.float32)


def _apply_layered_warp(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    center = points.mean(axis=0, keepdims=True)
    centered = points - center
    max_abs = np.max(np.abs(centered))
    if max_abs < 1e-6:
        return points.astype(np.float32)
    normed = centered / max_abs
    normed = _swirl(normed, strength=rng.uniform(0.6, 1.2))
    normed = _wave(
        normed,
        amp=rng.uniform(0.08, 0.16),
        freq=rng.uniform(4.0, 8.0),
        phase=rng.uniform(0.0, 2.0 * np.pi),
    )
    normed = _lens(normed, strength=rng.uniform(0.1, 0.25))
    return (normed * max_abs + center).astype(np.float32)


def _apply_subclusters(
    points: np.ndarray, rng: np.random.Generator, bounds: float
) -> np.ndarray:
    n = points.shape[0]
    if n < 10:
        return points.astype(np.float32)
    n_centers = int(rng.integers(6, 14))
    centers = points[rng.choice(n, size=n_centers, replace=False)]
    mask = rng.random(n) < rng.uniform(0.25, 0.45)
    labels = rng.integers(0, n_centers, size=n)
    alpha = rng.uniform(0.35, 0.6)
    noise = rng.normal(scale=bounds * 0.005, size=(mask.sum(), 2))
    out = points.copy()
    out[mask] = (
        out[mask] * (1.0 - alpha) + centers[labels[mask]] * alpha + noise
    )
    return out.astype(np.float32)


def _ridge_points(n: int, rng: np.random.Generator, bounds: float) -> np.ndarray:
    n_ctrl = int(rng.integers(3, 6))
    xs = np.linspace(bounds * 0.1, bounds * 0.9, n_ctrl)
    ys = bounds * 0.5 + rng.normal(scale=bounds * 0.2, size=n_ctrl)
    ctrl = np.stack([xs, ys], axis=1).astype(np.float32)
    center = np.array([bounds * 0.5, bounds * 0.5], dtype=np.float32)
    theta = rng.uniform(0.0, 2.0 * np.pi)
    rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32
    )
    ctrl = (ctrl - center) @ rot.T + center
    segs = ctrl[1:] - ctrl[:-1]
    seg_lens = np.sqrt((segs**2).sum(axis=1))
    total = seg_lens.sum()
    if total < 1e-6:
        return rng.uniform(0.0, bounds, size=(n, 2)).astype(np.float32)
    t = rng.random(n) * total
    cum = np.cumsum(seg_lens)
    seg_idx = np.searchsorted(cum, t)
    prev = np.concatenate([[0.0], cum[:-1]])
    local = (t - prev[seg_idx]) / (seg_lens[seg_idx] + 1e-6)
    pts = ctrl[seg_idx] + segs[seg_idx] * local[:, None]
    seg_dir = segs[seg_idx] / (seg_lens[seg_idx][:, None] + 1e-6)
    perp = np.stack([-seg_dir[:, 1], seg_dir[:, 0]], axis=1)
    jitter = rng.normal(scale=bounds * 0.02, size=(n, 1))
    pts = pts + perp * jitter
    return pts.astype(np.float32)


def _apply_ridges(
    points: np.ndarray, rng: np.random.Generator, bounds: float
) -> np.ndarray:
    n = points.shape[0]
    if n < 10:
        return points.astype(np.float32)
    frac = rng.uniform(0.12, 0.28)
    count = int(n * frac)
    ridge_pts = _ridge_points(count, rng, bounds)
    out = points.copy()
    idx = rng.choice(n, size=count, replace=False)
    out[idx] = ridge_pts
    return out.astype(np.float32)


def _apply_tissue_boundary(
    points: np.ndarray, rng: np.random.Generator, center: np.ndarray, bounds: float
) -> np.ndarray:
    out = points.copy()
    x = out[:, 0] - center[0]
    y = out[:, 1] - center[1]
    theta = rng.uniform(0.0, 2.0 * np.pi)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    xr = cos_t * x - sin_t * y
    yr = sin_t * x + cos_t * y
    sx = rng.uniform(0.7, 1.4)
    sy = rng.uniform(0.7, 1.4)
    xr /= sx
    yr /= sy
    ang = np.arctan2(yr, xr)
    r = np.sqrt(xr * xr + yr * yr)
    base = bounds * rng.uniform(0.18, 0.3)
    k1 = int(rng.integers(2, 5))
    k2 = int(rng.integers(5, 9))
    amp1 = rng.uniform(0.08, 0.14)
    amp2 = rng.uniform(0.04, 0.1)
    phase1 = rng.uniform(0.0, 2.0 * np.pi)
    phase2 = rng.uniform(0.0, 2.0 * np.pi)
    r_max = base * (1.0 + amp1 * np.sin(k1 * ang + phase1) + amp2 * np.sin(k2 * ang + phase2))
    squeeze = rng.uniform(0.55, 0.85)
    scale = np.ones_like(r)
    over = r > r_max
    scale[over] = 1.0 - (1.0 - (r_max[over] / (r[over] + 1e-6))) * squeeze
    xr *= scale
    yr *= scale
    edge_noise = rng.normal(scale=base * rng.uniform(0.015, 0.04), size=r.shape)
    radial = np.sqrt(xr * xr + yr * yr) + 1e-6
    xr += xr / radial * edge_noise
    yr += yr / radial * edge_noise
    xr *= sx
    yr *= sy
    out[:, 0] = cos_t * xr + sin_t * yr + center[0]
    out[:, 1] = -sin_t * xr + cos_t * yr + center[1]
    return out.astype(np.float32)


def _apply_dispersion(
    points: np.ndarray, rng: np.random.Generator, bounds: float
) -> np.ndarray:
    out = points.copy()
    n = out.shape[0]
    if n < 10:
        return out.astype(np.float32)
    center = out.mean(axis=0, keepdims=True)
    jitter = rng.normal(scale=bounds * rng.uniform(0.006, 0.012), size=out.shape)
    out = out + jitter
    halo_frac = rng.uniform(0.08, 0.18)
    halo_count = int(n * halo_frac)
    if halo_count > 0:
        halo_idx = rng.choice(n, size=halo_count, replace=False)
        vec = out[halo_idx] - center
        scale = rng.uniform(1.15, 1.4, size=(halo_count, 1))
        out[halo_idx] = center + vec * scale + rng.normal(
            scale=bounds * 0.01, size=(halo_count, 2)
        )
    outlier_frac = rng.uniform(0.02, 0.06)
    outlier_count = int(n * outlier_frac)
    if outlier_count > 0:
        outlier_idx = rng.choice(n, size=outlier_count, replace=False)
        out[outlier_idx] += rng.normal(scale=bounds * 0.06, size=(outlier_count, 2))
    return out.astype(np.float32)


def _apply_donut(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = points.copy()
    n = out.shape[0]
    if n < 10:
        return out.astype(np.float32)
    center = out.mean(axis=0, keepdims=True)
    vec = out - center
    r = np.sqrt((vec**2).sum(axis=1))
    max_r = np.percentile(r, 95)
    if max_r < 1e-6:
        return out.astype(np.float32)
    target = rng.uniform(0.45, 0.75) * max_r
    scale = rng.uniform(0.2, 0.5)
    donut_count = int(n * rng.uniform(0.5, 0.85))
    idx = rng.choice(n, size=donut_count, replace=False)
    r_sel = r[idx]
    r_new = target + (r_sel - r_sel.mean()) * scale
    r_new = np.maximum(r_new, max_r * 0.1)
    out[idx] = center + vec[idx] * (r_new / (r_sel + 1e-6))[:, None]
    return out.astype(np.float32)


def _apply_wave_field(
    points: np.ndarray, rng: np.random.Generator, bounds: float
) -> np.ndarray:
    out = points.copy()
    amp1 = bounds * rng.uniform(0.02, 0.05)
    amp2 = bounds * rng.uniform(0.015, 0.04)
    freq1 = rng.uniform(0.008, 0.02)
    freq2 = rng.uniform(0.01, 0.025)
    phase1 = rng.uniform(0.0, 2.0 * np.pi)
    phase2 = rng.uniform(0.0, 2.0 * np.pi)
    out[:, 0] += amp1 * np.sin(out[:, 1] * freq1 + phase1)
    out[:, 1] += amp2 * np.sin(out[:, 0] * freq2 + phase2)
    return out.astype(np.float32)


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

    pattern_pool = [
        "spiral",
        "ring",
        "wave_grid",
        "ribbon",
        "double_spiral",
        "arc",
        "constellation",
    ]
    rng.shuffle(pattern_pool)
    umap_pattern, tsne_pattern, pca_pattern = pattern_pool[:3]

    n_samples = 4
    sample_weights = rng.dirichlet(np.ones(n_samples))
    sample_sizes = rng.multinomial(n_cells, sample_weights)
    center = np.array([bounds * 0.5, bounds * 0.5], dtype=np.float32)
    orient = rng.uniform(0.0, 2.0 * np.pi)
    direction = np.array([np.cos(orient), np.sin(orient)], dtype=np.float32)
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    spacing = bounds * rng.uniform(0.16, 0.22)
    curve_amp = bounds * rng.uniform(0.08, 0.14)
    offsets = np.linspace(-(n_samples - 1) / 2.0, (n_samples - 1) / 2.0, n_samples)
    centers = []
    for t in offsets:
        arc = np.sin(t * np.pi / max(1.0, n_samples - 1))
        pos = center + direction * (t * spacing) + normal * (arc * curve_amp)
        pos += rng.normal(scale=bounds * 0.02, size=2)
        centers.append(pos)
    centers = np.array(centers, dtype=np.float32)
    centers = np.clip(centers, bounds * 0.15, bounds * 0.85)
    rng.shuffle(centers)
    sample_presets = [
        {
            "name": "donut_voids",
            "domains": (1, 1),
            "pattern_pool": ["ring", "arc"],
            "warp_pool": ["ripple", "shear"],
            "features": [
                lambda pts, rng=rng: _apply_donut(pts, rng),
                lambda pts, rng=rng, bounds=bounds: _apply_voids(pts, rng, bounds),
            ],
            "feature_names": ["donut", "voids"],
            "cluster_range": (5, 9),
            "span_range": (0.2, 0.3),
            "noise_range": (0.14, 0.28),
            "offset_scale": 0.05,
        },
        {
            "name": "ribbon_waves",
            "domains": (1, 2),
            "pattern_pool": ["ribbon", "wave_grid"],
            "warp_pool": ["wave", "shear"],
            "features": [
                lambda pts, rng=rng, bounds=bounds: _apply_gradient(pts, rng, bounds),
                lambda pts, rng=rng: _apply_anisotropic(pts, rng),
                lambda pts, rng=rng, bounds=bounds: _apply_wave_field(pts, rng, bounds),
            ],
            "feature_names": ["gradient", "anisotropic", "wave_field"],
            "cluster_range": (6, 12),
            "span_range": (0.24, 0.36),
            "noise_range": (0.16, 0.34),
            "offset_scale": 0.07,
        },
        {
            "name": "layered_subclusters",
            "domains": (1, 2),
            "pattern_pool": ["constellation", "arc"],
            "warp_pool": ["lens", "wave"],
            "features": [
                lambda pts, rng=rng: _apply_layered_warp(pts, rng),
                lambda pts, rng=rng, bounds=bounds: _apply_subclusters(pts, rng, bounds),
                lambda pts, rng=rng, bounds=bounds: _apply_wave_field(pts, rng, bounds),
            ],
            "feature_names": ["layered_warp", "subclusters", "wave_field"],
            "cluster_range": (6, 11),
            "span_range": (0.22, 0.32),
            "noise_range": (0.15, 0.32),
            "offset_scale": 0.06,
        },
        {
            "name": "ridge_filament",
            "domains": (1, 2),
            "pattern_pool": ["ribbon", "constellation"],
            "warp_pool": ["shear", "wave"],
            "features": [
                lambda pts, rng=rng, bounds=bounds: _apply_ridges(pts, rng, bounds),
                lambda pts, rng=rng, bounds=bounds: _apply_wave_field(pts, rng, bounds),
                lambda pts, rng=rng, bounds=bounds: _apply_gradient(pts, rng, bounds),
            ],
            "feature_names": ["ridges", "wave_field", "gradient"],
            "cluster_range": (5, 10),
            "span_range": (0.2, 0.32),
            "noise_range": (0.15, 0.3),
            "offset_scale": 0.07,
        },
    ]
    preset_order = rng.permutation(len(sample_presets))
    used_patterns: set[str] = set()
    used_warps: set[str] = set()

    spatial_base = np.zeros((n_cells, 2), dtype=np.float32)
    sample_labels = np.zeros(n_cells, dtype=np.int16)
    cursor = 0
    for s in range(n_samples):
        count = int(sample_sizes[s])
        if count <= 0:
            continue
        mode = sample_presets[preset_order[s % len(sample_presets)]]
        print(
            f"Sample {s} mode: {mode['name']} | features: {', '.join(mode['feature_names'])}"
        )
        domain_count = int(rng.integers(mode["domains"][0], mode["domains"][1] + 1))
        domain_weights = rng.dirichlet(np.ones(domain_count))
        domain_sizes = rng.multinomial(count, domain_weights)
        sample_coords = np.zeros((count, 2), dtype=np.float32)
        sample_cursor = 0
        for d in range(domain_count):
            d_count = int(domain_sizes[d])
            if d_count <= 0:
                continue
            pattern = rng.choice(mode["pattern_pool"])
            warp = rng.choice(mode["warp_pool"])
            used_patterns.add(pattern)
            used_warps.add(warp)
            cluster_min, cluster_max = mode.get("cluster_range", (4, 12))
            sample_clusters = int(rng.integers(cluster_min, cluster_max + 1))
            noise_min, noise_max = mode.get("noise_range", (0.16, 0.34))
            coords, _ = _make_clustered_coords(
                d_count,
                rng,
                1.0,
                pattern,
                sample_clusters,
                noise_min=noise_min,
                noise_max=noise_max,
            )
            span_min, span_max = mode.get("span_range", (0.2, 0.34))
            span = bounds * rng.uniform(span_min, span_max)
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
            elif warp == "shear":
                coords = _shear(
                    coords,
                    sx=rng.uniform(-0.35, 0.35),
                    sy=rng.uniform(-0.2, 0.2),
                )
            elif warp == "lens":
                coords = _lens(coords, strength=rng.uniform(0.12, 0.35))
            else:
                coords = _ripple(
                    coords,
                    amp=rng.uniform(0.18, 0.3) * span,
                    freq=rng.uniform(0.02, 0.05),
                    phase=rng.uniform(0.0, 2.0 * np.pi),
                )
            offset_scale = mode.get("offset_scale", 0.06)
            domain_offset = rng.normal(scale=bounds * offset_scale, size=2)
            coords += centers[s] + domain_offset
            coords += rng.normal(scale=bounds * 0.004, size=coords.shape)
            coords = np.clip(coords, 0.0, bounds)
            sample_coords[sample_cursor : sample_cursor + d_count] = coords
            sample_cursor += d_count
        for feature in mode["features"]:
            sample_coords = feature(sample_coords)
        sample_coords = _apply_dispersion(sample_coords, rng, bounds)
        sample_coords = centers[s] + (sample_coords - centers[s]) * 1.12
        sample_coords = np.clip(sample_coords, 0.0, bounds)
        spatial_base[cursor : cursor + count] = sample_coords
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
        sample_patterns=sorted(used_patterns),
        sample_warps=sorted(used_warps),
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
