#!/usr/bin/env python3
"""
master_mock_spatial.py

Generates an AnnData "spatial transcriptomics-like" dataset with:
- spatial tissue-like coordinates in obsm["spatial"] and obs["x"], obs["y"]
- embedding-like 2D spaces: obsm["X_umap"], obsm["X_tsne"], obsm["X_pca"]
- obs fields: sample, region, cell_type, cluster_{05,08,11,15,20}, score, embedding_1..embedding_n
- optional sparse gene counts matrix X

Additions for high-density rendering (generator-side):
- obs["render_alpha"]: density-compensated per-point alpha (dense regions get lower alpha)
- obs["render_size"]: optional per-point size multiplier derived from alpha
- obs["render_lod_100k"]: boolean mask for a stratified spatial LOD subset

Key improvements vs earlier variants:
1) Spatial: explicit biological motifs (glands, vessels, tumor, immune foci, layers) + boundary/compartments/dispersion.
2) Embeddings: anisotropic clusters + centroid push-out for separation + mild global structure.
3) Categorical realism: cell_type is spatially/region-informed AND embedding-cluster-biased so clusters don't look identical.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


# =========================
# Numeric utilities
# =========================

def _scale_center(points: np.ndarray, target_max: float) -> np.ndarray:
    """Center points and scale so max(abs(coord)) == target_max (robust for arbitrary shapes)."""
    pts = points.astype(np.float32, copy=False)
    centered = pts - pts.mean(axis=0, keepdims=True)
    max_abs = float(np.max(np.abs(centered)))
    if max_abs < 1e-6:
        return centered.astype(np.float32)
    return (centered / max_abs * np.float32(target_max)).astype(np.float32)


def _reflect_into_bounds(xy: np.ndarray, bounds: float) -> np.ndarray:
    """
    Reflect coordinates into [0, bounds] to avoid hard clipping 'walls'.
    """
    out = xy.astype(np.float32, copy=True)
    period = np.float32(2.0 * bounds)
    for j in range(2):
        v = out[:, j]
        v = np.mod(v, period)  # [0, 2*bounds)
        v = np.where(v > bounds, period - v, v)
        out[:, j] = v
    return out


def _rms_radius(points: np.ndarray) -> float:
    """O(n) scale proxy used to adapt warp magnitudes to the cloud size."""
    pts = points.astype(np.float32, copy=False)
    c = pts.mean(axis=0, keepdims=True)
    d = pts - c
    r2 = (d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]).mean(dtype=np.float64)
    return float(np.sqrt(max(r2, 1e-8)))


# =========================
# Generator-side rendering helpers (for high density)
# =========================

def _bin_index_2d(xy: np.ndarray, bin_size: float) -> np.ndarray:
    """
    Map each point to a 1D bin index using square bins of width bin_size.
    Uses a per-call local origin (xmin/ymin) to keep bin counts compact.
    """
    x = xy[:, 0].astype(np.float32, copy=False)
    y = xy[:, 1].astype(np.float32, copy=False)

    xmin = float(x.min())
    ymin = float(y.min())

    ix = np.floor((x - xmin) / np.float32(bin_size)).astype(np.int32)
    iy = np.floor((y - ymin) / np.float32(bin_size)).astype(np.int32)

    w = int(ix.max()) + 1  # bins per row
    idx = ix.astype(np.int64) + np.int64(w) * iy.astype(np.int64)
    return idx


def density_compensated_alpha(
    xy: np.ndarray,
    sample_labels: np.ndarray | None,
    bin_size: float = 2.0,
    gamma: float = 0.7,
    alpha_min: float = 0.02,
    pctl_norm: float = 99.0,
) -> np.ndarray:
    """
    Per-point alpha in (alpha_min..1], inversely proportional to local density.

    - bin_size is in the same units as xy (spatial coords). It should roughly match
      the "visual footprint" of a point in your renderer, expressed in spatial units.
    - gamma controls strength: ~0.5 mild, ~0.7 strong, ~1.0 aggressive.
    - pctl_norm stabilizes normalization (ignores extreme sparse bins).
    """
    n = xy.shape[0]
    if sample_labels is None:
        sample_labels = np.zeros(n, dtype=np.int16)

    out = np.empty(n, dtype=np.float32)
    lab = sample_labels.astype(np.int32, copy=False)

    for s in np.unique(lab):
        m = lab == s
        idx = _bin_index_2d(xy[m], bin_size=bin_size)

        counts = np.bincount(idx)  # count per bin
        local = counts[idx].astype(np.float32)  # local density per point

        # alpha ~ density^-gamma
        a = np.power(local, -np.float32(gamma))

        # normalize by a high percentile so a few very sparse bins don't dominate
        denom = np.float32(np.percentile(a, pctl_norm) + 1e-6)
        a = a / denom

        out[m] = np.clip(a, np.float32(alpha_min), np.float32(1.0)).astype(np.float32)

    return out


def density_compensated_size(alpha: np.ndarray, size_min: float = 0.25) -> np.ndarray:
    """
    Optional size multiplier derived from alpha.
    sqrt(alpha) gives a gentle reduction in dense regions without collapsing sizes.
    """
    r = np.sqrt(alpha.astype(np.float32, copy=False)).astype(np.float32)
    return np.clip(r, np.float32(size_min), np.float32(1.0)).astype(np.float32)


def lod_mask_stratified(
    xy: np.ndarray,
    target_n: int,
    sample_labels: np.ndarray | None,
    rng: np.random.Generator,
    bin_size: float = 2.0,
) -> np.ndarray:
    """
    Boolean mask selecting ~target_n points, stratified by spatial bins.
    Preserves boundaries/holes better than uniform random subsampling.

    Works per-sample, allocating target proportionally to sample size.
    """
    n = xy.shape[0]
    target_n = int(min(max(target_n, 0), n))

    if sample_labels is None:
        sample_labels = np.zeros(n, dtype=np.int16)
    lab = sample_labels.astype(np.int32, copy=False)

    keep = np.zeros(n, dtype=bool)
    if target_n == 0:
        return keep
    if target_n >= n:
        keep[:] = True
        return keep

    uniques, counts = np.unique(lab, return_counts=True)
    frac = counts / counts.sum()
    targets = np.floor(frac * target_n).astype(int)

    rem = int(target_n - targets.sum())
    if rem > 0:
        add = rng.choice(len(targets), size=rem, replace=True)
        np.add.at(targets, add, 1)

    for s, t_s in zip(uniques, targets):
        m = np.where(lab == s)[0]
        if m.size == 0 or t_s <= 0:
            continue
        if t_s >= m.size:
            keep[m] = True
            continue

        idx = _bin_index_2d(xy[m], bin_size=bin_size)

        # shuffle within sample to randomize choice within bins
        order = rng.permutation(m.size)
        idx_shuf = idx[order]

        # sort by bin id so bins form contiguous groups
        sort = np.argsort(idx_shuf, kind="mergesort")
        ord2 = order[sort]
        idx_sorted = idx_shuf[sort]

        # group boundaries
        change = np.r_[True, idx_sorted[1:] != idx_sorted[:-1]]
        n_bins = int(change.sum())
        if n_bins <= 0:
            chosen = rng.choice(m, size=t_s, replace=False)
            keep[chosen] = True
            continue

        # keep k points per non-empty bin (then trim to exact t_s)
        k = max(1, int(np.ceil(t_s / n_bins)))

        # start index of each group, broadcast to all items in group
        starts = np.maximum.accumulate(np.where(change, np.arange(m.size), 0))
        pos_in_group = np.arange(m.size) - starts
        keep_sorted = pos_in_group < k

        chosen = m[ord2[keep_sorted]]
        if chosen.size > t_s:
            drop = rng.choice(chosen.size, size=(chosen.size - t_s), replace=False)
            chosen = np.delete(chosen, drop)

        keep[chosen] = True

    return keep


# =========================
# Warps / tissue finishing
# =========================

def _wave(points: np.ndarray, amp: float, freq: float, phase: float) -> np.ndarray:
    out = points.astype(np.float32, copy=True)
    out[:, 1] += np.float32(amp) * np.sin(out[:, 0] * np.float32(freq) + np.float32(phase)).astype(np.float32)
    return out


def _apply_dispersion(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Adds jitter + halo + a few outliers, scaled to the point cloud.
    Reads as "capture noise" / "edge fuzz".
    """
    out = points.astype(np.float32, copy=True)
    n = out.shape[0]
    if n < 10:
        return out

    scale0 = _rms_radius(out)
    center = out.mean(axis=0, keepdims=True)

    # global jitter
    out += rng.normal(scale=scale0 * rng.uniform(0.010, 0.020), size=out.shape).astype(np.float32)

    # halo
    halo_count = int(n * rng.uniform(0.02, 0.06))
    if halo_count > 0:
        idx = rng.choice(n, size=halo_count, replace=False)
        vec = out[idx] - center
        out[idx] = center + vec * rng.uniform(1.03, 1.12, size=(halo_count, 1)).astype(np.float32)
        out[idx] += rng.normal(scale=scale0 * rng.uniform(0.02, 0.05), size=(halo_count, 2)).astype(np.float32)

    # outliers
    outlier_count = int(n * rng.uniform(0.00, 0.01))
    if outlier_count > 0:
        idx = rng.choice(n, size=outlier_count, replace=False)
        out[idx] += rng.normal(scale=scale0 * rng.uniform(0.06, 0.12), size=(outlier_count, 2)).astype(np.float32)

    return out


def _apply_tissue_boundary(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Applies a soft irregular boundary so the sample looks like a tissue island
    rather than an unbounded cloud.
    """
    out = points.astype(np.float32, copy=True)
    n = out.shape[0]
    if n < 50:
        return out

    center = out.mean(axis=0)
    x = out[:, 0] - center[0]
    y = out[:, 1] - center[1]

    # random rotation and anisotropy
    theta = rng.uniform(0.0, 2.0 * np.pi)
    ct, st = np.cos(theta), np.sin(theta)
    xr = ct * x - st * y
    yr = st * x + ct * y
    sx = rng.uniform(0.7, 1.4)
    sy = rng.uniform(0.7, 1.4)
    xr /= sx
    yr /= sy

    ang = np.arctan2(yr, xr)
    r = np.sqrt(xr * xr + yr * yr)

    # boundary radius adapted to cloud
    rms = float(np.sqrt(max((r * r).mean(dtype=np.float64), 1e-8)))
    base = rms * rng.uniform(1.25, 1.65)

    k1 = int(rng.integers(2, 5))
    k2 = int(rng.integers(5, 9))
    amp1 = rng.uniform(0.08, 0.14)
    amp2 = rng.uniform(0.04, 0.10)
    ph1 = rng.uniform(0.0, 2.0 * np.pi)
    ph2 = rng.uniform(0.0, 2.0 * np.pi)

    r_max = base * (1.0 + amp1 * np.sin(k1 * ang + ph1) + amp2 * np.sin(k2 * ang + ph2))
    squeeze = rng.uniform(0.55, 0.85)

    scale = np.ones_like(r, dtype=np.float32)
    over = r > r_max
    scale[over] = (1.0 - (1.0 - (r_max[over] / (r[over] + 1e-6))) * squeeze).astype(np.float32)

    xr *= scale
    yr *= scale

    # edge noise
    edge_noise = rng.normal(scale=base * rng.uniform(0.012, 0.035), size=r.shape).astype(np.float32)
    radial = np.sqrt(xr * xr + yr * yr) + 1e-6
    xr += xr / radial * edge_noise
    yr += yr / radial * edge_noise

    # undo anisotropy/rotation
    xr *= sx
    yr *= sy
    out[:, 0] = (ct * xr + st * yr + center[0]).astype(np.float32)
    out[:, 1] = (-st * xr + ct * yr + center[1]).astype(np.float32)

    return out


def _apply_compartments(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Lobule-like compartments (soft Voronoi) + mild affine differences per compartment.
    """
    out = points.astype(np.float32, copy=True)
    n = out.shape[0]
    if n < 200:
        return out

    n_seeds = int(rng.integers(3, 7))
    seeds = out[rng.choice(n, size=n_seeds, replace=False)]

    best_d2 = np.full(n, np.inf, dtype=np.float32)
    best = np.zeros(n, dtype=np.int16)
    for i in range(n_seeds):
        dx = out[:, 0] - seeds[i, 0]
        dy = out[:, 1] - seeds[i, 1]
        d2 = dx * dx + dy * dy
        m = d2 < best_d2
        best_d2[m] = d2[m]
        best[m] = i

    scale0 = _rms_radius(out)
    core = scale0 * rng.uniform(0.22, 0.35)
    inv_core2 = np.float32(1.0 / (core * core + 1e-6))

    for i in range(n_seeds):
        m = best == i
        if not np.any(m):
            continue
        loc = out[m] - seeds[i]

        th = rng.uniform(0.0, 2.0 * np.pi)
        ct = np.cos(th).astype(np.float32)
        st = np.sin(th).astype(np.float32)
        rot = np.array([[ct, -st], [st, ct]], dtype=np.float32)

        sc = np.diag(rng.uniform(0.78, 1.22, size=2).astype(np.float32))
        sh = np.array([[1.0, rng.uniform(-0.15, 0.15)], [0.0, 1.0]], dtype=np.float32)
        mat = rot @ (sh @ sc) @ rot.T
        loc = loc @ mat.T

        r2 = (loc[:, 0] * loc[:, 0] + loc[:, 1] * loc[:, 1]).astype(np.float32)
        w = np.exp(-r2 * inv_core2).astype(np.float32)
        loc *= (1.0 - w[:, None] * rng.uniform(0.06, 0.18)).astype(np.float32)

        out[m] = seeds[i] + loc

    out += rng.normal(scale=scale0 * 0.02, size=out.shape).astype(np.float32)
    return out


# =========================
# Cluster machinery (fast + anisotropic)
# =========================

def _cluster_centers(n_clusters: int, rng: np.random.Generator, bounds: float, pattern: str) -> np.ndarray:
    """
    Center patterns for cluster scaffolds (used for embeddings).
    bounds here is a "working scale", not final units.
    """
    center = np.array([bounds * 0.5, bounds * 0.5], dtype=np.float32)
    jitter = bounds * 0.02

    if pattern == "ring":
        angles = np.linspace(0.0, 2.0 * np.pi, n_clusters, endpoint=False)
        radius = rng.uniform(bounds * 0.25, bounds * 0.42)
        centers = np.stack([np.cos(angles), np.sin(angles)], axis=1) * radius + center
    elif pattern == "spiral":
        angles = np.linspace(0.0, 4.0 * np.pi, n_clusters)
        radii = np.linspace(bounds * 0.08, bounds * 0.45, n_clusters)
        centers = np.stack([np.cos(angles), np.sin(angles)], axis=1) * radii[:, None] + center
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
        centers = rng.uniform(bounds * 0.12, bounds * 0.88, size=(n_clusters, 2)).astype(np.float32)
        centers = centers * 0.8 + center * 0.2
    else:
        # mild ribbon
        t = np.linspace(0.0, 1.0, n_clusters, dtype=np.float32)
        x = bounds * (0.15 + 0.7 * t)
        y = bounds * 0.5 + np.sin(t * np.pi * 2.0) * (bounds * 0.16)
        centers = np.stack([x, y], axis=1).astype(np.float32)

    centers = centers + rng.normal(scale=jitter, size=centers.shape).astype(np.float32)
    return centers.astype(np.float32)


def _assign_clusters_dirichlet(n: int, k: int, rng: np.random.Generator, alpha: float = 1.2) -> np.ndarray:
    """Variable cluster sizes with tunable balance via Dirichlet(alpha)."""
    w = rng.dirichlet(np.ones(k, dtype=np.float32) * np.float32(alpha))
    return rng.choice(k, size=n, p=w).astype(np.int16)


def _make_anisotropic_clusters(
    n: int,
    rng: np.random.Generator,
    pattern: str,
    k: int,
    bounds: float = 1.0,
    noise_min: float = 0.02,
    noise_max: float = 0.08,
    size_alpha: float = 1.2,
    separate_between: float = 2.2,
    separate_within: float = 0.90,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast anisotropic Gaussian-ish clusters:
    - choose centers
    - per-cluster linear transforms for ellipse + shear
    - centroid push-out for separation
    """
    centers = _cluster_centers(k, rng, bounds, pattern)
    labels = _assign_clusters_dirichlet(n, k, rng, alpha=size_alpha)

    base = rng.uniform(bounds * noise_min, bounds * noise_max, size=k).astype(np.float32)
    sx = base * rng.uniform(0.65, 1.9, size=k).astype(np.float32)
    sy = base * rng.uniform(0.65, 1.9, size=k).astype(np.float32)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=k).astype(np.float32)
    c = np.cos(theta).astype(np.float32)
    s = np.sin(theta).astype(np.float32)
    shear = rng.uniform(-0.25, 0.25, size=k).astype(np.float32)

    # A = R @ [[sx, shear*sx],[0, sy]] in expanded form
    a00 = c * sx
    a01 = c * (shear * sx) - s * sy
    a10 = s * sx
    a11 = s * (shear * sx) + c * sy

    z = rng.normal(size=(n, 2)).astype(np.float32)
    li = labels.astype(np.int32, copy=False)
    dx = z[:, 0] * a00[li] + z[:, 1] * a01[li]
    dy = z[:, 0] * a10[li] + z[:, 1] * a11[li]
    pts = centers[li] + np.stack([dx, dy], axis=1).astype(np.float32)

    pts = _separate_clusters(pts, labels, between=separate_between, within=separate_within)
    return pts.astype(np.float32), labels.astype(np.int16)


def _separate_clusters(points: np.ndarray, labels: np.ndarray, between: float, within: float) -> np.ndarray:
    """
    Push clusters apart by moving centroids away from global center, keeping residuals.
    Vectorized via bincount (fast for 500k).
    """
    pts = points.astype(np.float32, copy=False)
    lab = labels.astype(np.int32, copy=False)
    k = int(lab.max()) + 1

    g = pts.mean(axis=0, keepdims=True).astype(np.float32)

    counts = np.bincount(lab, minlength=k).astype(np.float32) + 1e-6
    cx = np.bincount(lab, weights=pts[:, 0], minlength=k).astype(np.float32) / counts
    cy = np.bincount(lab, weights=pts[:, 1], minlength=k).astype(np.float32) / counts
    cent = np.stack([cx, cy], axis=1).astype(np.float32)

    res = pts - cent[lab]
    cent_new = g + (cent - g) * np.float32(between)
    out = cent_new[lab] + res * np.float32(within)
    return out.astype(np.float32)


def _cluster_labels_from_points(points: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    """Spatial grid-by-quantiles clustering (cheap, stable across n=500k)."""
    theta = rng.uniform(0.0, 2.0 * np.pi)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
    rotated = points.astype(np.float32) @ rot.T
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

    lab = (iy * kx + ix).astype(np.int16)
    if kx * ky > n_clusters:
        lab = np.minimum(lab, n_clusters - 1).astype(np.int16)
    return lab


# =========================
# Biology-inspired spatial generators
# (all return coords in [0,1] scale, plus region codes)
# =========================

@dataclass(frozen=True)
class SpatialMode:
    name: str


def _make_glandular(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gland-like rings with lumens:
    - points concentrated in annuli around multiple gland centers
    region codes:
      0=epithelium (gland ring)
      1=stroma (background)
    """
    n_glands = int(rng.integers(3, 7))
    centers = rng.uniform(0.18, 0.82, size=(n_glands, 2)).astype(np.float32)
    weights = rng.dirichlet(np.ones(n_glands, dtype=np.float32) * np.float32(0.8))
    assign = rng.choice(n_glands, size=n, p=weights).astype(np.int16)

    pts = np.zeros((n, 2), dtype=np.float32)
    region = np.ones(n, dtype=np.int16)  # default stroma

    for g in range(n_glands):
        m = assign == g
        ng = int(m.sum())
        if ng == 0:
            continue

        inner = rng.uniform(0.030, 0.060)  # lumen radius
        outer = rng.uniform(0.090, 0.150)  # gland outer radius

        ang = rng.uniform(0.0, 2.0 * np.pi, size=ng).astype(np.float32)
        # bias toward outer edge (epithelium)
        u = rng.uniform(0.0, 1.0, size=ng).astype(np.float32)
        rad = inner + (outer - inner) * (u ** np.float32(0.55))

        x = centers[g, 0] + rad * np.cos(ang)
        y = centers[g, 1] + rad * np.sin(ang)
        pts[m] = np.stack([x, y], axis=1).astype(np.float32)
        region[m] = 0

    # add some true background stroma points
    bg = rng.random(n) < rng.uniform(0.01, 0.01)
    pts[bg] = rng.uniform(0.05, 0.95, size=(int(bg.sum()), 2)).astype(np.float32)
    region[bg] = 1

    return pts, region


def _make_vessels(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vessel-like curvilinear tracks:
    - sample along several sinusoidal centerlines with thickness
    region codes:
      2=vessel-associated
      1=stroma (background)
    """
    n_v = int(rng.integers(4, 9))
    pts = np.zeros((n, 2), dtype=np.float32)
    region = np.ones(n, dtype=np.int16)  # stroma default

    weights = rng.dirichlet(np.ones(n_v, dtype=np.float32) * np.float32(0.9))
    assign = rng.choice(n_v, size=n, p=weights).astype(np.int16)

    for i in range(n_v):
        m = assign == i
        ni = int(m.sum())
        if ni == 0:
            continue

        start = rng.uniform(0.10, 0.90, size=2).astype(np.float32)
        direction = rng.normal(size=2).astype(np.float32)
        direction /= np.float32(np.linalg.norm(direction) + 1e-6)
        perp = np.array([-direction[1], direction[0]], dtype=np.float32)

        length = np.float32(rng.uniform(0.35, 0.65))
        t = rng.uniform(0.0, 1.0, size=ni).astype(np.float32)

        # centerline with curvature
        curv_amp = np.float32(rng.uniform(0.03, 0.08))
        curv_freq = np.float32(rng.uniform(1.0, 2.4))
        centerline = start[None, :] + (t[:, None] * length) * direction[None, :]
        centerline += (np.sin((t * (2.0 * np.pi * curv_freq)).astype(np.float32))[:, None] * curv_amp) * perp[None, :]

        # thickness
        width = np.float32(rng.uniform(0.006, 0.020))
        thickness = rng.normal(scale=width, size=(ni, 1)).astype(np.float32)
        jitter_long = rng.normal(scale=width * 0.35, size=(ni, 1)).astype(np.float32)

        pts[m] = centerline + thickness * perp[None, :] + jitter_long * direction[None, :]
        region[m] = 2

    # background points
    bg = rng.random(n) < rng.uniform(0.01, 0.01)
    pts[bg] = rng.uniform(0.05, 0.95, size=(int(bg.sum()), 2)).astype(np.float32)
    region[bg] = 1

    return pts, region


def _make_tumor(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tumor-like dense core + infiltrative margin.
    region codes:
      3=tumor_core
      4=tumor_margin
    """
    center = rng.uniform(0.30, 0.70, size=2).astype(np.float32)
    n_core = int(n * rng.uniform(0.55, 0.70))
    n_margin = n - n_core

    core_r = np.float32(rng.uniform(0.14, 0.22))
    ang = rng.uniform(0.0, 2.0 * np.pi, size=n_core).astype(np.float32)
    rad = (rng.uniform(0.0, 1.0, size=n_core).astype(np.float32) ** np.float32(0.5)) * core_r
    core = np.stack([center[0] + rad * np.cos(ang), center[1] + rad * np.sin(ang)], axis=1).astype(np.float32)
    core += rng.normal(scale=0.008, size=core.shape).astype(np.float32)

    margin_in = core_r
    margin_out = core_r * np.float32(rng.uniform(1.8, 2.6))
    angm = rng.uniform(0.0, 2.0 * np.pi, size=n_margin).astype(np.float32)
    # heavier tail outward
    exp = rng.exponential(scale=float((margin_out - margin_in) * 0.35), size=n_margin).astype(np.float32)
    radm = np.clip(margin_in + exp, margin_in, margin_out)
    margin = np.stack([center[0] + radm * np.cos(angm), center[1] + radm * np.sin(angm)], axis=1).astype(np.float32)
    margin += rng.normal(scale=0.025, size=margin.shape).astype(np.float32)

    pts = np.vstack([core, margin]).astype(np.float32)
    region = np.empty(n, dtype=np.int16)
    region[:n_core] = 3
    region[n_core:] = 4
    return pts, region


def _make_immune(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Immune-like foci plus scattered cells.
    region codes:
      5=immune_foci
      1=stroma (scattered/background)
    """
    n_foci = int(rng.integers(8, 16))
    centers = rng.uniform(0.15, 0.85, size=(n_foci, 2)).astype(np.float32)
    weights = rng.dirichlet(np.ones(n_foci, dtype=np.float32) * np.float32(0.5))
    assign = rng.choice(n_foci, size=n, p=weights).astype(np.int16)

    pts = np.zeros((n, 2), dtype=np.float32)
    region = np.ones(n, dtype=np.int16)  # stroma default

    for f in range(n_foci):
        m = assign == f
        nf = int(m.sum())
        if nf == 0:
            continue
        radius = np.float32(rng.uniform(0.020, 0.060))
        ang = rng.uniform(0.0, 2.0 * np.pi, size=nf).astype(np.float32)
        rad = (rng.uniform(0.0, 1.0, size=nf).astype(np.float32) ** np.float32(0.6)) * radius
        pts[m] = np.stack([centers[f, 0] + rad * np.cos(ang), centers[f, 1] + rad * np.sin(ang)], axis=1).astype(np.float32)
        region[m] = 5

    # scattered fraction
    scatter = rng.random(n) < np.float32(0.01)
    pts[scatter] = rng.uniform(0.05, 0.95, size=(int(scatter.sum()), 2)).astype(np.float32)
    region[scatter] = 1
    return pts, region


def _make_layers(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stratified layers (epithelium-like).
    region codes:
      10..(10+n_layers-1) for each layer
    """
    n_layers = int(rng.integers(3, 6))
    y_edges = np.linspace(0.10, 0.90, n_layers + 1, dtype=np.float32)

    weights = rng.dirichlet(np.ones(n_layers, dtype=np.float32))
    sizes = rng.multinomial(n, weights).astype(int)

    pts = np.zeros((n, 2), dtype=np.float32)
    region = np.zeros(n, dtype=np.int16)

    cur = 0
    for li in range(n_layers):
        cnt = int(sizes[li])
        if cnt <= 0:
            continue
        x = rng.uniform(0.05, 0.95, size=cnt).astype(np.float32)
        y = rng.uniform(float(y_edges[li]), float(y_edges[li + 1]), size=cnt).astype(np.float32)

        wave_amp = np.float32((y_edges[li + 1] - y_edges[li]) * 0.30)
        wave_freq = np.float32(rng.uniform(8.0, 16.0))  # in unit coords
        y = y + wave_amp * np.sin(x * wave_freq).astype(np.float32)

        pts[cur:cur + cnt] = np.stack([x, y], axis=1).astype(np.float32)
        region[cur:cur + cnt] = np.int16(10 + li)
        cur += cnt

    if cur < n:
        pts[cur:] = rng.uniform(0.05, 0.95, size=(n - cur, 2)).astype(np.float32)
        region[cur:] = np.int16(1)

    return pts, region


_REGION_NAMES: Dict[int, str] = {
    0: "gland_epithelium",
    1: "stroma",
    2: "vessel",
    3: "tumor_core",
    4: "tumor_margin",
    5: "immune_foci",
    10: "layer_0",
    11: "layer_1",
    12: "layer_2",
    13: "layer_3",
    14: "layer_4",
}


def _region_codes_to_names(region_codes: np.ndarray) -> np.ndarray:
    """Vectorized mapping with a fallback for unseen layer codes."""
    rc = region_codes.astype(np.int16, copy=False)
    out = np.empty(rc.shape[0], dtype=object)
    for code in np.unique(rc):
        if int(code) in _REGION_NAMES:
            out[rc == code] = _REGION_NAMES[int(code)]
        elif int(code) >= 10:
            out[rc == code] = f"layer_{int(code) - 10}"
        else:
            out[rc == code] = "other"
    return out


# =========================
# Cell type modeling
# =========================

def _default_cell_types(n_cell_types: int) -> List[str]:
    base = [
        "Epithelial",
        "Tumor",
        "Endothelial",
        "Fibroblast",
        "Myeloid",
        "T_cell",
        "B_cell",
        "NK_cell",
        "Smooth_muscle",
        "Pericyte",
        "Stem_like",
        "Other",
    ]
    if n_cell_types <= len(base):
        return base[:n_cell_types]
    # extend deterministically
    extra = [f"Type_{i:02d}" for i in range(len(base), n_cell_types)]
    return base + extra


def _region_priors(cell_types: List[str]) -> Dict[str, np.ndarray]:
    """
    Region -> probability vector over cell types.
    These priors drive spatial realism and ensure spatial plots are meaningful.
    """
    idx = {ct: i for i, ct in enumerate(cell_types)}
    k = len(cell_types)

    def v(**kwargs: float) -> np.ndarray:
        w = np.full(k, 1e-3, dtype=np.float32)
        for name, val in kwargs.items():
            if name in idx:
                w[idx[name]] = np.float32(val)
        w /= w.sum()
        return w

    priors: Dict[str, np.ndarray] = {
        "gland_epithelium": v(Epithelial=0.65, Fibroblast=0.12, Myeloid=0.08, T_cell=0.07, Endothelial=0.05, Other=0.03),
        "stroma": v(Fibroblast=0.38, Endothelial=0.12, Myeloid=0.14, T_cell=0.12, B_cell=0.07, Smooth_muscle=0.05, Pericyte=0.05, Other=0.07),
        "vessel": v(Endothelial=0.55, Pericyte=0.14, Smooth_muscle=0.12, Myeloid=0.07, Fibroblast=0.06, Other=0.06),
        "tumor_core": v(Tumor=0.70, Myeloid=0.10, Fibroblast=0.08, T_cell=0.05, Endothelial=0.03, Stem_like=0.02, Other=0.02),
        "tumor_margin": v(Tumor=0.42, T_cell=0.16, Myeloid=0.15, Fibroblast=0.10, Endothelial=0.07, NK_cell=0.05, Other=0.05),
        "immune_foci": v(T_cell=0.35, B_cell=0.22, Myeloid=0.20, NK_cell=0.10, Endothelial=0.05, Fibroblast=0.05, Other=0.03),
    }
    return priors


def _sample_cell_types_from_regions(
    region_names: np.ndarray,
    rng: np.random.Generator,
    cell_types: List[str],
) -> np.ndarray:
    priors = _region_priors(cell_types)
    k = len(cell_types)
    out = np.zeros(region_names.shape[0], dtype=np.int16)

    unique_regions = np.unique(region_names)
    for r in unique_regions:
        m = region_names == r
        n = int(m.sum())
        if n == 0:
            continue

        if r.startswith("layer_"):
            li = int(r.split("_")[1])
            epi = float(np.clip(0.60 - 0.10 * li, 0.20, 0.70))
            strom = 1.0 - epi
            idx = {ct: i for i, ct in enumerate(cell_types)}
            w = np.full(k, 1e-3, dtype=np.float32)
            if "Epithelial" in idx:
                w[idx["Epithelial"]] = np.float32(epi)
            for name, val in [("Fibroblast", 0.45), ("Endothelial", 0.15), ("Myeloid", 0.15), ("T_cell", 0.12), ("Other", 0.13)]:
                if name in idx:
                    w[idx[name]] = np.float32(strom * val)
            w /= w.sum()
        else:
            w = priors.get(r, None)
            if w is None:
                w = np.full(k, 1.0 / k, dtype=np.float32)

        out[m] = rng.choice(k, size=n, p=w).astype(np.int16)

    return out


def _bias_cell_types_by_embedding_clusters(
    cell_type_codes: np.ndarray,
    emb_labels: np.ndarray,
    rng: np.random.Generator,
    n_cell_types: int,
    strength: float = 0.65,
    concentration: float = 0.20,
    boost: float = 2.5,
) -> Tuple[np.ndarray, np.ndarray]:
    ct = cell_type_codes.astype(np.int16, copy=True)
    lab = emb_labels.astype(np.int16, copy=False)
    k = int(lab.max()) + 1

    probs = np.zeros((k, n_cell_types), dtype=np.float32)
    base_alpha = np.ones(n_cell_types, dtype=np.float32) * np.float32(concentration)

    for c in range(k):
        a = base_alpha.copy()
        n_boost = int(rng.integers(2, min(4, n_cell_types) + 1))
        picks = rng.choice(n_cell_types, size=n_boost, replace=False)
        a[picks] *= np.float32(boost)
        probs[c] = rng.dirichlet(a).astype(np.float32)

    resample = rng.random(ct.shape[0]) < np.float32(strength)
    if np.any(resample):
        idx = np.where(resample)[0]
        cidx = lab[idx].astype(np.int32)
        u = rng.random(idx.shape[0]).astype(np.float32)
        cdf = np.cumsum(probs[cidx], axis=1)
        new_types = (u[:, None] <= cdf).argmax(axis=1).astype(np.int16)
        ct[idx] = new_types

    return ct, probs


# =========================
# Embedding construction
# =========================

def _make_embedding(
    n: int,
    rng: np.random.Generator,
    base: np.ndarray,
    pattern: str,
    n_clusters: int,
    mix_cluster: float,
    noise: float,
    separate_between: float,
    separate_within: float,
    target_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    scaffold, labels = _make_anisotropic_clusters(
        n=n,
        rng=rng,
        pattern=pattern,
        k=n_clusters,
        bounds=1.0,
        noise_min=0.02,
        noise_max=0.08,
        size_alpha=1.1,
        separate_between=separate_between,
        separate_within=separate_within,
    )
    scaffold = _scale_center(scaffold, 1.0)

    b = _scale_center(base, 1.0)
    emb = (np.float32(1.0 - mix_cluster) * b + np.float32(mix_cluster) * scaffold).astype(np.float32)
    emb += rng.normal(scale=np.float32(noise), size=emb.shape).astype(np.float32)

    emb = _separate_clusters(emb, labels, between=separate_between, within=separate_within)
    emb = _scale_center(emb, target_max)
    return emb.astype(np.float32), labels.astype(np.int16)


# =========================
# Main generator
# =========================

def make_master_mock_spatial(
    n_cells: int = 500_000,
    n_embeddings: int = 10,
    n_genes: int = 50,
    expr_density: float = 0.15,
    n_cell_types: int = 12,
    seed: int | None = None,
    make_expression: bool = True,
) -> ad.AnnData:
    if n_cells < 10_000 or n_cells > 10_000_000:
        raise ValueError("n_cells must be between 10,000 and 10,000,000.")
    if not 0.0 <= expr_density <= 1.0:
        raise ValueError("expr_density must be in [0.0, 1.0].")
    if seed is None:
        seed = int(np.random.SeedSequence().entropy)
    rng = np.random.default_rng(seed)

    bounds = 1_000.0

    # ---- obs ----
    obs = pd.DataFrame(index=[f"cell_{i:06d}" for i in range(n_cells)])

    # ---- choose samples and spatial modes ----
    n_samples = 4
    sample_sizes = rng.multinomial(n_cells, rng.dirichlet(np.ones(n_samples, dtype=np.float32) * 100)).astype(int)

    # place sample centers in a loose arc / band (feels like separate tissue sections)
    center = np.array([bounds * 0.5, bounds * 0.5], dtype=np.float32)
    orient = rng.uniform(0.0, 2.0 * np.pi)
    direction = np.array([np.cos(orient), np.sin(orient)], dtype=np.float32)
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    spacing = bounds * rng.uniform(0.16, 0.22)
    curve_amp = bounds * rng.uniform(0.08, 0.14)

    offsets = np.linspace(-(n_samples - 1) / 2.0, (n_samples - 1) / 2.0, n_samples, dtype=np.float32)
    sample_centers = []
    for t in offsets:
        arc = np.sin(float(t) * np.pi / max(1.0, n_samples - 1)).astype(np.float32)
        pos = center + direction * (t * spacing) + normal * (arc * curve_amp)
        pos += rng.normal(scale=bounds * 0.02, size=2).astype(np.float32)
        sample_centers.append(pos)
    sample_centers = np.clip(np.array(sample_centers, dtype=np.float32), bounds * 0.20, bounds * 0.80)
    rng.shuffle(sample_centers)

    modes = [
        ("glandular", _make_glandular),
        ("vessels", _make_vessels),
        ("tumor", _make_tumor),
        ("immune", _make_immune),
        ("layered", _make_layers),
    ]
    rng.shuffle(modes)

    spatial_base = np.zeros((n_cells, 2), dtype=np.float32)
    sample_labels = np.zeros(n_cells, dtype=np.int16)
    region_codes = np.zeros(n_cells, dtype=np.int16)

    used_modes: List[str] = []
    cursor = 0
    for s in range(n_samples):
        cnt = int(sample_sizes[s])
        if cnt <= 0:
            continue

        mode_name, mode_fn = modes[s % len(modes)]
        used_modes.append(mode_name)

        pts_unit, reg = mode_fn(cnt, rng)

        # scale to a realistic span and position at sample center
        span = bounds * rng.uniform(0.38, 0.52)
        pts = _scale_center(pts_unit, float(span))
        pts += sample_centers[s]

        # tissue finishing steps
        pts = _apply_tissue_boundary(pts, rng)
        if rng.random() < 0.75:
            pts = _apply_compartments(pts, rng)
        pts = _apply_dispersion(pts, rng)
        pts = _apply_tissue_boundary(pts, rng)

        # subtle global wave field (texture)
        if rng.random() < 0.55:
            amp = _rms_radius(pts) * rng.uniform(0.10, 0.22)
            freq = rng.uniform(0.008, 0.020)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            pts[:, 0] += np.float32(amp) * np.sin(pts[:, 1] * np.float32(freq) + np.float32(phase)).astype(np.float32)
            pts[:, 1] += np.float32(amp * 0.85) * np.sin(pts[:, 0] * np.float32(freq * 1.2) + np.float32(phase * 0.7)).astype(np.float32)

        pts = _reflect_into_bounds(pts, bounds)

        spatial_base[cursor:cursor + cnt] = pts
        sample_labels[cursor:cursor + cnt] = np.int16(s)
        region_codes[cursor:cursor + cnt] = reg.astype(np.int16, copy=False)
        cursor += cnt

    # any remainder (shouldn't happen with multinomial, but keep safe)
    if cursor < n_cells:
        rem = n_cells - cursor
        spatial_base[cursor:] = rng.uniform(0.0, bounds, size=(rem, 2)).astype(np.float32)
        sample_labels[cursor:] = rng.integers(0, n_samples, size=rem).astype(np.int16)
        region_codes[cursor:] = np.int16(1)

    # separate samples slightly so they overlap less (then re-bound)
    spatial_base = _separate_clusters(spatial_base, sample_labels, between=1.22, within=1.00)
    spatial_base = _reflect_into_bounds(spatial_base, bounds)

    # ---- derive cluster labels from spatial organization ----
    cluster_counts = [5, 8, 11, 15, 20]
    obs["sample"] = pd.Categorical(sample_labels)
    for c in cluster_counts:
        obs[f"cluster_{c:02d}"] = pd.Categorical(_cluster_labels_from_points(spatial_base, c, rng))

    # region names
    region_names = _region_codes_to_names(region_codes)
    obs["region"] = pd.Categorical(region_names)

    # continuous field for coloring
    score = ((spatial_base[:, 0] - spatial_base[:, 0].mean()) / (spatial_base[:, 0].std() + 1e-6)).astype(np.float32)
    obs["score"] = score

    # final spatial coords with subtle warp
    spatial = spatial_base.copy()
    wave_amp = bounds * rng.uniform(0.03, 0.07)
    wave_freq = rng.uniform(0.003, 0.010)
    wave_phase = rng.uniform(0.0, 2.0 * np.pi)
    spatial = _wave(spatial, wave_amp, wave_freq, wave_phase)
    spatial[:, 0] += np.float32(wave_amp * 0.55) * np.cos(
        spatial[:, 1] * np.float32(wave_freq * 0.7) + np.float32(wave_phase * 0.6)
    ).astype(np.float32)
    spatial = _reflect_into_bounds(spatial, bounds)

    obs["x"] = spatial[:, 0].astype(np.float32)
    obs["y"] = spatial[:, 1].astype(np.float32)
    spatial_norm = (spatial / np.float32(bounds)).astype(np.float32)

    # ---- generator-side rendering fields (NEW) ----
    # Tune bin_size to your renderer's point footprint in spatial units.
    render_bin_size = 2.0
    render_gamma = 0.7
    render_alpha_min = 0.02

    alpha = density_compensated_alpha(
        xy=spatial,
        sample_labels=sample_labels,
        bin_size=render_bin_size,
        gamma=render_gamma,
        alpha_min=render_alpha_min,
        pctl_norm=99.0,
    )
    obs["render_alpha"] = alpha.astype(np.float32, copy=False)
    obs["render_size"] = density_compensated_size(alpha, size_min=0.25)

    # LOD mask for fast/clean rendering while preserving structures
    obs["render_lod_100k"] = lod_mask_stratified(
        xy=spatial,
        target_n=min(100_000, n_cells),
        sample_labels=sample_labels,
        rng=rng,
        bin_size=render_bin_size,
    )

    # ---- latent for global embedding structure ----
    latent = _scale_center(spatial_base, 1.0)
    th = rng.uniform(0.0, 2.0 * np.pi)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=np.float32)
    latent = (latent @ rot.T).astype(np.float32)

    # Base shapes
    umap_base = latent.copy()
    umap_base[:, 0] += np.float32(0.35) * np.sin(umap_base[:, 1] * np.float32(4.0)).astype(np.float32)
    umap_base[:, 1] += np.float32(0.25) * np.sin(umap_base[:, 0] * np.float32(3.2)).astype(np.float32)

    tsne_base = latent.copy()
    tsne_base[:, 0] = (tsne_base[:, 0] * np.float32(1.15) + np.float32(0.35) * np.sin(tsne_base[:, 1] * np.float32(5.2))).astype(np.float32)
    tsne_base[:, 1] = (tsne_base[:, 1] * np.float32(1.05) + np.float32(0.28) * np.sin(tsne_base[:, 0] * np.float32(4.4))).astype(np.float32)

    pca_base = latent.copy()
    pca_base[:, 0] = (pca_base[:, 0] * rng.uniform(2.0, 2.8)).astype(np.float32)
    pca_base[:, 1] = (pca_base[:, 1] * rng.uniform(0.55, 0.95)).astype(np.float32)
    pca_base[:, 0] = (pca_base[:, 0] + pca_base[:, 1] * rng.uniform(0.25, 0.65)).astype(np.float32)

    # Patterns for cluster scaffolds (chosen to vary)
    pattern_pool = ["spiral", "ring", "wave_grid", "constellation", "ribbon"]
    rng.shuffle(pattern_pool)
    umap_pattern, tsne_pattern, pca_pattern = pattern_pool[:3]

    # ---- embeddings ----
    n_umap_clusters = int(rng.integers(12, 20))
    n_tsne_clusters = int(rng.integers(18, 30))
    n_pca_clusters = int(rng.integers(7, 12))

    umap, umap_lab = _make_embedding(
        n=n_cells,
        rng=rng,
        base=umap_base,
        pattern=umap_pattern,
        n_clusters=n_umap_clusters,
        mix_cluster=0.3,
        noise=0.030,
        separate_between=2.35,
        separate_within=0.88,
        target_max=10.0,
    )
    tsne, tsne_lab = _make_embedding(
        n=n_cells,
        rng=rng,
        base=tsne_base,
        pattern=tsne_pattern,
        n_clusters=n_tsne_clusters,
        mix_cluster=0.84,
        noise=0.050,
        separate_between=2.65,
        separate_within=0.84,
        target_max=25.0,
    )
    pca, pca_lab = _make_embedding(
        n=n_cells,
        rng=rng,
        base=pca_base,
        pattern=pca_pattern,
        n_clusters=n_pca_clusters,
        mix_cluster=0.52,
        noise=0.040,
        separate_between=1.85,
        separate_within=0.93,
        target_max=6.0,
    )

    # ---- cell types (region-informed, then embedding-biased) ----
    cell_types = _default_cell_types(n_cell_types)
    ct_codes = _sample_cell_types_from_regions(region_names, rng, cell_types)

    ct_codes, umap_type_probs = _bias_cell_types_by_embedding_clusters(
        cell_type_codes=ct_codes,
        emb_labels=umap_lab,
        rng=rng,
        n_cell_types=n_cell_types,
        strength=0.65,
        concentration=0.18,
        boost=2.8,
    )
    obs["cell_type"] = pd.Categorical([cell_types[i] for i in ct_codes])

    # Optional embedding-cluster categoricals for debugging/plotting
    obs["umap_cluster"] = pd.Categorical(umap_lab)
    obs["tsne_cluster"] = pd.Categorical(tsne_lab)
    obs["pca_cluster"] = pd.Categorical(pca_lab)

    # ---- build embedding_n features in obs ----
    radial = np.sqrt((spatial_norm[:, 0] - 0.5) ** 2 + (spatial_norm[:, 1] - 0.5) ** 2).astype(np.float32)
    radial = (radial / (radial.max() + 1e-6)).astype(np.float32)
    score_norm = ((score - score.min()) / (score.max() - score.min() + 1e-6)).astype(np.float32)

    features = [
        umap[:, 0], umap[:, 1],
        tsne[:, 0], tsne[:, 1],
        pca[:, 0], pca[:, 1],
        spatial_norm[:, 0], spatial_norm[:, 1],
        radial, score_norm,
    ]

    emb = np.zeros((n_cells, n_embeddings), dtype=np.float32)
    base_cols = min(n_embeddings, len(features))
    for j in range(base_cols):
        emb[:, j] = features[j].astype(np.float32, copy=False)

    if n_embeddings > len(features):
        extra = rng.normal(scale=0.20, size=(n_cells, n_embeddings - len(features))).astype(np.float32)
        basis = np.column_stack(features).astype(np.float32)
        mix = rng.normal(scale=0.10, size=(basis.shape[1], extra.shape[1])).astype(np.float32)
        emb[:, len(features):] = (basis @ mix + extra).astype(np.float32)

    for j in range(n_embeddings):
        obs[f"embedding_{j + 1}"] = emb[:, j].astype(np.float32, copy=False)

    # ---- expression matrix X (sparse counts) ----
    if make_expression:
        spatial_clusters = int(rng.integers(8, 14))
        spatial_assign = _cluster_labels_from_points(spatial_base, spatial_clusters, rng).astype(np.int16)
        n_sc = int(spatial_assign.max()) + 1

        ct_primary = rng.integers(0, n_cell_types, size=n_genes).astype(np.int16)
        ct_secondary = rng.integers(0, n_cell_types, size=n_genes).astype(np.int16)

        sc_pref_a = rng.integers(0, n_sc, size=n_genes).astype(np.int16)
        sc_pref_b = rng.integers(0, n_sc, size=n_genes).astype(np.int16)

        gene_max = rng.integers(4, 10, size=n_genes).astype(np.int16)

        wf = rng.uniform(3.0, 7.0)
        ph = rng.uniform(0.0, 2.0 * np.pi)
        wave = (0.5 + 0.5 * np.sin(spatial_norm[:, 0] * wf * 2.0 * np.pi + ph) * np.cos(
            spatial_norm[:, 1] * wf * 2.0 * np.pi + ph * 0.7
        )).astype(np.float32)

        sc = spatial_assign.astype(np.int32, copy=False)
        sc_w = np.full((n_genes, n_sc), 0.18, dtype=np.float32)
        sc_w[np.arange(n_genes), sc_pref_a] = rng.uniform(0.55, 0.90, size=n_genes).astype(np.float32)
        sc_w[np.arange(n_genes), sc_pref_b] = np.maximum(
            sc_w[np.arange(n_genes), sc_pref_b],
            rng.uniform(0.35, 0.70, size=n_genes).astype(np.float32),
        )

        rows_list: List[np.ndarray] = []
        cols_list: List[np.ndarray] = []
        data_list: List[np.ndarray] = []

        for g in range(n_genes):
            primary = (ct_codes == ct_primary[g]).astype(np.float32)
            secondary = (ct_codes == ct_secondary[g]).astype(np.float32)
            ct_strength = (0.75 * primary + 0.35 * secondary).astype(np.float32)

            sc_strength = sc_w[g, sc].astype(np.float32)

            field = (0.40 * spatial_norm[:, 0] + 0.25 * (1.0 - radial) + 0.20 * wave + 0.15 * score_norm).astype(np.float32)
            field = np.clip(field, 0.0, 1.0)

            strength = np.clip(0.55 * ct_strength + 0.25 * sc_strength + 0.20 * field, 0.0, 1.0).astype(np.float32)
            lam = strength * np.float32(gene_max[g])

            m = rng.random(n_cells) < np.float32(expr_density)
            if not np.any(m):
                continue

            vals = np.round(lam[m]).astype(np.uint8)
            nz = vals > 0
            if not np.any(nz):
                continue

            rr = np.where(m)[0][nz].astype(np.int32)
            cc = np.full(rr.shape[0], g, dtype=np.int32)
            dd = vals[nz].astype(np.uint8)

            rows_list.append(rr)
            cols_list.append(cc)
            data_list.append(dd)

        if len(rows_list) == 0:
            X = sp.csr_matrix((n_cells, n_genes), dtype=np.float32)
        else:
            rows = np.concatenate(rows_list)
            cols = np.concatenate(cols_list)
            data = np.concatenate(data_list)
            X = sp.coo_matrix((data, (rows, cols)), shape=(n_cells, n_genes)).tocsr()
    else:
        X = sp.csr_matrix((n_cells, n_genes), dtype=np.float32)

    var = pd.DataFrame(index=[f"gene_{g:05d}" for g in range(n_genes)])

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = spatial.astype(np.float32)
    adata.obsm["X_umap"] = umap.astype(np.float32)
    adata.obsm["X_tsne"] = tsne.astype(np.float32)
    adata.obsm["X_pca"] = pca.astype(np.float32)

    adata.uns["mock_params"] = dict(
        seed=seed,
        n_cells=n_cells,
        n_genes=n_genes,
        n_embeddings=n_embeddings,
        expr_density=expr_density,
        make_expression=make_expression,
        n_samples=n_samples,
        used_spatial_modes=used_modes,
        umap_pattern=umap_pattern,
        tsne_pattern=tsne_pattern,
        pca_pattern=pca_pattern,
        n_umap_clusters=n_umap_clusters,
        n_tsne_clusters=n_tsne_clusters,
        n_pca_clusters=n_pca_clusters,
        n_cell_types=n_cell_types,
        cell_types=cell_types,
        umap_type_probs=umap_type_probs,
        render_bin_size=render_bin_size,
        render_gamma=render_gamma,
        render_alpha_min=render_alpha_min,
        render_lod_target=min(100_000, n_cells),
    )
    return adata


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("master_mock_spatial.h5ad"))
    ap.add_argument("--cells", type=int, default=500_000)
    ap.add_argument("--embeddings", type=int, default=10)
    ap.add_argument("--genes", type=int, default=50)
    ap.add_argument(
        "--density",
        type=float,
        default=0.15,
        help="Expression sparsity probability in [0, 1] (default: 0.15).",
    )
    ap.add_argument("--cell-types", type=int, default=12)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-expr", action="store_true", help="Skip expression matrix.")
    args = ap.parse_args()
    if not 0.0 <= args.density <= 1.0:
        ap.error("--density must be in [0, 1].")

    adata = make_master_mock_spatial(
        n_cells=args.cells,
        n_embeddings=args.embeddings,
        n_genes=args.genes,
        expr_density=args.density,
        n_cell_types=args.cell_types,
        seed=args.seed,
        make_expression=not args.no_expr,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.out)
    print(f"Wrote: {args.out}")
    print(adata)


if __name__ == "__main__":
    main()
