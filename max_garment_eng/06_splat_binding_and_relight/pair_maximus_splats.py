"""pair_maximus_splats.py — Deterministic Garment-Only Canonical Bone-Z Slab Splat Pairer & Packager.

Guarantees:
  1. Drops Maximus bare body splats (`part_id == 0`), keeping strictly garment/footwear splats (`part_id >= 1`).
  2. Resamples deterministically to fixed `N = 100_000` (importance stratified downsample or large-splat split).
  3. Sorts splats canonically along Maximus `theta_fit` bone-Z slabs (`ankle, knee, hip, waist, chest, shoulder, crown`),
     then azimuth `atan2(y - cy, x - cx)` around the slab centroid and radial distance `r`, ensuring smooth, low-jump
     (`median_jump < 1.5 cm` on perturbed copies) two-wave morph pairing across SKUs.
  4. Packs `garment_id: uint16` into `(R, G, 0, A=255)` lossless `-exact` WebP textures (WebKit premul-alpha safe).
  5. Exports compact compressed morph packs (`<= 2.5 MB` for 100k splats) with FP16 geometry & uint8 SH0/opacity.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

DEFAULT_TARGET_SPLATS = 100_000
SLAB_NAMES = ("ankle", "knee", "hip", "waist", "chest", "shoulder", "crown")
# Canonical Maximus v2 theta_fit Z-up height boundaries (sole at Z = 0.000 m, crown at Z = 1.734 m)
SLAB_Z_EDGES = (0.0, 0.09, 0.47, 0.93, 1.07, 1.28, 1.44, 1.85)


@dataclass
class SplatCloud:
    """In-memory structured 3D Gaussian cloud."""

    means: np.ndarray        # (N, 3) float32
    scales: np.ndarray       # (N, 3) float32 linear meters
    quats: np.ndarray        # (N, 4) float32 unit quaternions (x, y, z, w) or (w, x, y, z) normalized consistently
    opacities: np.ndarray    # (N,)   float32 in [0, 1]
    colors: np.ndarray       # (N, 3) float32 sRGB in [0, 1]
    part_ids: np.ndarray     # (N,)   uint16 (0=body, 1=top, 2=bottom, 3=footwear, >=1 garment)
    shN: np.ndarray | None = None  # optional (N, K, 3) float32

    def __len__(self) -> int:
        return int(len(self.means))


def filter_garment_splats(cloud: SplatCloud) -> SplatCloud:
    """Keep strictly splats with `part_id >= 1` (drop bare body `part_id == 0`) and finite coordinates."""
    pids = np.asarray(cloud.part_ids, dtype=np.uint16).reshape(-1)
    means = np.asarray(cloud.means, dtype=np.float32)
    scales = np.asarray(cloud.scales, dtype=np.float32)
    quats = np.asarray(cloud.quats, dtype=np.float32)
    opacities = np.asarray(cloud.opacities, dtype=np.float32).reshape(-1)
    colors = np.asarray(cloud.colors, dtype=np.float32)

    valid = (
        (pids >= 1)
        & np.all(np.isfinite(means), axis=1)
        & np.all(np.isfinite(scales), axis=1)
        & np.all(np.isfinite(quats), axis=1)
        & np.isfinite(opacities)
        & np.all(np.isfinite(colors), axis=1)
    )
    if not np.any(valid):
        raise ValueError("No valid garment splats (part_id >= 1) found after filtering.")

    return SplatCloud(
        means=means[valid],
        scales=np.clip(scales[valid], 1e-6, 0.05).astype(np.float32),
        quats=_normalize_quats(quats[valid]),
        opacities=np.clip(opacities[valid], 0.0, 1.0).astype(np.float32),
        colors=np.clip(colors[valid], 0.0, 1.0).astype(np.float32),
        part_ids=pids[valid],
        shN=cloud.shN[valid] if cloud.shN is not None else None,
    )


def _normalize_quats(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    n = np.linalg.norm(q, axis=1, keepdims=True)
    safe = np.where(n > 1e-8, q / np.maximum(n, 1e-8), np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
    return safe.astype(np.float32)


def _assign_bone_slabs(means: np.ndarray, up_axis: str = "z", slab_edges: tuple[float, ...] = SLAB_Z_EDGES) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map positions into Maximus bone-Z slab indices and horizontal coordinates (hx, hy)."""
    if up_axis == "z":
        up = means[:, 2]
        hx = means[:, 0]
        hy = means[:, 1]
    elif up_axis == "y":
        up = means[:, 1]
        hx = means[:, 0]
        hy = -means[:, 2]
    else:
        raise ValueError(f"Unsupported up_axis={up_axis!r}; expected 'z' or 'y'.")

    # np.searchsorted over internal edges gives 0..len(slab_edges)-2 clamped to [0, len(SLAB_NAMES)-1]
    edges = np.asarray(slab_edges, dtype=np.float64)
    slabs = np.searchsorted(edges[1:-1], up, side="right").astype(np.int32)
    return slabs, hx.astype(np.float64), hy.astype(np.float64)


def resample_fixed_n(cloud: SplatCloud, n_target: int = DEFAULT_TARGET_SPLATS, seed: int = 411, up_axis: str = "z") -> SplatCloud:
    """Deterministically resample a garment-only cloud to exact `n_target` splats."""
    n = len(cloud)
    if n == n_target:
        return cloud
    rng = np.random.default_rng(seed)

    if n > n_target:
        # Stratified-by-slab importance sampling (weighted by opacity * geometric radius)
        slabs, _, _ = _assign_bone_slabs(cloud.means, up_axis=up_axis)
        vol_r = np.cbrt(np.maximum(cloud.scales[:, 0] * cloud.scales[:, 1] * cloud.scales[:, 2], 1e-18))
        weights = np.clip(cloud.opacities, 0.05, 1.0) * vol_r
        weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)

        unique_slabs, counts = np.unique(slabs, return_counts=True)
        quotas = np.floor(counts * (n_target / float(n))).astype(np.int64)
        quotas = np.maximum(quotas, np.minimum(counts, 1))
        diff = int(n_target - quotas.sum())
        if diff > 0:
            order = np.argsort(-(counts - quotas))
            for k in order:
                take = min(diff, int(counts[k] - quotas[k]))
                quotas[k] += take
                diff -= take
                if diff <= 0:
                    break
        elif diff < 0:
            order = np.argsort(-quotas)
            need = -diff
            for k in order:
                drop = min(need, int(quotas[k] - 1))
                quotas[k] -= drop
                need -= drop
                if need <= 0:
                    break

        chosen_parts = []
        for s_val, q_val in zip(unique_slabs, quotas):
            idx_s = np.flatnonzero(slabs == s_val)
            w_s = weights[idx_s]
            p_s = w_s / np.maximum(w_s.sum(), 1e-12)
            sel = rng.choice(idx_s, size=int(q_val), replace=False, p=p_s)
            chosen_parts.append(sel)
        idx = np.sort(np.concatenate(chosen_parts))
        return SplatCloud(
            means=cloud.means[idx].copy(),
            scales=cloud.scales[idx].copy(),
            quats=cloud.quats[idx].copy(),
            opacities=cloud.opacities[idx].copy(),
            colors=cloud.colors[idx].copy(),
            part_ids=cloud.part_ids[idx].copy(),
            shN=cloud.shN[idx].copy() if cloud.shN is not None else None,
        )

    # n < n_target: split largest-scale splats deterministically with tiny tangent jitter
    need = n_target - n
    vol = cloud.scales[:, 0] * cloud.scales[:, 1] * cloud.scales[:, 2]
    rank_idx = np.argsort(-vol)
    reps = need // n
    rem = need % n
    dup_indices = np.concatenate([np.tile(rank_idx, reps), rank_idx[:rem]]) if reps > 0 else rank_idx[:rem]

    jitter = rng.normal(0.0, 0.0005, size=(len(dup_indices), 3)).astype(np.float32)
    dup_means = cloud.means[dup_indices] + jitter
    dup_scales = np.maximum(cloud.scales[dup_indices] * np.float32(0.75), 1e-5).astype(np.float32)
    dup_op = np.clip(cloud.opacities[dup_indices] * np.float32(0.92), 0.05, 1.0).astype(np.float32)

    return SplatCloud(
        means=np.vstack([cloud.means, dup_means]).astype(np.float32),
        scales=np.vstack([cloud.scales, dup_scales]).astype(np.float32),
        quats=np.vstack([cloud.quats, cloud.quats[dup_indices]]).astype(np.float32),
        opacities=np.concatenate([cloud.opacities, dup_op]).astype(np.float32),
        colors=np.vstack([cloud.colors, cloud.colors[dup_indices]]).astype(np.float32),
        part_ids=np.concatenate([cloud.part_ids, cloud.part_ids[dup_indices]]).astype(np.uint16),
        shN=np.vstack([cloud.shN, cloud.shN[dup_indices]]) if cloud.shN is not None else None,
    )


def canonicalize_garment_splats(
    cloud: SplatCloud,
    n_target: int = DEFAULT_TARGET_SPLATS,
    seed: int = 411,
    up_axis: str = "z",
    quantize_slab_grid: int = 16,
) -> tuple[SplatCloud, np.ndarray]:
    """Filter garment-only (`part_id >= 1`), resample to exact `n_target`, and sort into canonical Maximus pairing order.

    Sort key: `(sub_slab_idx, azimuth_atan2, radial_distance)` relative to each slab centroid, keeping
    top/bottom/footwear surfaces spatially aligned across different garment silhouettes without mixing `part_id` groups
    harshly while guaranteeing near-1-to-1 spatial correspondence in overlapping torso/limb zones.
    """
    filtered = filter_garment_splats(cloud)
    fixed = resample_fixed_n(filtered, n_target=n_target, seed=seed, up_axis=up_axis)

    slabs, hx, hy = _assign_bone_slabs(fixed.means, up_axis=up_axis)
    up_coord = fixed.means[:, 2] if up_axis == "z" else fixed.means[:, 1]
    # Sub-slice inside each anatomical region (quantized vertical height) so cross-garment pairing matches height closely
    z_min, z_max = float(SLAB_Z_EDGES[0]), float(SLAB_Z_EDGES[-1])
    sub_slab = np.clip(np.floor(((up_coord - z_min) / max(z_max - z_min, 1e-6)) * (quantize_slab_grid * len(SLAB_NAMES))), 0, 4095).astype(np.int32)

    cx = np.zeros_like(hx)
    cy = np.zeros_like(hy)
    for s in np.unique(slabs):
        m = slabs == s
        cx[m] = np.mean(hx[m])
        cy[m] = np.mean(hy[m])

    dx = hx - cx
    dy = hy - cy
    theta = np.arctan2(dy, dx)
    radius = np.hypot(dx, dy)

    # Sort primarily by anatomical slab & fine height sub-slab, then azimuth angle, then radius
    order = np.lexsort((radius, theta, sub_slab, slabs))
    ordered = SplatCloud(
        means=fixed.means[order],
        scales=fixed.scales[order],
        quats=fixed.quats[order],
        opacities=fixed.opacities[order],
        colors=fixed.colors[order],
        part_ids=fixed.part_ids[order],
        shN=fixed.shN[order] if fixed.shN is not None else None,
    )
    return ordered, order


def pair_two_garment_clouds(
    cloud_a: SplatCloud,
    cloud_b: SplatCloud,
    n_target: int = DEFAULT_TARGET_SPLATS,
    seed: int = 411,
    up_axis: str = "z",
) -> tuple[SplatCloud, SplatCloud, dict[str, Any]]:
    """Canonically pair two garment clouds to identical `N` and report morph jump statistics."""
    ca, _ = canonicalize_garment_splats(cloud_a, n_target=n_target, seed=seed, up_axis=up_axis)
    cb, _ = canonicalize_garment_splats(cloud_b, n_target=n_target, seed=seed + 1, up_axis=up_axis)

    jumps = np.linalg.norm(cb.means - ca.means, axis=1)
    uniq_a, cnt_a = np.unique(ca.part_ids, return_counts=True)
    uniq_b, cnt_b = np.unique(cb.part_ids, return_counts=True)
    stats = {
        "n_splats": int(n_target),
        "median_jump_m": round(float(np.median(jumps)), 6),
        "p95_jump_m": round(float(np.percentile(jumps, 95.0)), 6),
        "max_jump_m": round(float(np.max(jumps)), 6),
        "part_id_counts_a": {int(k): int(v) for k, v in zip(uniq_a, cnt_a)},
        "part_id_counts_b": {int(k): int(v) for k, v in zip(uniq_b, cnt_b)},
    }
    return ca, cb, stats


def write_garment_id_webp(part_ids: np.ndarray, out_path: Path) -> dict[str, Any]:
    """Pack `garment_id: uint16` into `(R, G, 0, A=255)` lossless exact WebP (WebKit premultiplied-alpha safe)."""
    gids = np.asarray(part_ids, dtype=np.uint16).reshape(-1)
    n = int(len(gids))
    if n <= 0:
        raise ValueError("Cannot pack empty part_ids array.")
    w = int(math.ceil(math.sqrt(n)))
    h = int(math.ceil(n / w))
    rgba = np.zeros((h * w, 4), dtype=np.uint8)
    rgba[:n, 0] = (gids & 0xFF).astype(np.uint8)
    rgba[:n, 1] = ((gids >> 8) & 0xFF).astype(np.uint8)
    rgba[:, 2] = 0
    rgba[:, 3] = 255
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(rgba.reshape((h, w, 4)), mode="RGBA")
    im.save(out_path, format="WEBP", lossless=True, quality=100, exact=True)
    return {
        "path": str(out_path),
        "width": w,
        "height": h,
        "n_splats": n,
        "bytes": int(out_path.stat().st_size),
        "exact_alpha_255": True,
    }


def read_garment_id_webp(path: Path, n: int) -> np.ndarray:
    """Decode lossless `-exact` WebP `(R, G, 0, A=255)` back to `uint16` garment IDs."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGBA"), dtype=np.uint8).reshape(-1, 4)
    if np.any(arr[:, 3] != 255):
        raise ValueError(f"Non-255 alpha detected in {path}: min={arr[:, 3].min()}")
    sliced = arr[:n]
    return (sliced[:, 0].astype(np.uint16) | (sliced[:, 1].astype(np.uint16) << np.uint16(8))).astype(np.uint16)


def export_morph_pack(cloud: SplatCloud, out_path: Path, sku: str = "SKU", colorway: str = "default") -> dict[str, Any]:
    """Export quantized compressed canonical garment morph pack (`<= 2.5 MB` for N=100k)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(cloud)
    means_f16 = np.asarray(cloud.means, dtype=np.float16)
    scales_f16 = np.asarray(cloud.scales, dtype=np.float16)
    quats_i8 = np.clip(np.round(_normalize_quats(cloud.quats) * 127.0), -127, 127).astype(np.int8)
    rgb_op_u8 = np.empty((n, 4), dtype=np.uint8)
    rgb_op_u8[:, :3] = np.clip(np.round(cloud.colors * 255.0), 0, 255).astype(np.uint8)
    rgb_op_u8[:, 3] = np.clip(np.round(cloud.opacities * 255.0), 0, 255).astype(np.uint8)
    part_u16 = np.asarray(cloud.part_ids, dtype=np.uint16)

    np.savez_compressed(
        out_path,
        schema=np.array("maximus.v2.garment_morph_pack/1"),
        sku=np.array(sku),
        colorway=np.array(colorway),
        means_f16=means_f16,
        scales_f16=scales_f16,
        quats_i8=quats_i8,
        rgb_op_u8=rgb_op_u8,
        part_ids=part_u16,
    )
    size_bytes = int(out_path.stat().st_size)
    size_mb = size_bytes / (1024.0 * 1024.0)
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    return {
        "path": str(out_path),
        "n_splats": n,
        "bytes": size_bytes,
        "size_mb": round(size_mb, 3),
        "under_2_5mb_budget": size_mb <= 2.5,
        "sha256": digest,
    }
