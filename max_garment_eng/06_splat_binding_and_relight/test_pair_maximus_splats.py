"""Unit tests for pair_maximus_splats.py (Workstream 1 Step 1.1)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from pair_maximus_splats import (
    DEFAULT_TARGET_SPLATS,
    SplatCloud,
    canonicalize_garment_splats,
    export_morph_pack,
    pair_two_garment_clouds,
    read_garment_id_webp,
    write_garment_id_webp,
)


def _make_synthetic_cylindrical_garment(
    n_total: int = 120_000,
    z_min: float = 0.85,
    z_max: float = 1.48,
    radius: float = 0.16,
    body_fraction: float = 0.25,
    seed: int = 11,
    include_bottoms: bool = False,
) -> SplatCloud:
    rng = np.random.default_rng(seed)
    n_body = int(n_total * body_fraction)
    n_cloth = n_total - n_body

    # Deterministic structured angles/heights covering cylinder evenly
    theta_g = rng.uniform(-np.pi, np.pi, size=n_cloth)
    z_g = rng.uniform(z_min, z_max, size=n_cloth)
    r_g = radius + rng.normal(0.0, 0.002, size=n_cloth)
    means_g = np.column_stack([r_g * np.cos(theta_g), r_g * np.sin(theta_g), z_g])

    means_b = rng.uniform([-0.15, -0.10, 0.0], [0.15, 0.10, 1.72], size=(n_body, 3))
    means = np.vstack([means_b, means_g]).astype(np.float32)

    scales = rng.uniform(0.002, 0.006, size=(n_total, 3)).astype(np.float32)
    quats = np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (n_total, 1))
    opacities = rng.uniform(0.55, 0.98, size=n_total).astype(np.float32)
    colors = rng.uniform(0.1, 0.95, size=(n_total, 3)).astype(np.float32)

    part_ids = np.zeros(n_total, dtype=np.uint16)
    if include_bottoms:
        bottom_mask = means[n_body:, 2] < 1.00
        g_ids = np.where(bottom_mask, 2, 1).astype(np.uint16)
        part_ids[n_body:] = g_ids
    else:
        part_ids[n_body:] = 1

    return SplatCloud(means=means, scales=scales, quats=quats, opacities=opacities, colors=colors, part_ids=part_ids)


def test_garment_only_filter_and_fixed_100k() -> None:
    cloud = _make_synthetic_cylindrical_garment(n_total=150_000, body_fraction=0.25, seed=7)
    out, _ = canonicalize_garment_splats(cloud, n_target=DEFAULT_TARGET_SPLATS, seed=411)
    assert len(out) == 100_000
    assert int(out.part_ids.min()) >= 1
    assert np.all(out.part_ids != 0)
    assert np.all(np.isfinite(out.means))

    # Also test upsampling when fewer than 100k garment splats exist
    small = _make_synthetic_cylindrical_garment(n_total=60_000, body_fraction=0.10, seed=9)
    out_up, _ = canonicalize_garment_splats(small, n_target=DEFAULT_TARGET_SPLATS, seed=411)
    assert len(out_up) == 100_000
    assert int(out_up.part_ids.min()) >= 1


def test_perturbed_copy_median_jump_under_1_5cm() -> None:
    rng = np.random.default_rng(123)
    base = _make_synthetic_cylindrical_garment(n_total=100_000, body_fraction=0.0, seed=55)
    # Slightly perturbed copy (+2mm drape expansion + +-2.5mm surface jitter) matching identical N
    noise = rng.normal(0.0, 0.0025, size=base.means.shape).astype(np.float32)
    radial = base.means.copy()
    radial[:, 2] = 0.0
    radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-6)
    perturbed = SplatCloud(
        means=(base.means + 0.002 * radial + noise).astype(np.float32),
        scales=base.scales.copy(),
        quats=base.quats.copy(),
        opacities=base.opacities.copy(),
        colors=base.colors.copy(),
        part_ids=base.part_ids.copy(),
    )
    ca, cb, stats = pair_two_garment_clouds(base, perturbed, n_target=100_000)
    assert len(ca) == 100_000 and len(cb) == 100_000
    assert stats["median_jump_m"] < 0.015, f"Expected median jump < 1.5 cm, got {stats['median_jump_m']} m"


def test_cross_geometry_pairing_preserves_parts() -> None:
    tee = _make_synthetic_cylindrical_garment(n_total=110_000, z_min=0.86, z_max=1.48, body_fraction=0.15, seed=21, include_bottoms=False)
    ensemble = _make_synthetic_cylindrical_garment(n_total=130_000, z_min=0.10, z_max=1.50, body_fraction=0.15, seed=22, include_bottoms=True)
    ca, cb, stats = pair_two_garment_clouds(tee, ensemble, n_target=100_000)
    assert len(ca) == 100_000 and len(cb) == 100_000
    assert set(stats["part_id_counts_a"].keys()) == {1}
    assert set(stats["part_id_counts_b"].keys()) == {1, 2}
    assert np.all(np.isfinite(cb.means))


def test_lossless_exact_webp_garment_id_roundtrip(tmp_path: Path) -> None:
    vals = np.array([1, 2, 3, 255, 256, 511, 1024, 4097, 65535] * 200, dtype=np.uint16)
    webp_path = tmp_path / "gid_exact.webp"
    info = write_garment_id_webp(vals, webp_path)
    assert info["exact_alpha_255"] is True
    decoded = read_garment_id_webp(webp_path, len(vals))
    assert np.array_equal(decoded, vals)


def test_morph_pack_under_2_5mb(tmp_path: Path) -> None:
    cloud = _make_synthetic_cylindrical_garment(n_total=100_000, body_fraction=0.0, seed=99)
    ca, _ = canonicalize_garment_splats(cloud, n_target=100_000)
    pack_path = tmp_path / "test_sku.morph.npz"
    info = export_morph_pack(ca, pack_path, sku="BS-RINGER-001", colorway="white-black")
    assert info["n_splats"] == 100_000
    assert info["size_mb"] <= 2.5, f"Morph pack too large: {info['size_mb']} MB"
    assert info["under_2_5mb_budget"] is True
