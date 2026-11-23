"""06_splat_binding_and_relight/gsplat_known_camera_runner.py — Track B Known-Camera & Track A SAM2 Runner.

Licence: the upstream dependency `nerfstudio-project/gsplat` is Apache-2.0; this wrapper script is
covered by this repository's LICENSE.

Target trainers: nerfstudio-project/gsplat (main CUDA 13 build; MCMCStrategy cap_max=600_000 primary,
DefaultStrategy(absgrad=True) fallback) and ArthurBrussee/brush (Rust/WebGPU fallback).
Zero COLMAP SfM for Track B: initializes Gaussians directly from explicit Maximus surface points
with (face_idx, u, v, w, garment_id) and reads Blender Cycles known-camera OpenCV transforms.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


@dataclass
class GsplatTrainSpec:
    data_dir: Path
    out_dir: Path
    init_points_ply: Path | None = None
    strategy: Literal["mcmc", "default"] = "mcmc"
    cap_max: int = 600_000
    max_steps: int = 30_000
    sh_degree: int = 3
    antialiased: bool = True
    absgrad: bool = True
    scale_reg: float = 0.01
    opacity_reg: float = 0.005
    aniso_reg: float = 0.005
    aniso_r_train: float = 8.0
    max_splat_scale_m: float = 0.015
    prune_every: int = 500
    packed: bool = False
    with_ut: bool = False
    with_eval3d: bool = False
    post_processing: str | None = None  # Keep None so sRGB Obsidian Noir blacks remain un-crushed
    random_bg: bool = True
    srgb_background: tuple[float, float, float] = (0.04, 0.04, 0.04)


def blender_c2w_to_opencv(T_c2w_bl: np.ndarray) -> np.ndarray:
    """Convert Blender Z-up (+X right, +Y up, -Z forward) c2w to OpenCV (+X right, +Y down, +Z forward)."""
    T = np.asarray(T_c2w_bl, dtype=np.float64)
    return T @ np.diag([1.0, -1.0, -1.0, 1.0])


def cam0_silhouette_iou_check(
    rendered_alpha: np.ndarray,
    mesh_silhouette: np.ndarray,
    out_diag_png: Path | None = None,
    iou_threshold: float = 0.98,
) -> dict[str, Any]:
    """Verify Cam-0 silhouette IoU >= 0.98; writes diagnostic overlay PNG and aborts if misaligned."""
    a_bin = np.asarray(rendered_alpha) > 0.5
    m_bin = np.asarray(mesh_silhouette) > 0.5
    inter = int(np.logical_and(a_bin, m_bin).sum())
    union = int(np.logical_or(a_bin, m_bin).sum())
    iou = float(inter / max(union, 1))
    passed = iou >= iou_threshold
    if out_diag_png is not None and (not passed or out_diag_png.name):
        from PIL import Image  # type: ignore

        h, w = a_bin.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[m_bin, 1] = 220   # Green = mesh silhouette
        rgb[a_bin, 0] = 220   # Red = rendered alpha (yellow = overlap)
        out_diag_png.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(out_diag_png)
    if not passed:
        raise RuntimeError(f"Cam-0 silhouette IoU={iou:.4f} < {iou_threshold:.2f} (see {out_diag_png})")
    return {"cam0_iou": round(iou, 5), "threshold": iou_threshold, "pass": passed, "diag_png": str(out_diag_png) if out_diag_png else None}


def build_gsplat_trainer_spec(spec: GsplatTrainSpec) -> dict[str, Any]:
    """Return gsplat MCMC (primary) or DefaultStrategy(absgrad=True) (fallback) trainer contract."""
    strategy_cls = (
        f"gsplat.strategy.MCMCStrategy(cap_max={int(spec.cap_max)}, noise_lr=5e5, refine_start_iter=500, refine_stop_iter=22_000, refine_every=100, min_opacity=0.005)"
        if spec.strategy == "mcmc"
        else "gsplat.strategy.DefaultStrategy(absgrad=True)"
    )
    return {
        "backend": "nerfstudio-project/gsplat@main (CUDA 13 + C++20)",
        "rasterize_fn": "gsplat.rendering.rasterization",
        "strategy_mode": spec.strategy,
        "strategy_cls": strategy_cls,
        "cap_max": int(spec.cap_max),
        "export_fn": "gsplat.export_splats(means=..., scales=..., quats=..., opacities=..., sh0=..., shN=..., format='ply', save_to=...)",
        "data_dir": str(spec.data_dir),
        "out_dir": str(spec.out_dir),
        "init_points_ply": str(spec.init_points_ply) if spec.init_points_ply else None,
        "max_steps": int(spec.max_steps),
        "sh_degree": int(spec.sh_degree),
        "antialiased": bool(spec.antialiased),
        "absgrad": bool(spec.absgrad),
        "scale_reg": float(spec.scale_reg),
        "opacity_reg": float(spec.opacity_reg),
        "aniso_reg": float(spec.aniso_reg),
        "aniso_r_train": float(spec.aniso_r_train),
        "max_splat_scale_m": float(spec.max_splat_scale_m),
        "prune_every": int(spec.prune_every),
        "guards": ["alive.any()", "hard_prune_gt_15mm", "data_factor=1_even_res", "randomized_solid_bg"],
        "post_processing": spec.post_processing,
        "random_bg": bool(spec.random_bg),
        "srgb_background": list(spec.srgb_background),
    }


def sample_explicit_surface_points(
    V: np.ndarray,
    F: np.ndarray,
    n_points: int = 150000,
    seed: int = 411,
    face_garment_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Sample explicit uniform-area surface points + normals + (face_idx, u, v, w, garment_id)."""
    rng = np.random.default_rng(seed)
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    normals_raw = np.cross(B - A, C - A)
    areas = 0.5 * np.linalg.norm(normals_raw, axis=1)
    probs = areas / np.maximum(areas.sum(), 1e-12)
    face_idx = rng.choice(len(F), size=n_points, p=probs)
    r1 = np.sqrt(rng.random(n_points))
    r2 = rng.random(n_points)
    u = 1.0 - r1
    v = r1 * (1.0 - r2)
    w = r1 * r2
    pts = u[:, None] * A[face_idx] + v[:, None] * B[face_idx] + w[:, None] * C[face_idx]
    nrm = normals_raw[face_idx] / np.maximum(np.linalg.norm(normals_raw[face_idx], axis=1, keepdims=True), 1e-12)
    if face_garment_ids is not None:
        gids = np.asarray(face_garment_ids, dtype=np.uint16)[face_idx]
    else:
        gids = np.zeros(n_points, dtype=np.uint16)
    binding = {
        "face_idx": face_idx.astype(np.int32),
        "u": u.astype(np.float32),
        "v": v.astype(np.float32),
        "w": w.astype(np.float32),
        "garment_id": gids,
    }
    return pts.astype(np.float32), nrm.astype(np.float32), binding


def write_ply_points(out_ply: Path, pts: np.ndarray, rgb: np.ndarray | None = None) -> Path:
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    if rgb is None:
        rgb = np.full((len(pts), 3), 128, dtype=np.uint8)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(pts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")
    rec = np.empty(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with out_ply.open("wb") as f:
        f.write(header)
        rec.tofile(f)
    return out_ply


def inspect_blender_transforms_json(transforms_json: Path) -> dict[str, Any]:
    data = json.loads(transforms_json.read_text(encoding="utf-8"))
    frames = data.get("frames", [])
    return {
        "camera_angle_x": data.get("camera_angle_x"),
        "fl_x": data.get("fl_x"),
        "fl_y": data.get("fl_y"),
        "w": data.get("w"),
        "h": data.get("h"),
        "frame_count": len(frames),
        "known_camera_zero_sfm": True,
    }


if __name__ == "__main__":
    V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    F = np.array([[0, 1, 2]], dtype=np.int64)
    pts, nrm, bind = sample_explicit_surface_points(V, F, n_points=512)
    spec = build_gsplat_trainer_spec(GsplatTrainSpec(data_dir=Path("cycles_120cam"), out_dir=Path("out_splats")))
    print(json.dumps({"sampled_points": len(pts), "binding_keys": sorted(bind), "normal_mean_z": float(nrm[:, 2].mean()), "api": spec}))
