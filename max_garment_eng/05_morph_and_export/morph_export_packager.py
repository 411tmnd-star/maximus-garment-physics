"""05_morph_and_export/morph_export_packager.py — Coupled 5-Key Size Morph Evaluator & < 1.5 MB Meshopt GLB Packager."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ENGINE_ROOT = Path(__file__).resolve().parents[1]
MAXIMUS_ROOT = ENGINE_ROOT.parent / "maximus_engine"
if str(MAXIMUS_ROOT) not in sys.path:
    sys.path.insert(0, str(MAXIMUS_ROOT))

import body_measure_engine as BME  # noqa: E402

GLB_BUDGET_BYTES = 2 * 1024 * 1024       # 2,097,152 bytes hard max
GLB_TARGET_BYTES = 1572864               # 1,572,864 bytes (< 1.5 MB target)
BARE_BODY_VERTEX_COUNT = 13380


def pack_tension_and_collar_into_color0(tension: np.ndarray, collar_mask: np.ndarray) -> np.ndarray:
    """Pack _TENSION into COLOR_0.r and collar mask into COLOR_0.g (uint8 RGBA) so gltfpack/meshopt preserves them."""
    t_u8 = np.clip(np.asarray(tension, dtype=np.float64) * 255.0, 0.0, 255.0).astype(np.uint8)
    c_u8 = (np.asarray(collar_mask, dtype=bool).astype(np.uint8)) * 255
    zeros = np.zeros_like(t_u8)
    alpha = np.full_like(t_u8, 255)
    return np.stack([t_u8, c_u8, zeros, alpha], axis=1)


def assert_bare_body_vertex_count(bare_vert_count: int) -> None:
    if int(bare_vert_count) != BARE_BODY_VERTEX_COUNT:
        raise AssertionError(f"Bare body vertex count mismatch: got {bare_vert_count}, expected {BARE_BODY_VERTEX_COUNT}")


def inspect_coupled_morphs(npz_path: Path = MAXIMUS_ROOT / "out" / "m9" / "coupled_morphs.npz") -> dict:
    d = np.load(npz_path, allow_pickle=True)
    keys = [str(k) for k in d["key_names"]]
    deltas_shirt = d["deltas_shirt"]
    V0 = d["V_shirt_0"]
    edges = d["edges"]
    L0 = d["L_rest"]
    stats = {}
    for idx, k in enumerate(keys):
        V_k = V0 + deltas_shirt[idx].astype(np.float64)
        L_k = np.linalg.norm(V_k[edges[:, 0]] - V_k[edges[:, 1]], axis=1)
        r = L_k / np.maximum(L0, 1e-12)
        stats[k] = {"p99_stretch": round(float(np.percentile(r, 99.0)), 4), "max_stretch": round(float(r.max()), 4)}
    return {"npz": str(npz_path), "verts": int(len(V0)), "keys": keys, "key_stretch_stats": stats}


def build_gltf_transform_cmd(src_glb: Path, dst_glb: Path) -> list[str]:
    """Build deterministic EXT_meshopt_compression command preserving morph targets, COLOR_0, and 152-bone skinning.

    Never uses Draco (which leaves morph targets uncompressed) and never uses --simplify/--weld.
    """
    exe = shutil.which("gltf-transform") or shutil.which("npx")
    base = [exe] if exe and exe.endswith("gltf-transform") else ["npx", "--yes", "@gltf-transform/cli@4.4.2"]
    return [
        *base,
        "optimize",
        str(src_glb),
        str(dst_glb),
        "--compress",
        "meshopt",
        "--meshopt-level",
        "high",
        "--sparse",
        "true",
        "--texture-compress",
        "webp",
        "--texture-size",
        "1024",
        "--simplify",
        "false",
        "--weld",
        "false",
    ]


def pack_glb(src_glb: Path, dst_glb: Path, bare_vert_count: int = BARE_BODY_VERTEX_COUNT, dry_run: bool = True) -> dict:
    assert_bare_body_vertex_count(bare_vert_count)
    cmd = build_gltf_transform_cmd(src_glb, dst_glb)
    if dry_run:
        return {
            "executed": False,
            "cmd": cmd,
            "compression": "EXT_meshopt_compression",
            "target_bytes": GLB_TARGET_BYTES,
            "budget_bytes": GLB_BUDGET_BYTES,
            "bare_body_verts": BARE_BODY_VERTEX_COUNT,
        }
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    out_bytes = dst_glb.stat().st_size if dst_glb.exists() else 0
    return {
        "executed": True,
        "returncode": proc.returncode,
        "out_bytes": out_bytes,
        "under_1_5mb_target": bool(0 < out_bytes < GLB_TARGET_BYTES),
        "under_2mb_budget": bool(0 < out_bytes < GLB_BUDGET_BYTES),
        "stderr_tail": proc.stderr[-800:],
    }


if __name__ == "__main__":
    bme = BME.BodyMeasureEngine()
    sample_fit = bme.measure_from_profile(175.0, 74.0, "average", pose="rest")
    morphs = inspect_coupled_morphs()
    dry_cmd = pack_glb(Path("maximus_v2_dressed_raw.glb"), Path("maximus_v2_dressed_lod1.glb"), dry_run=True)
    print(json.dumps({
        "sample_anthro_metrics": {k: sample_fit[k] for k in ("stature_cm", "chest_girth_cm", "waist_girth_cm", "hip_girth_cm") if k in sample_fit},
        "coupled_morphs": morphs,
        "meshopt_pack_plan": dry_cmd,
    }, indent=2))
