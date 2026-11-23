"""04_clearance_gates/clearance_gate_runner.py — C0 Gate Enforcer (G_STRETCH + Float64 GG-31 GWN + O-7 Collar Rim)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ENGINE_ROOT = Path(__file__).resolve().parents[1]
MAXIMUS_ROOT = ENGINE_ROOT.parent / "maximus_engine"
for _p in (MAXIMUS_ROOT, MAXIMUS_ROOT / "colab_stage"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import corpus_link  # noqa: E402
corpus_link.link(strict=os.environ.get("MAXIMUS_STRICT", "1") == "1")
import canon_geometry_kernels as K  # noqa: E402

G_STRETCH_P99_LIMIT = 1.08
G_STRETCH_MAX_LIMIT = 1.15
BODY_CLEARANCE_MIN_M = 0.003   # 3.0 mm
COLLAR_O7_MAX_M = 0.00025      # <= 0.25 mm collar seating per MASTER_VERIFICATION_CONTRACT.json


def evaluate_g_stretch(V_rest: np.ndarray, V_deformed: np.ndarray, edges: np.ndarray) -> dict[str, Any]:
    e_rest = np.linalg.norm(V_rest[edges[:, 0]] - V_rest[edges[:, 1]], axis=1)
    e_def = np.linalg.norm(V_deformed[edges[:, 0]] - V_deformed[edges[:, 1]], axis=1)
    ratios = e_def / np.maximum(e_rest, 1e-12)
    p99 = float(np.percentile(ratios, 99.0))
    rmax = float(ratios.max())
    rmin = float(ratios.min())
    return {
        "p99_ratio": round(p99, 4),
        "max_ratio": round(rmax, 4),
        "min_ratio": round(rmin, 4),
        "edges_gt_1_15": int((ratios > G_STRETCH_MAX_LIMIT).sum()),
        "pass": bool(p99 <= G_STRETCH_P99_LIMIT and rmax <= G_STRETCH_MAX_LIMIT),
    }


def evaluate_gwn_and_collar(
    V_garment: np.ndarray,
    V_body: np.ndarray,
    F_body: np.ndarray,
    collar_mask: np.ndarray | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    signed_m, _ = K.signed_distance_to_mesh(V_garment, V_body, F_body, k=16, device=device)
    inside = int((signed_m < 0.0).sum())
    min_mm = float(signed_m.min() * 1000.0)
    median_mm = float(np.median(signed_m) * 1000.0)
    res: dict[str, Any] = {
        "inside_gwn_count": inside,
        "min_signed_mm": round(min_mm, 3),
        "median_signed_mm": round(median_mm, 3),
        "body_barrier_pass": bool(inside == 0 and signed_m.min() >= BODY_CLEARANCE_MIN_M - 1e-4),
    }
    if collar_mask is not None and collar_mask.any():
        cs = signed_m[collar_mask]
        res["o7_collar"] = {
            "collar_inside": int((cs < 0.0).sum()),
            "collar_min_mm": round(float(cs.min() * 1000.0), 3),
            "collar_median_mm": round(float(np.median(cs) * 1000.0), 3),
            "o7_pass": bool((cs >= 0.0).all() and np.median(cs) <= COLLAR_O7_MAX_M),
        }
    return res


def smoke_test_m9_ringer() -> dict[str, Any]:
    npz = MAXIMUS_ROOT / "out" / "m9" / "hero_ringer_shell.npz"
    d = np.load(npz, allow_pickle=True)
    stretch = evaluate_g_stretch(d["V_orig"], d["V_fit"], d["edges"])
    # Fast subsample check on collar + 500 sampled garment verts to run in <2s on CPU
    idx = np.unique(np.concatenate([np.flatnonzero(d["collar_mask"]), np.linspace(0, len(d["V_fit"]) - 1, 300, dtype=int)]))
    gwn_sample = evaluate_gwn_and_collar(d["V_fit"][idx], d["body_ref_V"], d["body_F"], device="cpu")
    return {"npz": str(npz), "G_STRETCH": stretch, "gwn_cpu_sample": gwn_sample}


if __name__ == "__main__":
    print(json.dumps(smoke_test_m9_ringer(), indent=2))
