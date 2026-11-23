"""01_catalog_ingest: Validates Better Style SKUs and Maximus baseline meshes with trimesh + CATALOG_BASELINE_AUDIT.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

ENGINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ENGINE_ROOT.parents[1]
AUDIT_JSON = ENGINE_ROOT / "docs" / "CATALOG_BASELINE_AUDIT.json"
DEFAULT_OUT = Path(__file__).resolve().parent / "catalog_ingest_check.json"


def inspect_glb_with_trimesh(glb_path: Path) -> dict[str, Any]:
    scene = trimesh.load(str(glb_path), process=False)
    if isinstance(scene, trimesh.Scene):
        geoms = list(scene.geometry.values())
        verts = sum(len(g.vertices) for g in geoms if hasattr(g, "vertices"))
        faces = sum(len(g.faces) for g in geoms if hasattr(g, "faces"))
        watertight = all(bool(getattr(g, "is_watertight", False)) for g in geoms)
    else:
        verts = len(scene.vertices)
        faces = len(scene.faces)
        watertight = bool(getattr(scene, "is_watertight", False))
    return {
        "exists": True,
        "bytes": glb_path.stat().st_size,
        "verts": int(verts),
        "faces": int(faces),
        "watertight": watertight,
    }


def run_ingest(out_path: Path = DEFAULT_OUT) -> dict[str, Any]:
    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    sku_checks: dict[str, Any] = {}
    for sku, info in audit.get("skus", {}).items():
        blend_rel = info.get("master_blend", "")
        blend_path = REPO_ROOT / blend_rel
        q_rel = info.get("quality_json", "")
        q_path = (REPO_ROOT / q_rel) if q_rel else None
        sku_checks[sku] = {
            "title": info.get("title"),
            "master_blend_exists": blend_path.exists(),
            "master_blend_bytes": blend_path.stat().st_size if blend_path.exists() else 0,
            "quality_json_exists": bool(q_path and q_path.exists()),
            "hero_role": (
                "C0_AND_TRACK_A_HERO"
                if sku == "BS-RINGER-001"
                else ("Z1_ZIP_HERO_PREREQ_Z4a" if sku == "BS-TRACK-001" else "STANDARD")
            ),
            "known_defects_count": len(info.get("known_defects", [])),
        }

    lod1_glb = REPO_ROOT / "02_engines" / "maximus_engine" / "colab_stage" / "results" / "M6_game_canon" / "maximus_v2_game_lod1.glb"
    dressed_glb = REPO_ROOT / "02_engines" / "maximus_engine" / "out" / "m9" / "maximus_v2_dressed_ringer_lod1.glb"
    ringer_npz = REPO_ROOT / "02_engines" / "maximus_engine" / "out" / "m9" / "hero_ringer_shell.npz"

    shell_summary = {}
    if ringer_npz.exists():
        d = np.load(ringer_npz, allow_pickle=True)
        shell_summary = {
            "exists": True,
            "verts": int(d["V_orig"].shape[0]),
            "faces": int(d["F"].shape[0]),
            "edges": int(d["edges"].shape[0]),
            "collar_verts": int(d["collar_mask"].sum()),
            "body_verts": int(d["body_ref_V"].shape[0]),
            "body_faces": int(d["body_F"].shape[0]),
        }

    report = {
        "schema": "max_garment_eng.catalog_ingest_check/1",
        "audit_source": str(AUDIT_JSON),
        "trimesh_version": trimesh.__version__,
        "skus": sku_checks,
        "maximus_v2_game_lod1": inspect_glb_with_trimesh(lod1_glb) if lod1_glb.exists() else {"exists": False},
        "m9_dressed_ringer_lod1": inspect_glb_with_trimesh(dressed_glb) if dressed_glb.exists() else {"exists": False},
        "hero_ringer_shell_npz": shell_summary,
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    res = run_ingest()
    print(json.dumps({"skus_verified": len(res["skus"]), "trimesh": res["trimesh_version"], "lod1_bytes": res["maximus_v2_game_lod1"].get("bytes")}, indent=2))
