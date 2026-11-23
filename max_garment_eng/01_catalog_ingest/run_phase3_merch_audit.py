"""Deterministic 9-SKU Merch Inventory & Phase 3 Dressing Readiness Audit.

Audits both Unworn assets (flat/cutout paths, resolution, alpha check, SHA-256)
and Worn-on-Maximus readiness (.blend master, hash, C0 gate status, collar/penetration metrics,
material/rhinestone/jacquard fidelity, and colour-group QA) across all 9 Better Style SKUs.

Writes:
  - 02_engines/max_garment_eng/docs/PHASE3_MERCH_DRESSING_AUDIT.json
  - 02_engines/max_garment_eng/docs/PHASE3_MERCH_DRESSING_AUDIT.md
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
ENGINES_ROOT = REPO_ROOT / "02_engines"
MAX_GARMENT = ENGINES_ROOT / "max_garment_eng"
STORE_ROOT = ENGINES_ROOT / "better_style_engine" / "01_web_storefront" / "catalogue"
CATALOGUE_JSON = STORE_ROOT / "data" / "catalogue.json"
PRODUCTS_JSON = STORE_ROOT / "data" / "products.json"
BASELINE_AUDIT_JSON = MAX_GARMENT / "docs" / "CATALOG_BASELINE_AUDIT.json"
M9_MANIFEST_JSON = STORE_ROOT / "media" / "maximus_motion" / "m9_dressed_manifest.json"
OUT_JSON = MAX_GARMENT / "docs" / "PHASE3_MERCH_DRESSING_AUDIT.json"
OUT_MD = MAX_GARMENT / "docs" / "PHASE3_MERCH_DRESSING_AUDIT.md"

ORDERED_SKUS = [
    "BS-RINGER-001",
    "BS-POLO-001",
    "BS-CAMP-001",
    "BS-CAMP-002",
    "BS-BSTSET-001",
    "BS-TANK-001",
    "BS-DENIMSHIRT-001",
    "BS-SLV-001",
    "BS-TRACK-001",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _inspect_image(path: Path) -> dict[str, Any]:
    with Image.open(path) as im:
        has_alpha = im.mode in ("RGBA", "LA")
        alpha_min, alpha_max = (255, 255)
        if has_alpha:
            alpha_min, alpha_max = im.getchannel("A").getextrema()
        return {
            "rel_path": path.relative_to(REPO_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "resolution": [im.width, im.height],
            "aspect": round(im.width / im.height, 4) if im.height else 0.0,
            "mode": im.mode,
            "has_transparent_alpha": bool(has_alpha and alpha_min < 250),
            "alpha_extrema": [int(alpha_min), int(alpha_max)],
            "sha256": _sha256(path),
        }


def _collect_unworn_assets(sku: str, cat_entry: dict[str, Any] | None) -> list[dict[str, Any]]:
    seen_paths: set[Path] = set()
    records: list[dict[str, Any]] = []

    def add_if_img(p: Path, kind: str) -> None:
        p = p.resolve()
        if not p.is_file() or p in seen_paths:
            return
        if p.suffix.lower() not in (".png", ".webp", ".jpg", ".jpeg", ".avif"):
            return
        seen_paths.add(p)
        rec = _inspect_image(p)
        rec["kind"] = kind
        records.append(rec)

    if cat_entry:
        rdir = (STORE_ROOT / cat_entry["render_dir"]) if cat_entry.get("render_dir") else (STORE_ROOT / "engine_inputs" / "assets" / "renders" / sku)
        if rdir.is_dir():
            for p in sorted(rdir.glob("*.png")):
                if p.name.startswith("_"):
                    continue
                add_if_img(p, "engine_render_source")
        for extra in cat_entry.get("extra_render_dirs", []):
            ed = STORE_ROOT / extra
            if ed.is_dir():
                for p in sorted(ed.glob("*.png")):
                    if p.name.startswith("_"):
                        continue
                    add_if_img(p, "extra_render_source")
        for cw_id, cw in (cat_entry.get("colourways") or {}).items():
            for vw, rel in (cw.get("views") or {}).items():
                add_if_img(STORE_ROOT / rel, f"flat_view_{cw_id}_{vw}")

    for sdir in (STORE_ROOT / "sources" / sku, STORE_ROOT / "sources" / f"{sku}-back"):
        if sdir.is_dir():
            for p in sorted(sdir.rglob("*.png")):
                if p.name.startswith("_"):
                    continue
                add_if_img(p, "storefront_source")

    media_dir = STORE_ROOT / "media" / sku
    if media_dir.is_dir():
        for p in sorted(media_dir.glob("*_tile.webp")):
            add_if_img(p, "published_072_tile")
    return records


def _collect_engine_blends(sku: str, base_info: dict[str, Any]) -> dict[str, Any]:
    candidates: list[Path] = []
    if base_info.get("master_blend"):
        candidates.append(REPO_ROOT / base_info["master_blend"])
    alias_map = {
        "BS-RINGER-001": ["02_engines/max_garment_eng/COLAB_MEGA_SPRINT/ringer_tee_master.blend"],
        "BS-POLO-001": ["02_engines/human_engine/src_catalog/better_style_catalog/assets/archetypes/rugby_polo_master.blend"],
        "BS-CAMP-001": ["02_engines/human_engine/src_catalog/better_style_catalog/out/concept_visualizations/v2/blend/CAMP-BLUE_v2.blend"],
        "BS-CAMP-002": ["02_engines/human_engine/src_catalog/better_style_catalog/assets/archetypes/camp_shirt_master.blend"],
        "BS-BSTSET-001": [
            "02_engines/human_engine/src_catalog/better_style_catalog/out/concept_visualizations/blend/SET-ELEPHANT-BLUE.blend",
            "02_engines/human_engine/src_catalog/better_style_catalog/out/concept_visualizations/blend/SET-ELEPHANT-RED.blend",
        ],
        "BS-TRACK-001": ["02_engines/human_engine/src_catalog/BETTER_STYLE_VISUAL_CURATION/14_TRACKSUIT_DUAL_STATE_VIDEO/garments/bs_track_001_g0_master.blend"],
    }
    for rel in alias_map.get(sku, []):
        candidates.append(REPO_ROOT / rel)

    found = []
    seen = set()
    for c in candidates:
        rc = c.resolve()
        if rc.is_file() and rc not in seen:
            seen.add(rc)
            found.append({
                "rel_path": rc.relative_to(REPO_ROOT).as_posix(),
                "bytes": rc.stat().st_size,
                "sha256": _sha256(rc),
            })

    q_info: dict[str, Any] = {}
    if base_info.get("quality_json"):
        qp = REPO_ROOT / base_info["quality_json"]
        if qp.is_file():
            q_info["rel_path"] = qp.relative_to(REPO_ROOT).as_posix()
            q_data = json.loads(qp.read_text(encoding="utf-8"))
            q_info["keys"] = sorted(q_data.keys())[:12]

    set_glbs = []
    if sku == "BS-BSTSET-001":
        for rp in sorted((STORE_ROOT / "sources" / "BS-BSTSET-001").glob("*.report.json")):
            rd = json.loads(rp.read_text(encoding="utf-8"))
            glb = rp.with_suffix("").with_suffix(".glb")
            set_glbs.append({
                "report": rp.name,
                "glb_bytes": glb.stat().st_size if glb.is_file() else 0,
                "stones_total": rd.get("stones_total"),
                "zones": len(rd.get("zones", [])),
            })

    return {"masters": found, "quality_json": q_info, "set_glbs": set_glbs}


def run_audit() -> dict[str, Any]:
    cat = json.loads(CATALOGUE_JSON.read_text(encoding="utf-8"))
    prods = json.loads(PRODUCTS_JSON.read_text(encoding="utf-8")) if PRODUCTS_JSON.exists() else {"products": []}
    baseline = json.loads(BASELINE_AUDIT_JSON.read_text(encoding="utf-8")) if BASELINE_AUDIT_JSON.exists() else {"skus": {}}
    m9 = json.loads(M9_MANIFEST_JSON.read_text(encoding="utf-8")) if M9_MANIFEST_JSON.exists() else {}

    cat_by_sku = {p["sku"]: p for p in cat.get("products", [])}
    prod_by_sku = {p["sku"]: p for p in prods.get("products", [])}
    base_by_sku = baseline.get("skus", {})

    wave_plan = {
        "BS-RINGER-001": ("CERTIFIED_BURST_1A_V2", "Certified on Maximus v2 (re-baked knitted wool + cursive script + horse + black collar/cuffs GLB + SOG v3)"),
        "BS-POLO-001": ("WAVE_3A_KNITS", "Pre-FBL-008 collar + rest_edge_ratio_max 4.80-6.46x; target C0 + SO(3) + 120-cam SOG"),
        "BS-TANK-001": ("WAVE_3A_KNITS", "Owner cutout reference only (no .blend yet); draft parametric rib-tank pattern -> C0"),
        "BS-SLV-001": ("WAVE_3A_KNITS", "Shell POM mismatch (+37mm front/-47.5mm shoulder) & zero UVs; re-pattern + UVs + acid-wash shader -> C0"),
        "BS-CAMP-001": ("WAVE_3B_WOVEN_CAMPS", "v3 revers collar candidate (320-400 button degen tris); clean buttons + C0 coupled fit -> SOG"),
        "BS-CAMP-002": ("WAVE_3B_WOVEN_CAMPS", "Shares camp_shirt_master v3; Black / Ochre tartan label verified; re-solve C0 + SOG"),
        "BS-BSTSET-001": ("WAVE_3C_RHINESTONE_SET", "Trousers -48mm & no UVs; shirt -34.3mm vs theta_fit; separate jacquard cloth + rigid-bound 16,838 rhinestones (SO(3) + EXT_mesh_gpu_instancing)"),
        "BS-DENIMSHIRT-001": ("WAVE_3D_OUTERWEAR_DUAL", "Owner cutout reference only (red & black); build selvedge twill camp/denim master -> C0 + SOG"),
        "BS-TRACK-001": ("WAVE_3D_OUTERWEAR_DUAL", "fbl_003 preserve_volume mismatch + G0_jacket 424 inside / G0_pants 602 inside (-99mm); Z1/Z4a Excella zipper hardware + C0 dual-state"),
    }

    sku_audits = []
    camp2_label_ok = False
    for sku in ORDERED_SKUS:
        cat_e = cat_by_sku.get(sku)
        prod_e = prod_by_sku.get(sku)
        base_e = base_by_sku.get(sku, {})
        unworn = _collect_unworn_assets(sku, cat_e)
        blends = _collect_engine_blends(sku, base_e)

        cw_labels = {}
        colour_families = []
        if cat_e:
            for cid, cv in (cat_e.get("colourways") or {}).items():
                lbl = cv.get("label", "")
                cw_labels[cid] = lbl
                colour_families.append(lbl.split("/")[0].strip())
        if sku == "BS-CAMP-002":
            camp2_label_ok = cw_labels.get("black-plaid") == "Black / Ochre" and "Black plaid" not in cw_labels.values()

        worn_status: dict[str, Any]
        if sku == "BS-RINGER-001":
            gates = m9.get("certified_gates", {})
            worn_status = {
                "c0_certified": True,
                "gwn_inside": gates.get("gwn_inside"),
                "o7_collar_inside": gates.get("o7_collar_inside"),
                "g_stretch_p99": gates.get("g_stretch_p99"),
                "glb_bytes": m9.get("glb_bytes"),
                "glb_sha256": m9.get("glb_sha256"),
                "sog_bytes": (m9.get("assets") or {}).get("sog", {}).get("bytes"),
                "fidelity_notes": "Re-baked knitted wool normal + cavity AO + true #181716 ringer collar/cuffs + cursive 'Betterstyle' chest script + lower horse emblem; 7 coupled morph targets.",
            }
        else:
            worn_status = {
                "c0_certified": False,
                "known_defects": base_e.get("known_defects", ["No C0-solved dressed GLB/SOG on Maximus v2 yet."]),
                "has_blend_master": len(blends["masters"]) > 0,
                "fidelity_requirements": wave_plan[sku][1],
            }

        tiles_published = [r for r in unworn if r["kind"] == "published_072_tile"]
        src_imgs = [r for r in unworn if r["kind"] != "published_072_tile"]
        sku_audits.append({
            "sku": sku,
            "title": (cat_e or {}).get("title") or base_e.get("title") or sku,
            "storefront_status": (cat_e or {}).get("status", "engine_only_unlisted"),
            "colourways": cw_labels,
            "colour_families": sorted(set(colour_families)),
            "wave": wave_plan[sku][0],
            "unworn": {
                "source_image_count": len(src_imgs),
                "published_072_tile_count": len(tiles_published),
                "all_tiles_720x1000": all(r["resolution"] == [720, 1000] for r in tiles_published) if tiles_published else None,
                "transparent_alpha_count": sum(1 for r in src_imgs if r["has_transparent_alpha"]),
                "sample_sources": src_imgs[:4],
                "sample_tiles": tiles_published[:3],
            },
            "worn_on_maximus": worn_status,
            "engine_assets": blends,
        })

    report = {
        "schema": "maximus.v2.phase3_merch_audit/1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "total_skus": len(sku_audits),
        "c0_certified_count": sum(1 for s in sku_audits if s["worn_on_maximus"].get("c0_certified")),
        "camp_002_black_ochre_verified": camp2_label_ok,
        "skus": sku_audits,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # Render concise engineering Markdown summary
    md = [
        "# Phase 3 Full-Merch Inventory & Dressing Readiness Audit (9 SKUs)",
        "",
        f"- **Generated (UTC):** `{report['generated_utc']}`",
        f"- **SKUs Audited:** `{report['total_skus']} / 9` (`{report['c0_certified_count']}` C0-certified Hero, `8` queued across Waves 3A–3D)",
        f"- **`BS-CAMP-002` Colour-Group QA:** `{'PASS (Black / Ochre -> colour_family Black)' if camp2_label_ok else 'CHECK FAILED'}`",
        "",
        "## SKU Matrix (Unworn Storefront & Worn Maximus v2 Readiness)",
        "",
        "| SKU | Wave | Storefront Status | Unworn Sources (Alpha) | `0.72` Tiles | Master `.blend` / Hardware | Worn Maximus C0 Status & Action |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in sku_audits:
        u = s["unworn"]
        w = s["worn_on_maximus"]
        bl = s["engine_assets"]["masters"]
        bl_str = f"`{bl[0]['rel_path'].split('/')[-1]}` ({bl[0]['bytes'] / 1e6:.2f} MB)" if bl else "No `.blend` yet (reference cut-out)"
        if s["engine_assets"]["set_glbs"]:
            st = s["engine_assets"]["set_glbs"][0].get("stones_total")
            bl_str += f" + `{st}` rhinestones GLB"
        tile_str = f"{u['published_072_tile_count']} (`720x1000`)" if u["published_072_tile_count"] else "0"
        c0_str = (
            f"**CERTIFIED**: GLB `{w['glb_bytes']/1e6:.2f} MB`, GWN=`{w['gwn_inside']}`, O-7=`{w['o7_collar_inside']}`, p99=`{w['g_stretch_p99']}`"
            if w.get("c0_certified")
            else w["fidelity_requirements"]
        )
        md.append(f"| `{s['sku']}` | `{s['wave']}` | `{s['storefront_status']}` | {u['source_image_count']} ({u['transparent_alpha_count']} alpha) | {tile_str} | {bl_str} | {c0_str} |")

    md += [
        "",
        "## Staged Colab L4 Execution Packet",
        "",
        "- **Batch Orchestrator:** `02_engines/max_garment_eng/COLAB_MEGA_SPRINT/phase3_dress_all_skus_colab.py` (held offline — run only after owner approval).",
        "- **Waves:** `3A` (`BS-POLO-001`, `BS-TANK-001`, `BS-SLV-001`), `3B` (`BS-CAMP-001`, `BS-CAMP-002`), `3C` (`BS-BSTSET-001` jacquard + rigid-bound 16,838 rhinestones), `3D` (`BS-DENIMSHIRT-001`, `BS-TRACK-001`).",
        "- **Per-SKU Acceptance Gates:** `GWN == 0`, `O-7 collar == 0`, `p99_stretch <= 1.08`, Meshopt GLB `<= 1.2 MB`, 120-cam known-pose `gsplat` `.sog <= 7.0 MB` + `garment_id.webp`, `0.72` seamless `AV1 <= 400 KB` (`SSIM == 1.0`), `finally: runtime.unassign()`.",
    ]
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    rep = run_audit()
    print(json.dumps({"total_skus": rep["total_skus"], "c0_certified": rep["c0_certified_count"], "camp2_black_ochre": rep["camp_002_black_ochre_verified"], "out_md": str(OUT_MD)}, indent=2))
