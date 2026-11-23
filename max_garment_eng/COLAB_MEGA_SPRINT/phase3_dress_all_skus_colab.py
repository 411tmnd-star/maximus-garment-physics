"""phase3_dress_all_skus_colab.py — Staged Colab L4 8-SKU Multi-Wave Dressing, Splat & Morph-Pack Orchestrator.

Zero-burn offline until explicitly launched on an L4 runtime.
Supports `--dry-run` locally on CPU to validate 100% of local repaired .blend / .npz / texture inputs,
wave manifests, C0 / gsplat / morph-pack / video gates, and realistic L4 runtime breakdown.

Waves (all built atop Maximus v2 C0 SolverVBD fitting + SO(3) rigid hardware + canonical head-to-toe base bottoms):
  - Wave 3A-0 : BS-RINGER-001     (280 GSM combed cotton jersey; mohair archive stays offline untouched)
  - Wave 3A   : BS-POLO-001, BS-TANK-001, BS-SLV-001
  - Wave 3B   : BS-CAMP-001, BS-CAMP-002 (Black / Ochre tartan)
  - Wave 3C   : BS-BSTSET-001 (separated cloth + 16,838 rigid SO(3) stones), BS-DENIMSHIRT-001 (promoted to live burst)
  - Wave 3D   : BS-TRACK-001      (QUARANTINED_WAVE_3D — excluded from default 8-SKU live burst unless --include-quarantined)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
MAX_GARMENT = THIS_DIR.parent
REPO_ROOT = MAX_GARMENT.parents[1] if len(MAX_GARMENT.parents) > 1 else MAX_GARMENT
INGEST_DIR = MAX_GARMENT / "01_catalog_ingest"
BASE_WARDROBE_DIR = (THIS_DIR / "base_wardrobe") if (THIS_DIR / "base_wardrobe").is_dir() else (INGEST_DIR / "base_wardrobe")
REPAIRED_DIR = (THIS_DIR / "repaired_masters") if (THIS_DIR / "repaired_masters").is_dir() else (INGEST_DIR / "repaired_masters")
TEX_DIR = REPAIRED_DIR / "textures"

BS_ENGINE = REPO_ROOT / "02_engines" / "better_style_engine"
COLLAR_ASSETS = BS_ENGINE / "02_engine_3d" / "isolated_clones" / "better_style_catalog_collar" / "assets"
COMPILED_UVS = COLLAR_ASSETS / "compiled_uvs"
ART_CAMP = COLLAR_ASSETS / "artwork_camp"
SLV_ART = BS_ENGINE / "02_engine_3d" / "garment_engine" / "BS-SLV-001" / "artwork"
STONE_OUT = BS_ENGINE / "03_governance_and_evidence" / "better_style_execution_workspace" / "03_ENGINE" / "rhinestone_engine" / "garments" / "out" / "BS-BSTSET-001"
TANK_SRC = BS_ENGINE / "01_web_storefront" / "catalogue" / "sources" / "BS-TANK-001"
DENIM_SRC = BS_ENGINE / "01_web_storefront" / "catalogue" / "sources" / "BS-DENIMSHIRT-001"
TRACK_MASTER = REPO_ROOT / "02_engines" / "human_engine" / "engineering" / "CONTRACT_PREFLIGHT_2026-11-09" / "backup_track" / "bs_track_001_g0_master.blend"

DRIVE_ROOT = Path("/content/drive/MyDrive/maximus_v2")
DRIVE_RESULTS = DRIVE_ROOT / "results" / "PHASE3_WAVES"
# C0 SolverVBD drape horizon: 120 frames x dt 1/240 s = 0.5 s simulated (10 substeps/frame, substep 1/2400 s). The first
# canary used 6 frames (25 ms): the cloth was still falling and the PBD pass did the work.
C0_DRAPE_FRAMES = 120
LEDGER_PATH = DRIVE_ROOT / "results" / "CHECKPOINT_LEDGER.jsonl"
DRIVE_M4_ATLAS_BLEND = DRIVE_ROOT / "results" / "M4_canon_atlas" / "maximus_v2_master_psd.blend"
LOCAL_ATLAS_BLEND = THIS_DIR / "m4_atlas" / "maximus_v2_master_psd.blend"
REPO_ATLAS_BLEND = REPO_ROOT / "02_engines" / "maximus_engine" / "colab_stage" / "results" / "M4_canon_atlas" / "maximus_v2_master_psd.blend"

SHARED_LOCAL_INPUTS = {
    "maximus_v2_master_psd": (
        LOCAL_ATLAS_BLEND if LOCAL_ATLAS_BLEND.is_file()
        else (DRIVE_M4_ATLAS_BLEND if DRIVE_M4_ATLAS_BLEND.is_file() else REPO_ATLAS_BLEND)
    ),
    "hero_ringer_shell_npz": THIS_DIR / "inputs" / "hero_ringer_shell.npz",
    "m_theta_fit_json": THIS_DIR / "inputs" / "m_theta_fit.json",
    "base_trouser_obsidian_blend": BASE_WARDROBE_DIR / "BASE_TROUSER_OBSIDIAN.blend",
    "base_shoe_minimal_blend": BASE_WARDROBE_DIR / "BASE_SHOE_MINIMAL.blend",
}

LIVE_DEFAULT_WAVES = ["3A-0", "3A", "3B", "3C"]
WAVE_CHOICES = ["3A-0", "3A", "3A-polo", "3B", "3C", "3D"]

WAVE_SPECS: dict[str, list[dict[str, Any]]] = {
    "3A-0": [
        {
            "sku": "BS-RINGER-001",
            "title": "Better Style Ringer Tee (280 GSM Cotton Jersey)",
            "source_blend": REPAIRED_DIR / "BS-RINGER-001_cotton280_master.blend",
            "dress_with_base_bottoms": True,
            "colorways": ["white-black", "black-white", "red-white"],
            "texture_inputs": [
                TEX_DIR / "cotton_jersey_280gsm_normal.png",
                COMPILED_UVS / "BS-RINGER-001" / "front_albedo_light.png",
                COMPILED_UVS / "BS-RINGER-001" / "front_albedo_dark.png",
                COMPILED_UVS / "BS-RINGER-001" / "front_height.png",
            ],
            "special_constraints": [
                "280 GSM combed cotton jersey PBR shader + crisp ringer neck/cuff binding (MAT_BRUSHED_MOHAIR_ARCHIVE untouched offline)",
                "Head-to-toe dressing via BASE_TROUSER_OBSIDIAN (part_id=2) + BASE_SHOE_MINIMAL (part_id=3)",
                "Garment-only MCMC gsplat (N=100k..120k) + canonical Z-slab morph pack (<= 2.5 MB) + 2x 60s frozen-XYZ colorway SH rebakes",
            ],
            "est_l4_minutes": {"c0_glb": 1.2, "optix_120cam": 2.4, "gsplat_primary": 3.0, "colorway_rebakes": 2.0, "av1_h264_pack": 0.9},
        },
    ],
    "3A": [
        {
            "sku": "BS-POLO-001",
            "title": "Better Style Rugby Polo",
            "source_blend": REPAIRED_DIR / "BS-POLO-001_clean_master.blend",
            "dress_with_base_bottoms": True,
            "colorways": ["black-white", "burgundy-white"],
            "texture_inputs": [
                TEX_DIR / "cotton_jersey_280gsm_normal.png",
                COMPILED_UVS / "BS-POLO-001" / "front_albedo_dark.png",
                COMPILED_UVS / "BS-POLO-001" / "front_albedo_light.png",
                COMPILED_UVS / "BS-POLO-001" / "back_albedo_dark.png",
                COMPILED_UVS / "BS-POLO-001" / "back_albedo_light.png",
                COMPILED_UVS / "BS-POLO-001" / "front_height.png",
                COMPILED_UVS / "BS-POLO-001" / "back_height.png",
            ],
            "special_constraints": [
                "Regularized presim edges + 0-degen rigid SO(3) buttons + BASE_TROUSER_OBSIDIAN + BASE_SHOE_MINIMAL",
                "Garment-only gsplat + 1x 60s frozen-XYZ SH rebake for burgundy-white",
            ],
            "est_l4_minutes": {"c0_glb": 1.4, "optix_120cam": 2.4, "gsplat_primary": 3.0, "colorway_rebakes": 1.0, "av1_h264_pack": 0.9},
        },
        {
            "sku": "BS-TANK-001",
            "title": "Better Style Ribbed Tank",
            "source_blend": REPAIRED_DIR / "BS-TANK-001_parametric_master.blend",
            "dress_with_base_bottoms": True,
            "colorways": ["black", "white"],
            "texture_inputs": [
                TEX_DIR / "tank_2x2_rib_normal.png",
                TANK_SRC / "tank_black_front_v3.png",
                TANK_SRC / "tank_white_front_v3.png",
            ],
            "special_constraints": [
                "Parametric 2x2 rib-knit cotton shell from Ringer torso + BASE_TROUSER_OBSIDIAN + BASE_SHOE_MINIMAL",
                "Armhole & neck clearance O-7 <= 1.0mm collar / >= 2.5mm armhole + 1x 60s SH rebake",
            ],
            "est_l4_minutes": {"c0_glb": 1.0, "optix_120cam": 2.2, "gsplat_primary": 2.8, "colorway_rebakes": 1.0, "av1_h264_pack": 0.9},
        },
        {
            "sku": "BS-SLV-001",
            "title": "Scorched Monolith Sleeveless",
            "source_blend": REPAIRED_DIR / "BS-SLV-001_clean_master.blend",
            "dress_with_base_bottoms": True,
            "colorways": ["grey"],
            "texture_inputs": [
                SLV_ART / "flames_front_deboss_fill.png",
                SLV_ART / "flames_front_hd_ridge.png",
                SLV_ART / "shield_bst_base_twill.png",
                SLV_ART / "shield_bst_frame.png",
            ],
            "special_constraints": [
                "POM-02/03/04 corrected (0.735 / 0.747 / 0.520m) + clean UVMap + PBR deboss(-0.3mm)/silicone(+0.8mm) with 0 extruded flame verts",
                "Head-to-toe dressing via BASE_TROUSER_OBSIDIAN + BASE_SHOE_MINIMAL",
            ],
            "est_l4_minutes": {"c0_glb": 1.1, "optix_120cam": 2.3, "gsplat_primary": 2.8, "colorway_rebakes": 0.0, "av1_h264_pack": 0.9},
        },
    ],
    "3B": [
        {
            "sku": "BS-CAMP-001",
            "title": "Better Style Camp Shirt (Monogram Edition)",
            "source_blend": REPAIRED_DIR / "BS-CAMP-001_clean_master.blend",
            "dress_with_base_bottoms": True,
            "colorways": ["blue", "green", "ivory", "gold", "taupe"],
            "texture_inputs": [
                COMPILED_UVS / "BS-CAMP-001" / "front_albedo_dark.png",
                COMPILED_UVS / "BS-CAMP-001" / "front_albedo_light.png",
                COMPILED_UVS / "BS-CAMP-001" / "back_albedo_dark.png",
                COMPILED_UVS / "BS-CAMP-001" / "back_albedo_light.png",
                ART_CAMP / "lattice_tile.png",
                ART_CAMP / "back_script.png",
            ],
            "special_constraints": [
                "v3 revers collar + art_center_y=0.47 locked + 0-degen rigid buttons + BASE_TROUSER_OBSIDIAN + BASE_SHOE_MINIMAL",
                "Primary gsplat (blue) + 4x 60s frozen-XYZ colorway SH rebakes (green, ivory, gold, taupe)",
            ],
            "est_l4_minutes": {"c0_glb": 1.4, "optix_120cam": 2.5, "gsplat_primary": 3.0, "colorway_rebakes": 4.0, "av1_h264_pack": 1.0},
        },
        {
            "sku": "BS-CAMP-002",
            "title": "Better Style Camp Shirt (Contrast Tartan Edition)",
            "source_blend": REPAIRED_DIR / "BS-CAMP-002_clean_master.blend",
            "dress_with_base_bottoms": True,
            "colorways": ["black-plaid"],
            "texture_inputs": [
                ART_CAMP / "tartan_tile.png",
                COMPILED_UVS / "BS-CAMP-002" / "front_albedo_dark.png",
                COMPILED_UVS / "BS-CAMP-002" / "back_albedo_dark.png",
                COMPILED_UVS / "BS-CAMP-002" / "front_height.png",
            ],
            "special_constraints": [
                "Verified label: Black / Ochre (colour_family=Black) + seamless tartan_tile UV + 0-degen rigid buttons",
                "Head-to-toe dressing via BASE_TROUSER_OBSIDIAN + BASE_SHOE_MINIMAL",
            ],
            "est_l4_minutes": {"c0_glb": 1.4, "optix_120cam": 2.4, "gsplat_primary": 3.0, "colorway_rebakes": 0.0, "av1_h264_pack": 0.9},
        },
    ],
    "3C": [
        {
            "sku": "BS-BSTSET-001",
            "title": "BST Camp Set (Jacquard + 5-Zone Contour Pave)",
            "source_blend": REPAIRED_DIR / "BS-BSTSET-001_clean_master.blend",
            "dress_with_base_bottoms": False,  # Ships own separated trousers (part_id=2); pairs w/ BASE_SHOE_MINIMAL (part_id=3)
            "include_base_footwear": True,
            "colorways": ["blue", "red"],
            "texture_inputs": [
                REPAIRED_DIR / "BS-BSTSET-001_stones_manifest.json",
                STONE_OUT / "BS-BSTSET-001_shirt_stones.npz",
                STONE_OUT / "BS-BSTSET-001_trousers_stones.npz",
                TEX_DIR / "denim_12oz_twill_normal.png",
            ],
            "special_constraints": [
                "Separated shirt + trouser shells (0 inside verts, clean trouser UV) + BASE_SHOE_MINIMAL footwear",
                "Rigid-bind 16,838 rhinestones (16,301 shirt + 537 trouser) via SO(3) triangle frames + EXT_mesh_gpu_instancing",
                "Separated tonal jacquard weave shader + 1x 60s SH rebake (red)",
            ],
            "est_l4_minutes": {"c0_glb": 1.8, "optix_120cam": 2.6, "gsplat_primary": 3.2, "colorway_rebakes": 1.0, "av1_h264_pack": 1.0},
        },
        {
            "sku": "BS-DENIMSHIRT-001",
            "title": "Better Style Denim Shirt (12 oz Heavyweight Twill)",
            "source_blend": REPAIRED_DIR / "BS-DENIMSHIRT-001_parametric_master.blend",
            "dress_with_base_bottoms": True,
            "colorways": ["black", "red"],
            "texture_inputs": [
                TEX_DIR / "denim_12oz_twill_normal.png",
                DENIM_SRC / "BS-DENIMSHIRT-001_black_floating_front_alpha.png",
                DENIM_SRC / "BS-DENIMSHIRT-001_black_floating_back_alpha.png",
                DENIM_SRC / "BS-DENIMSHIRT-001_red_floating_front_alpha.png",
                DENIM_SRC / "BS-DENIMSHIRT-001_red_floating_back_alpha.png",
            ],
            "special_constraints": [
                "Promoted from Wave 3D -> Wave 3C live 8-SKU burst",
                "12 oz long-sleeve point-collar denim master w/ rolled cuffs + pocket + 7 rigid snap buttons + BASE_TROUSER_OBSIDIAN + BASE_SHOE_MINIMAL",
            ],
            "est_l4_minutes": {"c0_glb": 1.5, "optix_120cam": 2.5, "gsplat_primary": 3.0, "colorway_rebakes": 1.0, "av1_h264_pack": 0.9},
        },
    ],
    "3D": [
        {
            "sku": "BS-TRACK-001",
            "title": "Better Style Tracksuit (Dual-State + Excella #5) [QUARANTINED]",
            "quarantined": True,
            "source_blend": TRACK_MASTER,
            "dress_with_base_bottoms": False,
            "colorways": ["black"],
            "texture_inputs": [],
            "special_constraints": [
                "QUARANTINED_WAVE_3D: deferred out of live 8-SKU burst; requires preserve_volume=False recovery + frame-1 extraction + Excella Z1/Z4a hardware",
            ],
            "est_l4_minutes": {"c0_glb": 0.0, "optix_120cam": 0.0, "gsplat_primary": 0.0, "colorway_rebakes": 0.0, "av1_h264_pack": 0.0},
        },
    ],
}


def resolve_selection(
    waves: list[str],
    sku_tokens: list[str] | None = None,
) -> tuple[list[str], set[str] | None, list[str]]:
    """Expand wave aliases (e.g. `3A-polo` -> `3A` w/ `BS-POLO-001`) and comma/space `--skus` filters."""
    explicit_skus: list[str] = []
    if sku_tokens:
        for tok in sku_tokens:
            for item in tok.split(","):
                item = item.strip()
                if item:
                    explicit_skus.append(item)

    norm_waves: list[str] = []
    alias_skus: set[str] = set()
    has_non_alias_wave = False
    for w in waves:
        if w == "3A-polo":
            if "3A" not in norm_waves:
                norm_waves.append("3A")
            alias_skus.add("BS-POLO-001")
        else:
            if w not in norm_waves:
                norm_waves.append(w)
            if w != "3D":
                has_non_alias_wave = True

    allowed_skus: set[str] | None = None
    if explicit_skus:
        allowed_skus = set(explicit_skus)
        # Ensure all waves needed for explicit SKUs are included in canonical order
        sku_to_wave = {spec["sku"]: w for w, specs in WAVE_SPECS.items() for spec in specs}
        unknown = [s for s in explicit_skus if s not in sku_to_wave]
        if unknown:
            raise ValueError(f"Unknown SKU(s) in --skus: {unknown}. Known: {sorted(sku_to_wave)}")
        needed_waves = [sku_to_wave[s] for s in explicit_skus]
        ordered_waves: list[str] = []
        for w in list(WAVE_SPECS.keys()):
            if w in needed_waves or w == "3D":
                ordered_waves.append(w)
        norm_waves = ordered_waves
    elif alias_skus and not has_non_alias_wave:
        allowed_skus = alias_skus
    elif alias_skus:
        # Both e.g. 3A-0, 3B and 3A-polo were passed: include all SKUs in non-alias waves + BS-POLO-001 in 3A
        allowed_skus = set(alias_skus)
        for w in norm_waves:
            if w != "3A" and w != "3D":
                for spec in WAVE_SPECS[w]:
                    allowed_skus.add(spec["sku"])

    if "3D" not in norm_waves:
        norm_waves.append("3D")
    return norm_waves, allowed_skus, explicit_skus


def verify_local_inputs(
    waves: list[str],
    include_quarantined: bool = False,
    allowed_skus: set[str] | None = None,
) -> dict[str, Any]:
    """Verify 100% of shared & selected per-SKU local input files exist on disk."""
    checked: list[dict[str, Any]] = []
    missing: list[str] = []

    for label, p in SHARED_LOCAL_INPUTS.items():
        ok = p.is_file()
        checked.append({"scope": "shared", "label": label, "path": str(p), "exists": ok})
        if not ok:
            missing.append(str(p))

    for w in waves:
        for spec in WAVE_SPECS[w]:
            if allowed_skus is not None and spec["sku"] not in allowed_skus:
                continue
            if spec.get("quarantined") and not include_quarantined:
                continue
            sb = Path(spec["source_blend"])
            ok_b = sb.is_file()
            checked.append({"scope": spec["sku"], "label": "source_blend", "path": str(sb), "exists": ok_b})
            if not ok_b:
                missing.append(str(sb))
            for tp in spec.get("texture_inputs", []):
                p = Path(tp)
                ok_t = p.is_file()
                checked.append({"scope": spec["sku"], "label": p.name, "path": str(p), "exists": ok_t})
                if not ok_t:
                    missing.append(str(p))

    return {
        "total_paths_checked": len(checked),
        "resolved_paths": sum(1 for c in checked if c["exists"]),
        "missing_count": len(missing),
        "missing_paths": missing,
        "resolution_pct": 100.0 if not missing else round(100.0 * (len(checked) - len(missing)) / max(1, len(checked)), 2),
    }


def estimate_l4_runtime(plan_items: list[dict[str, Any]]) -> dict[str, Any]:
    per_sku = {}
    totals = {"c0_glb": 0.0, "optix_120cam": 0.0, "gsplat_primary": 0.0, "colorway_rebakes": 0.0, "av1_h264_pack": 0.0}
    for item in plan_items:
        est = item.get("est_l4_minutes", {})
        s_tot = sum(float(est.get(k, 0.0)) for k in totals)
        per_sku[item["sku"]] = {**est, "sku_total_min": round(s_tot, 2)}
        for k in totals:
            totals[k] += float(est.get(k, 0.0))
    bootstrap_min = 6.5  # One-time venv + Blender 5.2 + cached/stubbed gsplat wheel + Vulkan setup
    active_min = sum(totals.values())
    total_with_overhead_min = round(bootstrap_min + active_min * 1.08, 1)  # +8% Drive SHA-256 sync buffer
    return {
        "gpu_target": "NVIDIA L4 (24 GB VRAM, Ada sm_89)",
        "bootstrap_once_min": bootstrap_min,
        "stage_breakdown_min": {k: round(v, 2) for k, v in totals.items()},
        "active_pipeline_min": round(active_min, 2),
        "total_estimated_min_with_buffer": total_with_overhead_min,
        "est_colab_compute_units_l4": round((total_with_overhead_min / 60.0) * 1.95, 2),
        "per_sku_min": per_sku,
    }


def build_execution_plan(
    waves: list[str],
    include_quarantined: bool = False,
    allowed_skus: set[str] | None = None,
    explicit_skus_order: list[str] | None = None,
) -> dict[str, Any]:
    plan_items = []
    skipped_items = []
    quarantined_items = []
    for w in waves:
        for spec in WAVE_SPECS[w]:
            entry = {
                "wave": w,
                "sku": spec["sku"],
                "title": spec["title"],
                "source_blend": str(spec["source_blend"]),
                "dress_with_base_bottoms": bool(spec.get("dress_with_base_bottoms", False)),
                "include_base_footwear": bool(spec.get("include_base_footwear", True)),
                "colorways": spec["colorways"],
                "texture_inputs": [str(x) for x in spec.get("texture_inputs", [])],
                "special_constraints": spec["special_constraints"],
                "est_l4_minutes": spec["est_l4_minutes"],
            }
            if spec.get("quarantined") and not include_quarantined:
                quarantined_items.append({**entry, "status": "QUARANTINED_WAVE_3D"})
            elif allowed_skus is not None and spec["sku"] not in allowed_skus:
                skipped_items.append({"wave": w, "sku": spec["sku"], "status": "SKIPPED_NOT_IN_BURST_FILTER"})
            else:
                plan_items.append(entry)

    if explicit_skus_order:
        order_idx = {s: i for i, s in enumerate(explicit_skus_order)}
        plan_items.sort(key=lambda x: order_idx.get(x["sku"], 999))

    return {
        "schema": "maximus.v2.phase3_batch_plan/2",
        "selected_waves": waves,
        "selected_skus_order": [x["sku"] for x in plan_items],
        "skipped_skus": skipped_items,
        "live_sku_count": len(plan_items),
        "quarantined_sku_count": len(quarantined_items),
        "pinned_stack": {
            "torch": "2.14.1+cu130",
            "newton": "1.8.0 (SolverVBD ONLY; SolverStyle3D prohibited)",
            "pyuipc": "0.0.29",
            "splat_transform": "@playcanvas/splat-transform@3.12.0 (--gpu-backend vulkan -H 1 --sh-palette-size 4096 -i 2 w/ bounded fallback)",
            "vulkan_icd": "/etc/vulkan/icd.d/nvidia_icd.json (libvulkan1)",
        },
        "pipeline_stages": [
            "1_C0_SOLVEVBD_AND_SO3_HARDWARE_HEAD_TO_TOE",
            "2_OPTIX_TURNTABLE_120CAM_OBJECT_ID_AND_072_REEL",
            "3_GSPLAT_GARMENT_ONLY_MCMC_AND_CANONICAL_MORPH_PACK",
            "4_FROZEN_XYZ_COLORWAY_SH_REBAKE_4X60S_AND_CALLOUT_ANCHORS",
            "5_MESHOPT_GLB_AND_DRIVE_SHA256_MANIFEST",
        ],
        "hard_gates": {
            "gwn_inside": 0,
            "g_stretch_p99_max": 1.08,
            "g_stretch_max": 1.15,
            "o7_collar_max_mm": 1.0,
            "glb_max_mb": 1.2,
            "morph_pack_max_mb": 2.5,
            "av1_max_kb": 400,
            "h264_max_kb": 550,
            "grid_aspect": "720x1000 (0.72)",
            "cam0_iou_min": 0.98,
            "object_id_pass": {"body": 0, "top": 1, "bottom": 2, "footwear": 3},
            "gsplat_n_target": "120000..150000 cap / 100000 canonical morph",
            "aniso_r_train": 8.0,
            "lambda_aniso": 5e-3,
            "post_processing": None,
            "garment_id_webp_exact": "(R, G, 0, A=255) lossless exact=True (no -m reorder)",
            "colorway_sh_rebake_sec": 60,
            "drive_sha256_manifest": True,
            "per_sku_checkpoint_isolation": True,
            "teardown": "finally: drive.flush_and_unmount(); runtime.unassign()",
        },
        "cache_reuse": {
            "wheels": "/content/drive/MyDrive/maximus_v2/cache/wheels/gsplat-1.6.0-cp312-cp312-linux_x86_64.whl",
            "warp_kernels": "/content/drive/MyDrive/maximus_v2/cache/warp/1.18.0",
            "m4_canon_atlas": "/content/drive/MyDrive/maximus_v2/results/M4_canon_atlas/",
        },
        "shared_inputs": {k: str(v) for k, v in SHARED_LOCAL_INPUTS.items()},
        "items": plan_items,
        "quarantined_items": quarantined_items,
    }


def _is_sku_checkpointed(sku: str) -> bool:
    """Verify from Drive ledger whether `sku` already passed and promoted intact."""
    marker = DRIVE_RESULTS / sku / "SKU_COMPLETE.json"
    if not marker.is_file() or not LEDGER_PATH.is_file():
        return False
    try:
        marker_payload = json.loads(marker.read_text(encoding="utf-8"))
        if marker_payload.get("status") != "PASSED":
            return False
    except Exception:
        return False
    stage_tag = f"PHASE3_{sku}"
    expected: dict[str, str] = {}
    for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("stage") == stage_tag:
            expected[rec["file"]] = rec["sha256"]
    if str(marker) not in expected:
        return False
    for fpath_str, want_sha in expected.items():
        fp = Path(fpath_str)
        if not fp.is_file():
            return False
        h = hashlib.sha256()
        with fp.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != want_sha:
            return False
    return True


def _pack_meshopt_glb_under_1_2mb(raw_glb: Path, lod1_glb: Path, env: dict[str, str], run_fn: Any) -> dict[str, Any]:
    """Compress raw GLB with EXT_meshopt_compression strictly under 1.2 MB (1,200,000 B)."""
    import shutil as _sh
    exe = _sh.which("gltf-transform", path=env.get("PATH")) or _sh.which("npx")
    assert exe is not None, "gltf-transform CLI missing from env PATH"
    attempts = []
    for tex_sz in (1024, 512, 256):
        run_fn([
            exe, "optimize", str(raw_glb), str(lod1_glb),
            "--compress", "meshopt", "--meshopt-level", "high", "--sparse", "true",
            "--texture-compress", "webp", "--texture-size", str(tex_sz),
            "--simplify", "false", "--weld", "false",
        ], env=env)
        sz = lod1_glb.stat().st_size if lod1_glb.is_file() else 0
        attempts.append({"texture_size": tex_sz, "bytes": sz})
        if 0 < sz <= 1_200_000:
            break
    final_bytes = lod1_glb.stat().st_size if lod1_glb.is_file() else 0
    return {
        "lod1_glb": str(lod1_glb),
        "bytes": final_bytes,
        "mb": round(final_bytes / (1024.0 * 1024.0), 3),
        "under_1_2mb_budget": bool(0 < final_bytes <= 1_200_000),
        "attempts": attempts,
    }


def _build_all8_qa_sheet_and_scorecard(
    plan_items: list[dict[str, Any]],
    results_by_sku: dict[str, dict[str, Any]],
    out_dir: Path,
    canary_aborted: bool = False,
) -> tuple[Path, Path]:
    """Emit unified 4x2 PHASE3_ALL8_QA_SHEET.png + PHASE3_GATE_SCORECARD.json."""
    from PIL import Image, ImageDraw  # type: ignore

    out_dir.mkdir(parents=True, exist_ok=True)
    tile_w, tile_h = 540, 750
    cols, rows = 4, 2
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), (14, 15, 19))
    draw = ImageDraw.Draw(sheet)

    scorecard_items: list[dict[str, Any]] = []
    for idx, item in enumerate(plan_items[: cols * rows]):
        sku = item["sku"]
        res = results_by_sku.get(sku, {"sku": sku, "status": "NOT_RUN" if canary_aborted else "MISSING"})
        x0 = (idx % cols) * tile_w
        y0 = (idx // cols) * tile_h
        still_p = Path(res.get("hero_still", "")) if res.get("hero_still") else Path("")
        if still_p.is_file():
            with Image.open(still_p) as im:
                sheet.paste(im.convert("RGB").resize((tile_w, tile_h)), (x0, y0))
        else:
            draw.rectangle([(x0 + 8, y0 + 8), (x0 + tile_w - 8, y0 + tile_h - 8)], outline=(64, 68, 78), width=2)

        status = res.get("status", "UNKNOWN")
        banner_col = (22, 101, 52) if status in ("PASSED", "SKIPPED_CHECKPOINTED") else ((153, 27, 27) if status == "FAILED" else (63, 63, 70))
        draw.rectangle([(x0, y0), (x0 + tile_w, y0 + 44)], fill=banner_col)
        draw.text((x0 + 12, y0 + 14), f"{sku}  [{status}]", fill=(248, 250, 252))

        c0 = res.get("c0_gates", {})
        b2 = res.get("b2_metrics", {})
        glb = res.get("glb_pack", {})
        b1 = res.get("b1_report", {})
        draw.rectangle([(x0, y0 + tile_h - 82), (x0 + tile_w, y0 + tile_h)], fill=(12, 13, 17))
        l1_txt = (
            f"GWN={c0.get('GG31_GWN', {}).get('inside_count', '-')} "
            f"STR={c0.get('G_STRETCH', {}).get('p99_ratio', '-')}/{c0.get('G_STRETCH', {}).get('max_ratio', '-')} "
            f"COL_in={c0.get('O7_COLLAR', {}).get('collar_inside', '-')} "
            f"GLB={glb.get('mb', '-')}MB"
        )
        l2_txt = (
            f"PSNR={b2.get('psnr_db', '-')} SSIM={b2.get('ssim', '-')} "
            f"LPIPS={b2.get('lpips', '-')} N={b2.get('n_splats', '-')} "
            f"IoU={b1.get('cam0_iou_gate', {}).get('cam0_iou', '-')} SOG={b2.get('splat_transform', {}).get('sog_mb', '-')}MB"
        )
        draw.text((x0 + 12, y0 + tile_h - 68), l1_txt, fill=(228, 228, 231))
        draw.text((x0 + 12, y0 + tile_h - 38), l2_txt, fill=(161, 161, 170))
        scorecard_items.append(res)

    qa_png = out_dir / "PHASE3_ALL8_QA_SHEET.png"
    sheet.save(qa_png)
    scorecard_json = out_dir / "PHASE3_GATE_SCORECARD.json"
    scorecard_payload = {
        "schema": "maximus.v2.phase3_gate_scorecard/1",
        "canary_aborted": canary_aborted,
        "total_queued": len(plan_items),
        "passed_count": sum(1 for r in scorecard_items if r.get("status") in ("PASSED", "SKIPPED_CHECKPOINTED")),
        "failed_count": sum(1 for r in scorecard_items if r.get("status") == "FAILED"),
        "skus": scorecard_items,
        "qa_sheet": str(qa_png),
    }
    scorecard_json.write_text(json.dumps(scorecard_payload, indent=2) + "\n", encoding="utf-8")
    return qa_png, scorecard_json


def _run_single_sku_colab(
    item: dict[str, Any],
    local_stage: Path,
    master_blend: Path,
    py: Path,
    blender_exe: Path,
    env: dict[str, str],
    work_sku_dir: Path,
    run_fn: Any,
    c0_only: bool = False,
) -> dict[str, Any]:
    """Execute full 6-stage SolverVBD + GLB<=1.2MB + 120-cam OptiX + garment-only 3DGS + Vulkan SOG + Morph Pack.

    c0_only=True (canary): stop after stage 2 (extract + C0 SolverVBD + LayerCollider PBD + G_LAYER_ORDER gates). The C0 gate
    report, solved NPZ and extract NPZ are copied to certified/ even when the gates fail (then the failure is re-raised)."""
    import shutil as _sh

    sku = item["sku"]
    primary_cw = item["colorways"][0]
    work_sku_dir.mkdir(parents=True, exist_ok=True)
    src_blend = local_stage / "repaired_masters" / Path(item["source_blend"]).name
    if not src_blend.is_file():
        src_blend = Path(item["source_blend"])
    assert src_blend.is_file(), f"Missing staged SKU master blend: {src_blend}"

    base_trouser = local_stage / "base_wardrobe" / "BASE_TROUSER_OBSIDIAN.blend"
    base_shoe = local_stage / "base_wardrobe" / "BASE_SHOE_MINIMAL.blend"
    extract_dir = work_sku_dir / "extract"
    c0_dir = work_sku_dir / "c0"
    b1_dir = work_sku_dir / "b1_b2"
    cert_dir = work_sku_dir / "certified"
    for d in (extract_dir, c0_dir, b1_dir, cert_dir):
        d.mkdir(parents=True, exist_ok=True)

    extract_npz = extract_dir / f"{sku}_extract.npz"
    solved_npz = c0_dir / "c0_solved.npz"
    raw_glb = work_sku_dir / f"maximus_v2_{sku}_raw.glb"
    lod1_glb = cert_dir / f"maximus_v2_{sku}_lod1.glb"

    # 1. Extract theta_fit-aligned garment + canonical head-to-toe bottoms/footwear
    ext_cmd = [
        str(blender_exe), "-b", str(master_blend), "--factory-startup", "--python-exit-code", "3",
        "--python", "scripts/run_phase3_sku_blender.py", "--",
        "--mode", "extract", "--sku", sku, "--sku-blend", str(src_blend),
        "--base-trouser-blend", str(base_trouser), "--base-shoe-blend", str(base_shoe),
        "--hero-npz", "inputs/hero_ringer_shell.npz", "--theta-fit-json", "inputs/m_theta_fit.json",
        "--extract-npz", str(extract_npz), "--out-dir", str(extract_dir),
    ]
    if item.get("dress_with_base_bottoms", False):
        ext_cmd.append("--dress-base-bottoms")
    if item.get("include_base_footwear", True):
        ext_cmd.append("--include-base-footwear")
    run_fn(ext_cmd, env=env)

    # 2. C0 SolverVBD + PBD strain & GWN clearance gates
    c0_cmd = [
        str(py), "scripts/run_phase3_c0_sku.py",
        "--extract-npz", str(extract_npz), "--out-dir", str(c0_dir), "--sku", sku, "--n-frames", str(C0_DRAPE_FRAMES),
    ]
    if c0_only:
        c0_exc: Exception | None = None
        try:
            run_fn(c0_cmd, env=env)
        except Exception as exc:  # gates failed or solver error: still promote whatever evidence exists
            c0_exc = exc
        for p in (c0_dir / "C0_GATE_REPORT.json", c0_dir / "c0_solved.npz", extract_npz):
            if p.is_file():
                _sh.copy2(p, cert_dir / p.name)
        if c0_exc is not None:
            raise c0_exc
        c0_report = json.loads((c0_dir / "C0_GATE_REPORT.json").read_text(encoding="utf-8"))
        res = {"sku": sku, "wave": item["wave"], "mode": "C0_CANARY", "status": "C0_PASS" if c0_report["newton_vbd_gates"].get("c0_all_pass") else "C0_FAIL",
               "vbd_run": c0_report.get("vbd_run"), "newton_vbd_gates": c0_report["newton_vbd_gates"]}
        (cert_dir / "C0_CANARY_COMPLETE.json").write_text(json.dumps(res, indent=2, default=str) + "\n", encoding="utf-8")
        return res
    run_fn(c0_cmd, env=env)
    c0_report = json.loads((c0_dir / "C0_GATE_REPORT.json").read_text(encoding="utf-8"))
    c0_gates = c0_report["newton_vbd_gates"]

    # 3. B1 garment surface seeding (120k) + 120-cam schedule + 3D bone callout anchors
    run_fn([
        str(py), "scripts/run_b1_b2_known_camera_splat.py",
        "--extract-npz", str(extract_npz), "--solved-npz", str(solved_npz),
        "--sku", sku, "--colorway", primary_cw, "--n-points", "120000", "--cap-max", "135000",
        "--out-dir", str(b1_dir), "--n-views", "120", "--strategy", "mcmc", "--phase", "b1",
    ], env=env)

    # 4. Blender GLB export, 720x1000 hero still, & 120-cam OptiX + Object-ID masks (Cam-0 IoU >= 0.98)
    ren_cmd = [
        str(blender_exe), "-b", str(master_blend), "--factory-startup", "--python-exit-code", "3",
        "--python", "scripts/run_phase3_sku_blender.py", "--",
        "--mode", "render_and_glb", "--cycles-device", "OPTIX", "--samples", "48",
        "--sku", sku, "--sku-blend", str(src_blend),
        "--base-trouser-blend", str(base_trouser), "--base-shoe-blend", str(base_shoe),
        "--hero-npz", "inputs/hero_ringer_shell.npz", "--theta-fit-json", "inputs/m_theta_fit.json",
        "--extract-npz", str(extract_npz), "--solved-npz", str(solved_npz),
        "--transforms-json", str(b1_dir / "transforms_cv.json"),
        "--raw-glb", str(raw_glb), "--out-dir", str(b1_dir),
    ]
    if item.get("dress_with_base_bottoms", False):
        ren_cmd.append("--dress-base-bottoms")
    if item.get("include_base_footwear", True):
        ren_cmd.append("--include-base-footwear")
    run_fn(ren_cmd, env=env)
    b1_report = json.loads((b1_dir / "B1_120CAM_REPORT.json").read_text(encoding="utf-8"))

    # 5. Meshopt GLB optimization strictly <= 1.2 MB
    glb_pack = _pack_meshopt_glb_under_1_2mb(raw_glb, lod1_glb, env, run_fn)
    (cert_dir / "GLB_PACK_REPORT.json").write_text(json.dumps(glb_pack, indent=2) + "\n", encoding="utf-8")

    # 6. B2 garment-only 3DGS MCMC (cap_max=135k), Vulkan SOG, exact=True WebP & 100k paired morph pack
    run_fn([
        str(py), "scripts/run_b1_b2_known_camera_splat.py",
        "--extract-npz", str(extract_npz), "--solved-npz", str(solved_npz),
        "--sku", sku, "--colorway", primary_cw, "--n-points", "120000", "--cap-max", "135000",
        "--max-steps", "6000", "--out-dir", str(b1_dir), "--n-views", "120", "--strategy", "mcmc", "--phase", "b2", "--train-live",
    ], env=env)
    b1_b2_plan = json.loads((b1_dir / "B1_B2_PLAN.json").read_text(encoding="utf-8"))
    b2_metrics = b1_b2_plan["b2_metrics"]

    # Gather compact certified deliverables into cert_dir (dropping raw renders/tar after staging)
    gs_out = b1_dir / "gsplat_out"
    for p in list(gs_out.glob("*")) + [
        c0_dir / "C0_GATE_REPORT.json",
        b1_dir / "B1_120CAM_REPORT.json",
        b1_dir / "B1_B2_PLAN.json",
        b1_dir / "still_01_front.png",
        b1_dir / "cam0_iou_diag.png",
        b1_dir / f"{sku}_callout_anchors.json",
    ]:
        if p.is_file() and p.suffix.lower() != ".ply" and p.name != f"{sku}_frozen_xyz_ckpt.pt":
            _sh.copy2(p, cert_dir / p.name)
    # Remove heavy intermediate images/masks/tar (~170 MB) so /content disk stays clean across 8 SKUs
    for heavy in (b1_dir / "images", b1_dir / "masks", b1_dir / "b1_120cam_dataset.tar", raw_glb, gs_out / "splats.ply"):
        if heavy.is_dir():
            _sh.rmtree(heavy, ignore_errors=True)
        elif heavy.is_file():
            heavy.unlink(missing_ok=True)

    c0_ok = bool(c0_gates.get("c0_all_pass", False))
    glb_ok = bool(glb_pack.get("under_1_2mb_budget", False))
    iou_ok = bool(b1_report.get("cam0_iou_gate", {}).get("pass", False))
    gs_ok = bool(b2_metrics.get("gates", {}).get("all_3dgs_gates_pass", False))
    all_gates_pass = bool(c0_ok and glb_ok and iou_ok and gs_ok)
    status = "PASSED" if all_gates_pass else "FAILED"

    complete_payload = {
        "sku": sku,
        "wave": item["wave"],
        "status": status,
        "all_gates_pass": all_gates_pass,
        "c0_ok": c0_ok,
        "glb_ok": glb_ok,
        "iou_ok": iou_ok,
        "gs_ok": gs_ok,
        "staged_blend": str(src_blend),
        "colorways": item["colorways"],
        "hero_still": str(cert_dir / "still_01_front.png"),
        "c0_gates": c0_gates,
        "glb_pack": glb_pack,
        "b1_report": b1_report,
        "b2_metrics": b2_metrics,
    }
    (cert_dir / "SKU_COMPLETE.json").write_text(json.dumps(complete_payload, indent=2) + "\n", encoding="utf-8")
    if not all_gates_pass:
        raise RuntimeError(
            f"[{sku}] Hard gate failed: c0_ok={c0_ok}, glb_ok={glb_ok}, iou_ok={iou_ok}, gs_ok={gs_ok}"
        )
    return complete_payload


def execute_on_colab_l4(
    waves: list[str],
    include_quarantined: bool = False,
    allowed_skus: set[str] | None = None,
    explicit_skus_order: list[str] | None = None,
    c0_only: bool = False,
) -> None:
    try:
        from google.colab import drive, runtime  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Colab runtime packages missing; run with --dry-run on local CPU.") from exc

    import os
    import traceback
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from COLAB_MEGA_SPRINT_CELL import (  # type: ignore
        _run,
        atomic_promote_to_drive,
        bootstrap_part1,
        ensure_drive_mounted,
        install_gsplat_cached_or_build,
        stage_to_local_nvme,
    )

    drive.mount("/content/drive", force_remount=False)
    results_by_sku: dict[str, dict[str, Any]] = {}
    canary_aborted = False
    try:
        local_stage, master_blend, surface_atlas, _ = stage_to_local_nvme()
        assert master_blend.is_file() and surface_atlas.is_file(), "Missing M4_canon_atlas assets on Drive"
        os.chdir(local_stage)
        _run(["sha256sum", "-c", "MANIFEST_SHA256.txt"])
        _run(["nvidia-smi"])
        py, env, blender_exe = bootstrap_part1(local_stage)
        if not c0_only:  # the C0 canary needs Newton/warp + Blender only, not gsplat (saves up to 30 min of L4)
            gsplat_ok = install_gsplat_cached_or_build(py, env, timeout_sec=1800)
            assert gsplat_ok, "Cached gsplat wheel unavailable or build timed out"

        plan = build_execution_plan(
            waves,
            include_quarantined=include_quarantined,
            allowed_skus=allowed_skus,
            explicit_skus_order=explicit_skus_order,
        )
        plan_local = Path("/content/work/PHASE3/PHASE3_EXECUTION_PLAN.json")
        plan_local.parent.mkdir(parents=True, exist_ok=True)
        plan_local.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        atomic_promote_to_drive(plan_local, DRIVE_RESULTS / "PHASE3_EXECUTION_PLAN.json", "PHASE3_PLAN")

        for item in plan["items"]:
            sku = item["sku"]
            if not c0_only and _is_sku_checkpointed(sku):
                ckpt_json = DRIVE_RESULTS / sku / "SKU_COMPLETE.json"
                prior = json.loads(ckpt_json.read_text(encoding="utf-8"))
                prior["status"] = "SKIPPED_CHECKPOINTED"
                results_by_sku[sku] = prior
                print(f"[PHASE3] {sku}: already verified in Drive ledger, skipping.", flush=True)
                continue

            sku_work_dir = Path("/content/work/PHASE3") / sku
            cert_dir = sku_work_dir / "certified"
            try:
                cert_dir.mkdir(parents=True, exist_ok=True)
                res = _run_single_sku_colab(item, local_stage, master_blend, py, blender_exe, env, sku_work_dir, _run, c0_only=c0_only)
                dest_root = (DRIVE_RESULTS / "C0_CANARY") if c0_only else DRIVE_RESULTS
                for p in sorted(cert_dir.glob("*")):
                    if p.is_file() and not p.name.endswith(".tmp"):
                        atomic_promote_to_drive(p, dest_root / sku / p.name, f"PHASE3_{sku}")
                results_by_sku[sku] = res
            except Exception as sku_exc:
                err_file = cert_dir / "SKU_ERROR.json"
                cert_dir.mkdir(parents=True, exist_ok=True)
                err_payload = {"sku": sku, "status": "FAILED", "error": str(sku_exc), "traceback": traceback.format_exc()}
                err_file.write_text(json.dumps(err_payload, indent=2) + "\n", encoding="utf-8")
                dest_root = (DRIVE_RESULTS / "C0_CANARY") if c0_only else DRIVE_RESULTS
                for p in sorted(cert_dir.glob("*")):
                    if p.is_file() and not p.name.endswith(".tmp"):
                        atomic_promote_to_drive(p, dest_root / sku / p.name, f"PHASE3_{sku}_ERR")
                partial_res = {"sku": sku, "wave": item["wave"], "status": "FAILED", "error": str(sku_exc)}
                if (cert_dir / "SKU_COMPLETE.json").is_file():
                    partial_res = json.loads((cert_dir / "SKU_COMPLETE.json").read_text(encoding="utf-8"))
                    partial_res["error"] = str(sku_exc)
                results_by_sku[sku] = partial_res
                print(f"[PHASE3 ISOLATED ERROR] {sku} failed ({sku_exc}).", flush=True)
                if sku == "BS-RINGER-001":
                    canary_aborted = True
                    print("[CANARY FAIL-FAST] BS-RINGER-001 failed gates; aborting remaining 7 SKUs cleanly.", flush=True)
                    break

        if c0_only:
            canary_json = Path("/content/work/PHASE3/C0_CANARY_SUMMARY.json")
            canary_json.write_text(json.dumps(results_by_sku, indent=2, default=str) + "\n", encoding="utf-8")
            atomic_promote_to_drive(canary_json, DRIVE_RESULTS / "C0_CANARY" / "C0_CANARY_SUMMARY.json", "PHASE3_C0_CANARY")
            if canary_aborted or any(r.get("status") != "C0_PASS" for r in results_by_sku.values()):
                raise RuntimeError("C0 canary failed; inspect C0_CANARY/<sku>/C0_GATE_REPORT.json on Drive.")
            return
        qa_png, sc_json = _build_all8_qa_sheet_and_scorecard(
            plan["items"], results_by_sku, Path("/content/work/PHASE3"), canary_aborted=canary_aborted
        )
        atomic_promote_to_drive(qa_png, DRIVE_RESULTS / "PHASE3_ALL8_QA_SHEET.png", "PHASE3_QA")
        atomic_promote_to_drive(sc_json, DRIVE_RESULTS / "PHASE3_GATE_SCORECARD.json", "PHASE3_QA")
        if canary_aborted:
            raise RuntimeError("Canary BS-RINGER-001 failed hard gates; inspect PHASE3_GATE_SCORECARD.json.")
    finally:
        try:
            ensure_drive_mounted()
            drive.flush_and_unmount()
        except Exception:
            pass
        runtime.unassign()


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 3 Full-Catalog Colab L4 Multi-Wave Dressing Orchestrator")
    ap.add_argument("--waves", nargs="+", default=LIVE_DEFAULT_WAVES, choices=WAVE_CHOICES)
    ap.add_argument("--skus", nargs="+", default=None, help="Optional comma- or space-separated SKU filter (e.g. BS-RINGER-001,BS-POLO-001,BS-CAMP-001,BS-CAMP-002)")
    ap.add_argument("--include-quarantined", action="store_true", help="Include Wave 3D quarantined BS-TRACK-001.")
    ap.add_argument("--dry-run", action="store_true", help="Validate all local inputs, waves, and runtime locally without GPU.")
    ap.add_argument("--c0-only", action="store_true",
                    help="C0 canary: extract + SolverVBD + G_LAYER_ORDER only (no gsplat install, renders, GLB or splats). "
                         "Combine with --skus BS-RINGER-001.")
    args = ap.parse_args()
    if args.c0_only and not args.skus:
        ap.error("--c0-only is a canary mode: name the SKU(s) with --skus (e.g. --skus BS-RINGER-001)")

    waves_for_plan, allowed_skus, explicit_skus = resolve_selection(args.waves, args.skus)
    plan = build_execution_plan(
        waves_for_plan,
        include_quarantined=args.include_quarantined,
        allowed_skus=allowed_skus,
        explicit_skus_order=explicit_skus,
    )
    if args.dry_run:
        path_check = verify_local_inputs(waves_for_plan, include_quarantined=args.include_quarantined, allowed_skus=allowed_skus)
        runtime_est = estimate_l4_runtime(plan["items"])
        out = {
            "mode": "C0_CANARY (extract + SolverVBD + G_LAYER_ORDER only)" if args.c0_only else "FULL_PIPELINE",
            "dry_run_status": "PASS" if path_check["missing_count"] == 0 else "FAIL_MISSING_INPUTS",
            "local_input_verification": path_check,
            "l4_runtime_estimate": runtime_est,
            "plan": plan,
        }
        print(json.dumps(out, indent=2))
        return 0 if path_check["missing_count"] == 0 else 2

    execute_on_colab_l4(
        waves_for_plan,
        include_quarantined=args.include_quarantined,
        allowed_skus=allowed_skus,
        explicit_skus_order=explicit_skus,
        c0_only=args.c0_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
