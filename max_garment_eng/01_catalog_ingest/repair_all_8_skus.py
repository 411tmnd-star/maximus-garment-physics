"""repair_all_8_skus.py — CPU Local Repair & Parametric Master Builder for All 8 Live SKUs (Zero bpy.ops, $0 GPU).

Outputs cleanly into 02_engines/max_garment_eng/01_catalog_ingest/repaired_masters/:
  1. BS-RINGER-001_cotton280_master.blend      (280 GSM combed cotton jersey; MAT_BRUSHED_MOHAIR_ARCHIVE untouched)
  2. BS-POLO-001_clean_master.blend            (regularized edges + 0-degen rigid buttons + theta_fit aligned)
  3. BS-CAMP-001_clean_master.blend            (v3 revers collar + 0-degen rigid buttons + art_center_y=0.47)
  4. BS-CAMP-002_clean_master.blend            (v3 revers collar + seamless tartan_tile UV + Black / Ochre)
  5. BS-SLV-001_clean_master.blend             (POM-02/03/04 fixed + clean UVMap + PBR deboss/ridge, 0 extruded flame verts)
  6. BS-BSTSET-001_clean_master.blend          (separated shirt + trouser + clean trouser UV + 0 inside verts + 16,838 stones manifest)
  7. BS-TANK-001_parametric_master.blend       (parametric ribbed tank shell from Ringer torso + 2x2 rib normal map)
  8. BS-DENIMSHIRT-001_parametric_master.blend (parametric 12 oz long-sleeve button-down denim shirt on theta_fit)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy  # type: ignore
import bmesh  # type: ignore
import numpy as np
from mathutils import Vector  # type: ignore

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _garment_repair_lib import (  # noqa: E402
    HERO_RINGER_NPZ,
    REPO_ROOT,
    append_objects_from_blend,
    apply_remesh_or_decimate_depsgraph,
    ensure_clean_panel_uv,
    extract_button_centers,
    finalize_clean_garment_mesh,
    load_theta_fit_body,
    make_pbr_material,
    pack_image_datablock,
    push_out_mesh_from_body,
    rebuild_clean_buttons,
    reset_empty_scene,
    save_blend,
    signed_clearance_and_inside_bvh,
)

OUT_DIR = THIS_DIR / "repaired_masters"
TEX_DIR = OUT_DIR / "textures"
BS_ENGINE = REPO_ROOT / "02_engines" / "better_style_engine"
COLLAR_CLONE = BS_ENGINE / "02_engine_3d" / "isolated_clones" / "better_style_catalog_collar"
COMPILED_UVS = COLLAR_CLONE / "assets" / "compiled_uvs"
ART_CAMP = COLLAR_CLONE / "assets" / "artwork_camp"
ARCHETYPES = COLLAR_CLONE / "assets" / "archetypes"
SLV_ROOT = BS_ENGINE / "02_engine_3d" / "garment_engine" / "BS-SLV-001"
STONE_OUT = BS_ENGINE / "03_governance_and_evidence" / "better_style_execution_workspace" / "03_ENGINE" / "rhinestone_engine" / "garments" / "out" / "BS-BSTSET-001"
STONE_WORK = BS_ENGINE / "03_governance_and_evidence" / "better_style_execution_workspace" / "03_ENGINE" / "rhinestone_engine" / "garments" / "work"
ELEPHANT_BLUE_BLEND = COLLAR_CLONE / "out" / "concept_visualizations" / "blend" / "SET-ELEPHANT-BLUE.blend"
TANK_SRC = BS_ENGINE / "01_web_storefront" / "catalogue" / "sources" / "BS-TANK-001"
DENIM_SRC = BS_ENGINE / "01_web_storefront" / "catalogue" / "sources" / "BS-DENIMSHIRT-001"


def _copy_object_preserve_clean(name: str, verts: np.ndarray, faces: np.ndarray) -> bpy.types.Object:
    me = bpy.data.meshes.new(name + "_Mesh")
    me.from_pydata([tuple(v) for v in verts.tolist()], [], [tuple(f) for f in faces.tolist()])
    me.update()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _assign_uv_array_from_tri_uvs(me: bpy.types.Mesh, uv_name: str, uv_arr: np.ndarray) -> None:
    """Assign either per-loop (len(loops), 2) or per-vertex (len(verts), 2) UVs onto `me`."""
    uv_layer = me.uv_layers.new(name=uv_name)
    arr = np.asarray(uv_arr, dtype=np.float32)
    if len(arr) == len(me.loops):
        uv_layer.data.foreach_set("uv", arr.reshape(-1).tolist())
        return
    data = uv_layer.data
    for poly in me.polygons:
        for li in poly.loop_indices:
            vi = me.loops[li].vertex_index
            data[li].uv = (float(arr[vi, 0]), float(arr[vi, 1]))


def _uv_valid(me: bpy.types.Mesh) -> bool:
    if not me.uv_layers:
        return False
    uv = me.uv_layers[0].data
    if len(uv) == 0:
        return False
    coords = np.empty(len(uv) * 2, dtype=np.float64)
    uv.foreach_get("uv", coords)
    if not np.all(np.isfinite(coords)):
        return False
    coords = coords.reshape(-1, 2)
    span = coords.max(axis=0) - coords.min(axis=0)
    if float(span.min()) < 1e-4:
        return False
    return True


def _keep_largest_component(me: bpy.types.Mesh) -> None:
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    visited = set()
    comps = []
    for seed in bm.verts:
        if seed.index in visited:
            continue
        comp = []
        st = [seed]
        visited.add(seed.index)
        while st:
            v = st.pop()
            comp.append(v)
            for e in v.link_edges:
                ov = e.other_vert(v)
                if ov.index not in visited:
                    visited.add(ov.index)
                    st.append(ov)
        comps.append(comp)
    if len(comps) > 1:
        comps.sort(key=len, reverse=True)
        drop_verts = [v for comp in comps[1:] for v in comp]
        bmesh.ops.delete(bm, geom=drop_verts, context="VERTS")
        bm.to_mesh(me)
    bm.free()
    me.update()


def _align_garment_group_xy_to_body(objs: list[bpy.types.Object], body_ctx: dict[str, Any]) -> list[float]:
    """Align garment group's chest X/Y center cleanly onto theta_fit torso X/Y center if offset > 4mm."""
    all_pts = []
    for o in objs:
        V = np.empty(len(o.data.vertices) * 3, dtype=np.float64)
        o.data.vertices.foreach_get("co", V)
        all_pts.append(V.reshape(-1, 3))
    pts = np.vstack(all_pts)
    bV = body_ctx["V"]
    chest_b = bV[(bV[:, 2] >= 1.15) & (bV[:, 2] <= 1.38) & (np.abs(bV[:, 0]) < 0.19)]
    chest_g = pts[(pts[:, 2] >= 1.15) & (pts[:, 2] <= 1.38) & (np.abs(pts[:, 0]) < 0.22)]
    if len(chest_g) < 20 or len(chest_b) < 20:
        return [0.0, 0.0, 0.0]
    dx = float(np.mean(chest_b[:, 0]) - np.mean(chest_g[:, 0]))
    dy = float(np.mean(chest_b[:, 1]) - np.mean(chest_g[:, 1]))
    # Only apply small rigid centering shift if garment is off-center (> 3mm); keep designed Z vertical drape
    shift = np.array([dx if abs(dx) > 0.003 else 0.0, dy if abs(dy) > 0.003 else 0.0, 0.0], dtype=np.float32)
    if np.linalg.norm(shift) > 0.0:
        for o in objs:
            V = np.empty(len(o.data.vertices) * 3, dtype=np.float32)
            o.data.vertices.foreach_get("co", V)
            V = V.reshape(-1, 3) + shift
            o.data.vertices.foreach_set("co", V.ravel())
            o.data.update()
    return [round(float(x), 5) for x in shift.tolist()]


def build_bs_ringer_001(body_ctx: dict[str, Any]) -> dict[str, Any]:
    reset_empty_scene()
    d = np.load(HERO_RINGER_NPZ, allow_pickle=True)
    V_fit = np.asarray(d["V_fit"], dtype=np.float64)
    F = np.asarray(d["F"], dtype=np.int64)
    collar_mask = np.asarray(d["collar_mask"], dtype=bool)
    vg_names = [str(n) for n in d["vg_names"]]
    W_vg = np.asarray(d["W_vg"], dtype=np.float64)

    obj = _copy_object_preserve_clean("BS_RINGER_001", V_fit, F)
    me = obj.data

    # Preserve UVMap_Fabric (also primary UVMap at index 0) + Art UVs
    if "uv|UVMap_Fabric" in d.files:
        _assign_uv_array_from_tri_uvs(me, "UVMap", np.asarray(d["uv|UVMap_Fabric"], dtype=np.float64))
    if "uv|UVMap_Art_Front" in d.files:
        _assign_uv_array_from_tri_uvs(me, "UVMap_Art_Front", np.asarray(d["uv|UVMap_Art_Front"], dtype=np.float64))
    if "uv|UVMap_Art_Back" in d.files:
        _assign_uv_array_from_tri_uvs(me, "UVMap_Art_Back", np.asarray(d["uv|UVMap_Art_Back"], dtype=np.float64))
    if not _uv_valid(me):
        ensure_clean_panel_uv(me, uv_name="UVMap")

    uv_dir = COMPILED_UVS / "BS-RINGER-001"
    tex_light = uv_dir / "front_albedo_light.png"
    tex_dark = uv_dir / "front_albedo_dark.png"
    tex_height = uv_dir / "front_height.png"
    cotton_nrm = TEX_DIR / "cotton_jersey_280gsm_normal.png"
    for req in (tex_light, tex_dark, tex_height, cotton_nrm):
        if not req.is_file():
            raise FileNotFoundError(f"Missing Ringer texture: {req}")

    # Bind multi-slot 280 GSM Combed Cotton Jersey materials BEFORE finalize_clean_garment_mesh so tri_mats survive cleanly
    mat_white = make_pbr_material(
        "MAT_COTTON280_WHITE_BLACK",
        base_rgba=(0.88, 0.87, 0.84, 1.0),
        roughness=0.80,
        sheen=0.06,
        normal_tex=cotton_nrm,
        normal_strength=0.16,
        front_decal_tex=tex_light,
        front_uv_map="UVMap_Art_Front",
    )
    mat_rib_black = make_pbr_material(
        "MAT_COTTON280_RINGER_RIB_BLACK",
        base_rgba=(0.016, 0.017, 0.020, 1.0),
        roughness=0.82,
        sheen=0.06,
        normal_tex=cotton_nrm,
        normal_strength=0.20,
    )
    mat_black = make_pbr_material(
        "MAT_COTTON280_BLACK_WHITE",
        base_rgba=(0.025, 0.026, 0.030, 1.0),
        roughness=0.80,
        sheen=0.08,
        normal_tex=cotton_nrm,
        normal_strength=0.18,
        front_decal_tex=tex_dark,
        front_uv_map="UVMap_Art_Front",
    )
    mat_red = make_pbr_material(
        "MAT_COTTON280_RED_WHITE",
        base_rgba=(0.48, 0.04, 0.06, 1.0),
        roughness=0.78,
        sheen=0.08,
        normal_tex=cotton_nrm,
        normal_strength=0.18,
        front_decal_tex=tex_dark,
        front_uv_map="UVMap_Art_Front",
    )
    for m in (mat_white, mat_rib_black, mat_black, mat_red):
        me.materials.append(m)

    if "tri_mats" in d.files and len(d["tri_mats"]) == len(me.polygons):
        tm = np.asarray(d["tri_mats"], dtype=np.int32)
        slot_ids = np.where(tm == 0, 0, 1).astype(np.int32)
        me.polygons.foreach_set("material_index", slot_ids.tolist())

    # Populate vertex groups
    for gi, gname in enumerate(vg_names):
        vg = obj.vertex_groups.new(name=gname)
        for vi in np.flatnonzero(W_vg[:, gi] > 1e-4):
            vg.add([int(vi)], float(W_vg[vi, gi]), "REPLACE")

    # Gentle clearance clamp ONLY on non-collar inside/grazing vertices so collar stays >= +0.25mm O-7
    finalize_clean_garment_mesh(
        me,
        body_ctx,
        min_clearance=0.0025,
        collar_guard=True,
        max_edge_m=0.024,
        min_edge_m=0.0018,
    )
    if not _uv_valid(me):
        ensure_clean_panel_uv(me, uv_name="UVMap")

    mi_after = np.empty(len(me.polygons), dtype=np.int32)
    me.polygons.foreach_get("material_index", mi_after)
    assert int((mi_after == 0).sum()) > 0 and int((mi_after == 1).sum()) > 0, (
        f"Ringer material slots collapsed after finalize: counts={np.unique(mi_after, return_counts=True)}"
    )

    obj["sku"] = "BS-RINGER-001"
    obj["part_id"] = 1
    obj["fabric_spec"] = "280 GSM combed cotton jersey"
    obj["colorway_presets"] = ["white-black", "black-white", "red-white"]
    obj["texture_refs"] = [str(tex_light), str(tex_dark), str(tex_height), str(cotton_nrm)]

    out_path = OUT_DIR / "BS-RINGER-001_cotton280_master.blend"
    save_blend(out_path)
    return {"path": str(out_path), "verts": len(me.vertices), "faces": len(me.polygons), "textures": list(obj["texture_refs"])}


def _repair_shirt_master_blend(
    src_blend: Path,
    sku: str,
    decimate_tee_ratio: float,
    decimate_collar_ratio: float,
    min_clearance: float,
    mat_builder_fn,
    art_center_y: float | None = None,
) -> dict[str, Any]:
    reset_empty_scene()
    objs = append_objects_from_blend(src_blend, ["Tee", "collar", "placket_overlap", "placket_buttons"])
    by_name = {o.name: o for o in objs}
    tee = by_name["Tee"]
    collar = by_name["collar"]
    placket = by_name["placket_overlap"]
    old_btn = by_name["placket_buttons"]

    # Extract button locations BEFORE deleting degenerate button object
    btn_centers = extract_button_centers(old_btn)
    bpy.data.objects.remove(old_btn, do_unlink=True)

    # Decimate 575k-587k high-poly cloth into regular ~17k-22k sim-ready geometry via depsgraph
    apply_remesh_or_decimate_depsgraph(tee, ratio=decimate_tee_ratio)
    apply_remesh_or_decimate_depsgraph(collar, ratio=decimate_collar_ratio)

    cloth_objs = [tee, collar, placket]
    _align_garment_group_xy_to_body(cloth_objs, load_theta_fit_body_cached())

    body_ctx = load_theta_fit_body_cached()
    for co in cloth_objs:
        finalize_clean_garment_mesh(co.data, body_ctx, min_clearance=min_clearance)
        if not _uv_valid(co.data):
            ensure_clean_panel_uv(co.data, uv_name="UVMap")
        else:
            co.data.uv_layers[0].name = "UVMap"
        co["part_id"] = 1
        co["sku"] = sku
        if art_center_y is not None:
            co["art_center_y"] = float(art_center_y)

    # Align buttons slightly outside placket surface so 0 inside verts
    adj_buttons = []
    for c, n in btn_centers:
        c2 = Vector((c.x, c.y, c.z))
        # Ensure button center sits at least min_clearance + 1mm outside body
        s_arr, n_arr, _ = signed_clearance_and_inside_bvh(np.array([[c2.x, c2.y, c2.z]]), body_ctx)
        if float(s_arr[0]) < (min_clearance + 0.001):
            c2 += Vector(n_arr[0].tolist()) * float((min_clearance + 0.0015) - s_arr[0])
        adj_buttons.append((c2, n))

    btn_mat = make_pbr_material(f"MAT_{sku}_BUTTON", base_rgba=(0.04, 0.04, 0.045, 1.0), roughness=0.32, sheen=0.02)
    buttons_obj = rebuild_clean_buttons(adj_buttons, name="placket_buttons", mat=btn_mat)
    push_out_mesh_from_body(buttons_obj.data, body_ctx, min_clearance=min_clearance, smooth_iters=0)
    buttons_obj["sku"] = sku
    buttons_obj["part_id"] = 1

    tex_refs = mat_builder_fn(tee, collar, placket, buttons_obj)
    return {
        "sku": sku,
        "verts_total": sum(len(o.data.vertices) for o in (tee, collar, placket, buttons_obj)),
        "buttons": len(adj_buttons),
        "textures": tex_refs,
    }


_BODY_CACHE = None


def load_theta_fit_body_cached() -> dict[str, Any]:
    global _BODY_CACHE
    if _BODY_CACHE is None:
        _BODY_CACHE = load_theta_fit_body()
    return _BODY_CACHE


def build_bs_polo_001() -> dict[str, Any]:
    uv_dir = COMPILED_UVS / "BS-POLO-001"
    reqs = [
        uv_dir / "front_albedo_dark.png",
        uv_dir / "front_albedo_light.png",
        uv_dir / "back_albedo_dark.png",
        uv_dir / "back_albedo_light.png",
        uv_dir / "front_height.png",
        uv_dir / "back_height.png",
        TEX_DIR / "cotton_jersey_280gsm_normal.png",
    ]
    for r in reqs:
        if not r.is_file():
            raise FileNotFoundError(f"Missing Polo texture: {r}")

    def _mats(tee, collar, placket, buttons):
        m_bw = make_pbr_material(
            "MAT_POLO_BLACK_WHITE",
            (0.022, 0.023, 0.026, 1.0),
            roughness=0.76,
            normal_tex=reqs[-1],
            normal_strength=0.22,
            front_decal_tex=reqs[0],
            back_decal_tex=reqs[2],
        )
        m_burg = make_pbr_material(
            "MAT_POLO_BURGUNDY_WHITE",
            (0.24, 0.035, 0.06, 1.0),
            roughness=0.76,
            normal_tex=reqs[-1],
            normal_strength=0.22,
            front_decal_tex=reqs[0],
            back_decal_tex=reqs[2],
        )
        m_col = make_pbr_material("MAT_POLO_COLLAR", (0.90, 0.89, 0.86, 1.0), roughness=0.78)
        tee.data.materials.clear()
        tee.data.materials.append(m_bw)
        tee.data.materials.append(m_burg)
        collar.data.materials.clear()
        collar.data.materials.append(m_col)
        placket.data.materials.clear()
        placket.data.materials.append(m_col)
        tee["colorway_presets"] = ["black-white", "burgundy-white"]
        tee["texture_refs"] = [str(p) for p in reqs]
        return [str(p) for p in reqs]

    res = _repair_shirt_master_blend(
        ARCHETYPES / "rugby_polo_master.blend",
        "BS-POLO-001",
        decimate_tee_ratio=0.032,
        decimate_collar_ratio=0.080,
        min_clearance=0.0030,
        mat_builder_fn=_mats,
    )
    out_path = OUT_DIR / "BS-POLO-001_clean_master.blend"
    save_blend(out_path)
    res["path"] = str(out_path)
    return res


def build_bs_camp_001() -> dict[str, Any]:
    uv_dir = COMPILED_UVS / "BS-CAMP-001"
    reqs = [
        uv_dir / "front_albedo_dark.png",
        uv_dir / "front_albedo_light.png",
        uv_dir / "back_albedo_dark.png",
        uv_dir / "back_albedo_light.png",
        ART_CAMP / "lattice_tile.png",
        ART_CAMP / "back_script.png",
    ]
    for r in reqs:
        if not r.is_file():
            raise FileNotFoundError(f"Missing Camp-001 texture: {r}")

    def _mats(tee, collar, placket, buttons):
        presets = [
            ("blue", (0.06, 0.16, 0.36, 1.0), reqs[0], reqs[2]),
            ("green", (0.09, 0.22, 0.14, 1.0), reqs[0], reqs[2]),
            ("ivory", (0.86, 0.83, 0.77, 1.0), reqs[1], reqs[3]),
            ("gold", (0.72, 0.56, 0.24, 1.0), reqs[0], reqs[2]),
            ("taupe", (0.44, 0.39, 0.34, 1.0), reqs[0], reqs[2]),
        ]
        tee.data.materials.clear()
        for cname, rgba, ftx, btx in presets:
            tee.data.materials.append(
                make_pbr_material(
                    f"MAT_CAMP001_{cname.upper()}",
                    rgba,
                    roughness=0.72,
                    sheen=0.14,
                    front_decal_tex=ftx,
                    back_decal_tex=btx,
                )
            )
        m_trim = make_pbr_material("MAT_CAMP001_REVERS_COLLAR", (0.06, 0.16, 0.36, 1.0), roughness=0.70)
        collar.data.materials.clear()
        collar.data.materials.append(m_trim)
        placket.data.materials.clear()
        placket.data.materials.append(m_trim)
        tee["colorway_presets"] = [c for c, _, _, _ in presets]
        tee["texture_refs"] = [str(p) for p in reqs]
        return [str(p) for p in reqs]

    res = _repair_shirt_master_blend(
        ARCHETYPES / "camp_shirt_master.blend",
        "BS-CAMP-001",
        decimate_tee_ratio=0.032,
        decimate_collar_ratio=0.075,
        min_clearance=0.0030,
        mat_builder_fn=_mats,
        art_center_y=0.47,
    )
    out_path = OUT_DIR / "BS-CAMP-001_clean_master.blend"
    save_blend(out_path)
    res["path"] = str(out_path)
    return res


def build_bs_camp_002() -> dict[str, Any]:
    uv_dir = COMPILED_UVS / "BS-CAMP-002"
    reqs = [
        ART_CAMP / "tartan_tile.png",
        uv_dir / "front_albedo_dark.png",
        uv_dir / "back_albedo_dark.png",
        uv_dir / "front_height.png",
    ]
    for r in reqs:
        if not r.is_file():
            raise FileNotFoundError(f"Missing Camp-002 texture: {r}")

    def _mats(tee, collar, placket, buttons):
        m_plaid = make_pbr_material(
            "MAT_CAMP002_BLACK_OCHRE",
            (0.04, 0.04, 0.045, 1.0),
            roughness=0.74,
            sheen=0.15,
            base_albedo_tex=reqs[0],
            base_uv_scale=4.0,
            front_decal_tex=reqs[1],
            back_decal_tex=reqs[2],
        )
        m_collar = make_pbr_material("MAT_CAMP002_COLLAR_OCHRE", (0.68, 0.50, 0.20, 1.0), roughness=0.72)
        tee.data.materials.clear()
        tee.data.materials.append(m_plaid)
        collar.data.materials.clear()
        collar.data.materials.append(m_collar)
        placket.data.materials.clear()
        placket.data.materials.append(m_collar)
        tee["colorway_presets"] = ["black-plaid"]
        tee["verified_label"] = "Black / Ochre"
        tee["colour_family"] = "Black"
        tee["texture_refs"] = [str(p) for p in reqs]
        return [str(p) for p in reqs]

    res = _repair_shirt_master_blend(
        ARCHETYPES / "camp_shirt_master.blend",
        "BS-CAMP-002",
        decimate_tee_ratio=0.032,
        decimate_collar_ratio=0.075,
        min_clearance=0.0030,
        mat_builder_fn=_mats,
        art_center_y=0.47,
    )
    out_path = OUT_DIR / "BS-CAMP-002_clean_master.blend"
    save_blend(out_path)
    res["path"] = str(out_path)
    return res


def build_bs_slv_001() -> dict[str, Any]:
    reset_empty_scene()
    body_ctx = load_theta_fit_body_cached()
    objs = append_objects_from_blend(SLV_ROOT / "out" / "BS-SLV-001_shell.blend", ["BS-SLV-001"])
    obj = objs[0]
    me = obj.data

    # Fix POM-02 (front len -> 0.735m), POM-03 (back len -> 0.747m), POM-04 (shoulder width -> 0.520m)
    # Measured raw: shoulder_w = 0.4937m (+0.0263m needed), front Z span = 0.7793m (-0.0443m), back Z span = 0.7837m (-0.0367m)
    V = np.empty(len(me.vertices) * 3, dtype=np.float64)
    me.vertices.foreach_get("co", V)
    V = V.reshape(-1, 3)

    # 1. Shoulder width POM-04 correction to 0.520m on shoulder cap region (Z >= 1.36, |X| > 0.14)
    sh_mask = (V[:, 2] >= 1.37) & (V[:, 2] <= 1.48)
    cur_sh_w = float(V[sh_mask, 0].max() - V[sh_mask, 0].min()) if sh_mask.any() else 0.494
    target_sh_w = 0.520
    sh_scale = target_sh_w / max(cur_sh_w, 1e-4)
    if abs(sh_scale - 1.0) > 1e-3:
        z_sh_w = np.clip((V[:, 2] - 1.24) / (1.42 - 1.24), 0.0, 1.0)
        x_outer_w = np.clip((np.abs(V[:, 0]) - 0.10) / 0.12, 0.0, 1.0)
        w_sh = (z_sh_w * z_sh_w * (3.0 - 2.0 * z_sh_w)) * x_outer_w
        V[:, 0] *= (1.0 + (sh_scale - 1.0) * w_sh)

    # 2. Hem length POM-02 (front=0.735m) & POM-03 (back=0.747m) keeping neck/collar anchored at Z_max
    z_top = float(V[:, 2].max())
    front_m = V[:, 1] < -0.02
    back_m = V[:, 1] > 0.02
    cur_front_len = float(z_top - V[front_m, 2].min())
    cur_back_len = float(z_top - V[back_m, 2].min())
    f_ratio = 0.735 / max(cur_front_len, 1e-4)
    b_ratio = 0.747 / max(cur_back_len, 1e-4)
    # Smoothly blend front vs back vertical scaling based on Y coordinate so left/right seams stay C1 continuous
    side_blend = np.clip((-V[:, 1] + 0.05) / 0.10, 0.0, 1.0)  # 1 at front, 0 at back
    vert_scale = b_ratio * (1.0 - side_blend) + f_ratio * side_blend
    # Apply vertical warp progressively below shoulder/chest anchor (Z < 1.32m) so collar/armhole stays invariant!
    anchor_w = np.clip((1.35 - V[:, 2]) / (1.35 - V[:, 2].min()), 0.0, 1.0)
    z_from_top = z_top - V[:, 2]
    V[:, 2] = z_top - z_from_top * (1.0 + (vert_scale - 1.0) * anchor_w)

    me.vertices.foreach_set("co", V.astype(np.float32).ravel())
    me.update()

    _align_garment_group_xy_to_body([obj], body_ctx)
    finalize_clean_garment_mesh(me, body_ctx, min_clearance=0.0025)
    ensure_clean_panel_uv(me, uv_name="UVMap")

    # Bind Acid-Wash Grey PBR + Deboss (-0.3mm) + Silicone HD Ridge (+0.8mm) bump chain (0 extruded flame verts!)
    art = SLV_ROOT / "artwork"
    reqs = [
        art / "flames_front_deboss_fill.png",
        art / "flames_front_hd_ridge.png",
        art / "shield_bst_base_twill.png",
        art / "shield_bst_frame.png",
    ]
    for r in reqs:
        if not r.is_file():
            raise FileNotFoundError(f"Missing BS-SLV-001 artwork: {r}")

    mat = bpy.data.materials.new(name="MAT_SLV_ACID_GREY_DEBOSS")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.16, 0.17, 0.19, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.80
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    # Combine deboss (-0.3mm) + relief (+0.8mm) in shader bump nodes
    tex_deb = nt.nodes.new("ShaderNodeTexImage")
    tex_deb.image = bpy.data.images.load(str(reqs[0].resolve()), check_existing=True)
    tex_deb.image.colorspace_settings.name = "Non-Color"
    pack_image_datablock(tex_deb.image)
    tex_hd = nt.nodes.new("ShaderNodeTexImage")
    tex_hd.image = bpy.data.images.load(str(reqs[1].resolve()), check_existing=True)
    tex_hd.image.colorspace_settings.name = "Non-Color"
    pack_image_datablock(tex_hd.image)

    b1 = nt.nodes.new("ShaderNodeBump")
    b1.inputs["Strength"].default_value = 0.45
    b1.inputs["Distance"].default_value = -0.0003  # -0.3 mm deboss
    b2 = nt.nodes.new("ShaderNodeBump")
    b2.inputs["Strength"].default_value = 0.65
    b2.inputs["Distance"].default_value = 0.0008   # +0.8 mm silicone relief
    nt.links.new(tex_deb.outputs["Color"], b1.inputs["Height"])
    nt.links.new(b1.outputs["Normal"], b2.inputs["Normal"])
    nt.links.new(tex_hd.outputs["Color"], b2.inputs["Height"])
    nt.links.new(b2.outputs["Normal"], bsdf.inputs["Normal"])

    me.materials.clear()
    me.materials.append(mat)

    V_chk = np.empty(len(me.vertices) * 3, dtype=np.float64)
    me.vertices.foreach_get("co", V_chk)
    V_chk = V_chk.reshape(-1, 3)
    sh_chk = V_chk[(V_chk[:, 2] >= 1.37) & (V_chk[:, 2] <= 1.48)]
    pom_sh = float(sh_chk[:, 0].max() - sh_chk[:, 0].min()) if sh_chk.any() else 0.0
    pom_fl = float(V_chk[:, 2].max() - V_chk[V_chk[:, 1] < -0.02, 2].min())
    pom_bl = float(V_chk[:, 2].max() - V_chk[V_chk[:, 1] > 0.02, 2].min())

    obj["sku"] = "BS-SLV-001"
    obj["part_id"] = 1
    obj["extruded_flame_verts"] = 0
    obj["pom_measured"] = {"shoulder_m": round(pom_sh, 4), "front_len_m": round(pom_fl, 4), "back_len_m": round(pom_bl, 4)}
    obj["texture_refs"] = [str(r) for r in reqs]
    obj["colorway_presets"] = ["grey"]

    out_path = OUT_DIR / "BS-SLV-001_clean_master.blend"
    save_blend(out_path)
    return {"path": str(out_path), "verts": len(me.vertices), "faces": len(me.polygons), "pom": dict(obj["pom_measured"]), "extruded_flame_verts": 0}


def build_bs_bstset_001() -> dict[str, Any]:
    reset_empty_scene()
    body_ctx = load_theta_fit_body_cached()

    # 1. Clean separated Shirt from camp_rebuilt.blend
    shirt_objs = append_objects_from_blend(STONE_WORK / "camp_rebuilt.blend", ["Tee", "collar", "placket_overlap", "placket_buttons"])
    by_n = {o.name: o for o in shirt_objs}
    tee = by_n["Tee"]
    collar = by_n["collar"]
    placket = by_n["placket_overlap"]
    old_btn = by_n["placket_buttons"]
    btn_centers = extract_button_centers(old_btn)
    bpy.data.objects.remove(old_btn, do_unlink=True)

    apply_remesh_or_decimate_depsgraph(tee, ratio=0.032)
    apply_remesh_or_decimate_depsgraph(collar, ratio=0.075)
    tee.name = "BS_BSTSET_001_Shirt"
    collar.name = "BS_BSTSET_001_Collar"
    placket.name = "BS_BSTSET_001_Placket"

    _align_garment_group_xy_to_body([tee, collar, placket], body_ctx)
    for so in (tee, collar, placket):
        finalize_clean_garment_mesh(so.data, body_ctx, min_clearance=0.0035)
        if not _uv_valid(so.data):
            ensure_clean_panel_uv(so.data, uv_name="UVMap")
        else:
            so.data.uv_layers[0].name = "UVMap"
        so["part_id"] = 1
        so["sku"] = "BS-BSTSET-001"

    m_btn = make_pbr_material("MAT_BSTSET_BUTTON", (0.04, 0.04, 0.05, 1.0), roughness=0.28)
    btn_obj = rebuild_clean_buttons(btn_centers, name="BS_BSTSET_001_Buttons", mat=m_btn)
    push_out_mesh_from_body(btn_obj.data, body_ctx, min_clearance=0.0035, smooth_iters=0)
    btn_obj["sku"] = "BS-BSTSET-001"
    btn_obj["part_id"] = 1

    # 2. Clean separated Trousers from SET-ELEPHANT-BLUE.blend (8,401 verts -> clean UV + zero overlap vs theta_fit)
    tr_objs = append_objects_from_blend(ELEPHANT_BLUE_BLEND, ["SET-ELEPHANT-BLUE-TROUSERS__body"])
    trouser = tr_objs[0]
    trouser.name = "BS_BSTSET_001_Trousers"
    # Note: SET-ELEPHANT trousers were slightly wide/forward; center X/Y onto pelvis of theta_fit first!
    Vt = np.empty(len(trouser.data.vertices) * 3, dtype=np.float64)
    trouser.data.vertices.foreach_get("co", Vt)
    Vt = Vt.reshape(-1, 3)
    b_pelvis = body_ctx["V"][(body_ctx["V"][:, 2] >= 0.80) & (body_ctx["V"][:, 2] <= 1.05) & (np.abs(body_ctx["V"][:, 0]) < 0.175)]
    tr_pelvis = Vt[(Vt[:, 2] >= 0.80) & (Vt[:, 2] <= 1.05)]
    if len(tr_pelvis) and len(b_pelvis):
        Vt[:, 0] += float(np.mean(b_pelvis[:, 0]) - np.mean(tr_pelvis[:, 0]))
        Vt[:, 1] += float(np.mean(b_pelvis[:, 1]) - np.mean(tr_pelvis[:, 1]))
        trouser.data.vertices.foreach_set("co", Vt.astype(np.float32).ravel())
        trouser.data.update()

    finalize_clean_garment_mesh(trouser.data, body_ctx, min_clearance=0.0040)
    ensure_clean_panel_uv(trouser.data, uv_name="UVMap", split_legs=True)
    trouser["part_id"] = 2
    trouser["sku"] = "BS-BSTSET-001"

    # Separate tonal jacquard weave PBR shader from rigid SO(3)-bound rhinestones
    jacq_tex = TEX_DIR / "denim_12oz_twill_normal.png"
    m_blue = make_pbr_material("MAT_BSTSET_JACQUARD_BLUE", (0.045, 0.12, 0.30, 1.0), roughness=0.68, sheen=0.22, normal_tex=jacq_tex, normal_strength=0.20)
    m_red = make_pbr_material("MAT_BSTSET_JACQUARD_RED", (0.32, 0.04, 0.06, 1.0), roughness=0.68, sheen=0.22, normal_tex=jacq_tex, normal_strength=0.20)
    for co in (tee, collar, placket, trouser):
        co.data.materials.clear()
        co.data.materials.append(m_blue)
        co.data.materials.append(m_red)

    # Verify rigid stone files
    s_npz = STONE_OUT / "BS-BSTSET-001_shirt_stones.npz"
    t_npz = STONE_OUT / "BS-BSTSET-001_trousers_stones.npz"
    sd = np.load(s_npz, allow_pickle=True)
    td = np.load(t_npz, allow_pickle=True)
    n_shirt_stones = int(len(sd["P"]))
    n_tr_stones = int(len(td["P"]))
    assert n_shirt_stones + n_tr_stones == 16838, f"Expected 16838 stones, got {n_shirt_stones + n_tr_stones}"

    stones_manifest = {
        "schema": "maximus.v2.bstset_stones_manifest/1",
        "sku": "BS-BSTSET-001",
        "binding_mode": "SO(3)_triangle_frame_EXT_mesh_gpu_instancing",
        "shirt_stones_npz": str(s_npz),
        "trouser_stones_npz": str(t_npz),
        "shirt_stones_count": n_shirt_stones,
        "trouser_stones_count": n_tr_stones,
        "stones_total": n_shirt_stones + n_tr_stones,
        "keys": list(sd.files),
    }
    manifest_path = OUT_DIR / "BS-BSTSET-001_stones_manifest.json"
    manifest_path.write_text(json.dumps(stones_manifest, indent=2) + "\n", encoding="utf-8")

    tee["colorway_presets"] = ["blue", "red"]
    tee["stones_manifest"] = str(manifest_path)
    tee["texture_refs"] = [str(jacq_tex), str(s_npz), str(t_npz)]

    out_path = OUT_DIR / "BS-BSTSET-001_clean_master.blend"
    save_blend(out_path)
    return {
        "path": str(out_path),
        "shirt_verts": len(tee.data.vertices),
        "trouser_verts": len(trouser.data.vertices),
        "buttons": len(btn_centers),
        "stones_total": 16838,
        "stones_manifest": str(manifest_path),
    }


def build_bs_tank_001() -> dict[str, Any]:
    reset_empty_scene()
    body_ctx = load_theta_fit_body_cached()
    d = np.load(HERO_RINGER_NPZ, allow_pickle=True)
    V_fit = np.asarray(d["V_fit"], dtype=np.float64)
    F = np.asarray(d["F"], dtype=np.int64)

    # Filter faces cleanly into sculpted ribbed tank shell (no sleeves, wide athletic armholes, scooped front/back neckline)
    bm = bmesh.new()
    verts = [bm.verts.new(tuple(v)) for v in V_fit.tolist()]
    for f in F.tolist():
        tri = (verts[f[0]], verts[f[1]], verts[f[2]])
        c = (tri[0].co + tri[1].co + tri[2].co) / 3.0
        # Drop sleeves & shape athletic tapered armhole edge
        armhole_frac = max(0.0, min(1.0, (c.z - 1.24) / (1.46 - 1.24)))
        x_lim = 0.172 - 0.042 * math.sin(math.pi * 0.5 * armhole_frac)
        if c.z > 1.32 and abs(c.x) > x_lim:
            continue
        if abs(c.x) > 0.185:
            continue
        # Scoop front neckline (y < 0) & upper back neckline (y >= 0) while preserving shoulder straps (|x| >= 0.088)
        if c.y < 0.0 and abs(c.x) < 0.088:
            z_neck_f = 1.375 + 0.115 * ((abs(c.x) / 0.088) ** 2)
            if c.z > z_neck_f:
                continue
        if c.y >= 0.0 and abs(c.x) < 0.082:
            z_neck_b = 1.418 + 0.080 * ((abs(c.x) / 0.082) ** 2)
            if c.z > z_neck_b:
                continue
        try:
            bm.faces.new(tri)
        except ValueError:
            pass

    bmesh.ops.delete(bm, geom=[v for v in bm.verts if len(v.link_faces) == 0], context="VERTS")
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    me = bpy.data.meshes.new("BS_TANK_001_Mesh")
    bm.to_mesh(me)
    bm.free()

    _keep_largest_component(me)
    finalize_clean_garment_mesh(me, body_ctx, min_clearance=0.0028)
    ensure_clean_panel_uv(me, uv_name="UVMap")

    rib_nrm = TEX_DIR / "tank_2x2_rib_normal.png"
    ref_blk = TANK_SRC / "tank_black_front_v3.png"
    ref_wht = TANK_SRC / "tank_white_front_v3.png"
    for r in (rib_nrm, ref_blk, ref_wht):
        if not r.is_file():
            raise FileNotFoundError(f"Missing Tank texture/ref: {r}")

    m_blk = make_pbr_material("MAT_TANK_2X2_BLACK", (0.025, 0.026, 0.030, 1.0), roughness=0.76, sheen=0.10, normal_tex=rib_nrm, normal_strength=0.45)
    m_wht = make_pbr_material("MAT_TANK_2X2_WHITE", (0.91, 0.90, 0.87, 1.0), roughness=0.76, sheen=0.10, normal_tex=rib_nrm, normal_strength=0.45)
    me.materials.append(m_blk)
    me.materials.append(m_wht)

    obj = bpy.data.objects.new("BS_TANK_001", me)
    bpy.context.scene.collection.objects.link(obj)
    obj["sku"] = "BS-TANK-001"
    obj["part_id"] = 1
    obj["fabric_spec"] = "Parametric 2x2 rib-knit cotton"
    obj["colorway_presets"] = ["black", "white"]
    obj["texture_refs"] = [str(rib_nrm), str(ref_blk), str(ref_wht)]

    out_path = OUT_DIR / "BS-TANK-001_parametric_master.blend"
    save_blend(out_path)
    return {"path": str(out_path), "verts": len(me.vertices), "faces": len(me.polygons)}


def _radial_ring_around_centerline(pts_3d: np.ndarray, axis_dir: np.ndarray, origin: np.ndarray, n_bins: int, ease: float, min_r: float) -> np.ndarray:
    """Create smooth closed ring perpendicular to arm axis around body arm vertices."""
    u = np.cross(axis_dir, np.array([0.0, 1.0, 0.0]))
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(axis_dir, np.array([0.0, 0.0, 1.0]))
    u /= np.linalg.norm(u)
    v = np.cross(axis_dir, u)
    v /= np.linalg.norm(v)

    thetas = np.linspace(0.0, 2.0 * np.pi, n_bins, endpoint=False)
    radii = np.full(n_bins, min_r, dtype=np.float64)
    if len(pts_3d):
        rel = pts_3d - origin[None, :]
        cu = rel @ u
        cv = rel @ v
        ang = np.mod(np.arctan2(cv, cu), 2.0 * np.pi)
        dist = np.hypot(cu, cv)
        bw = (2.0 * np.pi) / n_bins
        for k, th in enumerate(thetas):
            da = np.abs(np.arctan2(np.sin(ang - th), np.cos(ang - th)))
            m = da <= 1.3 * bw
            if m.any():
                radii[k] = max(radii[k], float(dist[m].max()) + ease)
            else:
                radii[k] = max(radii[k], float(np.median(dist)) + ease)
    mean_r = float(np.mean(radii))
    radii = np.maximum(radii, 0.55 * mean_r + 0.45 * radii)
    for _ in range(3):
        sm = 0.25 * np.roll(radii, 1) + 0.5 * radii + 0.25 * np.roll(radii, -1)
        radii = np.maximum(sm, radii)
    return np.array([origin + radii[k] * (math.cos(th) * u + math.sin(th) * v) for k, th in enumerate(thetas)], dtype=np.float64)


def build_bs_denimshirt_001() -> dict[str, Any]:
    reset_empty_scene()
    body_ctx = load_theta_fit_body_cached()
    bV = body_ctx["V"]

    # Build structured heavyweight 12 oz long-sleeve button-down denim shirt around theta_fit:
    # 1. Main torso + hip overshirt shell (Z in [0.755..1.475], 28 rings x 44 bins around torso-only |x|<0.185)
    bm = bmesh.new()
    n_torso_bins = 44
    z_torso = np.linspace(0.760, 1.475, 28)
    torso_bV = bV[np.abs(bV[:, 0]) < 0.185]
    torso_rings = []
    for zi, z in enumerate(z_torso):
        sl = torso_bV[np.abs(torso_bV[:, 2] - z) < 0.028]
        if len(sl) < 10:
            sl = torso_bV[np.abs(torso_bV[:, 2] - z) < 0.06]
        cx = float(np.mean(sl[:, 0]))
        cy = float(np.mean(sl[:, 1]))
        thetas = np.linspace(0.0, 2.0 * np.pi, n_torso_bins, endpoint=False)
        radii = np.full(n_torso_bins, 0.10, dtype=np.float64)
        ease = 0.011 + 0.005 * (1.0 - zi / max(1, len(z_torso) - 1))
        dx, dy = sl[:, 0] - cx, sl[:, 1] - cy
        ang = np.mod(np.arctan2(dy, dx), 2.0 * np.pi)
        dist = np.hypot(dx, dy)
        bw = (2.0 * np.pi) / n_torso_bins
        for k, th in enumerate(thetas):
            da = np.abs(np.arctan2(np.sin(ang - th), np.cos(ang - th)))
            m = da <= 1.3 * bw
            radii[k] = float(dist[m].max()) + ease if m.any() else float(np.median(dist)) + ease
        # Neck taper cleanly at top ring
        for _ in range(3):
            radii = np.maximum(0.25 * np.roll(radii, 1) + 0.5 * radii + 0.25 * np.roll(radii, -1), radii)
        ring_v = [bm.verts.new((float(cx + radii[k] * math.cos(th)), float(cy + radii[k] * math.sin(th)), float(z))) for k, th in enumerate(thetas)]
        torso_rings.append(ring_v)

    for ri in range(len(torso_rings) - 1):
        r0, r1 = torso_rings[ri], torso_rings[ri + 1]
        for k in range(n_torso_bins):
            k2 = (k + 1) % n_torso_bins
            bm.faces.new((r0[k], r0[k2], r1[k2], r1[k]))

    # 2. Structured Point Collar stand + fold around neck (Z in [1.465..1.545], 5 rings x 32 bins)
    neck_V = bV[(bV[:, 2] >= 1.455) & (bV[:, 2] <= 1.545) & (np.abs(bV[:, 0]) < 0.095)]
    ncx, ncy = float(np.mean(neck_V[:, 0])), float(np.mean(neck_V[:, 1]))
    n_col_bins = 32
    col_rings = []
    for ci, (z_c, r_extra) in enumerate([(1.465, 0.020), (1.490, 0.014), (1.518, 0.015), (1.535, 0.022), (1.505, 0.032)]):
        sl_n = neck_V[np.abs(neck_V[:, 2] - min(z_c, 1.53)) < 0.025]
        thetas = np.linspace(0.0, 2.0 * np.pi, n_col_bins, endpoint=False)
        base_r = np.percentile(np.hypot(sl_n[:, 0] - ncx, sl_n[:, 1] - ncy), 95) if len(sl_n) else 0.065
        # Pointed collar drop at front (sin(theta) < -0.5) on outer folded rim
        ring_c = []
        for th in thetas:
            front_point = max(0.0, -math.sin(th) - 0.35) if ci == 4 else 0.0
            rad = float(base_r + r_extra)
            zv = z_c - 0.018 * front_point
            ring_c.append(bm.verts.new((float(ncx + rad * math.cos(th)), float(ncy + rad * math.sin(th)), float(zv))))
        col_rings.append(ring_c)
    for ri in range(len(col_rings) - 1):
        r0, r1 = col_rings[ri], col_rings[ri + 1]
        for k in range(n_col_bins):
            k2 = (k + 1) % n_col_bins
            bm.faces.new((r0[k], r0[k2], r1[k2], r1[k]))

    # 3. Full Long Sleeves + Rolled Cuffs down left & right theta_fit arms from shoulder to wrists (Z=0.875..1.385)
    n_arm_bins = 28
    for sign in (-1.0, 1.0):
        arm_V = bV[(bV[:, 0] * sign >= 0.185) & (bV[:, 2] >= 0.84) & (bV[:, 2] <= 1.40)]
        z_arm = np.linspace(0.875, 1.385, 34)
        centers = []
        for z in z_arm:
            sl = arm_V[np.abs(arm_V[:, 2] - z) < 0.032]
            if len(sl) < 4:
                sl = arm_V[np.abs(arm_V[:, 2] - z) < 0.065]
            centers.append(np.median(sl, axis=0))
        centers = np.array(centers, dtype=np.float64)
        arm_rings = []
        for ai, z in enumerate(z_arm):
            sl = arm_V[np.abs(arm_V[:, 2] - z) < 0.032]
            if len(sl) < 4:
                sl = arm_V[np.abs(arm_V[:, 2] - z) < 0.065]
            dist_c = np.linalg.norm(sl - centers[ai][None, :], axis=1)
            sl_tight = sl[dist_c <= 0.080]
            if len(sl_tight) >= 4:
                sl = sl_tight
            if ai == 0:
                ax_dir = centers[1] - centers[0]
            elif ai == len(z_arm) - 1:
                ax_dir = centers[-1] - centers[-2]
            else:
                ax_dir = centers[ai + 1] - centers[ai - 1]
            ax_dir /= max(float(np.linalg.norm(ax_dir)), 1e-8)
            is_cuff = ai <= 3
            ease_arm = 0.014 if is_cuff else 0.012
            pts_ring = _radial_ring_around_centerline(sl, ax_dir, centers[ai], n_arm_bins, ease=ease_arm, min_r=0.044)
            rel = pts_ring - centers[ai][None, :]
            rl = np.linalg.norm(rel, axis=1, keepdims=True)
            pts_ring = centers[ai][None, :] + rel * np.minimum(1.0, 0.075 / np.maximum(rl, 1e-6))
            arm_rings.append([bm.verts.new(tuple(p)) for p in pts_ring.tolist()])
        for ri in range(len(arm_rings) - 1):
            r0, r1 = arm_rings[ri], arm_rings[ri + 1]
            for k in range(n_arm_bins):
                k2 = (k + 1) % n_arm_bins
                bm.faces.new((r0[k], r0[k2], r1[k2], r1[k]))

    # 4. Left Chest Pocket Patch (subdivided 5x5 grid so zero 50mm spanning edges + conforms to chest!)
    pk_z = np.linspace(1.21, 1.32, 5)
    pk_x = np.linspace(0.065, 0.115, 5)
    pk_rows = []
    for pz in pk_z:
        row = []
        for px in pk_x:
            sl_p = bV[(np.abs(bV[:, 0] - px) < 0.025) & (np.abs(bV[:, 2] - pz) < 0.025) & (bV[:, 1] < 0.0)]
            py = float(sl_p[:, 1].min() - 0.022) if len(sl_p) else -0.160
            row.append(bm.verts.new((float(px), py, float(pz))))
        pk_rows.append(row)
    for ri in range(len(pk_rows) - 1):
        for ci in range(len(pk_rows[ri]) - 1):
            bm.faces.new((pk_rows[ri][ci], pk_rows[ri][ci + 1], pk_rows[ri + 1][ci + 1], pk_rows[ri + 1][ci]))

    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    me = bpy.data.meshes.new("BS_DENIMSHIRT_001_Mesh")
    bm.to_mesh(me)
    bm.free()

    finalize_clean_garment_mesh(me, body_ctx, min_clearance=0.0045, max_edge_m=0.022, min_edge_m=0.0020)
    ensure_clean_panel_uv(me, uv_name="UVMap")

    denim_nrm = TEX_DIR / "denim_12oz_twill_normal.png"
    refs = [
        DENIM_SRC / "BS-DENIMSHIRT-001_black_floating_front_alpha.png",
        DENIM_SRC / "BS-DENIMSHIRT-001_black_floating_back_alpha.png",
        DENIM_SRC / "BS-DENIMSHIRT-001_red_floating_front_alpha.png",
        DENIM_SRC / "BS-DENIMSHIRT-001_red_floating_back_alpha.png",
        denim_nrm,
    ]
    for r in refs:
        if not r.is_file():
            raise FileNotFoundError(f"Missing Denim asset: {r}")

    m_blk = make_pbr_material("MAT_DENIM_12OZ_BLACK", (0.026, 0.028, 0.032, 1.0), roughness=0.78, sheen=0.08, normal_tex=denim_nrm, normal_strength=0.35)
    m_red = make_pbr_material("MAT_DENIM_12OZ_RED", (0.40, 0.045, 0.055, 1.0), roughness=0.78, sheen=0.08, normal_tex=denim_nrm, normal_strength=0.35)
    me.materials.append(m_blk)
    me.materials.append(m_red)

    obj = bpy.data.objects.new("BS_DENIMSHIRT_001", me)
    bpy.context.scene.collection.objects.link(obj)
    obj["sku"] = "BS-DENIMSHIRT-001"
    obj["part_id"] = 1
    obj["fabric_spec"] = "12 oz heavyweight selvedge twill denim"
    obj["colorway_presets"] = ["black", "red"]
    obj["texture_refs"] = [str(r) for r in refs]

    # Add 7 matte enamel snap buttons down front placket via rigid SO(3) group builder
    btn_list = []
    for bz in np.linspace(0.86, 1.43, 7):
        sl_f = bV[(np.abs(bV[:, 0]) < 0.02) & (np.abs(bV[:, 2] - bz) < 0.02) & (bV[:, 1] < 0.0)]
        py = float(sl_f[:, 1].min() - 0.019) if len(sl_f) else -0.160
        btn_list.append((Vector((0.0, py, float(bz))), Vector((0.0, -1.0, 0.0))))
    m_snap = make_pbr_material("MAT_DENIM_SNAP_ENAMEL", (0.04, 0.04, 0.045, 1.0), roughness=0.25, metallic=0.35)
    snap_obj = rebuild_clean_buttons(btn_list, name="BS_DENIMSHIRT_001_Buttons", mat=m_snap, radius=0.0064)
    push_out_mesh_from_body(snap_obj.data, body_ctx, min_clearance=0.0045, smooth_iters=0)
    snap_obj["sku"] = "BS-DENIMSHIRT-001"
    snap_obj["part_id"] = 1

    out_path = OUT_DIR / "BS-DENIMSHIRT-001_parametric_master.blend"
    save_blend(out_path)
    return {"path": str(out_path), "verts": len(me.vertices), "faces": len(me.polygons), "buttons": len(btn_list)}


BUILDERS = {
    "BS-RINGER-001": lambda: build_bs_ringer_001(load_theta_fit_body_cached()),
    "BS-POLO-001": build_bs_polo_001,
    "BS-CAMP-001": build_bs_camp_001,
    "BS-CAMP-002": build_bs_camp_002,
    "BS-SLV-001": build_bs_slv_001,
    "BS-BSTSET-001": build_bs_bstset_001,
    "BS-TANK-001": build_bs_tank_001,
    "BS-DENIMSHIRT-001": build_bs_denimshirt_001,
}


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--sku", nargs="+", default=list(BUILDERS.keys()), choices=list(BUILDERS.keys()))
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for sku in args.sku:
        summary[sku] = BUILDERS[sku]()
    if set(args.sku) == set(BUILDERS.keys()):
        manifest_path = OUT_DIR / "repaired_sku_manifest.json"
        manifest_path.write_text(json.dumps(summary, indent=2, default=lambda x: dict(x) if hasattr(x, "keys") else list(x)) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=lambda x: dict(x) if hasattr(x, "keys") else list(x)))


if __name__ == "__main__":
    main()
