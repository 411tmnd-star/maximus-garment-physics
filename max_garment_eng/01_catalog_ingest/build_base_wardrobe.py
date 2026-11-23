"""build_base_wardrobe.py — Build Canonical Head-to-Toe Base Bottoms for Maximus v2 (Zero bpy.ops).

Emits into 02_engines/max_garment_eng/01_catalog_ingest/base_wardrobe/:
  - BASE_TROUSER_OBSIDIAN.blend  (tailored straight-leg obsidian trousers, part_id=2, zero penetration, valid UVMap)
  - BASE_SHOE_MINIMAL.blend      (minimalist architectural footwear, Z_sole = 0.000 m, part_id=3, valid UVMap)

Run with Blender 5.2.3 LTS headless:
  blender.exe --background --factory-startup --python 02_engines/max_garment_eng/01_catalog_ingest/build_base_wardrobe.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy  # type: ignore
import bmesh  # type: ignore
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _garment_repair_lib import (  # noqa: E402
    clean_topology_and_outward_normals,
    ensure_clean_panel_uv,
    load_theta_fit_body,
    make_pbr_material,
    push_out_mesh_from_body,
    regularize_cloth_edges,
    reset_empty_scene,
    save_blend,
)

BASE_DIR = THIS_DIR / "base_wardrobe"


def _radial_envelope(pts_xy: np.ndarray, cx: float, cy: float, n_bins: int, ease: float, min_r: float, floor_blend: float = 0.55) -> np.ndarray:
    """Compute smooth star-shaped radial ring (n_bins, 2) enclosing `pts_xy` around `(cx, cy)`."""
    thetas = np.linspace(0.0, 2.0 * np.pi, n_bins, endpoint=False)
    radii = np.full(n_bins, min_r, dtype=np.float64)
    if len(pts_xy) > 0:
        dx = pts_xy[:, 0] - cx
        dy = pts_xy[:, 1] - cy
        ang = np.mod(np.arctan2(dy, dx), 2.0 * np.pi)
        dist = np.hypot(dx, dy)
        bin_w = (2.0 * np.pi) / n_bins
        for k in range(n_bins):
            t0 = thetas[k]
            d_ang = np.abs(np.arctan2(np.sin(ang - t0), np.cos(ang - t0)))
            m = d_ang <= 1.25 * bin_w
            if m.any():
                radii[k] = max(radii[k], float(dist[m].max()) + ease)
            else:
                radii[k] = max(radii[k], float(np.median(dist)) + ease)
    # Enforce tailored drape smoothness by blending toward circular envelope while never violating max radius
    mean_r = float(np.mean(radii))
    tailored = np.maximum(radii, floor_blend * mean_r + (1.0 - floor_blend) * radii)
    # Circular Gaussian blur twice (without shrinking below `radii`)
    smooth = tailored.copy()
    for _ in range(3):
        smooth = 0.25 * np.roll(smooth, 1) + 0.50 * smooth + 0.25 * np.roll(smooth, -1)
        smooth = np.maximum(smooth, radii)
    return np.column_stack([cx + smooth * np.cos(thetas), cy + smooth * np.sin(thetas)])


def build_base_trouser_obsidian(body_ctx: dict) -> dict:
    reset_empty_scene()
    body_V = body_ctx["V"]
    # Strictly exclude hanging arms (|x| >= 0.175) when sampling leg/hip/waist body hull
    mask_central = np.abs(body_V[:, 0]) <= 0.175
    central_V = body_V[mask_central]

    bm = bmesh.new()
    n_leg_bins = 40  # per leg
    z_legs = np.linspace(0.088, 0.775, 44)
    z_waist = np.linspace(0.815, 0.995, 16)

    leg_rings_L: list[list[bmesh.types.BMVert]] = []
    leg_rings_R: list[list[bmesh.types.BMVert]] = []

    # Build L (x < 0) and R (x > 0) straight-leg trouser tubes
    for zi, z in enumerate(z_legs):
        frac_up = zi / max(1, len(z_legs) - 1)
        # Straight tailored leg: looser at calf/ankle (+18mm), snugger near thigh (+10mm)
        ease_leg = 0.018 * (1.0 - 0.42 * frac_up)
        sl_L = central_V[(np.abs(central_V[:, 2] - z) < 0.025) & (central_V[:, 0] < -0.003)]
        sl_R = central_V[(np.abs(central_V[:, 2] - z) < 0.025) & (central_V[:, 0] > 0.003)]
        if len(sl_L) < 6:
            sl_L = central_V[(np.abs(central_V[:, 2] - z) < 0.055) & (central_V[:, 0] < 0.0)]
        if len(sl_R) < 6:
            sl_R = central_V[(np.abs(central_V[:, 2] - z) < 0.055) & (central_V[:, 0] > 0.0)]

        cx_L = float(np.mean(sl_L[:, 0])) if len(sl_L) else -0.095
        cy_L = float(np.mean(sl_L[:, 1])) if len(sl_L) else -0.015
        cx_R = float(np.mean(sl_R[:, 0])) if len(sl_R) else 0.095
        cy_R = float(np.mean(sl_R[:, 1])) if len(sl_R) else -0.015

        fb_leg = 0.65 - 0.35 * max(0.0, (float(z) - 0.66) / (0.775 - 0.66))
        ring_L_xy = _radial_envelope(sl_L[:, :2], cx_L, cy_L, n_leg_bins, ease=ease_leg, min_r=0.056, floor_blend=fb_leg)
        ring_R_xy = _radial_envelope(sl_R[:, :2], cx_R, cy_R, n_leg_bins, ease=ease_leg, min_r=0.056, floor_blend=fb_leg)
        # Prevent midline overlap between left & right thighs
        ring_L_xy[:, 0] = np.minimum(ring_L_xy[:, 0], -0.003)
        ring_R_xy[:, 0] = np.maximum(ring_R_xy[:, 0], 0.003)

        vL = [bm.verts.new((float(x), float(y), float(z))) for x, y in ring_L_xy]
        vR = [bm.verts.new((float(x), float(y), float(z))) for x, y in ring_R_xy]
        leg_rings_L.append(vL)
        leg_rings_R.append(vR)

    for ri in range(len(leg_rings_L) - 1):
        r0L, r1L = leg_rings_L[ri], leg_rings_L[ri + 1]
        r0R, r1R = leg_rings_R[ri], leg_rings_R[ri + 1]
        for k in range(n_leg_bins):
            k2 = (k + 1) % n_leg_bins
            bm.faces.new((r0L[k], r0L[k2], r1L[k2], r1L[k]))
            bm.faces.new((r0R[k], r0R[k2], r1R[k2], r1R[k]))

    # Crotch transition saddle: bridge top of L ring (28 verts) + top of R ring (28 verts)
    # Wait: even cleaner manifold bridge: trace top of L around outer side and top of R around outer side!
    # At top of legs: bin k=0 is angle 0 (+X, inner for L, outer for R).
    # For Left leg (cx < 0): k=0 is inner-right (+X toward midline), k=n//4 is back (+Y), k=n//2 is outer-left (-X), k=3n//4 is front (-Y).
    # For Right leg (cx > 0): k=0 is outer-right (+X), k=n//4 is back (+Y), k=n//2 is inner-left (-X toward midline), k=3n//4 is front (-Y).
    top_L = leg_rings_L[-1]
    top_R = leg_rings_R[-1]
    q = n_leg_bins // 4  # 7 (since n_leg_bins=28)

    # Shared figure-8 waist ring vertices at z = 0.815: size = n_leg_bins (28 -> 2 * (28//2) outer loop)
    # Build first waist ring (0.815) with exact 28 angular slots aligned symmetrically so inner edges stitch to crotch saddle!
    # Left outer half: top_L indices q .. 3*q inclusive (15 vertices from back-top to front-top via outer-left -X)
    # Right outer half: top_R indices 3*q .. n+q inclusive mod n (15 vertices from front-top to back-top via outer-right +X)
    # Inner L arc: top_L indices 3*q .. n+q (15 vertices from front-L to back-L through midline-L)
    # Inner R arc: top_R indices q .. 3*q (15 vertices from back-R to front-R through midline-R)
    # Stitching inner L (i=0..14) directly to reversed inner R (i=14..0) closes the crotch saddle cleanly in one zipper strip!
    inner_L = [top_L[(3 * q + i) % n_leg_bins] for i in range(2 * q + 1)]  # front-L (3q) -> midline -> back-L (q)
    inner_R_fwd = [top_R[(q + i) % n_leg_bins] for i in range(2 * q + 1)]  # back-R (q) -> midline -> front-R (3q)
    inner_R_match = list(reversed(inner_R_fwd))  # front-R (3q) -> midline -> back-R (q)

    # Create central crotch seam bridge vertices midway between inner_L and inner_R_match (excluding endpoints which connect to waist!)
    # Actually, zipping inner_L directly to inner_R_match with a single smooth crotch centerline row of verts at x=0 avoids any pinched quads:
    crotch_mid = []
    for i in range(2 * q + 1):
        pL = inner_L[i].co
        pR = inner_R_match[i].co
        z_saddle = 0.788 - 0.010 * math.sin(math.pi * i / (2 * q))
        crotch_mid.append(bm.verts.new((0.0, 0.5 * (pL.y + pR.y), z_saddle)))

    for i in range(2 * q):
        bm.faces.new((inner_L[i], inner_L[i + 1], crotch_mid[i + 1], crotch_mid[i]))
        bm.faces.new((crotch_mid[i], crotch_mid[i + 1], inner_R_match[i + 1], inner_R_match[i]))

    # Now trace the unified 56-vertex perimeter around the top of the united crotch saddle:
    # From front-crotch (crotch_mid[0]) -> front-L (inner_L[0]=top_L[3q]) -> outer-L -> back-L (top_L[q]) -> back-crotch (crotch_mid[-1])
    # -> back-R (top_R[q]) -> outer-R -> front-R (top_R[3q]) -> back to front-crotch!
    left_outer_arc = [top_L[(q + i) % n_leg_bins] for i in range(2 * q + 1)]   # back-L (q) -> outer-L (2q) -> front-L (3q)
    right_outer_arc = [top_R[(3 * q + i) % n_leg_bins] for i in range(2 * q + 1)]  # front-R (3q) -> outer-R (0) -> back-R (q)
    # Trace strictly CCW seen from +Z top:
    # Start front-mid (0, -y): right_outer_arc (front-R -> +X outer -> back-R) -> back-mid -> left_outer_arc (back-L -> -X outer -> front-L) -> front-mid
    united_bottom = [crotch_mid[0]] + right_outer_arc + [crotch_mid[-1]] + left_outer_arc
    # Notice length = 1 + 15 + 1 + 15 = 32 vertices cleanly! Wait: angles of those 32 vertices ordered CCW from front (-pi/2):
    n_waist_bins = len(united_bottom)
    waist_rings: list[list[bmesh.types.BMVert]] = [united_bottom]

    for wi, z in enumerate(z_waist):
        frac_w = (wi + 1) / float(len(z_waist))
        ease_w = 0.0055 - 0.0015 * frac_w  # Snug tailored hip/waistband (4.0..5.5mm) so top hems drape outside cleanly
        sl_W = central_V[np.abs(central_V[:, 2] - z) < 0.028]
        if len(sl_W) < 10:
            sl_W = central_V[np.abs(central_V[:, 2] - z) < 0.055]
        cx_W = float(np.mean(sl_W[:, 0])) if len(sl_W) else 0.0
        cy_W = float(np.mean(sl_W[:, 1])) if len(sl_W) else -0.025
        # Use target angles matching united_bottom starting at front (-pi/2) CCW
        base_angles = np.linspace(-0.5 * np.pi, 1.5 * np.pi, n_waist_bins, endpoint=False)
        pts_xy = sl_W[:, :2]
        radii = np.full(n_waist_bins, 0.085, dtype=np.float64)
        dx = pts_xy[:, 0] - cx_W
        dy = pts_xy[:, 1] - cy_W
        ang = np.arctan2(dy, dx)
        dist = np.hypot(dx, dy)
        bin_w = (2.0 * np.pi) / n_waist_bins
        for k, th in enumerate(base_angles):
            d_ang = np.abs(np.arctan2(np.sin(ang - th), np.cos(ang - th)))
            m = d_ang <= 1.25 * bin_w
            if m.any():
                radii[k] = max(radii[k], float(dist[m].max()) + ease_w)
            else:
                radii[k] = max(radii[k], float(np.median(dist)) + ease_w)
        smooth_r = radii.copy()
        for _ in range(3):
            smooth_r = 0.25 * np.roll(smooth_r, 1) + 0.50 * smooth_r + 0.25 * np.roll(smooth_r, -1)
            smooth_r = np.maximum(smooth_r, radii)
        w_ring = [
            bm.verts.new((float(cx_W + smooth_r[k] * math.cos(th)), float(cy_W + smooth_r[k] * math.sin(th)), float(z)))
            for k, th in enumerate(base_angles)
        ]
        waist_rings.append(w_ring)

    for ri in range(len(waist_rings) - 1):
        r0, r1 = waist_rings[ri], waist_rings[ri + 1]
        for k in range(n_waist_bins):
            k2 = (k + 1) % n_waist_bins
            bm.faces.new((r0[k], r0[k2], r1[k2], r1[k]))

    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    me = bpy.data.meshes.new("BASE_TROUSER_OBSIDIAN_Mesh")
    bm.to_mesh(me)
    bm.free()

    mat = make_pbr_material(
        "MAT_BASE_TROUSER_OBSIDIAN",
        base_rgba=(0.012, 0.013, 0.016, 1.0),
        roughness=0.86,
        sheen=0.06,
    )
    me.materials.append(mat)
    obj = bpy.data.objects.new("BASE_TROUSER_OBSIDIAN", me)
    bpy.context.scene.collection.objects.link(obj)
    obj["part_id"] = 2
    obj["sku"] = "BASE_TROUSER_OBSIDIAN"

    clean_topology_and_outward_normals(me, body_ctx)
    push_out_mesh_from_body(me, body_ctx, min_clearance=0.0040, smooth_iters=2)
    edge_stats = regularize_cloth_edges(me, max_local_ratio=1.24, max_iters=12)
    push_out_mesh_from_body(me, body_ctx, min_clearance=0.0040, smooth_iters=1)
    clean_topology_and_outward_normals(me, body_ctx)
    uv_stats = ensure_clean_panel_uv(me, uv_name="UVMap", split_legs=True)

    out_path = BASE_DIR / "BASE_TROUSER_OBSIDIAN.blend"
    save_blend(out_path)
    return {"path": str(out_path), "verts": len(me.vertices), "faces": len(me.polygons), "edge_stats": edge_stats, "uv": uv_stats}


def build_base_shoe_minimal(body_ctx: dict) -> dict:
    reset_empty_scene()
    body_V = body_ctx["V"]
    bm = bmesh.new()
    n_bins = 28

    # Vertical levels from flat outsole bottom (Z = 0.000) to minimalist ankle collar (Z = 0.105)
    z_levels = [0.000, 0.014, 0.030, 0.048, 0.068, 0.088, 0.105]
    # Ease shapes clean derby/sneaker sole silhouette at base and ankle collar at top
    ease_levels = [0.010, 0.009, 0.007, 0.006, 0.005, 0.004, 0.004]

    for side_sign in (-1.0, 1.0):
        foot_V = body_V[(body_V[:, 2] <= 0.118) & ((body_V[:, 0] * side_sign) > 0.005)]
        sole_slice = foot_V[foot_V[:, 2] <= 0.042]
        cx0 = float(np.mean(sole_slice[:, 0]))
        cy0 = float(np.mean(sole_slice[:, 1]))
        rings = []
        for z, ease in zip(z_levels, ease_levels):
            # Take body cross-section around height z (wider window near sole so whole toebox silhouette is included)
            win = 0.025 if z < 0.035 else 0.020
            sl = foot_V[np.abs(foot_V[:, 2] - z) <= win]
            if len(sl) < 8:
                sl = sole_slice if z < 0.05 else foot_V[foot_V[:, 2] >= 0.06]
            cx = float(np.mean(sl[:, 0])) if z >= 0.05 else cx0
            cy = float(np.mean(sl[:, 1])) if z >= 0.05 else cy0
            xy = _radial_envelope(sl[:, :2], cx, cy, n_bins, ease=ease, min_r=0.042, floor_blend=0.35)
            # Match sole silhouette subtly at ground levels
            rings.append([bm.verts.new((float(x), float(y), float(z))) for x, y in xy])

        # Close bottom outsole flat at exact Z_sole = 0.000 m using 2 balanced concentric inset rings (clean open ankle collar!)
        r0 = rings[0]
        r0_xy = np.array([[v.co.x, v.co.y] for v in r0], dtype=np.float64)
        sole_rings = [r0]
        for s_scale in (0.68, 0.36):
            ring_in = [
                bm.verts.new((float(cx0 + s_scale * (p[0] - cx0)), float(cy0 + s_scale * (p[1] - cy0)), 0.000))
                for p in r0_xy
            ]
            sole_rings.append(ring_in)
        for s_i in range(len(sole_rings) - 1):
            ra, rb = sole_rings[s_i], sole_rings[s_i + 1]
            for k in range(n_bins):
                k2 = (k + 1) % n_bins
                bm.faces.new((ra[k2], ra[k], rb[k], rb[k2]))
        sole_c = bm.verts.new((float(cx0), float(cy0), 0.000))
        inner_r = sole_rings[-1]
        for k in range(n_bins):
            k2 = (k + 1) % n_bins
            bm.faces.new((sole_c, inner_r[k2], inner_r[k]))

        for ri in range(len(rings) - 1):
            ra, rb = rings[ri], rings[ri + 1]
            for k in range(n_bins):
                k2 = (k + 1) % n_bins
                bm.faces.new((ra[k], ra[k2], rb[k2], rb[k]))

    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    me = bpy.data.meshes.new("BASE_SHOE_MINIMAL_Mesh")
    bm.to_mesh(me)
    bm.free()

    mat = make_pbr_material(
        "MAT_BASE_SHOE_MINIMAL",
        base_rgba=(0.012, 0.013, 0.016, 1.0),
        roughness=0.38,
        sheen=0.04,
    )
    me.materials.append(mat)
    obj = bpy.data.objects.new("BASE_SHOE_MINIMAL", me)
    bpy.context.scene.collection.objects.link(obj)
    obj["part_id"] = 3
    obj["sku"] = "BASE_SHOE_MINIMAL"

    clean_topology_and_outward_normals(me, body_ctx)
    # Push upper footwear outward keeping outsole bottom verts anchored at Z_sole >= 0.000
    V = np.empty(len(me.vertices) * 3, dtype=np.float64)
    me.vertices.foreach_get("co", V)
    V = V.reshape(-1, 3)
    sole_mask = V[:, 2] < 0.005
    push_out_mesh_from_body(me, body_ctx, min_clearance=0.0035, smooth_iters=1)
    me.vertices.foreach_get("co", V.ravel())
    V[sole_mask, 2] = 0.000  # Pin sole strictly to Z = 0.000 m
    me.vertices.foreach_set("co", V.astype(np.float32).ravel())
    me.update()

    edge_stats = regularize_cloth_edges(me, max_local_ratio=1.20, max_iters=40, body_ctx=body_ctx, min_clearance=0.0020)
    # Rigidly translate vertically so sole bottom stays exact Z = 0.000 m with zero local edge distortion
    V2 = np.empty(len(me.vertices) * 3, dtype=np.float64)
    me.vertices.foreach_get("co", V2)
    V2 = V2.reshape(-1, 3)
    z_shift = float(V2[:, 2].min())
    if abs(z_shift) > 1e-6:
        V2[:, 2] -= z_shift
        me.vertices.foreach_set("co", V2.astype(np.float32).ravel())
        me.update()

    clean_topology_and_outward_normals(me, body_ctx)
    uv_stats = ensure_clean_panel_uv(me, uv_name="UVMap", split_legs=True)

    out_path = BASE_DIR / "BASE_SHOE_MINIMAL.blend"
    save_blend(out_path)
    return {
        "path": str(out_path),
        "verts": len(me.vertices),
        "faces": len(me.polygons),
        "sole_z_min": round(float(V2[:, 2].min()), 5),
        "edge_stats": edge_stats,
        "uv": uv_stats,
    }


def main() -> None:
    import json
    body_ctx = load_theta_fit_body()
    res_trouser = build_base_trouser_obsidian(body_ctx)
    res_shoe = build_base_shoe_minimal(body_ctx)
    print(json.dumps({"BASE_TROUSER_OBSIDIAN": res_trouser, "BASE_SHOE_MINIMAL": res_shoe}, indent=2))


if __name__ == "__main__":
    main()
