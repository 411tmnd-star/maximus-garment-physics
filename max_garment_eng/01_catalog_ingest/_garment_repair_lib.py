"""_garment_repair_lib.py — Shared pure-bpy.data + bmesh + NumPy + io_ops_gateway library for Stage 1A SKU repair.

STRICT ZERO-bpy.ops DISCIPLINE:
  - Never calls bpy.ops.* anywhere in this module (loads via bpy.data.libraries.load, evaluates modifiers via depsgraph,
    edits topology/UVs/normals via bmesh + NumPy + mathutils.bvhtree, saves via io_ops_gateway.save_copy).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import bpy  # type: ignore
import bmesh  # type: ignore
import numpy as np
from mathutils import Vector  # type: ignore
from mathutils.bvhtree import BVHTree  # type: ignore

MAX_GARMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MAX_GARMENT_ROOT.parents[1]
MAXIMUS_PIPELINE = REPO_ROOT / "02_engines" / "maximus_engine" / "pipeline"
HERO_RINGER_NPZ = MAX_GARMENT_ROOT / "COLAB_MEGA_SPRINT" / "inputs" / "hero_ringer_shell.npz"

if str(MAXIMUS_PIPELINE) not in sys.path:
    sys.path.insert(0, str(MAXIMUS_PIPELINE))
import io_ops_gateway  # noqa: E402


def reset_empty_scene() -> None:
    """Clear all objects/meshes/materials from current blend data cleanly without bpy.ops."""
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for me in list(bpy.data.meshes):
        bpy.data.meshes.remove(me, do_unlink=True)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat, do_unlink=True)
    for img in list(bpy.data.images):
        bpy.data.images.remove(img, do_unlink=True)


def save_blend(path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    io_ops_gateway.save_copy(str(path))


def load_theta_fit_body() -> dict[str, Any]:
    """Load canonical Maximus v2 theta_fit body (13,380 verts) + BVHTree + outward vertex normals."""
    d = np.load(HERO_RINGER_NPZ, allow_pickle=True)
    V = np.asarray(d["body_ref_V"], dtype=np.float64)
    F = np.asarray(d["body_F"], dtype=np.int64)
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    fn = np.cross(B - A, C - A)
    VN = np.zeros_like(V)
    for k in range(3):
        np.add.at(VN, F[:, k], fn)
    ln = np.linalg.norm(VN, axis=1, keepdims=True)
    VN = VN / np.maximum(ln, 1e-12)
    bvh = BVHTree.FromPolygons([tuple(v) for v in V.tolist()], [tuple(f) for f in F.tolist()])
    return {"V": V, "F": F, "VN": VN, "bvh": bvh}


def append_objects_from_blend(blend_path: Path, object_names: list[str] | None = None) -> list[Any]:
    """Append objects from another .blend via bpy.data.libraries.load (zero bpy.ops)."""
    scene_col = bpy.context.scene.collection
    with bpy.data.libraries.load(str(blend_path), link=False) as (data_from, data_to):
        avail = list(data_from.objects)
        if object_names is None:
            data_to.objects = avail
        else:
            want = set(object_names)
            data_to.objects = [n for n in avail if n in want]
    loaded = []
    for obj in data_to.objects:
        if obj is not None:
            if obj.name not in scene_col.objects:
                scene_col.objects.link(obj)
            # Bake any non-identity object transform onto mesh coordinates cleanly
            if obj.type == "MESH" and not np.allclose(np.asarray(obj.matrix_world), np.eye(4), atol=1e-6):
                mw = obj.matrix_world.copy()
                obj.data.transform(mw)
                obj.matrix_world.identity()
            loaded.append(obj)
    return loaded


def generalized_winding_number_cpu(pts: np.ndarray, body_V: np.ndarray, body_F: np.ndarray, batch: int = 256) -> np.ndarray:
    """Exact Float64 Van Oosterom & Strakhov solid-angle GWN against theta_fit body (zero SciPy required)."""
    pts = np.asarray(pts, dtype=np.float64)
    A0 = body_V[body_F[:, 0]]
    B0 = body_V[body_F[:, 1]]
    C0 = body_V[body_F[:, 2]]
    out = np.empty(len(pts), dtype=np.float64)
    four_pi = 4.0 * math.pi
    for s in range(0, len(pts), batch):
        P = pts[s : s + batch][:, None, :]  # (B, 1, 3)
        a = A0[None, :, :] - P
        b = B0[None, :, :] - P
        c = C0[None, :, :] - P
        la = np.maximum(np.linalg.norm(a, axis=2), 1e-12)
        lb = np.maximum(np.linalg.norm(b, axis=2), 1e-12)
        lc = np.maximum(np.linalg.norm(c, axis=2), 1e-12)
        det = np.einsum("ijk,ijk->ij", a, np.cross(b, c))
        den = (
            la * lb * lc
            + np.einsum("ijk,ijk->ij", a, b) * lc
            + np.einsum("ijk,ijk->ij", b, c) * la
            + np.einsum("ijk,ijk->ij", c, a) * lb
        )
        omega = 2.0 * np.arctan2(det, den)
        out[s : s + len(P)] = omega.sum(axis=1) / four_pi
    return out


def signed_clearance_and_inside_bvh(
    pts: np.ndarray,
    body_ctx: dict[str, Any],
    z_ignore_inside_margin: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute signed distance & closest surface normal using Blender's C++ BVHTree (sub-millisecond per mesh)."""
    bvh: BVHTree = body_ctx["bvh"]
    body_V: np.ndarray = body_ctx["V"]
    body_F: np.ndarray = body_ctx["F"]
    body_VN: np.ndarray = body_ctx["VN"]
    signed = np.empty(len(pts), dtype=np.float64)
    closest_pts = np.empty_like(pts, dtype=np.float64)
    closest_nrm = np.empty_like(pts, dtype=np.float64)
    for i, p in enumerate(pts):
        loc, nrm, face_idx, dist = bvh.find_nearest(Vector((float(p[0]), float(p[1]), float(p[2]))))
        if loc is None:
            signed[i] = 1.0
            closest_pts[i] = p
            closest_nrm[i] = np.array([0.0, 0.0, 1.0])
            continue
        loc_np = np.array([loc.x, loc.y, loc.z], dtype=np.float64)
        # Smooth vertex-interpolated outward normal avoids flat-face crease ambiguities
        tri = body_F[face_idx]
        vn = (body_VN[tri[0]] + body_VN[tri[1]] + body_VN[tri[2]]) / 3.0
        vn /= max(float(np.linalg.norm(vn)), 1e-12)
        diff = p - loc_np
        sgn = 1.0 if float(np.dot(diff, vn)) >= 0.0 else -1.0
        signed[i] = sgn * float(dist)
        closest_pts[i] = loc_np
        closest_nrm[i] = vn
    inside_mask = signed < -float(z_ignore_inside_margin)
    return signed, closest_nrm, inside_mask


def _signed_for_point(p: np.ndarray, body_ctx: dict[str, Any]) -> tuple[float, np.ndarray]:
    """Compute signed distance and smooth body normal for a single 3D query point."""
    bvh: BVHTree = body_ctx["bvh"]
    body_F: np.ndarray = body_ctx["F"]
    body_VN: np.ndarray = body_ctx["VN"]
    loc, _, face_idx, dist = bvh.find_nearest(Vector((float(p[0]), float(p[1]), float(p[2]))))
    if loc is None:
        return 1.0, np.array([0.0, 0.0, 1.0], dtype=np.float64)
    loc_np = np.array([loc.x, loc.y, loc.z], dtype=np.float64)
    tri = body_F[face_idx]
    vn = (body_VN[tri[0]] + body_VN[tri[1]] + body_VN[tri[2]]) / 3.0
    vn /= max(float(np.linalg.norm(vn)), 1e-12)
    sgn = 1.0 if float(np.dot(p - loc_np, vn)) >= 0.0 else -1.0
    return sgn * float(dist), vn


def _resolve_min_req(
    V: np.ndarray,
    min_clearance: float,
    min_clearance_per_vert: np.ndarray | None,
    collar_guard: bool,
) -> np.ndarray:
    """Resolve per-vertex minimum body clearance array dynamically from current vertex positions."""
    if min_clearance_per_vert is not None and len(min_clearance_per_vert) == len(V):
        return np.asarray(min_clearance_per_vert, dtype=np.float64)
    if collar_guard:
        is_col = (V[:, 2] >= 1.44) & (np.hypot(V[:, 0], V[:, 1] + 0.02) <= 0.115)
        return np.where(is_col, 0.00025, float(min_clearance))
    return np.full(len(V), float(min_clearance), dtype=np.float64)


def _clamp_converge_outside_body(
    V: np.ndarray,
    body_ctx: dict[str, Any],
    min_clearance: float,
    pinned_z_above: float | None = None,
    min_clearance_per_vert: np.ndarray | None = None,
    collar_guard: bool = False,
    max_passes: int = 8,
) -> np.ndarray:
    """Iteratively project inside vertices outside theta_fit, including coherent concave-crease escape."""
    moved_mask = np.zeros(len(V), dtype=bool)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    for p_idx in range(max_passes):
        signed, nrm, _ = signed_clearance_and_inside_bvh(V, body_ctx)
        min_req = _resolve_min_req(V, min_clearance, min_clearance_per_vert, collar_guard)
        need = signed < min_req
        if pinned_z_above is not None:
            need = need & (V[:, 2] < float(pinned_z_above))
        if not need.any():
            break
        moved_mask |= need

        if p_idx < 3:
            V[need] += (min_req[need] + 0.0005 - signed[need])[:, None] * nrm[need]
        else:
            # Side-coherent concave crease escape so adjacent vertices escape together without ripping
            for vi in np.flatnonzero(need):
                p0 = V[vi].copy()
                req = float(min_req[vi]) + 0.0005
                sx = 1.0 if p0[0] >= 0.0 else -1.0
                sy = 1.0 if p0[1] >= -0.01 else -1.0
                cands = (
                    nrm[vi],
                    np.array([0.0, sy, 0.0], dtype=np.float64),
                    np.array([0.0, 0.0, -1.0], dtype=np.float64),
                    np.array([0.0, sy * inv_sqrt2, -inv_sqrt2], dtype=np.float64),
                    np.array([sx, 0.0, 0.0], dtype=np.float64),
                    np.array([0.0, 0.0, 1.0], dtype=np.float64),
                    np.array([0.0, -sy, 0.0], dtype=np.float64),
                )
                best_s = float(signed[vi])
                best_p = p0
                found = False
                for step in (0.002, 0.004, 0.007, 0.011, 0.016):
                    for d in cands:
                        pc = p0 + step * d
                        sc, _ = _signed_for_point(pc, body_ctx)
                        if sc > best_s:
                            best_s = sc
                            best_p = pc
                        if sc >= req:
                            best_p = pc
                            found = True
                            break
                    if found:
                        break
                V[vi] = best_p

    return moved_mask


def push_out_mesh_from_body(
    me: bpy.types.Mesh,
    body_ctx: dict[str, Any],
    min_clearance: float = 0.0035,
    smooth_iters: int = 2,
    pinned_z_above: float | None = None,
    min_clearance_per_vert: np.ndarray | None = None,
    collar_guard: bool = False,
) -> int:
    """Push any vertex inside or closer than `min_clearance` to theta_fit outward with concave crease escape."""
    V = np.empty(len(me.vertices) * 3, dtype=np.float64)
    me.vertices.foreach_get("co", V)
    V = V.reshape(-1, 3)

    moved_mask = _clamp_converge_outside_body(
        V,
        body_ctx,
        min_clearance=min_clearance,
        pinned_z_above=pinned_z_above,
        min_clearance_per_vert=min_clearance_per_vert,
        collar_guard=collar_guard,
        max_passes=8,
    )
    n_moved = int(moved_mask.sum())
    if n_moved == 0:
        return 0

    # Gentle 1-ring smoothing only on moved vertices to prevent facets while preserving min clearance
    if smooth_iters > 0:
        adj: list[list[int]] = [[] for _ in range(len(V))]
        for e in me.edges:
            a, b = e.vertices[0], e.vertices[1]
            adj[a].append(b)
            adj[b].append(a)
        moved_idx = np.flatnonzero(moved_mask)
        for _ in range(smooth_iters):
            V_new = V.copy()
            for vi in moved_idx:
                nb = adj[vi]
                if nb:
                    avg = V[nb].mean(axis=0)
                    V_new[vi] = 0.65 * V[vi] + 0.35 * avg
            V = V_new
            extra = _clamp_converge_outside_body(
                V,
                body_ctx,
                min_clearance=min_clearance,
                pinned_z_above=pinned_z_above,
                min_clearance_per_vert=min_clearance_per_vert,
                collar_guard=collar_guard,
                max_passes=6,
            )
            moved_mask |= extra
            moved_idx = np.flatnonzero(moved_mask)

    me.vertices.foreach_set("co", V.astype(np.float32).ravel())
    me.update()
    return int(moved_mask.sum())


def clean_topology_and_outward_normals(me: bpy.types.Mesh, body_ctx: dict[str, Any] | None = None) -> dict[str, int]:
    """Remove degenerate/zero-area triangles, ensure manifold-clean outward normals via bmesh (zero bpy.ops)."""
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=list(bm.faces))
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=1e-5)
    bmesh.ops.dissolve_degenerate(bm, edges=list(bm.edges), dist=1e-5)

    bm.faces.ensure_lookup_table()
    bad_faces = []
    for f in bm.faces:
        if not f.is_valid or len(f.verts) < 3 or f.calc_area() < 1e-9:
            bad_faces.append(f)
            continue
        lens = [e.calc_length() for e in f.edges]
        if min(lens) < 1e-6:
            bad_faces.append(f)
    if bad_faces:
        bmesh.ops.delete(bm, geom=bad_faces, context="FACES")

    orphan_verts = [v for v in bm.verts if len(v.link_faces) == 0]
    if orphan_verts:
        bmesh.ops.delete(bm, geom=orphan_verts, context="VERTS")

    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))

    # Flip any island whose normals point inward relative to body centroid or theta_fit BVH
    if body_ctx is not None and len(bm.faces) > 0:
        bm.faces.ensure_lookup_table()
        bvh: BVHTree = body_ctx["bvh"]
        # Flood connected islands
        visited = set()
        for seed in bm.faces:
            if seed.index in visited:
                continue
            island = []
            stack = [seed]
            visited.add(seed.index)
            while stack:
                cf = stack.pop()
                island.append(cf)
                for e in cf.edges:
                    for nf in e.link_faces:
                        if nf.index not in visited:
                            visited.add(nf.index)
                            stack.append(nf)
            # Sample up to 64 faces from island to vote outward vs inward
            step = max(1, len(island) // 64)
            dots = []
            for f in island[::step]:
                c = f.calc_center_median()
                loc, nrm, _, _ = bvh.find_nearest(c)
                if loc is not None and nrm is not None:
                    # Reference direction from closest body point to face center (or body normal if very close)
                    rad = c - loc
                    ref = rad.normalized() if rad.length > 0.002 else nrm
                    dots.append(float(f.normal.dot(ref)))
            if dots and float(np.median(dots)) < 0.0:
                bmesh.ops.reverse_faces(bm, faces=island)

    bm.to_mesh(me)
    bm.free()
    me.update()
    return {"verts": len(me.vertices), "faces": len(me.polygons)}


def _edge_stats(me: bpy.types.Mesh) -> dict[str, float]:
    V = np.empty(len(me.vertices) * 3, dtype=np.float64)
    me.vertices.foreach_get("co", V)
    V = V.reshape(-1, 3)
    if len(me.edges) == 0:
        return {"p50_mm": 0.0, "local_ratio_p99": 1.0, "local_ratio_max": 1.0}
    E = np.array([[e.vertices[0], e.vertices[1]] for e in me.edges], dtype=np.int64)
    L = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    # Local vertex neighbor edge ratio: each edge length divided by incident vertex mean edge length
    deg = np.zeros(len(V), dtype=np.float64)
    sum_l = np.zeros(len(V), dtype=np.float64)
    np.add.at(sum_l, E[:, 0], L)
    np.add.at(sum_l, E[:, 1], L)
    np.add.at(deg, E[:, 0], 1.0)
    np.add.at(deg, E[:, 1], 1.0)
    v_mean = sum_l / np.maximum(deg, 1.0)
    e_local_ref = 0.5 * (v_mean[E[:, 0]] + v_mean[E[:, 1]])
    ratio_local = L / np.maximum(e_local_ref, 1e-9)
    return {
        "p50_mm": round(float(np.median(L) * 1000.0), 3),
        "local_ratio_p99": round(float(np.percentile(ratio_local, 99.0)), 4),
        "local_ratio_max": round(float(np.max(ratio_local)), 4),
    }


def apply_remesh_or_decimate_depsgraph(obj: bpy.types.Object, ratio: float = 0.06) -> None:
    """Decimate high-poly cloth via Modifier + depsgraph evaluated copy (ZERO bpy.ops!)."""
    mod = obj.modifiers.new(name="CollapseTmp", type="DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = float(ratio)
    mod.use_collapse_triangulate = True
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    new_me = bpy.data.meshes.new_from_object(ev)
    old_me = obj.data
    new_me.name = old_me.name + "_sim"
    for mat in old_me.materials:
        new_me.materials.append(mat)
    obj.modifiers.remove(mod)
    obj.data = new_me
    if old_me.users == 0:
        bpy.data.meshes.remove(old_me)


def _compute_presim_stretch(
    V: np.ndarray,
    F: np.ndarray,
    E: np.ndarray,
    VN_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact tangential equilibrium stretch ratios matching local_preflight_all_skus."""
    n_v = len(V)
    if VN_override is not None and len(VN_override) == n_v:
        VN = VN_override
    else:
        VN = np.zeros_like(V)
        if len(F) > 0:
            A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
            fn = np.cross(B - A, C - A)
            for k in range(3):
                np.add.at(VN, F[:, k], fn)
        ln = np.linalg.norm(VN, axis=1, keepdims=True)
        VN = VN / np.maximum(ln, 1e-12)

    nbr_sum = np.zeros_like(V)
    deg = np.zeros(n_v, dtype=np.float64)
    np.add.at(nbr_sum, E[:, 0], V[E[:, 1]])
    np.add.at(nbr_sum, E[:, 1], V[E[:, 0]])
    np.add.at(deg, E[:, 0], 1.0)
    np.add.at(deg, E[:, 1], 1.0)
    delta = (nbr_sum / np.maximum(deg[:, None], 1.0)) - V
    tang = delta - np.sum(delta * VN, axis=1, keepdims=True) * VN
    V_eq = V + 0.15 * tang
    L = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    L_eq = np.linalg.norm(V_eq[E[:, 0]] - V_eq[E[:, 1]], axis=1)
    rr = np.maximum(L / np.maximum(L_eq, 1e-6), L_eq / np.maximum(L, 1e-6))
    return rr, tang, L


def _extract_mesh_topology(me: bpy.types.Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract float64 vertices, triangle indices, undirected edges, and boundary vertex mask."""
    n_v = len(me.vertices)
    V = np.empty(n_v * 3, dtype=np.float64)
    me.vertices.foreach_get("co", V)
    V = V.reshape(-1, 3)
    tris = []
    for p in me.polygons:
        vs = p.vertices
        if len(vs) == 3:
            tris.append((vs[0], vs[1], vs[2]))
        elif len(vs) > 3:
            for k in range(1, len(vs) - 1):
                tris.append((vs[0], vs[k], vs[k + 1]))
    F = np.array(tris, dtype=np.int64) if tris else np.zeros((0, 3), dtype=np.int64)
    E = np.array([[e.vertices[0], e.vertices[1]] for e in me.edges], dtype=np.int64)
    bm_b = bmesh.new()
    bm_b.from_mesh(me)
    is_bnd = np.array([v.is_boundary for v in bm_b.verts], dtype=bool)
    bm_b.free()
    return V, F, E, is_bnd


def _collapse_disjoint_edges(
    bm: bmesh.types.BMesh,
    cand_edges: list[bmesh.types.BMEdge],
    allow_sharp_collapse: bool = False,
) -> int:
    """Safely collapse a maximal vertex-disjoint subset of manifold-safe edges using bmesh.ops.collapse."""
    used_verts: set[int] = set()
    chosen: list[bmesh.types.BMEdge] = []
    cand_edges.sort(key=lambda e: e.calc_length())
    for e in cand_edges:
        if not e.is_valid:
            continue
        v0, v1 = e.verts[0], e.verts[1]
        if v0.index in used_verts or v1.index in used_verts:
            continue
        # Link condition prevents non-manifold pinches / closed-tube collapses
        n0 = {e2.other_vert(v0).index for e2 in v0.link_edges}
        n1 = {e3.other_vert(v1).index for e3 in v1.link_edges}
        n_shared = len(n0 & n1)
        if n_shared > len(e.link_faces):
            continue
        # Protect 2-ply beveled seams during Phase 1 unless edge is sub-0.6mm or outlier cleanup is enabled
        if (not allow_sharp_collapse) and e.calc_length() >= 0.0006 and float(v0.normal.dot(v1.normal)) < -0.15:
            continue
        chosen.append(e)
        used_verts.add(v0.index)
        used_verts.add(v1.index)

    if chosen:
        bmesh.ops.collapse(bm, edges=chosen, uvs=True)
    return len(chosen)


def _cleanup_bm_post_edit(bm: bmesh.types.BMesh) -> None:
    """Triangulate quads, dissolve sub-micron slivers, purge zero-area faces & orphans."""
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])
    bmesh.ops.dissolve_degenerate(bm, edges=list(bm.edges), dist=1e-6)
    bm.faces.ensure_lookup_table()
    bad_f = [f for f in bm.faces if not f.is_valid or len(f.verts) < 3 or f.calc_area() < 1e-9]
    if bad_f:
        bmesh.ops.delete(bm, geom=bad_f, context="FACES")
    orphan_v = [v for v in bm.verts if len(v.link_faces) == 0]
    if orphan_v:
        bmesh.ops.delete(bm, geom=orphan_v, context="VERTS")
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))


def _relax_stretch_steps(
    V: np.ndarray,
    F: np.ndarray,
    E: np.ndarray,
    is_bnd: np.ndarray,
    max_stretch: float,
    iters: int,
    VN_fixed: np.ndarray | None = None,
    global_warm_iters: int = 0,
    body_ctx: dict[str, Any] | None = None,
    guard_clearance: float = 0.0008,
    min_clearance_per_vert: np.ndarray | None = None,
    collar_guard: bool = False,
) -> float:
    """Run body-projected damped tangential equilibrium relaxation steps and return final max stretch."""
    n_v = len(V)
    rr_max = 1.0
    for it in range(iters):
        rr, tang, _ = _compute_presim_stretch(V, F, E, VN_override=VN_fixed)
        rr_max = float(rr.max())
        if rr_max <= max_stretch:
            break
        bad_e = rr > (max_stretch - 0.02)
        bad_v = np.zeros(n_v, dtype=bool)
        bad_v[E[bad_e, 0]] = True
        bad_v[E[bad_e, 1]] = True
        bad_ring = bad_v.copy()
        bad_ring[E[bad_v[E[:, 0]], 1]] = True
        bad_ring[E[bad_v[E[:, 1]], 0]] = True

        if it < global_warm_iters:
            mask_in = ~is_bnd
        else:
            mask_in = bad_ring & ~is_bnd
        V[mask_in] += 0.28 * tang[mask_in]
        mask_bnd = bad_v & is_bnd
        V[mask_bnd] += 0.24 * tang[mask_bnd]

        if body_ctx is not None:
            active_mask = mask_in | mask_bnd
            moved_idx = np.flatnonzero(active_mask)
            if len(moved_idx):
                signed, nrm, _ = signed_clearance_and_inside_bvh(V[moved_idx], body_ctx)
                if min_clearance_per_vert is not None and len(min_clearance_per_vert) == n_v:
                    min_req = np.asarray(min_clearance_per_vert, dtype=np.float64)[moved_idx]
                else:
                    min_req = _resolve_min_req(V[moved_idx], guard_clearance, None, collar_guard)
                dip = signed < min_req
                if dip.any():
                    vi_dip = moved_idx[dip]
                    V[vi_dip] += (min_req[dip] + 0.0002 - signed[dip])[:, None] * nrm[dip]
    return rr_max


def regularize_cloth_edges(
    me: bpy.types.Mesh,
    max_local_ratio: float = 1.22,
    max_iters: int = 35,
    max_edge_m: float = 0.022,
    min_edge_m: float = 0.0020,
    max_stretch: float | None = None,
    body_ctx: dict[str, Any] | None = None,
    min_clearance: float = 0.0030,
    min_clearance_per_vert: np.ndarray | None = None,
    collar_guard: bool = False,
) -> dict[str, Any]:
    """Equalize cloth edges via body-wrapped subdivide_edges, link-safe collapse, and projected relaxation."""
    target_stretch = float(max_local_ratio if max_stretch is None else max_stretch)
    guard_clearance = min(0.0008, float(min_clearance))

    # Phase 1: Geodesic-wrapped topology normalization (split > max_edge_m, wrap outside body, collapse < min_edge_m)
    for _pass in range(6):
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.normal_update()
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        if len(bm.edges) == 0:
            bm.free()
            return _edge_stats(me)

        long_edges = [e for e in bm.edges if e.is_valid and e.calc_length() > max_edge_m]
        if long_edges:
            bmesh.ops.subdivide_edges(
                bm,
                edges=long_edges[:15000],
                cuts=1,
                use_grid_fill=False,
                smooth=0.0,
                use_single_edge=True,
            )
            bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])

        bm.normal_update()
        bm.edges.ensure_lookup_table()
        short_edges = [e for e in bm.edges if e.is_valid and e.calc_length() < min_edge_m]
        n_col = _collapse_disjoint_edges(bm, short_edges, allow_sharp_collapse=False) if short_edges else 0

        _cleanup_bm_post_edit(bm)
        bm.to_mesh(me)
        bm.free()
        me.update()
        if not long_edges and n_col == 0:
            break
        if body_ctx is not None:
            push_out_mesh_from_body(
                me,
                body_ctx,
                min_clearance=min_clearance,
                smooth_iters=1,
                min_clearance_per_vert=min_clearance_per_vert,
                collar_guard=collar_guard,
            )

    # Phase 2: Body-projected equilibrium relaxation + short-pinch-only outlier collapse rounds
    for r_idx in range(5):
        n_v = len(me.vertices)
        if n_v == 0 or len(me.edges) == 0:
            return _edge_stats(me)
        V, F, E, is_bnd = _extract_mesh_topology(me)
        _relax_stretch_steps(
            V,
            F,
            E,
            is_bnd,
            max_stretch=target_stretch,
            iters=max_iters,
            global_warm_iters=8 if r_idx == 0 else 0,
            body_ctx=body_ctx,
            guard_clearance=guard_clearance,
            min_clearance_per_vert=min_clearance_per_vert,
            collar_guard=collar_guard,
        )
        me.vertices.foreach_set("co", V.astype(np.float32).ravel())
        me.update()

        # Verify against Blender's exact vertex normals and refine with fixed VN if needed
        VN_flat = np.empty(n_v * 3, dtype=np.float64)
        me.vertices.foreach_get("normal", VN_flat)
        VN_me = VN_flat.reshape(-1, 3)
        rr_exact, _, L_exact = _compute_presim_stretch(V, F, E, VN_override=VN_me)
        if float(rr_exact.max()) <= target_stretch:
            break

        _relax_stretch_steps(
            V,
            F,
            E,
            is_bnd,
            max_stretch=target_stretch,
            iters=18,
            VN_fixed=VN_me,
            body_ctx=body_ctx,
            guard_clearance=guard_clearance,
            min_clearance_per_vert=min_clearance_per_vert,
            collar_guard=collar_guard,
        )
        me.vertices.foreach_set("co", V.astype(np.float32).ravel())
        me.update()
        me.vertices.foreach_get("normal", VN_flat)
        VN_me = VN_flat.reshape(-1, 3)
        rr_exact, _, L_exact = _compute_presim_stretch(V, F, E, VN_override=VN_me)
        if float(rr_exact.max()) <= target_stretch:
            break

        # Collapse ONLY genuinely short pinched outlier edges or split residual over-long outlier edges
        med_L = float(np.median(L_exact)) if len(L_exact) else 0.006
        short_cap = min(0.0045, 0.70 * med_L)
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.normal_update()
        bm.edges.ensure_lookup_table()
        bm_edges = list(bm.edges)
        bad_indices = np.flatnonzero(rr_exact > target_stretch)
        split_cands = [bm_edges[ei] for ei in bad_indices if ei < len(bm_edges) and L_exact[ei] > max_edge_m * 0.85]
        if split_cands:
            bmesh.ops.subdivide_edges(
                bm,
                edges=split_cands,
                cuts=1,
                use_grid_fill=False,
                smooth=0.0,
                use_single_edge=True,
            )
            bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])
            bm.normal_update()
            bm.edges.ensure_lookup_table()
        short_out = [
            bm_edges[ei]
            for ei in bad_indices
            if ei < len(bm_edges) and bm_edges[ei].is_valid and L_exact[ei] <= short_cap
        ]
        n_col = _collapse_disjoint_edges(bm, short_out, allow_sharp_collapse=True) if short_out else 0
        if not split_cands and n_col == 0:
            bm.free()
            break
        _cleanup_bm_post_edit(bm)
        bm.to_mesh(me)
        bm.free()
        me.update()
        if body_ctx is not None:
            push_out_mesh_from_body(
                me,
                body_ctx,
                min_clearance=guard_clearance,
                smooth_iters=0,
                min_clearance_per_vert=min_clearance_per_vert,
                collar_guard=collar_guard,
            )

    return _edge_stats(me)


def finalize_clean_garment_mesh(
    me: bpy.types.Mesh,
    body_ctx: dict[str, Any],
    min_clearance: float = 0.0030,
    min_clearance_per_vert: np.ndarray | None = None,
    collar_guard: bool = False,
    max_edge_m: float = 0.022,
    min_edge_m: float = 0.0020,
) -> dict[str, Any]:
    """Run geodesic-wrapped topology cleanup, edge regularization, and lightweight projected polish."""
    clean_topology_and_outward_normals(me, body_ctx)
    push_out_mesh_from_body(
        me,
        body_ctx,
        min_clearance=min_clearance,
        smooth_iters=2,
        min_clearance_per_vert=min_clearance_per_vert,
        collar_guard=collar_guard,
    )
    regularize_cloth_edges(
        me,
        max_stretch=1.21,
        max_iters=40,
        max_edge_m=max_edge_m,
        min_edge_m=min_edge_m,
        body_ctx=body_ctx,
        min_clearance=min_clearance,
        min_clearance_per_vert=min_clearance_per_vert,
        collar_guard=collar_guard,
    )
    clean_topology_and_outward_normals(me, body_ctx)

    guard_clearance = min(0.0008, float(min_clearance))
    for _rnd in range(3):
        moved = push_out_mesh_from_body(
            me,
            body_ctx,
            min_clearance=guard_clearance,
            smooth_iters=0,
            min_clearance_per_vert=min_clearance_per_vert,
            collar_guard=collar_guard,
        )
        n_v = len(me.vertices)
        if n_v == 0 or len(me.edges) == 0:
            break
        V, F, E, is_bnd = _extract_mesh_topology(me)
        VN_me = np.empty(n_v * 3, dtype=np.float64)
        me.vertices.foreach_get("normal", VN_me)
        VN_me = VN_me.reshape(-1, 3)
        rr_exact, _, _ = _compute_presim_stretch(V, F, E, VN_override=VN_me)
        if moved == 0 and float(rr_exact.max()) <= 1.23:
            break
        _relax_stretch_steps(
            V,
            F,
            E,
            is_bnd,
            max_stretch=1.20,
            iters=18,
            VN_fixed=VN_me,
            body_ctx=body_ctx,
            guard_clearance=guard_clearance,
            min_clearance_per_vert=min_clearance_per_vert,
            collar_guard=collar_guard,
        )
        me.vertices.foreach_set("co", V.astype(np.float32).ravel())
        me.update()

    # Final hard inside-body guard at penetration-free clearance without disturbing converged edges
    push_out_mesh_from_body(
        me,
        body_ctx,
        min_clearance=guard_clearance,
        smooth_iters=0,
        min_clearance_per_vert=min_clearance_per_vert,
        collar_guard=collar_guard,
    )
    me.update()
    return _edge_stats(me)


def ensure_clean_panel_uv(me: bpy.types.Mesh, uv_name: str = "UVMap", split_legs: bool = False) -> dict[str, Any]:
    """Guarantee a valid non-overlapping front/back panel UVMap on every face in pure bmesh (zero NaN/zero-area UV)."""
    if not me.uv_layers:
        me.uv_layers.new(name=uv_name)
    else:
        me.uv_layers[0].name = uv_name

    bm = bmesh.new()
    bm.from_mesh(me)
    uv_layer = bm.loops.layers.uv.verify()

    coords = np.array([v.co[:] for v in bm.verts], dtype=np.float64)
    z_min, z_max = float(coords[:, 2].min()), float(coords[:, 2].max())
    z_span = max(z_max - z_min, 1e-4)
    cx = float(coords[:, 0].mean())
    cy = float(coords[:, 1].mean())

    for f in bm.faces:
        fc = f.calc_center_median()
        is_back = fc.y > cy
        for loop in f.loops:
            co = loop.vert.co
            vz = (co.z - z_min) / z_span * 0.92 + 0.04
            if split_legs:
                # 4 quadrants for trousers: front-L, front-R, back-L, back-R -> zero leg overlap!
                dx = co.x - (-0.09 if co.x < cx else 0.09)
                dy = co.y - cy
                ang = math.atan2(dy, dx)  # [-pi, pi]
                ang_u = (ang / (2.0 * math.pi)) + 0.5
                quad_off = (0.0 if co.x < cx else 0.25) + (0.0 if not is_back else 0.50)
                ux = quad_off + 0.02 + ang_u * 0.21
            else:
                # Cylindrical front [0.02..0.48] / back [0.52..0.98] projection
                ang = math.atan2(co.y - cy, co.x - cx)
                u_norm = (ang + math.pi) / (2.0 * math.pi)
                if not is_back:
                    ux = 0.02 + ((co.x - coords[:, 0].min()) / max(coords[:, 0].max() - coords[:, 0].min(), 1e-4)) * 0.46
                else:
                    ux = 0.52 + (1.0 - (co.x - coords[:, 0].min()) / max(coords[:, 0].max() - coords[:, 0].min(), 1e-4)) * 0.46
                # Add slight cylindrical curvature offset to prevent edge-on zero-area UVs under sleeves/sides
                ux += 0.025 * math.sin(u_norm * 2.0 * math.pi)
            loop[uv_layer].uv = (max(0.005, min(0.995, ux)), max(0.005, min(0.995, vz)))

    bm.to_mesh(me)
    bm.free()
    me.update()
    return {"uv_name": me.uv_layers[0].name, "faces": len(me.polygons)}


def rebuild_clean_buttons(
    centroids_and_normals: list[tuple[Vector, Vector]],
    name: str = "placket_buttons",
    mat: bpy.types.Material | None = None,
    radius: float = 0.0062,
    thickness: float = 0.0022,
) -> bpy.types.Object:
    """Create 0-degenerate-tri beveled cylindrical buttons with clean UVMap & rigid `button_0..k` vertex groups."""
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.verify()
    v_groups_map: list[tuple[int, list[int]]] = []

    for idx, (center, nrm) in enumerate(centroids_and_normals):
        n_vec = nrm.normalized() if nrm.length > 1e-6 else Vector((0.0, -1.0, 0.0))
        ref = Vector((0.0, 0.0, 1.0)) if abs(n_vec.z) < 0.9 else Vector((1.0, 0.0, 0.0))
        t1 = ref.cross(n_vec).normalized()
        t2 = n_vec.cross(t1).normalized()

        # Construct 14-gon flat disk extruded cleanly into beveled cylinder (0 degenerate tris!)
        segs = 14
        ring_bot_idx = []
        ring_top_idx = []
        for s in range(segs):
            ang = 2.0 * math.pi * s / segs
            offset = (math.cos(ang) * t1 + math.sin(ang) * t2) * radius
            vb = bm.verts.new(center - 0.35 * thickness * n_vec + offset)
            vt = bm.verts.new(center + 0.65 * thickness * n_vec + offset * 0.92)
            ring_bot_idx.append(vb)
            ring_top_idx.append(vt)
        vc_bot = bm.verts.new(center - 0.35 * thickness * n_vec)
        vc_top = bm.verts.new(center + 0.65 * thickness * n_vec)

        created_verts = ring_bot_idx + ring_top_idx + [vc_bot, vc_top]
        for s in range(segs):
            s2 = (s + 1) % segs
            bm.faces.new((ring_top_idx[s], ring_top_idx[s2], vc_top))
            bm.faces.new((ring_bot_idx[s2], ring_bot_idx[s], vc_bot))
            bm.faces.new((ring_bot_idx[s], ring_bot_idx[s2], ring_top_idx[s2]))
            bm.faces.new((ring_bot_idx[s], ring_top_idx[s2], ring_top_idx[s]))
        v_groups_map.append((idx, created_verts))

    bm.verts.index_update()
    group_vert_indices = {idx: [v.index for v in verts] for idx, verts in v_groups_map}
    for f in bm.faces:
        for loop in f.loops:
            loop[uv_layer].uv = (0.5 + 0.3 * math.cos(loop.vert.index), 0.5 + 0.3 * math.sin(loop.vert.index))

    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    me = bpy.data.meshes.new(name + "_Mesh")
    bm.to_mesh(me)
    bm.free()
    if me.uv_layers:
        me.uv_layers[0].name = "UVMap"
    if mat is not None:
        me.materials.append(mat)
    me.update()

    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    for idx, v_ids in group_vert_indices.items():
        vg = obj.vertex_groups.new(name=f"button_{idx:02d}")
        vg.add(v_ids, 1.0, "REPLACE")
    obj["part_id"] = 1
    obj["rigid_so3_buttons"] = True
    return obj


def extract_button_centers(src_btn_obj: bpy.types.Object) -> list[tuple[Vector, Vector]]:
    """Extract connected button island centroids + outward front normal from noisy button geometry."""
    me = src_btn_obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    visited = set()
    buttons = []
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
        if len(comp) >= 4:
            c = sum((v.co for v in comp), Vector((0.0, 0.0, 0.0))) / float(len(comp))
            buttons.append((c, Vector((0.0, -1.0, 0.0))))
    bm.free()
    buttons.sort(key=lambda item: -item[0].z)
    return buttons


def pack_image_datablock(img: bpy.types.Image) -> None:
    """Embed external image bytes into the .blend file via pure bpy.data (Linux/Colab portable)."""
    if img is not None and not img.packed_file:
        try:
            img.pack()
        except Exception:
            pass


def _load_image_has_transparency(img: bpy.types.Image) -> bool:
    """Detect whether an image carries meaningful alpha < 0.99 (e.g., RGBA decals)."""
    if img.channels < 4:
        return False
    n_px = int(img.size[0]) * int(img.size[1])
    if n_px <= 0:
        return False
    buf = np.empty(n_px * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    return bool(buf[3::4].min() < 0.99)


def _wire_srgb_tex_node(
    nt: bpy.types.NodeTree,
    tex_path: Path,
    uv_name: str,
    extension: str = "CLIP",
    scale: float = 1.0,
) -> bpy.types.Node:
    """Create a packed sRGB ShaderNodeTexImage wired from `ShaderNodeUVMap` (+ optional Mapping)."""
    uv_node = nt.nodes.new("ShaderNodeUVMap")
    uv_node.uv_map = uv_name
    vec_out = uv_node.outputs["UV"]
    if abs(float(scale) - 1.0) > 1e-4:
        mp = nt.nodes.new("ShaderNodeMapping")
        mp.inputs["Scale"].default_value = (float(scale), float(scale), 1.0)
        nt.links.new(vec_out, mp.inputs["Vector"])
        vec_out = mp.outputs["Vector"]

    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(Path(tex_path).resolve()), check_existing=True)
    tex.image.colorspace_settings.name = "sRGB"
    tex.extension = extension
    pack_image_datablock(tex.image)
    nt.links.new(vec_out, tex.inputs["Vector"])
    return tex


def make_pbr_material(
    name: str,
    base_rgba: tuple[float, float, float, float],
    roughness: float = 0.80,
    sheen: float = 0.12,
    normal_tex: Path | None = None,
    normal_strength: float = 0.25,
    albedo_tex: Path | None = None,
    metallic: float = 0.0,
    base_albedo_tex: Path | None = None,
    base_uv_map: str = "UVMap",
    base_uv_scale: float = 1.0,
    front_decal_tex: Path | None = None,
    front_uv_map: str = "UVMap_Art_Front",
    back_decal_tex: Path | None = None,
    back_uv_map: str = "UVMap_Art_Back",
    normal_uv_scale: float = 6.0,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = base_rgba
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if "Sheen Weight" in bsdf.inputs:
        bsdf.inputs["Sheen Weight"].default_value = sheen
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    # Backward-compatible `albedo_tex`: auto-detect RGBA decals vs opaque base fabric textures.
    if albedo_tex is not None and Path(albedo_tex).is_file() and base_albedo_tex is None and front_decal_tex is None:
        probe_img = bpy.data.images.load(str(Path(albedo_tex).resolve()), check_existing=True)
        is_decal = _load_image_has_transparency(probe_img) or ("compiled_uvs" in Path(albedo_tex).parts)
        if is_decal:
            front_decal_tex = albedo_tex
        else:
            base_albedo_tex = albedo_tex

    color_socket = None
    if base_albedo_tex is not None and Path(base_albedo_tex).is_file():
        t_base = _wire_srgb_tex_node(nt, Path(base_albedo_tex), base_uv_map, extension="REPEAT", scale=base_uv_scale)
        color_socket = t_base.outputs["Color"]

    for decal_tex, uv_map in ((front_decal_tex, front_uv_map), (back_decal_tex, back_uv_map)):
        if decal_tex is None or not Path(decal_tex).is_file():
            continue
        t_dec = _wire_srgb_tex_node(nt, Path(decal_tex), uv_map, extension="CLIP", scale=1.0)
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "MIX"
        # Blender 5.2 ShaderNodeMix RGBA socket indices: 0=Factor(VALUE), 6=A(RGBA), 7=B(RGBA), out 2=Result(RGBA)
        nt.links.new(t_dec.outputs["Alpha"], mix.inputs[0])
        if color_socket is not None:
            nt.links.new(color_socket, mix.inputs[6])
        else:
            mix.inputs[6].default_value = base_rgba
        nt.links.new(t_dec.outputs["Color"], mix.inputs[7])
        color_socket = mix.outputs[2]

    if color_socket is not None:
        nt.links.new(color_socket, bsdf.inputs["Base Color"])

    if normal_tex is not None and Path(normal_tex).is_file():
        uv_nrm = nt.nodes.new("ShaderNodeUVMap")
        uv_nrm.uv_map = base_uv_map
        mp_nrm = nt.nodes.new("ShaderNodeMapping")
        mp_nrm.inputs["Scale"].default_value = (float(normal_uv_scale), float(normal_uv_scale), 1.0)
        nt.links.new(uv_nrm.outputs["UV"], mp_nrm.inputs["Vector"])

        ntex = nt.nodes.new("ShaderNodeTexImage")
        ntex.image = bpy.data.images.load(str(Path(normal_tex).resolve()), check_existing=True)
        ntex.image.colorspace_settings.name = "Non-Color"
        ntex.extension = "REPEAT"
        pack_image_datablock(ntex.image)
        nt.links.new(mp_nrm.outputs["Vector"], ntex.inputs["Vector"])

        nmap = nt.nodes.new("ShaderNodeNormalMap")
        nmap.inputs["Strength"].default_value = float(normal_strength)
        nt.links.new(ntex.outputs["Color"], nmap.inputs["Color"])
        nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

    return mat
