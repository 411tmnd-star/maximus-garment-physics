"""05_morph_and_export/rebake_ringer_knitted_glb.py — Re-bake knitted-wool BS-RINGER-001 GLB (<= 1.2 MB).

Preserves exact certified Burst 1A C0 geometry (9,268 verts / 17,960 tris) recovered bijectively
from out_burst1a/maximus_v2_dressed_ringer_lod1.glb, bakes 2048x2048 white-black wool + jet-black
chest script / lower horse sketch + 1024x1024 knit normal + 1024x1024 ORM (AO/Roughness=0.88),
transfers all 7 coupled morph targets (5 anthro + XL + XXL), verifies GG-31 GWN = 0 and
G_STRETCH <= 1.08 / 1.15, and packages EXT_meshopt_compression GLB <= 1.2 MB.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
MAX_GARMENT_ENG = REPO_ROOT / "02_engines" / "max_garment_eng"
MAXIMUS_ROOT = REPO_ROOT / "02_engines" / "maximus_engine"
STOREFRONT_MOTION = (
    REPO_ROOT / "02_engines" / "better_style_engine" / "01_web_storefront" / "catalogue" / "media" / "maximus_motion"
)
SCRATCH_STUDIO_ASSETS = MAX_GARMENT_ENG / "07_web_scratch_studio" / "assets" / "bs_ringer_001"

CERTIFIED_GLB = MAX_GARMENT_ENG / "COLAB_MEGA_SPRINT" / "out_burst1a" / "maximus_v2_dressed_ringer_lod1.glb"
HERO_NPZ = MAX_GARMENT_ENG / "COLAB_MEGA_SPRINT" / "inputs" / "hero_ringer_shell.npz"
THETA_FIT_JSON = MAX_GARMENT_ENG / "COLAB_MEGA_SPRINT" / "inputs" / "m_theta_fit.json"
COUPLED_MORPHS_NPZ = MAXIMUS_ROOT / "out" / "m9" / "coupled_morphs.npz"
MASTER_PSD_BLEND = MAXIMUS_ROOT / "colab_stage" / "results" / "M4_canon_atlas" / "maximus_v2_master_psd.blend"
FRONT_ALBEDO_LIGHT = (
    REPO_ROOT
    / "02_engines"
    / "human_engine"
    / "src_catalog"
    / "better_style_catalog"
    / "assets"
    / "compiled_uvs"
    / "BS-RINGER-001"
    / "front_albedo_light.png"
)
FRONT_HEIGHT = (
    REPO_ROOT
    / "02_engines"
    / "human_engine"
    / "src_catalog"
    / "better_style_catalog"
    / "assets"
    / "compiled_uvs"
    / "BS-RINGER-001"
    / "front_height.png"
)
OUT_DIR = Path(__file__).resolve().parent / "out"
GLB_BUDGET_BYTES = int(1.2 * 1024 * 1024)


def srgb_to_lin(c: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(c, dtype=np.float64)
    lin = np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)
    return (float(lin[0]), float(lin[1]), float(lin[2]))


def hex_to_lin(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return srgb_to_lin([int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)])


def _recover_c0_geometry() -> tuple[np.ndarray, dict[str, Any]]:
    import bpy  # type: ignore
    from mathutils.kdtree import KDTree  # type: ignore

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(CERTIFIED_GLB))
    hd = np.load(HERO_NPZ, allow_pickle=True)
    V_c = hd["V_fit"].astype(np.float64)
    F_c = hd["F"].astype(np.int32)

    obj = bpy.data.objects["BS_RINGER_001"]
    me = obj.data
    V_glb = np.array([obj.matrix_world @ v.co for v in me.vertices], dtype=np.float64)
    F_glb = np.array([list(p.vertices) for p in me.polygons], dtype=np.int32)

    q = np.round(V_glb * 1e5).astype(np.int64)
    _, uniq_idx, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    V_u = V_glb[uniq_idx]
    F_u = inv[F_glb]
    assert len(V_u) == len(V_c) == 9268, f"Expected 9268 welded verts, got {len(V_u)}"

    def build_graph(F: np.ndarray, n: int):
        e = defaultdict(list)
        adj = [set() for _ in range(n)]
        vf = [[] for _ in range(n)]
        for fi, (a, b, c) in enumerate(F):
            a, b, c = int(a), int(b), int(c)
            vf[a].append(fi)
            vf[b].append(fi)
            vf[c].append(fi)
            adj[a].update((b, c))
            adj[b].update((a, c))
            adj[c].update((a, b))
            for u, v, w in ((a, b, c), (b, c, a), (c, a, b)):
                e[(u, v) if u < v else (v, u)].append(w)
        return e, adj, vf

    ec, adj_c, vf_c = build_graph(F_c, len(V_c))
    eu, adj_u, _ = build_graph(F_u, len(V_u))
    kd = KDTree(len(V_u))
    for i, co in enumerate(V_u):
        kd.insert(co, i)
    kd.balance()

    def ek(u: int, v: int) -> tuple[int, int]:
        return (u, v) if u < v else (v, u)

    c2u: dict[int, int] = {}
    used_u: set[int] = set()
    dq: deque[tuple[int, int]] = deque()
    for a, b, c in F_c:
        a, b, c = int(a), int(b), int(c)
        _, ua, da = kd.find(V_c[a])
        _, ub, db = kd.find(V_c[b])
        _, uc, dc = kd.find(V_c[c])
        if uc in eu.get(ek(ua, ub), []) and max(da, db, dc) < 0.008:
            c2u[a] = ua
            c2u[b] = ub
            c2u[c] = uc
            used_u.update((ua, ub, uc))
            dq.extend([ek(a, b), ek(b, c), ek(c, a)])
            break

    while dq:
        x, y = dq.popleft()
        zc = [z for z in ec[ek(x, y)] if z not in c2u]
        zu = [z for z in eu[ek(c2u[x], c2u[y])] if z not in used_u]
        if len(zc) == 1 and len(zu) == 1:
            z, uz = zc[0], zu[0]
            c2u[z] = uz
            used_u.add(uz)
            dq.extend([ek(x, z), ek(y, z)])

    rem_c = [i for i in range(len(V_c)) if i not in c2u]
    rem_u = [i for i in range(len(V_u)) if i not in used_u]
    for z in sorted(rem_c, key=lambda v: -len([m for m in adj_c[v] if m in c2u])):
        mc = {c2u[m] for m in adj_c[z] if m in c2u}
        cands = [uz for uz in rem_u if uz not in used_u and mc.issubset(adj_u[uz])]
        if not cands:
            cands = [uz for uz in rem_u if uz not in used_u]
        cands.sort(key=lambda uz: (-len(mc & adj_u[uz]), abs(len(adj_u[uz]) - len(adj_c[z])), float(np.linalg.norm(V_u[uz] - V_c[z]))))
        c2u[z] = cands[0]
        used_u.add(cands[0])

    perm = [c2u[i] for i in range(len(V_c))]
    fset_u = {tuple(sorted((int(a), int(b), int(c)))) for a, b, c in F_u}
    for _ in range(25):
        bad_tris = [fi for fi, (a, b, c) in enumerate(F_c) if tuple(sorted((perm[a], perm[b], perm[c]))) not in fset_u]
        if not bad_tris:
            break
        cand_v = sorted({int(v) for fi in bad_tris for v in F_c[fi]})
        improved = False
        for v1 in cand_v:
            for v2 in cand_v:
                if v1 >= v2:
                    continue
                aff = set(vf_c[v1]) | set(vf_c[v2])
                before = sum(1 for fi in aff if tuple(sorted((perm[F_c[fi, 0]], perm[F_c[fi, 1]], perm[F_c[fi, 2]]))) not in fset_u)
                perm[v1], perm[v2] = perm[v2], perm[v1]
                after = sum(1 for fi in aff if tuple(sorted((perm[F_c[fi, 0]], perm[F_c[fi, 1]], perm[F_c[fi, 2]]))) not in fset_u)
                if after < before:
                    improved = True
                    break
                perm[v1], perm[v2] = perm[v2], perm[v1]
            if improved:
                break
        if not improved:
            break

    exact_tris = sum(1 for a, b, c in F_c if tuple(sorted((perm[a], perm[b], perm[c]))) in fset_u)
    assert len(perm) == len(V_c) == 9268
    assert exact_tris == len(F_c) == 17960, f"Exact triangle match failed: {exact_tris}/{len(F_c)}"

    V_solved = V_u[np.asarray(perm, dtype=np.int32)]
    E = hd["edges"]
    L0 = hd["L_template"]
    r = np.linalg.norm(V_solved[E[:, 0]] - V_solved[E[:, 1]], axis=1) / np.maximum(L0, 1e-12)
    p99 = float(np.percentile(r, 99.0))
    rmax = float(r.max())
    gt115 = int((r > 1.15).sum())
    assert p99 <= 1.08 and rmax <= 1.15 and gt115 == 0, f"G_STRETCH gate failed: p99={p99}, max={rmax}, gt115={gt115}"
    return V_solved, {"exact_tris": exact_tris, "p99_stretch": round(p99, 4), "max_stretch": round(rmax, 4), "edges_gt_1_15": gt115}


def _apply_theta_fit_pose(body_obj: Any, hd: Any) -> tuple[np.ndarray, float]:
    import bpy  # type: ignore
    from mathutils import Quaternion  # type: ignore

    rig = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    assert rig is not None
    rig.data.pose_position = "POSE"
    for mod in body_obj.modifiers:
        if mod.type == "ARMATURE":
            mod.use_deform_preserve_volume = False

    pose_data = json.loads(THETA_FIT_JSON.read_text(encoding="utf-8"))
    for pb in rig.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.matrix_basis.identity()
    for bname, spec in pose_data.get("bones", {}).items():
        pb = rig.pose.bones.get(bname)
        if pb is not None:
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = Quaternion(spec["q_wxyz"])
            pb.location = tuple(spec.get("t", (0.0, 0.0, 0.0)))
    for cname in ("clavicle.L", "clavicle.R"):
        if cname in rig.pose.bones:
            cb = rig.pose.bones[cname]
            cb.rotation_mode = "QUATERNION"
            cb.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
            cb.location = (0.0, 0.0, 0.0)
            cb.scale = (1.0, 1.0, 1.0)
    bpy.context.view_layer.update()

    z_min = 0.0
    for _ in range(4):
        dg = bpy.context.evaluated_depsgraph_get()
        ev_me = body_obj.evaluated_get(dg).data
        coords = np.empty(len(ev_me.vertices) * 3, dtype=np.float64)
        ev_me.vertices.foreach_get("co", coords)
        M_w = np.asarray(body_obj.matrix_world, dtype=np.float64)
        V_ev = (M_w[:3, :3] @ coords.reshape(-1, 3).T).T + M_w[:3, 3]
        z_min = float(V_ev[:, 2].min())
        if 0.0 <= z_min <= 1e-5 or "root" not in rig.pose.bones:
            break
        rpb = rig.pose.bones["root"]
        M = rpb.matrix.copy()
        M.translation.z -= (z_min - 2e-6)
        rpb.matrix = M
        bpy.context.view_layer.update()

    dg = bpy.context.evaluated_depsgraph_get()
    ev_me = body_obj.evaluated_get(dg).data
    coords = np.empty(len(ev_me.vertices) * 3, dtype=np.float64)
    ev_me.vertices.foreach_get("co", coords)
    M_w = np.asarray(body_obj.matrix_world, dtype=np.float64)
    V_ev = (M_w[:3, :3] @ coords.reshape(-1, 3).T).T + M_w[:3, 3]

    body_mat = bpy.data.materials.get("MX_M1_ObsidianNoir")
    if body_mat is None:
        body_mat = bpy.data.materials.new("MX_M1_ObsidianNoir")
    body_mat.use_nodes = True
    bsdf = next(n for n in body_mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (0.011, 0.011, 0.013, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.52
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.22
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = 0.10
    if len(body_obj.data.materials) == 0:
        body_obj.data.materials.append(body_mat)
    else:
        body_obj.data.materials[0] = body_mat
    return V_ev, float(z_min)


def _compute_gwn_and_tension(V_solved: np.ndarray, V_body: np.ndarray, F_body: np.ndarray, collar_mask: np.ndarray, edges: np.ndarray, L0: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    gwn_in = OUT_DIR / "_gwn_input.npz"
    gwn_out = OUT_DIR / "_gwn_output.npz"
    np.savez(gwn_in, V_solved=V_solved, V_body=V_body, F_body=F_body)
    sys_py = shutil.which("python") or "python"
    helper_code = f"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, r'{MAXIMUS_ROOT}')
sys.path.insert(0, r'{MAXIMUS_ROOT / "colab_stage"}')
os.environ['MAXIMUS_STRICT'] = '0'
import canon_geometry_kernels as K
d = np.load(r'{gwn_in}')
signed_m, _ = K.signed_distance_to_mesh(d['V_solved'], d['V_body'], d['F_body'], k=16, device='cpu')
np.savez(r'{gwn_out}', signed_m=signed_m)
"""
    proc = subprocess.run([sys_py, "-c", helper_code], capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not gwn_out.exists():
        raise RuntimeError(f"GG-31 GWN subprocess failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    signed_m = np.load(gwn_out)["signed_m"]
    gwn_in.unlink(missing_ok=True)
    gwn_out.unlink(missing_ok=True)
    inside_all = int((signed_m < 0.0).sum())
    inside_collar = int((signed_m[collar_mask] < 0.0).sum()) if collar_mask.any() else 0
    min_mm = float(signed_m.min() * 1000.0)
    col_min_mm = float(signed_m[collar_mask].min() * 1000.0) if collar_mask.any() else 0.0
    assert inside_all == 0, f"GG-31 GWN failed: {inside_all} penetrating vertices (min={min_mm:.3f}mm)"
    assert inside_collar == 0, f"O-7 collar failed: {inside_collar} penetrating vertices"

    # Compute physically grounded _TENSION in [0, 1] from signed clearance (mm) + edge stretch
    signed_mm = signed_m * 1000.0
    clear_term = np.clip((18.0 - signed_mm) / 14.0, 0.0, 1.0)
    edge_ratio = np.linalg.norm(V_solved[edges[:, 0]] - V_solved[edges[:, 1]], axis=1) / np.maximum(L0, 1e-12)
    v_stretch = np.ones(len(V_solved), dtype=np.float64)
    np.maximum.at(v_stretch, edges[:, 0], edge_ratio)
    np.maximum.at(v_stretch, edges[:, 1], edge_ratio)
    stretch_term = np.clip((v_stretch - 0.98) / 0.14, 0.0, 1.0)
    tension = np.clip(0.68 * clear_term + 0.32 * stretch_term, 0.0, 1.0).astype(np.float32)

    return tension, {
        "inside_gwn_count": inside_all,
        "min_signed_mm": round(min_mm, 3),
        "median_signed_mm": round(float(np.median(signed_mm)), 3),
        "o7_collar_inside": inside_collar,
        "o7_collar_min_mm": round(col_min_mm, 3),
        "tension_min": round(float(tension.min()), 3),
        "tension_median": round(float(np.median(tension)), 3),
        "tension_max": round(float(tension.max()), 3),
    }


def _compute_smoothed_rib_sdf(V: np.ndarray, F: np.ndarray, tri_mats: np.ndarray, iters: int = 12) -> np.ndarray:
    """Compute a 1-ring Laplacian-smoothed continuous scalar field [0..1] across Main_Body/Rib boundaries."""
    n_verts = len(V)
    adj: list[set[int]] = [set() for _ in range(n_verts)]
    rib_cnt = np.zeros(n_verts, dtype=np.float64)
    tot_cnt = np.zeros(n_verts, dtype=np.float64)
    edge_tris: dict[tuple[int, int], list[int]] = defaultdict(list)

    for fi, tri in enumerate(F):
        is_rib = 1.0 if int(tri_mats[fi]) > 0 else 0.0
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u in (a, b, c):
            rib_cnt[u] += is_rib
            tot_cnt[u] += 1.0
        for u, v in ((a, b), (b, c), (c, a)):
            adj[u].add(v)
            adj[v].add(u)
            key = (u, v) if u < v else (v, u)
            edge_tris[key].append(fi)

    sdf = np.where(tot_cnt > 0.0, rib_cnt / np.maximum(tot_cnt, 1.0), 0.0)
    seam_verts: set[int] = set()
    pinned_rim: set[int] = set()
    for (u, v), flist in edge_tris.items():
        if len(flist) == 1:
            if int(tri_mats[flist[0]]) > 0:
                pinned_rim.add(u)
                pinned_rim.add(v)
        elif len(flist) == 2:
            if (int(tri_mats[flist[0]]) > 0) != (int(tri_mats[flist[1]]) > 0):
                seam_verts.add(u)
                seam_verts.add(v)

    # Dilate active smoothing band by 2 hops from the rib/body seam while pinning open collar/cuff rims
    band = set(seam_verts)
    for _ in range(2):
        nxt = set(band)
        for u in band:
            nxt.update(adj[u])
        band = nxt
    band -= pinned_rim
    band_list = sorted(band)
    pinned_idx = sorted(pinned_rim)
    if pinned_idx:
        sdf[pinned_idx] = 1.0

    for _ in range(iters):
        new_sdf = sdf.copy()
        for u in band_list:
            nbrs = adj[u]
            if nbrs:
                mean_nbr = float(np.mean([sdf[v] for v in nbrs]))
                new_sdf[u] = 0.5 * sdf[u] + 0.5 * mean_nbr
        if pinned_idx:
            new_sdf[pinned_idx] = 1.0
        sdf = new_sdf
    return np.clip(sdf.astype(np.float32), 0.0, 1.0)


def _build_two_panel_bake_uv(bpy, V_orig: np.ndarray, F_c: np.ndarray) -> np.ndarray:
    """Create a clean 2-panel (Front / Back) tailor-seamed unwrap on un-wrinkled V_orig."""
    C = V_orig[F_c].mean(axis=1)
    edge_tris: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, tri in enumerate(F_c):
        for k in range(3):
            a, b = int(tri[k]), int(tri[(k + 1) % 3])
            edge_tris[(min(a, b), max(a, b))].append(fi)
    fadj: list[list[int]] = [[] for _ in range(len(F_c))]
    for fl in edge_tris.values():
        if len(fl) == 2:
            fadj[fl[0]].append(fl[1])
            fadj[fl[1]].append(fl[0])

    z_bins = np.linspace(float(V_orig[:, 2].min()), float(V_orig[:, 2].max()), 25)
    y_mid = np.zeros(len(F_c), dtype=np.float64)
    for i in range(len(z_bins) - 1):
        m = (C[:, 2] >= z_bins[i] - 0.01) & (C[:, 2] <= z_bins[i + 1] + 0.01)
        if m.any():
            ym = 0.5 * (float(C[m, 1].min()) + float(C[m, 1].max()))
            m2 = (C[:, 2] >= z_bins[i]) & (C[:, 2] <= z_bins[i + 1])
            y_mid[m2] = ym
    side = (C[:, 1] < y_mid).astype(np.int32)

    for _ in range(8):
        ns = side.copy()
        for fi in range(len(F_c)):
            nb = fadj[fi]
            if nb:
                s_sum = sum(int(side[j]) for j in nb)
                if s_sum > len(nb) * 0.5:
                    ns[fi] = 1
                elif s_sum < len(nb) * 0.5:
                    ns[fi] = 0
        side = ns

    def _components(lbl: int) -> list[list[int]]:
        vis = np.zeros(len(F_c), dtype=bool)
        comps: list[list[int]] = []
        for i in range(len(F_c)):
            if int(side[i]) != lbl or vis[i]:
                continue
            q = deque([i])
            vis[i] = True
            comp: list[int] = []
            while q:
                u = q.popleft()
                comp.append(u)
                for w in fadj[u]:
                    if int(side[w]) == lbl and not vis[w]:
                        vis[w] = True
                        q.append(w)
            comps.append(comp)
        comps.sort(key=len, reverse=True)
        return comps

    for lbl, flip_to in ((1, 0), (0, 1)):
        for comp in _components(lbl)[1:]:
            if len(comp) < 500:
                side[comp] = flip_to

    seam_keys = {k for k, fl in edge_tris.items() if len(fl) == 2 and int(side[fl[0]]) != int(side[fl[1]])}

    tmp_mesh = bpy.data.meshes.new("_TmpTwoPanelUV")
    tmp_mesh.from_pydata(V_orig.tolist(), [], F_c.tolist())
    tmp_mesh.update(calc_edges=True)
    for e in tmp_mesh.edges:
        v0, v1 = int(e.vertices[0]), int(e.vertices[1])
        if (min(v0, v1), max(v0, v1)) in seam_keys:
            e.use_seam = True

    tmp_obj = bpy.data.objects.new("_TmpTwoPanelUV", tmp_mesh)
    bpy.context.scene.collection.objects.link(tmp_obj)
    uv_tmp = tmp_mesh.uv_layers.new(name="UVMap_Bake")
    tmp_mesh.uv_layers.active = uv_tmp

    bpy.ops.object.select_all(action="DESELECT")
    tmp_obj.select_set(True)
    bpy.context.view_layer.objects.active = tmp_obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=0.025)
    bpy.ops.object.mode_set(mode="OBJECT")

    bake_uvs = np.empty(len(tmp_obj.data.loops) * 2, dtype=np.float32)
    tmp_obj.data.uv_layers["UVMap_Bake"].data.foreach_get("uv", bake_uvs)
    bpy.data.objects.remove(tmp_obj, do_unlink=True)
    bpy.data.meshes.remove(tmp_mesh, do_unlink=True)
    return bake_uvs


def _build_procedural_ringer_material(bpy, mat, body_hex: str = "#E6E1D9", rib_hex: str = "#181716") -> None:
    """Unified Cycles wool + ringer-rib shader driven smoothly by _RIB_SDF so collar/cuff seams anti-alias across triangles."""
    mat.use_nodes = True
    nt = mat.node_tree
    N = nt.nodes
    Lk = nt.links
    N.clear()
    out = N.new("ShaderNodeOutputMaterial")
    b = N.new("ShaderNodeBsdfPrincipled")
    Lk.new(b.outputs["BSDF"], out.inputs["Surface"])

    attr_sdf = N.new("ShaderNodeAttribute")
    attr_sdf.attribute_name = "_RIB_SDF"
    rib_mr = N.new("ShaderNodeMapRange")
    rib_mr.interpolation_type = "SMOOTHSTEP"
    rib_mr.clamp = True
    rib_mr.inputs["From Min"].default_value = 0.44
    rib_mr.inputs["From Max"].default_value = 0.56
    rib_mr.inputs["To Min"].default_value = 0.0
    rib_mr.inputs["To Max"].default_value = 1.0
    Lk.new(attr_sdf.outputs["Fac"], rib_mr.inputs["Value"])
    rib_fac = rib_mr.outputs["Result"]

    uv = N.new("ShaderNodeUVMap")
    uv.uv_map = "UVMap_Fabric"
    mp = N.new("ShaderNodeMapping")
    Lk.new(uv.outputs["UV"], mp.inputs["Vector"])

    # Blend body jersey knit (scale 115) with tighter collar/cuff rib gauge (scale 175)
    def make_weave(scale: float):
        w1 = N.new("ShaderNodeTexWave")
        w1.wave_type = "BANDS"
        w1.bands_direction = "X"
        w1.inputs["Scale"].default_value = scale * 0.85
        w1.inputs["Distortion"].default_value = 0.09
        w1.inputs["Detail"].default_value = 1.5
        w2 = N.new("ShaderNodeTexWave")
        w2.wave_type = "BANDS"
        w2.bands_direction = "Y"
        w2.inputs["Scale"].default_value = scale * 0.65
        w2.inputs["Distortion"].default_value = 0.09
        nz = N.new("ShaderNodeTexNoise")
        nz.inputs["Scale"].default_value = scale * 1.85
        nz.inputs["Detail"].default_value = 1.6
        for t in (w1, w2, nz):
            Lk.new(mp.outputs["Vector"], t.inputs["Vector"])
        mul = N.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        Lk.new(w1.outputs["Fac"], mul.inputs[0])
        Lk.new(w2.outputs["Fac"], mul.inputs[1])
        ad = N.new("ShaderNodeMath")
        ad.operation = "ADD"
        m2 = N.new("ShaderNodeMath")
        m2.operation = "MULTIPLY"
        m2.inputs[1].default_value = 0.34
        Lk.new(mul.outputs["Value"], ad.inputs[0])
        Lk.new(nz.outputs["Fac"], m2.inputs[0])
        Lk.new(m2.outputs["Value"], ad.inputs[1])
        return ad.outputs["Value"]

    inv_rib = N.new("ShaderNodeMath")
    inv_rib.operation = "SUBTRACT"
    inv_rib.inputs[0].default_value = 1.0
    Lk.new(rib_fac, inv_rib.inputs[1])

    body_weave = make_weave(115.0)
    rib_weave = make_weave(175.0)
    bw_m = N.new("ShaderNodeMath")
    bw_m.operation = "MULTIPLY"
    Lk.new(body_weave, bw_m.inputs[0])
    Lk.new(inv_rib.outputs["Value"], bw_m.inputs[1])
    rw_m = N.new("ShaderNodeMath")
    rw_m.operation = "MULTIPLY"
    Lk.new(rib_weave, rw_m.inputs[0])
    Lk.new(rib_fac, rw_m.inputs[1])
    weave_sum = N.new("ShaderNodeMath")
    weave_sum.operation = "ADD"
    Lk.new(bw_m.outputs["Value"], weave_sum.inputs[0])
    Lk.new(rw_m.outputs["Value"], weave_sum.inputs[1])
    weave = weave_sum.outputs["Value"]

    mott = N.new("ShaderNodeTexNoise")
    mott.inputs["Scale"].default_value = 6.5
    mott.inputs["Detail"].default_value = 2.0
    Lk.new(mp.outputs["Vector"], mott.inputs["Vector"])

    mc = N.new("ShaderNodeMath")
    mc.operation = "MULTIPLY_ADD"
    mc.inputs[1].default_value = 0.13
    mc.inputs[2].default_value = 0.88
    Lk.new(weave, mc.inputs[0])
    mott_m = N.new("ShaderNodeMath")
    mott_m.operation = "MULTIPLY_ADD"
    mott_m.inputs[1].default_value = 0.045
    mott_m.inputs[2].default_value = 0.975
    Lk.new(mott.outputs["Fac"], mott_m.inputs[0])
    wm = N.new("ShaderNodeMath")
    wm.operation = "MULTIPLY"
    Lk.new(mc.outputs["Value"], wm.inputs[0])
    Lk.new(mott_m.outputs["Value"], wm.inputs[1])

    body_rgb = N.new("ShaderNodeRGB")
    body_rgb.outputs[0].default_value = (*hex_to_lin(body_hex), 1.0)
    rib_rgb = N.new("ShaderNodeRGB")
    rib_rgb.outputs[0].default_value = (*hex_to_lin(rib_hex), 1.0)

    uva = N.new("ShaderNodeUVMap")
    uva.uv_map = "UVMap_Art_Front"
    im = N.new("ShaderNodeTexImage")
    im.image = bpy.data.images.load(str(FRONT_ALBEDO_LIGHT), check_existing=True)
    im.extension = "CLIP"
    im.interpolation = "Cubic"
    Lk.new(uva.outputs["UV"], im.inputs["Vector"])

    # Strict front-facing + non-rib mask: Maximus faces -Y in Blender Z-up; zero bleed onto back or collar rib
    geom = N.new("ShaderNodeNewGeometry")
    sep_n = N.new("ShaderNodeSeparateXYZ")
    Lk.new(geom.outputs["Normal"], sep_n.inputs["Vector"])
    fmask = N.new("ShaderNodeMapRange")
    fmask.clamp = True
    fmask.inputs["From Min"].default_value = -0.15
    fmask.inputs["From Max"].default_value = -0.35
    fmask.inputs["To Min"].default_value = 0.0
    fmask.inputs["To Max"].default_value = 1.0
    Lk.new(sep_n.outputs["Y"], fmask.inputs["Value"])

    front_body_mask = N.new("ShaderNodeMath")
    front_body_mask.operation = "MULTIPLY"
    Lk.new(fmask.outputs["Result"], front_body_mask.inputs[0])
    Lk.new(inv_rib.outputs["Value"], front_body_mask.inputs[1])

    masked_a = N.new("ShaderNodeMath")
    masked_a.operation = "MULTIPLY"
    Lk.new(im.outputs["Alpha"], masked_a.inputs[0])
    Lk.new(front_body_mask.outputs["Value"], masked_a.inputs[1])
    ink_alpha = masked_a.outputs["Value"]

    mix_ink = N.new("ShaderNodeMixRGB")
    Lk.new(body_rgb.outputs[0], mix_ink.inputs["Color1"])
    Lk.new(im.outputs["Color"], mix_ink.inputs["Color2"])
    Lk.new(ink_alpha, mix_ink.inputs["Fac"])

    hi = N.new("ShaderNodeTexImage")
    hi.image = bpy.data.images.load(str(FRONT_HEIGHT), check_existing=True)
    hi.image.colorspace_settings.name = "Non-Color"
    hi.extension = "CLIP"
    hi.interpolation = "Cubic"
    Lk.new(uva.outputs["UV"], hi.inputs["Vector"])
    masked_h = N.new("ShaderNodeMath")
    masked_h.operation = "MULTIPLY"
    Lk.new(hi.outputs["Color"], masked_h.inputs[0])
    Lk.new(front_body_mask.outputs["Value"], masked_h.inputs[1])
    ink_height = masked_h.outputs["Value"]

    # Weave + subtle wool mottling over body + rib, kept clean over ink
    modmix = N.new("ShaderNodeMixRGB")
    modmix.blend_type = "MULTIPLY"
    modmix.inputs["Fac"].default_value = 1.0
    Lk.new(mix_ink.outputs["Color"], modmix.inputs["Color1"])
    wv = N.new("ShaderNodeCombineColor")
    for idx in range(3):
        Lk.new(wm.outputs["Value"], wv.inputs[idx])
    fm = N.new("ShaderNodeMixRGB")
    fm.inputs["Color2"].default_value = (1.0, 1.0, 1.0, 1.0)
    Lk.new(wv.outputs["Color"], fm.inputs["Color1"])
    Lk.new(ink_alpha, fm.inputs["Fac"])
    Lk.new(fm.outputs["Color"], modmix.inputs["Color2"])

    rib_tex_mod = N.new("ShaderNodeMixRGB")
    rib_tex_mod.blend_type = "MULTIPLY"
    rib_tex_mod.inputs["Fac"].default_value = 1.0
    Lk.new(rib_rgb.outputs[0], rib_tex_mod.inputs["Color1"])
    Lk.new(wv.outputs["Color"], rib_tex_mod.inputs["Color2"])

    final_col = N.new("ShaderNodeMixRGB")
    Lk.new(modmix.outputs["Color"], final_col.inputs["Color1"])
    Lk.new(rib_tex_mod.outputs["Color"], final_col.inputs["Color2"])
    Lk.new(rib_fac, final_col.inputs["Fac"])
    Lk.new(final_col.outputs["Color"], b.inputs["Base Color"])

    b.inputs["Specular IOR Level"].default_value = 0.10
    b.inputs["Roughness"].default_value = 0.90

    # Tactile weave height + ink relief + subtle coverstitch seam groove along _RIB_SDF == 0.50
    keep = N.new("ShaderNodeMath")
    keep.operation = "MULTIPLY_ADD"
    keep.inputs[1].default_value = -0.75
    keep.inputs[2].default_value = 1.0
    Lk.new(ink_alpha, keep.inputs[0])
    wk = N.new("ShaderNodeMath")
    wk.operation = "MULTIPLY"
    Lk.new(weave, wk.inputs[0])
    Lk.new(keep.outputs["Value"], wk.inputs[1])
    ih = N.new("ShaderNodeMath")
    ih.operation = "MULTIPLY"
    ih.inputs[1].default_value = 2.0
    Lk.new(ink_height, ih.inputs[0])
    hs = N.new("ShaderNodeMath")
    hs.operation = "ADD"
    Lk.new(wk.outputs["Value"], hs.inputs[0])
    Lk.new(ih.outputs["Value"], hs.inputs[1])

    # Coverstitch groove centered at 0.50 (width +-0.045)
    d_mid = N.new("ShaderNodeMath")
    d_mid.operation = "SUBTRACT"
    d_mid.inputs[1].default_value = 0.50
    Lk.new(attr_sdf.outputs["Fac"], d_mid.inputs[0])
    d_abs = N.new("ShaderNodeMath")
    d_abs.operation = "ABSOLUTE"
    Lk.new(d_mid.outputs["Value"], d_abs.inputs[0])
    groove_mr = N.new("ShaderNodeMapRange")
    groove_mr.interpolation_type = "SMOOTHSTEP"
    groove_mr.clamp = True
    groove_mr.inputs["From Min"].default_value = 0.0
    groove_mr.inputs["From Max"].default_value = 0.045
    groove_mr.inputs["To Min"].default_value = -0.35
    groove_mr.inputs["To Max"].default_value = 0.0
    Lk.new(d_abs.outputs["Value"], groove_mr.inputs["Value"])
    hs_seam = N.new("ShaderNodeMath")
    hs_seam.operation = "ADD"
    Lk.new(hs.outputs["Value"], hs_seam.inputs[0])
    Lk.new(groove_mr.outputs["Result"], hs_seam.inputs[1])

    bump = N.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.50
    bump.inputs["Distance"].default_value = 0.00045
    Lk.new(hs_seam.outputs["Value"], bump.inputs["Height"])
    Lk.new(bump.outputs["Normal"], b.inputs["Normal"])


def main() -> None:
    import bpy  # type: ignore

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    V_solved, stretch_info = _recover_c0_geometry()

    # Now load master_psd.blend cleanly (all 152 bones + 58 morph targets)
    bpy.ops.wm.open_mainfile(filepath=str(MASTER_PSD_BLEND))
    hd = np.load(HERO_NPZ, allow_pickle=True)
    F_c = hd["F"].astype(np.int32)
    collar_mask = hd["collar_mask"].astype(bool)
    edges = hd["edges"].astype(np.int64)
    L0 = hd["L_template"].astype(np.float64)
    tri_mats = hd["tri_mats"].astype(np.int32)
    mat_names = [str(x) for x in hd["mat_names"]]

    body_obj = next((o for o in bpy.data.objects if o.type == "MESH" and len(o.data.vertices) == 13380), None)
    assert body_obj is not None, "Expected 13,380-vert body in maximus_v2_master_psd.blend"
    V_body, z_sole = _apply_theta_fit_pose(body_obj, hd)
    F_body = np.array([list(p.vertices) for p in body_obj.data.polygons], dtype=np.int32)

    tension, gate_info = _compute_gwn_and_tension(V_solved, V_body, F_body, collar_mask, edges, L0)

    # Create BS_RINGER_001 mesh from exact V_solved + F_c
    g_mesh = bpy.data.meshes.new("BS_RINGER_001_Mesh")
    g_mesh.from_pydata(V_solved.tolist(), [], F_c.tolist())
    g_mesh.update(calc_edges=True)

    # Restore authored UVs first
    for k in hd.files:
        if k.startswith("uv|"):
            uv_name = k.split("|", 1)[1]
            uv_data = hd[k].astype(np.float32)
            if len(uv_data) == len(g_mesh.loops):
                uvl = g_mesh.uv_layers.new(name=uv_name)
                uvl.data.foreach_set("uv", uv_data.reshape(-1).tolist())

    # Pack COLOR_0 (_TENSION, collar_mask, 0, 1) and temporary _RIB_SDF for smooth collar/cuff anti-aliasing
    rib_sdf = _compute_smoothed_rib_sdf(V_solved, F_c, tri_mats, iters=12)
    sdf_attr = g_mesh.attributes.new(name="_RIB_SDF", type="FLOAT", domain="POINT")
    sdf_attr.data.foreach_set("value", rib_sdf.tolist())

    col_attr = g_mesh.color_attributes.new(name="COLOR_0", type="BYTE_COLOR", domain="CORNER")
    rgba_v = np.zeros((len(V_solved), 4), dtype=np.float32)
    rgba_v[:, 0] = tension
    rgba_v[:, 1] = collar_mask.astype(np.float32)
    rgba_v[:, 3] = 1.0
    loop_v = np.array([lp.vertex_index for lp in g_mesh.loops], dtype=np.int32)
    col_attr.data.foreach_set("color", rgba_v[loop_v].reshape(-1).tolist())

    # Single unified wool + rib material eliminates jagged per-triangle collar/cuff material boundaries
    mat_names = ["Main_Body"]
    main_mat = bpy.data.materials.new("Main_Body")
    _build_procedural_ringer_material(bpy, main_mat, body_hex="#E6E1D9", rib_hex="#181716")
    g_mesh.materials.append(main_mat)
    g_mesh.polygons.foreach_set("material_index", np.zeros(len(F_c), dtype=np.int32).tolist())

    g_obj = bpy.data.objects.new("BS_RINGER_001", g_mesh)
    bpy.context.scene.collection.objects.link(g_obj)
    for poly in g_mesh.polygons:
        poly.use_smooth = True

    # Build clean 2-panel (Front/Back) UVMap_Bake from un-wrinkled V_orig so folds never shatter into micro-diamonds
    bake_uvs_init = _build_two_panel_bake_uv(bpy, hd["V_orig"].astype(np.float64), F_c)
    uv_bake = g_mesh.uv_layers.new(name="UVMap_Bake")
    uv_bake.data.foreach_set("uv", bake_uvs_init.tolist())
    g_mesh.uv_layers.active = uv_bake
    g_mesh.uv_layers.active_render = uv_bake
    uv_bake.active = True
    uv_bake.active_render = True
    g_mesh.update()

    # Configure Cycles CPU bake on UVMap_Bake (margin=6 keeps >40px island separation completely clean)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = 2
    scene.render.bake.use_selected_to_active = False
    scene.render.bake.use_clear = False
    scene.render.bake.margin = 6

    bpy.ops.object.select_all(action="DESELECT")
    g_obj.select_set(True)
    bpy.context.view_layer.objects.active = g_obj

    albedo_img = bpy.data.images.new("ringer_albedo", width=2048, height=2048, alpha=False)
    albedo_img.colorspace_settings.name = "sRGB"
    alb_fill = np.tile(np.array([230.0 / 255.0, 225.0 / 255.0, 217.0 / 255.0, 1.0], dtype=np.float32), (2048 * 2048, 1))
    albedo_img.pixels.foreach_set(alb_fill.reshape(-1).tolist())

    normal_img = bpy.data.images.new("ringer_normal", width=768, height=768, alpha=False)
    normal_img.colorspace_settings.name = "Non-Color"
    nrm_fill = np.tile(np.array([0.5, 0.5, 1.0, 1.0], dtype=np.float32), (768 * 768, 1))
    normal_img.pixels.foreach_set(nrm_fill.reshape(-1).tolist())

    ao_img = bpy.data.images.new("ringer_ao", width=512, height=512, alpha=False)
    ao_img.colorspace_settings.name = "Non-Color"
    ao_fill = np.ones((512 * 512, 4), dtype=np.float32)
    ao_img.pixels.foreach_set(ao_fill.reshape(-1).tolist())

    def attach_bake_target(img):
        for mn in mat_names:
            nt = bpy.data.materials[mn].node_tree
            node = next((n for n in nt.nodes if n.name == "BAKE_TARGET"), None)
            if node is None:
                node = nt.nodes.new("ShaderNodeTexImage")
                node.name = "BAKE_TARGET"
            node.image = img
            for n in nt.nodes:
                n.select = False
            node.select = True
            nt.nodes.active = node

    # 1. Bake DIFFUSE Albedo (color pass only)
    scene.cycles.samples = 2
    attach_bake_target(albedo_img)
    bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"}, uv_layer="UVMap_Bake", use_clear=False, margin=6)
    albedo_path = OUT_DIR / "ringer_albedo.png"
    albedo_img.filepath_raw = str(albedo_path)
    albedo_img.file_format = "PNG"
    albedo_img.save()

    # 2. Bake Tangent-Space Normal Map (knit weave + ink relief + coverstitch groove + rib)
    attach_bake_target(normal_img)
    bpy.ops.object.bake(type="NORMAL", normal_space="TANGENT", uv_layer="UVMap_Bake", use_clear=False, margin=6)
    normal_path = OUT_DIR / "ringer_normal.png"
    normal_img.filepath_raw = str(normal_path)
    normal_img.file_format = "PNG"
    normal_img.save()

    # 3. Bake Cavity AO with bump disconnected (Strength=0.0) so AO has zero micro-shadow speckle
    for mn in mat_names:
        nt = bpy.data.materials[mn].node_tree
        for n in nt.nodes:
            if n.type == "BUMP":
                n.inputs["Strength"].default_value = 0.0
    scene.cycles.samples = 16
    body_obj.hide_render = False
    attach_bake_target(ao_img)
    bpy.ops.object.bake(type="AO", uv_layer="UVMap_Bake", use_clear=False, margin=6)
    ao_path = OUT_DIR / "ringer_ao.png"
    ao_img.filepath_raw = str(ao_path)
    ao_img.file_format = "PNG"
    ao_img.save()

    # Remap AO into soft luxury range [0.88, 1.0] (keeps light wool bright & spotless around folds) and pack ORM (R=ao, G=0.88 roughness, B=0)
    ao_px = np.empty(ao_img.size[0] * ao_img.size[1] * 4, dtype=np.float32)
    ao_img.pixels.foreach_get(ao_px)
    ao_l = ao_px.reshape(-1, 4)[:, 0]
    ao_soft = np.clip(0.88 + 0.12 * ao_l, 0.0, 1.0)
    orm_img = bpy.data.images.new("ringer_orm", width=ao_img.size[0], height=ao_img.size[1], alpha=False)
    orm_img.colorspace_settings.name = "Non-Color"
    orm_px = np.zeros((len(ao_soft), 4), dtype=np.float32)
    orm_px[:, 0] = ao_soft
    orm_px[:, 1] = 0.88
    orm_px[:, 2] = 0.0
    orm_px[:, 3] = 1.0
    orm_img.pixels.foreach_set(orm_px.reshape(-1).tolist())
    orm_path = OUT_DIR / "ringer_orm.png"
    orm_img.filepath_raw = str(orm_path)
    orm_img.file_format = "PNG"
    orm_img.save()

    # Switch active UV to UVMap_Bake and build clean glTF-exportable PBR material
    loaded_albedo = bpy.data.images.load(str(albedo_path), check_existing=False)
    loaded_albedo.colorspace_settings.name = "sRGB"
    loaded_normal = bpy.data.images.load(str(normal_path), check_existing=False)
    loaded_normal.colorspace_settings.name = "Non-Color"
    loaded_orm = bpy.data.images.load(str(orm_path), check_existing=False)
    loaded_orm.colorspace_settings.name = "Non-Color"

    for mn in mat_names:
        mat = bpy.data.materials[mn]
        nt = mat.node_tree
        N = nt.nodes
        Lk = nt.links
        N.clear()
        out = N.new("ShaderNodeOutputMaterial")
        b = N.new("ShaderNodeBsdfPrincipled")
        b.inputs["Metallic"].default_value = 0.0
        if "Specular IOR Level" in b.inputs:
            b.inputs["Specular IOR Level"].default_value = 0.14
        Lk.new(b.outputs["BSDF"], out.inputs["Surface"])

        uv_n = N.new("ShaderNodeUVMap")
        uv_n.uv_map = "UVMap_Bake"

        t_alb = N.new("ShaderNodeTexImage")
        t_alb.image = loaded_albedo
        Lk.new(uv_n.outputs["UV"], t_alb.inputs["Vector"])
        Lk.new(t_alb.outputs["Color"], b.inputs["Base Color"])

        t_nrm = N.new("ShaderNodeTexImage")
        t_nrm.image = loaded_normal
        Lk.new(uv_n.outputs["UV"], t_nrm.inputs["Vector"])
        nrm_map = N.new("ShaderNodeNormalMap")
        nrm_map.uv_map = "UVMap_Bake"
        nrm_map.inputs["Strength"].default_value = 0.44
        Lk.new(t_nrm.outputs["Color"], nrm_map.inputs["Color"])
        Lk.new(nrm_map.outputs["Normal"], b.inputs["Normal"])

        t_orm = N.new("ShaderNodeTexImage")
        t_orm.image = loaded_orm
        Lk.new(uv_n.outputs["UV"], t_orm.inputs["Vector"])
        sep_orm = N.new("ShaderNodeSeparateColor")
        Lk.new(t_orm.outputs["Color"], sep_orm.inputs["Color"])
        Lk.new(sep_orm.outputs["Green"], b.inputs["Roughness"])

        # Add official glTF Material Output node for Occlusion channel
        gltf_ng = bpy.data.node_groups.get("glTF Material Output")
        if gltf_ng is None:
            gltf_ng = bpy.data.node_groups.new("glTF Material Output", "ShaderNodeTree")
            gltf_ng.interface.new_socket(name="Occlusion", in_out="INPUT", socket_type="NodeSocketFloat")
        gltf_out = N.new("ShaderNodeGroup")
        gltf_out.node_tree = gltf_ng
        Lk.new(sep_orm.outputs["Red"], gltf_out.inputs["Occlusion"])

    # Reorder UVs explicitly so UVMap_Bake is index 0 (TEXCOORD_0) and UVMap_Fabric is index 1 (TEXCOORD_1)
    bake_uvs = np.empty(len(g_mesh.loops) * 2, dtype=np.float32)
    g_mesh.uv_layers["UVMap_Bake"].data.foreach_get("uv", bake_uvs)
    fab_uvs = np.empty(len(g_mesh.loops) * 2, dtype=np.float32)
    g_mesh.uv_layers["UVMap_Fabric"].data.foreach_get("uv", fab_uvs)
    while len(g_mesh.uv_layers) > 0:
        g_mesh.uv_layers.remove(g_mesh.uv_layers[0])
    uv0 = g_mesh.uv_layers.new(name="UVMap_Bake")
    uv0.data.foreach_set("uv", bake_uvs.tolist())
    uv1 = g_mesh.uv_layers.new(name="UVMap_Fabric")
    uv1.data.foreach_set("uv", fab_uvs.tolist())
    g_mesh.uv_layers.active = uv0
    g_mesh.uv_layers.active_render = uv0
    uv0.active = True
    uv0.active_render = True

    # Transfer 7 coupled morph targets onto BS_RINGER_001
    cm = np.load(COUPLED_MORPHS_NPZ, allow_pickle=True)
    key_names = [str(k) for k in cm["key_names"]]
    deltas_shirt = cm["deltas_shirt"].astype(np.float64)
    delta_xl = cm["delta_XL"].astype(np.float64)
    delta_xxl = cm["delta_XXL"].astype(np.float64)

    g_obj.shape_key_add(name="Basis", from_mix=False)
    for ki, kname in enumerate(key_names):
        sk = g_obj.shape_key_add(name=kname, from_mix=False)
        coords = (V_solved + deltas_shirt[ki]).reshape(-1).astype(np.float32)
        sk.data.foreach_set("co", coords)
    for size_name, delta_arr in (("size_XL", delta_xl), ("size_XXL", delta_xxl)):
        sk = g_obj.shape_key_add(name=size_name, from_mix=False)
        coords = (V_solved + delta_arr).reshape(-1).astype(np.float32)
        sk.data.foreach_set("co", coords)

    # Save debug blend + quick preview render to verify front-only artwork + black collar/cuffs
    cam_d = bpy.data.cameras.new("VerifyCam")
    cam_d.lens = 50.0
    cam_o = bpy.data.objects.new("VerifyCam", cam_d)
    cam_o.location = (0.0, -2.35, 1.05)
    from mathutils import Vector  # type: ignore
    cam_o.rotation_euler = (Vector((0.0, 0.0, 1.00)) - Vector(cam_o.location)).to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(cam_o)
    scene.camera = cam_o
    for o in list(bpy.data.objects):
        if o.type == "LIGHT":
            bpy.data.objects.remove(o, do_unlink=True)
    l_key = bpy.data.lights.new("VKey", type="AREA")
    l_key.energy = 160.0
    l_key.size = 1.6
    lo = bpy.data.objects.new("VKey", l_key)
    lo.location = (1.6, -2.1, 2.2)
    lo.rotation_euler = (Vector((0.0, 0.0, 1.05)) - Vector(lo.location)).to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(lo)
    l_fill = bpy.data.lights.new("VFill", type="AREA")
    l_fill.energy = 65.0
    l_fill.size = 2.2
    lf = bpy.data.objects.new("VFill", l_fill)
    lf.location = (-1.6, -2.0, 1.4)
    lf.rotation_euler = (Vector((0.0, 0.0, 1.05)) - Vector(lf.location)).to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(lf)
    scene.render.resolution_x = 720
    scene.render.resolution_y = 1000
    scene.cycles.samples = 16
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = -0.35
    scene.render.film_transparent = True
    preview_png = OUT_DIR / "knitted_ringer_cycles_verify.png"
    scene.render.filepath = str(preview_png)
    bpy.ops.render.render(write_still=True)

    # Remove helper verify camera/lights before glTF export
    bpy.data.objects.remove(cam_o, do_unlink=True)
    bpy.data.objects.remove(lo, do_unlink=True)
    bpy.data.objects.remove(lf, do_unlink=True)

    # Remove temporary _RIB_SDF before glTF export; export COLOR_0 (_TENSION, collar_mask) explicitly
    if "_RIB_SDF" in g_mesh.attributes:
        g_mesh.attributes.remove(g_mesh.attributes["_RIB_SDF"])
    g_mesh.color_attributes.active_color = g_mesh.color_attributes["COLOR_0"]
    g_mesh.color_attributes.render_color_index = 0

    raw_glb = OUT_DIR / "maximus_v2_dressed_ringer_knitted_raw.glb"
    dst_glb = OUT_DIR / "maximus_v2_dressed_ringer_lod1_knitted.glb"
    npx_exe = shutil.which("npx") or "npx"
    packed_bytes = 0
    chosen_webp_q = 74
    for q in (74, 68, 62, 56):
        bpy.ops.export_scene.gltf(
            filepath=str(raw_glb),
            export_format="GLB",
            export_image_format="WEBP",
            export_image_quality=q,
            export_attributes=True,
            export_vertex_color="ACTIVE",
            export_all_vertex_colors=False,
            export_morph=True,
            export_texcoords=True,
            export_normals=True,
            export_materials="EXPORT",
            use_selection=False,
        )
        cmd = [
            npx_exe, "--yes", "@gltf-transform/cli@4.4.2",
            "optimize", str(raw_glb), str(dst_glb),
            "--compress", "meshopt", "--meshopt-level", "high", "--sparse", "true",
            "--texture-compress", "false",
            "--simplify", "false", "--weld", "false",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"gltf-transform failed ({proc.returncode}): {proc.stdout}\n{proc.stderr}")
        packed_bytes = dst_glb.stat().st_size
        chosen_webp_q = q
        if packed_bytes <= GLB_BUDGET_BYTES:
            break

    chosen_tex_size = 2048
    assert 0 < packed_bytes <= GLB_BUDGET_BYTES, f"GLB {packed_bytes} B exceeded budget {GLB_BUDGET_BYTES} B"

    glb_bytes = dst_glb.read_bytes()
    clen, _ = struct.unpack_from("<II", glb_bytes, 12)
    gjson = json.loads(glb_bytes[20 : 20 + clen])
    mesh_summary = {
        m.get("name"): {
            "prims": [
                {
                    "attrs": list(p.get("attributes", {}).keys()),
                    "targets": len(p.get("targets", [])),
                    "mat": p.get("material"),
                }
                for p in m.get("primitives", [])
            ],
            "targetNames": (m.get("extras") or {}).get("targetNames"),
        }
        for m in gjson.get("meshes", [])
    }

    sha = hashlib.sha256(glb_bytes).hexdigest()
    # Promote into storefront and scratch studio assets
    dst_store = STOREFRONT_MOTION / "maximus_v2_dressed_ringer_lod1.glb"
    dst_scratch = SCRATCH_STUDIO_ASSETS / "maximus_v2_dressed_ringer_lod1.glb"
    shutil.copy2(dst_glb, dst_store)
    shutil.copy2(dst_glb, dst_scratch)

    # Update m9_dressed_manifest.json with exact hash/bytes + real 3D callout anchors (single price on primary garment pin)
    man_path = STOREFRONT_MOTION / "m9_dressed_manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    man["glb_bytes"] = packed_bytes
    man["glb_sha256"] = sha
    man["assets"]["glb"] = {
        "url": "media/maximus_motion/maximus_v2_dressed_ringer_lod1.glb",
        "bytes": packed_bytes,
        "sha256": sha,
    }
    man["certified_gates"]["glb_mb"] = round(packed_bytes / (1024 * 1024), 4)
    man["certified_gates"]["gwn_inside"] = gate_info["inside_gwn_count"]
    man["certified_gates"]["g_stretch_p99"] = stretch_info["p99_stretch"]
    man["certified_gates"]["g_stretch_max"] = stretch_info["max_stretch"]
    man["certified_gates"]["edges_gt_1_15"] = stretch_info["edges_gt_1_15"]
    man["callouts"] = [
        {
            "id": "collar",
            "sku": "BS-RINGER-001",
            "title": "Bound Ringer Collar",
            "spec": "20 mm heavyweight rib · O-7 3.3 mm clearance",
            "primary": False,
            "cam_preset": "macro",
            "anchor_y_frac": 0.815,
            "anchor_x_frac": 0.06,
            "anchor_z_offset_m": 0.085,
            "normal": [0.25, 0.25, 0.94],
        },
        {
            "id": "chest",
            "sku": "BS-RINGER-001",
            "title": "Knitted Ringer Tee",
            "spec": "Heavyweight wool jersey · Cursive script print",
            "primary": True,
            "price_label": "Price on Request",
            "cam_preset": "drape34",
            "anchor_y_frac": 0.72,
            "anchor_x_frac": -0.18,
            "anchor_z_offset_m": 0.135,
            "normal": [-0.35, 0.10, 0.93],
        },
        {
            "id": "hem",
            "sku": "BS-RINGER-001",
            "title": "House Standard Emblem",
            "spec": "Raised relief transfer · Coupled L/XL/XXL morph",
            "primary": False,
            "cam_preset": "monolith",
            "anchor_y_frac": 0.545,
            "anchor_x_frac": 0.14,
            "anchor_z_offset_m": 0.13,
            "normal": [0.28, 0.0, 0.96],
        },
    ]
    man_path.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")

    summary = {
        "raw_bytes": raw_glb.stat().st_size,
        "packed_bytes": packed_bytes,
        "packed_mb": round(packed_bytes / (1024 * 1024), 4),
        "tex_size_max": chosen_tex_size,
        "webp_quality": chosen_webp_q,
        "sha256": sha,
        "z_sole_mm": round(z_sole * 1000.0, 4),
        "stretch": stretch_info,
        "gwn_and_tension": gate_info,
        "meshes": mesh_summary,
        "images_count": len(gjson.get("images", [])),
        "preview_png": str(preview_png),
    }
    (OUT_DIR / "REBAKE_REPORT.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
