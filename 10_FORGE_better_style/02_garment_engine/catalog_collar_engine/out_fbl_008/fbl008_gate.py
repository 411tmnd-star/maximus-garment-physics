"""FBL-008 gate: collar stand seating report for a built archetype master.  Read-only (never saves); does NOT import collar_builder.

  python -I fbl008_gate.py <out.json> <label> <master.blend> <quality.json> <ring.npz | -> [<label> <master.blend> <quality.json> <ring.npz | ->] ...

Definitions are those of REQ-01 remediation/req01_gate.py unless stated:
  stand      pure-stand vertices of object `collar` (collar_stand weight > 0, collar_leaf weight == 0)
  theta      atan2(x - cx, -(y - cy)) about the pure-stand centroid; 0 = centre front, +-180 = centre back
  ring       REQ-01 dynamic definition: 2-degree columns, per column the stand vertices within 1.0 mm of that column's lowest z.
             When a ring.npz (key `ring_index`, collar vertex indices captured by the build BEFORE projection) is given, that STATIC set is the
             gated ring and the dynamic ring is reported next to it.
  G1_raw     ring vertex -> nearest Tee VERTEX < 1.0 mm                                   (logged for comparison only, see O-7)
  G1_surface ring vertex -> nearest point of a 0.25 mm point cloud sampled on the Tee neckline surface < 1.0 mm      (O-7; gate >= 99 %)
  G2         ring vertex -> nearest Tee surface point (BVH) <= 1.0 mm                     (gate 100 %; the REQ-01 window [0.1, 1.0] is also logged)
  G3         ring vertex not inside the shell solid: (v - q) . n_hit >= 0                 (gate 100 %)
  body       collar vertices (ALL of them) with (v - q) . n_hit < 0 against object `Mannequin`   (gate 0)
  height     per 2-degree column: z range of the pure-stand vertices; min / max of that per populated 30-degree sector.  The REQ-01 audit figure
             (z range of all pure-stand vertices of a 30-degree sector, which also contains the neckline's own slope inside the sector) is logged too.
"""
import sys, os, json, hashlib
import numpy as np
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from scipy.spatial import cKDTree

HI, LO, STEP, BAND = 1.0, 0.1, 0.00025, 0.030


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''): h.update(ch)
    return h.hexdigest()


def arr(o):
    m = o.data; a = np.empty(len(m.vertices) * 3); m.vertices.foreach_get('co', a); return a.reshape(-1, 3)


def weights(o, name):
    gi = o.vertex_groups[name].index
    return np.array([next((g.weight for g in v.groups if g.group == gi), 0.0) for v in o.data.vertices])


def tris(o):
    m = o.data; m.calc_loop_triangles(); t = np.empty(len(m.loop_triangles) * 3, dtype=np.int32); m.loop_triangles.foreach_get('vertices', t); return t.reshape(-1, 3)


def r(x, n=3): return round(float(x), n)


def sample_surface(V, T, step):
    """Barycentric lattice on every triangle, lattice pitch <= step along the longest edge."""
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    L = np.maximum.reduce([np.linalg.norm(B - A, axis=1), np.linalg.norm(C - B, axis=1), np.linalg.norm(A - C, axis=1)])
    n = np.maximum(1, np.ceil(L / step).astype(int)); out = []
    for k in np.unique(n):
        m = n == k; i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing='ij'); ok = (i + j) <= k
        u = (i[ok] / k)[None, :, None]; v = (j[ok] / k)[None, :, None]
        out.append((A[m][:, None] * (1 - u - v) + B[m][:, None] * u + C[m][:, None] * v).reshape(-1, 3))
    return np.vstack(out)


def measure(path, qpath, ringpath):
    h0 = sha(path); bpy.ops.wm.open_mainfile(filepath=path, load_ui=False)
    Tee, Man, C = bpy.data.objects['Tee'], bpy.data.objects['Mannequin'], bpy.data.objects['collar']
    tv = arr(Tee); tt = tris(Tee); tee_kd = cKDTree(tv); tee_bvh = BVHTree.FromPolygons([tuple(v) for v in tv], [tuple(int(x) for x in t) for t in tt])
    mv = arr(Man); mt = tris(Man); body_bvh = BVHTree.FromPolygons([tuple(v) for v in mv], [tuple(int(x) for x in t) for t in mt])
    c = arr(C); ws = weights(C, 'collar_stand'); wl = weights(C, 'collar_leaf'); idx = np.flatnonzero((ws > 0) & (wl == 0)); P = c[idx]; cen = P[:, :2].mean(0)
    th = np.degrees(np.arctan2(P[:, 0] - cen[0], -(P[:, 1] - cen[1]))); col = np.floor((th + 180.0) / 2.0).astype(int)
    dyn = []
    for k in np.unique(col):
        m = np.flatnonzero(col == k); dyn += list(m[P[m, 2] <= P[m, 2].min() + 0.001])
    dyn_idx = np.array(sorted(idx[dyn]))
    static = None
    if ringpath and ringpath != '-':
        static = np.sort(np.load(ringpath)['ring_index'].astype(int))
    ring_idx = static if static is not None else dyn_idx
    R = c[ring_idx]; tr = np.degrees(np.arctan2(R[:, 0] - cen[0], -(R[:, 1] - cen[1])))
    dv = 1e3 * tee_kd.query(R)[0]
    ds = np.zeros(len(R)); side = np.zeros(len(R))
    for i, p in enumerate(R):
        h = tee_bvh.find_nearest(Vector(p)); d = p - np.array(h[0]); ds[i] = 1e3 * np.linalg.norm(d); side[i] = float(d @ np.array(h[1]))
    near = np.zeros(len(tv), bool)                                                # O-7 cloud: Tee triangles with a vertex within BAND of the ring
    for lst in tee_kd.query_ball_point(R, BAND): near[lst] = True
    Tn = tt[near[tt].any(1)]; cloud = sample_surface(tv, Tn, STEP); dc = 1e3 * cKDTree(cloud).query(R)[0]
    sec = np.floor((tr + 180.0) / 10.0).astype(int).clip(0, 35); table = []
    for s in range(36):
        m = sec == s
        table.append(None if not m.any() else dict(sector_from_deg=-180 + 10 * s, n=int(m.sum()), vertex_mm_min=r(dv[m].min()), vertex_mm_median=r(np.median(dv[m])), vertex_mm_max=r(dv[m].max()),
                                                    surface_mm_min=r(ds[m].min()), surface_mm_median=r(np.median(ds[m])), surface_mm_max=r(ds[m].max()), inside_shell=int((side[m] < 0).sum())))
    n = len(R); g1r = int((dv < HI).sum()); g1s = int((dc < HI).sum()); g2 = int((ds <= HI).sum()); g2w = int(((ds >= LO) & (ds <= HI)).sum()); g3 = int((side >= 0).sum())
    inside_tee = inside_body = 0; body_min = 1e9
    for p in c:
        h = tee_bvh.find_nearest(Vector(p))
        if h[0] is not None and float((p - np.array(h[0])) @ np.array(h[1])) < -1e-5: inside_tee += 1
        hb = body_bvh.find_nearest(Vector(p))
        if hb[0] is not None:
            sd = float((p - np.array(hb[0])) @ np.array(hb[1])); body_min = min(body_min, sd)
            if sd < 0: inside_body += 1
    sec30 = np.floor((th + 180.0) / 30.0).astype(int).clip(0, 11); hts = []; colh = {}
    for k in np.unique(col):
        m = col == k
        if m.sum() >= 4: colh[int(k)] = (1e3 * float(P[m, 2].max() - P[m, 2].min()), int(sec30[m][0]))
    for k in range(12):
        m = sec30 == k; hh = [v[0] for v in colh.values() if v[1] == k]
        hts.append(None if not m.any() else dict(sector_from_deg=-180 + 30 * k, column_height_mm_min=r(min(hh), 2), column_height_mm_max=r(max(hh), 2), column_height_mm_median=r(np.median(hh), 2),
                                                   req01_sector_z_range_mm=r(1e3 * (P[m, 2].max() - P[m, 2].min()), 2)))
    me = C.data; q = json.load(open(qpath)) if qpath and qpath != '-' and os.path.exists(qpath) else {}
    cq = (q.get('construction') or {}).get('collar', {})
    assert sha(path) == h0
    return dict(file=os.path.relpath(path).replace('\\', '/'), sha256=h0, ring_definition='static indices captured by the build before projection' if static is not None else 'REQ-01 dynamic (2-deg columns, within 1 mm of lowest z)',
                ring_vertices=int(n), ring_vertices_req01_dynamic=int(len(dyn_idx)), static_ring_equals_dynamic=(bool(np.array_equal(static, dyn_idx)) if static is not None else None), stand_vertices=int(len(idx)),
                G1_raw_vertex_lt_1mm=dict(pass_count=g1r, of=n, pct=r(100 * g1r / n, 2)),
                G1_surface_O7_lt_1mm=dict(pass_count=g1s, of=n, pct=r(100 * g1s / n, 2), cloud_points=int(len(cloud)), cloud_step_mm=1e3 * STEP, max_mm=r(dc.max())),
                G2_surface_le_1mm=dict(pass_count=g2, of=n, pct=r(100 * g2 / n, 2)), G2_req01_window_0p1_to_1p0mm=dict(pass_count=g2w, of=n, pct=r(100 * g2w / n, 2)),
                G3_ring_outside_shell=dict(pass_count=g3, of=n, pct=r(100 * g3 / n, 2), inside=int(n - g3)),
                vertex_mm=dict(min=r(dv.min()), median=r(np.median(dv)), p95=r(np.percentile(dv, 95)), max=r(dv.max())),
                surface_mm=dict(min=r(ds.min()), median=r(np.median(ds)), p95=r(np.percentile(ds, 95)), max=r(ds.max())),
                tee_vertex_spacing_near_ring_mm=dict(median=r(np.median(1e3 * cKDTree(tv[near]).query(tv[near], k=2)[0][:, 1])), max=r((1e3 * cKDTree(tv[near]).query(tv[near], k=2)[0][:, 1]).max())),
                sectors_present=int(sum(1 for t in table if t)), sectors_empty=[-180 + 10 * s for s, t in enumerate(table) if t is None], sector_table_10deg=table,
                collar_vertices_inside_mannequin=int(inside_body), collar_min_signed_dist_to_mannequin_mm=r(1e3 * body_min), audit_collar_vertices_inside_tee_solid=int(inside_tee),
                stand_height_per_30deg_sector=hts,
                leaf_min_quad_area_ratio=cq.get('leaf_min_quad_area_ratio'), build_quality_gate_passed=q.get('gate_passed'), build_collar_report=cq,
                contract=dict(vertices=len(me.vertices), faces=len(me.polygons), vertex_groups=sorted(g.name for g in C.vertex_groups), material_slots=[s.material.name if s.material else None for s in C.material_slots],
                              uv_layers=[u.name for u in me.uv_layers], modifiers=len(C.modifiers), tee_vertices=len(tv), tee_faces=len(Tee.data.polygons),
                              garment_objects=sorted(o.name for o in bpy.data.collections['Garment'].objects) if 'Garment' in bpy.data.collections else None))


out_path = os.path.abspath(sys.argv[1]); a = sys.argv[2:]; res = {}
for k in range(0, len(a), 4):
    label, path, qpath, ringpath = a[k], os.path.abspath(a[k + 1]), a[k + 2], a[k + 3]
    m = measure(path, qpath, ringpath); res[label] = m
    print('%-26s ring=%5d (dyn %5d)  G1_raw=%6.2f%%  G1_surf=%6.2f%%  G2=%6.2f%%  G3=%6.2f%%  surface_mm min/med/max=%s/%s/%s  inside_body=%d  inside_tee=%d  leaf_ratio=%s  verts=%d faces=%d' % (
        label, m['ring_vertices'], m['ring_vertices_req01_dynamic'], m['G1_raw_vertex_lt_1mm']['pct'], m['G1_surface_O7_lt_1mm']['pct'], m['G2_surface_le_1mm']['pct'], m['G3_ring_outside_shell']['pct'],
        m['surface_mm']['min'], m['surface_mm']['median'], m['surface_mm']['max'], m['collar_vertices_inside_mannequin'], m['audit_collar_vertices_inside_tee_solid'], m['leaf_min_quad_area_ratio'],
        m['contract']['vertices'], m['contract']['faces']), flush=True)
    print('   stand height per 30deg (col min/max | req01 z-range):', ' '.join('-' if h is None else '%s/%s|%s' % (h['column_height_mm_min'], h['column_height_mm_max'], h['req01_sector_z_range_mm']) for h in m['stand_height_per_30deg_sector']), flush=True)
json.dump(res, open(out_path, 'w'), indent=1, sort_keys=True)
