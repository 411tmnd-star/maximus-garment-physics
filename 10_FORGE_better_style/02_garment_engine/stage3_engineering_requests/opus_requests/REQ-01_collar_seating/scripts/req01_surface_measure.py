"""REQ-01 independent measurement (read-only; never saves). Blender 5.2 binary, numpy + mathutils only (no scipy, no probe code).
Run: blender -b --factory-startup --python req01_surface_measure.py -- <out.json> <blend> [...]

1. Tee faces are split into OUTER / INNER sheet / EDGE (rim) by the sign of  n_face . (c_face - nearest point on Mannequin):
   cos > +0.3 -> OUTER (normal points away from the body), cos < -0.3 -> INNER, otherwise EDGE (solidify rim / neckline edge faces).
   One BVHTree per class plus one of all Tee triangles.
2. Stand BOTTOM ring: collar vertices with collar_stand weight > 0 and collar_leaf weight == 0 ("pure stand"); binned into 2-degree columns
   around the neck (angle convention identical to the probe: XY centroid of all collar_stand vertices, theta = atan2(x - cx, -(y - cy)),
   0 = centre front, +-180 = centre back); in each column the vertices within 1.0 mm of that column's lowest z are the bottom ring.
3. For each bottom-ring vertex: BVH find_nearest (true nearest SURFACE point) to ALL / OUTER / INNER / EDGE; the vector to the nearest ALL point
   is decomposed into vertical (dz) and horizontal (dh) parts; the signed distance along the OUTER hit normal is recorded.
4. placket_overlap and placket_buttons: every vertex -> nearest point on the OUTER sheet (unsigned + signed along hit normal).
Sector groups: centre-back |theta| >= 135, sides 45 <= |theta| < 135, centre-front |theta| < 45.  All distances reported in mm.
"""
import bpy, sys, json, hashlib, math
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

argv = sys.argv[sys.argv.index('--') + 1:]
out_path, files = argv[0], argv[1:]


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''): h.update(ch)
    return h.hexdigest()


def world_verts(ob):
    me = ob.data; co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get('co', co); co = co.reshape(-1, 3)
    M = np.array(ob.matrix_world); return co @ M[:3, :3].T + M[:3, 3]


def stats(a):
    a = np.asarray(a, float)
    if a.size == 0: return None
    return dict(n=int(a.size), min=round(float(a.min()), 2), median=round(float(np.median(a)), 2),
                p95=round(float(np.percentile(a, 95)), 2), max=round(float(a.max()), 2))


def group_weights(ob, name):
    g = ob.vertex_groups.get(name); w = np.zeros(len(ob.data.vertices))
    if g is None: return w
    for v in ob.data.vertices:
        for x in v.groups:
            if x.group == g.index: w[v.index] = x.weight
    return w


def nearest_all(bvh, P):
    d = np.full(len(P), np.nan); loc = np.full((len(P), 3), np.nan); nrm = np.full((len(P), 3), np.nan)
    for i, p in enumerate(P):
        h = bvh.find_nearest(Vector(p))
        if h[0] is not None: loc[i] = h[0]; nrm[i] = h[1]; d[i] = h[3]
    return d, loc, nrm


res = {'_meta': dict(blender=bpy.app.version_string, numpy=np.__version__)}
for f in files:
    h0 = sha(f); bpy.ops.wm.open_mainfile(filepath=f, load_ui=False)
    tee = bpy.data.objects['Tee']; col = bpy.data.objects['collar']; man = bpy.data.objects['Mannequin']
    # ---- body BVH
    MV = world_verts(man); man.data.calc_loop_triangles()
    body = BVHTree.FromPolygons([tuple(v) for v in MV], [tuple(t.vertices) for t in man.data.loop_triangles])
    # ---- Tee sheet classification
    TV = world_verts(tee); me = tee.data; me.calc_loop_triangles()
    npoly = len(me.polygons); C = np.empty(npoly * 3); N = np.empty(npoly * 3)
    me.polygons.foreach_get('center', C); me.polygons.foreach_get('normal', N); C = C.reshape(-1, 3); N = N.reshape(-1, 3)
    cosv = np.zeros(npoly); dbody = np.zeros(npoly)
    for k in range(npoly):
        h = body.find_nearest(Vector(C[k]))
        if h[0] is None: continue
        r = C[k] - np.array(h[0]); L = np.linalg.norm(r); dbody[k] = L
        cosv[k] = float(N[k] @ r / L) if L > 1e-9 else 0.0
    cls = np.where(cosv > 0.3, 0, np.where(cosv < -0.3, 1, 2))   # 0 outer, 1 inner, 2 edge
    tri_v = np.empty(len(me.loop_triangles) * 3, int); me.loop_triangles.foreach_get('vertices', tri_v); tri_v = tri_v.reshape(-1, 3)
    tri_p = np.empty(len(me.loop_triangles), int); me.loop_triangles.foreach_get('polygon_index', tri_p)
    TVt = [tuple(v) for v in TV]
    bv = {'all': BVHTree.FromPolygons(TVt, [tuple(t) for t in tri_v])}
    for name, c in (('outer', 0), ('inner', 1), ('edge', 2)):
        sel = tri_v[cls[tri_p] == c]; bv[name] = BVHTree.FromPolygons(TVt, [tuple(t) for t in sel]) if len(sel) else None
    sheet = dict(faces_outer=int((cls == 0).sum()), faces_inner=int((cls == 1).sum()), faces_edge=int((cls == 2).sum()),
                 median_face_to_body_mm_outer=round(1e3 * float(np.median(dbody[cls == 0])), 2),
                 median_face_to_body_mm_inner=round(1e3 * float(np.median(dbody[cls == 1])), 2))
    # ---- stand bottom ring
    CV = world_verts(col); ws = group_weights(col, 'collar_stand'); wl = group_weights(col, 'collar_leaf')
    stand_all = np.flatnonzero(ws > 0); pure = np.flatnonzero((ws > 0) & (wl == 0))
    cxy = CV[stand_all, :2].mean(0)
    def theta(P): return np.degrees(np.arctan2(P[:, 0] - cxy[0], -(P[:, 1] - cxy[1])))
    thp = theta(CV[pure]); b2 = np.floor((thp + 180) / 2).astype(int)
    base = []
    for b in np.unique(b2):
        m = pure[b2 == b]; z = CV[m, 2]; base += list(m[z <= z.min() + 0.001])
    base = np.array(sorted(set(base))); PB = CV[base]; thb = theta(PB)
    out = dict(sha256=h0, tee_verts=len(TV), tee_sheets=sheet, collar_stand_members=int(len(stand_all)), pure_stand_verts=int(len(pure)),
               bottom_ring_verts=int(len(base)), bottom_ring_columns_2deg=int(len(np.unique(b2))),
               bottom_ring_z_range_m=[round(float(PB[:, 2].min()), 4), round(float(PB[:, 2].max()), 4)])
    D = {}
    for name in ('all', 'outer', 'inner', 'edge'):
        D[name] = nearest_all(bv[name], PB) if bv[name] is not None else (np.full(len(PB), np.nan),) * 3
    d_all, loc_all, _ = D['all']; vec = PB - loc_all
    dz = vec[:, 2]; dh = np.linalg.norm(vec[:, :2], axis=1)
    sgn_outer = np.einsum('ij,ij->i', PB - D['outer'][1], D['outer'][2])
    # which sheet is nearest
    stack = np.vstack([D['outer'][0], D['inner'][0], D['edge'][0]]); stack = np.where(np.isnan(stack), np.inf, stack); nearest_cls = np.argmin(stack, 0)
    groups = dict(centre_back=np.abs(thb) >= 135, sides=(np.abs(thb) >= 45) & (np.abs(thb) < 135), centre_front=np.abs(thb) < 45)
    tab = {}
    for g, m in groups.items():
        tab[g] = dict(n_verts=int(m.sum()),
                      to_all_mm=stats(1e3 * d_all[m]), to_outer_mm=stats(1e3 * D['outer'][0][m]), to_inner_mm=stats(1e3 * D['inner'][0][m]),
                      to_edge_mm=stats(1e3 * D['edge'][0][m]), signed_vs_outer_normal_mm=stats(1e3 * sgn_outer[m]),
                      dz_to_nearest_mm=stats(1e3 * dz[m]), dh_to_nearest_mm=stats(1e3 * dh[m]),
                      nearest_sheet_counts=dict(outer=int((nearest_cls[m] == 0).sum()), inner=int((nearest_cls[m] == 1).sum()), edge=int((nearest_cls[m] == 2).sum())))
    out['bottom_ring_by_group'] = tab
    sec = ((thb + 180) // 10).astype(int).clip(0, 35); per = []
    for s in range(36):
        m = sec == s
        per.append(None if not m.any() else dict(min_all=round(1e3 * float(d_all[m].min()), 2), median_all=round(1e3 * float(np.median(d_all[m])), 2),
                                                 min_outer=round(1e3 * float(np.nanmin(D['outer'][0][m])), 2), min_inner=round(1e3 * float(np.nanmin(D['inner'][0][m])), 2),
                                                 median_dz=round(1e3 * float(np.median(dz[m])), 2), median_dh=round(1e3 * float(np.median(dh[m])), 2)))
    out['bottom_ring_sector_10deg_from_back_-180'] = per
    # whole-stand nearest-surface min per sector (same vertex set as the probe, but surface not vertex) for a like-for-like check
    PS = CV[stand_all]; ths = theta(PS); ds, _, _ = nearest_all(bv['all'], PS); ss = ((ths + 180) // 10).astype(int).clip(0, 35)
    out['whole_stand_surface_min_sector_10deg_mm'] = [None if not (ss == s).any() else round(1e3 * float(ds[ss == s].min()), 2) for s in range(36)]
    # ---- placket + buttons
    for nm in ('placket_overlap', 'placket_buttons'):
        ob = bpy.data.objects.get(nm)
        if ob is None: out[nm] = None; continue
        P = world_verts(ob); d, loc, nrm = nearest_all(bv['outer'], P); sd = np.einsum('ij,ij->i', P - loc, nrm)
        da, _, _ = nearest_all(bv['all'], P)
        out[nm] = dict(n_verts=len(P), to_outer_mm=stats(1e3 * d), pct_within_5mm=round(100 * float((d <= 0.005).mean()), 2),
                       pct_within_2mm=round(100 * float((d <= 0.002).mean()), 2), signed_vs_outer_normal_mm=stats(1e3 * sd),
                       to_all_tee_mm=stats(1e3 * da), z_range_m=[round(float(P[:, 2].min()), 4), round(float(P[:, 2].max()), 4)])
    out['unchanged'] = sha(f) == h0
    res[f] = out; print('MEAS', f.replace('\\', '/').split('/')[-1], json.dumps(tab)[:900])
json.dump(res, open(out_path, 'w'), indent=1)
