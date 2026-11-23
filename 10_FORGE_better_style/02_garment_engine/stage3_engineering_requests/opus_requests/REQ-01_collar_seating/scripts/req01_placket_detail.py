"""REQ-01 placket detail (read-only; never saves). blender -b --factory-startup --python req01_placket_detail.py -- <out.json> <blend> [...]
Where are the placket vertices > 5 mm from the Tee outer sheet (z range, rows from the top), versus the collar-stand bottom ring height near
centre front; and how far are button vertices from the placket surface.  Outer sheet = Tee faces with n . (c - nearest Mannequin point) > 0.3|c - p|."""
import bpy, sys, json, hashlib
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

argv = sys.argv[sys.argv.index('--') + 1:]; out_path, files = argv[0], argv[1:]


def wv(ob):
    co = np.empty(len(ob.data.vertices) * 3); ob.data.vertices.foreach_get('co', co); co = co.reshape(-1, 3)
    M = np.array(ob.matrix_world); return co @ M[:3, :3].T + M[:3, 3]


def tris(ob):
    ob.data.calc_loop_triangles(); t = np.empty(len(ob.data.loop_triangles) * 3, int); ob.data.loop_triangles.foreach_get('vertices', t)
    p = np.empty(len(ob.data.loop_triangles), int); ob.data.loop_triangles.foreach_get('polygon_index', p); return t.reshape(-1, 3), p


res = {}
for f in files:
    bpy.ops.wm.open_mainfile(filepath=f, load_ui=False)
    O = bpy.data.objects; man = O['Mannequin']; tee = O['Tee']
    MV = wv(man); mt, _ = tris(man); body = BVHTree.FromPolygons([tuple(v) for v in MV], [tuple(x) for x in mt])
    TV = wv(tee); tt, tp = tris(tee); me = tee.data; n = len(me.polygons)
    C = np.empty(n * 3); N = np.empty(n * 3); me.polygons.foreach_get('center', C); me.polygons.foreach_get('normal', N); C = C.reshape(-1, 3); N = N.reshape(-1, 3)
    near = (C[:, 2] > 0.85) & (C[:, 1] < 0.05)         # front of the shirt from below the camp placket bottom (0.976 m) up
    outer = np.zeros(n, bool)
    for k in np.flatnonzero(near):
        h = body.find_nearest(Vector(C[k]))
        if h[0] is None: continue
        r = C[k] - np.array(h[0]); L = np.linalg.norm(r); outer[k] = L > 1e-9 and N[k] @ r > 0.3 * L
    bo = BVHTree.FromPolygons([tuple(v) for v in TV], [tuple(x) for x in tt[outer[tp]]])
    pk = O['placket_overlap']; P = wv(pk); d = np.array([bo.find_nearest(Vector(p))[3] for p in P])
    far = d > 0.005
    col = O['collar']; CV = wv(col); g = col.vertex_groups['collar_stand'].index
    st = np.array([v.index for v in col.data.vertices if any(x.group == g and x.weight > 0 for x in v.groups)])
    S = CV[st]; cf = S[np.abs(S[:, 0] - P[:, 0].mean()) < 0.04]
    pkt = BVHTree.FromPolygons([tuple(v) for v in P], [tuple(x) for x in tris(pk)[0]])
    B = wv(O['placket_buttons']); db = np.array([pkt.find_nearest(Vector(p))[3] for p in B])
    res[f] = dict(placket_top_z=round(float(P[:, 2].max()), 4), placket_bottom_z=round(float(P[:, 2].min()), 4), far_gt5mm_verts=int(far.sum()), far_z_range=[round(float(P[far, 2].min()), 4), round(float(P[far, 2].max()), 4)] if far.any() else None,
                  far_band_below_top_mm=round(1e3 * float(P[:, 2].max() - P[far, 2].min()), 1) if far.any() else None,
                  dist_by_band_below_top_mm={'%d-%d' % (a, b): round(1e3 * float(d[(P[:, 2].max() - P[:, 2] >= a / 1e3) & (P[:, 2].max() - P[:, 2] < b / 1e3)].max()), 2) for a, b in ((0, 5), (5, 10), (10, 15), (15, 20), (20, 40), (40, 1000))},
                  stand_lowest_z_within_4cm_of_cf_x=round(float(cf[:, 2].min()), 4) if len(cf) else None,
                  buttons_to_placket_mm=dict(min=round(1e3 * float(db.min()), 2), median=round(1e3 * float(np.median(db)), 2), max=round(1e3 * float(db.max()), 2)))
    print('PKT', f.replace('\\', '/').split('/')[-1], json.dumps(res[f]))
json.dump(res, open(out_path, 'w'), indent=1)
