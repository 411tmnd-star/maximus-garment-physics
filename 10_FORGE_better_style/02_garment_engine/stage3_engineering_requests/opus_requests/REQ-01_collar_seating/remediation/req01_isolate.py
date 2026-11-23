"""REQ-01 step 2: parameter isolation. Read-only (never saves).  python -I req01_isolate.py <out.json> <blend> [...]
For each master: stand bottom ring (pure stand, 2-deg columns, within 1 mm of lowest z) -> distance to Mannequin (is the hard-coded 0.003 m
body-clearance push in collar_builder.build_leaf_object binding?) and the Tee point nearest each ring vertex (the shell neckline) -> its distance to Mannequin."""
import bpy, sys, json, numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
out, files = sys.argv[-1 - len(sys.argv[sys.argv.index('--'):]) if False else 1], sys.argv[2:]

def V(o):
    m = o.data; a = np.empty(len(m.vertices) * 3); m.vertices.foreach_get('co', a); return a.reshape(-1, 3)
def bvh(o):
    m = o.data; m.calc_loop_triangles(); return BVHTree.FromPolygons([tuple(v.co) for v in m.vertices], [tuple(t.vertices) for t in m.loop_triangles])
def st(a): a = np.asarray(a); return dict(n=int(a.size), min=round(float(a.min()), 2), median=round(float(np.median(a)), 2), p95=round(float(np.percentile(a, 95)), 2), max=round(float(a.max()), 2))
res = {}
for f in files:
    bpy.ops.wm.open_mainfile(filepath=f, load_ui=False)
    T, C, M = (bpy.data.objects[n] for n in ('Tee', 'collar', 'Mannequin')); c = V(C); bt, bm = bvh(T), bvh(M)
    g = C.vertex_groups['collar_stand'].index; gl = C.vertex_groups['collar_leaf'].index
    ws = np.array([next((x.weight for x in v.groups if x.group == g), 0) for v in C.data.vertices]); wl = np.array([next((x.weight for x in v.groups if x.group == gl), 0) for v in C.data.vertices])
    idx = np.flatnonzero((ws > 0) & (wl == 0)); P = c[idx]; cen = P[:, :2].mean(0)
    th = np.degrees(np.arctan2(P[:, 0] - cen[0], -(P[:, 1] - cen[1]))); col = np.floor((th + 180) / 2).astype(int)
    ring = []
    for k in np.unique(col):
        m = np.flatnonzero(col == k); z = P[m, 2]; ring += list(m[z <= z.min() + 0.001])
    ring = np.array(ring); R = P[ring]; tr = th[ring]
    d_body, d_tee_body, dz, signed_up = [], [], [], []
    for p in R:
        hb = bm.find_nearest(Vector(p)); d_body.append(1e3 * (Vector(p) - hb[0]).length)
        ht = bt.find_nearest(Vector(p)); q = ht[0]; hq = bm.find_nearest(q); d_tee_body.append(1e3 * (q - hq[0]).length); dz.append(1e3 * (p[2] - q[2]))
    d_body, d_tee_body, dz = map(np.array, (d_body, d_tee_body, dz))
    grp = {'CB': np.abs(tr) >= 135, 'sides': (np.abs(tr) >= 45) & (np.abs(tr) < 135), 'CF': np.abs(tr) < 45}
    res[f.split('/')[-1]] = {k: dict(ring_n=int(m.sum()), stand_ring_to_body_mm=st(d_body[m]) if m.any() else None, shell_neckline_to_body_mm=st(d_tee_body[m]) if m.any() else None,
                              stand_base_dz_above_neckline_mm=st(dz[m]) if m.any() else None, frac_ring_body_clearance_lt_3p2mm=round(float((d_body[m] < 3.2).mean()), 3) if m.any() else None) for k, m in grp.items()}
    print(f.split('/')[-1], json.dumps(res[f.split('/')[-1]], indent=0)[:1800])
json.dump(res, open(out, 'w'), indent=1)
