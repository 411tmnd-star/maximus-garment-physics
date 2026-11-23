"""REQ-01 gate: independent clearance report for the collar stand's sewn bottom ring against the Tee shell.  Read-only; deterministic output.
Does NOT import collar_builder: ring definition, distances and thresholds are re-implemented here.

  python -I req01_gate.py <out.json> <regen_summary.json> <master.blend> [...]          (bpy module + scipy)

Ring (same definition as the REQ-01 M2 measurement): pure-stand vertices (collar_stand weight > 0, collar_leaf weight == 0), binned in 2-degree columns
round the neck (theta = atan2(x - cx, -(y - cy)); 0 = centre front, +-180 = centre back); per column the vertices within 1.0 mm of that column's lowest z.
Per ring vertex:  vertex distance = scipy cKDTree nearest TEE VERTEX;  surface distance = BVH nearest Tee surface point;  side = sign of
(v - q) . n_hit (negative = inside the shell solid).
GATE (every item must hold for 100 % of ring vertices, per master):
  G1  cKDTree vertex distance < 1.0 mm
  G2  surface distance in [0.1 mm, 1.0 mm]
  G3  no ring vertex on the inside of the shell (side >= 0)
  G4  all 36 x 10-degree sectors that contain stand vertices are present in the table (coverage; sectors with no stand are listed, not failed)"""
import sys, os, json, hashlib
import numpy as np
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from scipy.spatial import cKDTree

out_path, summary_path, files = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), [os.path.abspath(a) for a in sys.argv[3:]]
S = json.load(open(summary_path))
LO, HI = 0.1, 1.0


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


def tri_bvh(o):
    m = o.data; m.calc_loop_triangles(); return BVHTree.FromPolygons([tuple(v.co) for v in m.vertices], [tuple(t.vertices) for t in m.loop_triangles])


def r(x, n=3): return round(float(x), n)


def measure(Tee, tee_kd, tee_bvh, body_bvh, C):
    c = arr(C); ws = weights(C, 'collar_stand'); wl = weights(C, 'collar_leaf'); idx = np.flatnonzero((ws > 0) & (wl == 0)); P = c[idx]; cen = P[:, :2].mean(0)
    th = np.degrees(np.arctan2(P[:, 0] - cen[0], -(P[:, 1] - cen[1]))); col = np.floor((th + 180.0) / 2.0).astype(int)
    ring = []
    for k in np.unique(col):
        m = np.flatnonzero(col == k); ring += list(m[P[m, 2] <= P[m, 2].min() + 0.001])
    ring = np.array(sorted(ring)); R = P[ring]; tr = th[ring]
    dv = 1e3 * tee_kd.query(R)[0]
    ds = np.zeros(len(R)); side = np.zeros(len(R))
    for i, p in enumerate(R):
        h = tee_bvh.find_nearest(Vector(p)); d = p - np.array(h[0]); ds[i] = 1e3 * np.linalg.norm(d); side[i] = float(d @ np.array(h[1]))
    sec = np.floor((tr + 180.0) / 10.0).astype(int).clip(0, 35); table = []
    for s in range(36):
        m = sec == s
        table.append(None if not m.any() else dict(sector_from_deg=-180 + 10 * s, n=int(m.sum()), vertex_mm_min=r(dv[m].min()), vertex_mm_median=r(np.median(dv[m])), vertex_mm_max=r(dv[m].max()),
                                                    surface_mm_min=r(ds[m].min()), surface_mm_max=r(ds[m].max())))
    g1 = int((dv < HI).sum()); g2 = int(((ds >= LO) & (ds <= HI)).sum()); g3 = int((side >= 0).sum()); n = len(R)
    # whole-collar intersection audit (informational): any collar vertex on the inside of the Tee solid / inside the Mannequin
    inside_tee = inside_body = 0
    for p in c:
        h = tee_bvh.find_nearest(Vector(p))
        if h[0] is not None and float((p - np.array(h[0])) @ np.array(h[1])) < -1e-5: inside_tee += 1
        hb = body_bvh.find_nearest(Vector(p))
        if hb[0] is not None and float((p - np.array(hb[0])) @ np.array(hb[1])) < 0: inside_body += 1
    ps = P; hts = []
    for k in range(12):                                                          # stand height per 30-degree sector (informational: seating must not squash or stretch it)
        m = np.floor((th + 180.0) / 30.0).astype(int).clip(0, 11) == k
        hts.append(None if not m.any() else r(1e3 * (P[m, 2].max() - P[m, 2].min()), 2))
    return dict(stand_vertices=int(len(idx)), total_collar_vertices=int(len(c)), ring_vertices=int(n),
                G1_vertex_dist_lt_1mm=dict(pass_count=g1, of=int(n), pct=r(100 * g1 / n, 4)),
                G2_surface_dist_in_0p1_to_1p0mm=dict(pass_count=g2, of=int(n), pct=r(100 * g2 / n, 4)),
                G3_ring_outside_shell=dict(pass_count=g3, of=int(n), pct=r(100 * g3 / n, 4)),
                vertex_mm=dict(min=r(dv.min()), median=r(np.median(dv)), p95=r(np.percentile(dv, 95)), max=r(dv.max())),
                surface_mm=dict(min=r(ds.min()), median=r(np.median(ds)), p95=r(np.percentile(ds, 95)), max=r(ds.max())),
                sectors_present=int(sum(1 for t in table if t)), sectors_empty=[-180 + 10 * s for s, t in enumerate(table) if t is None],
                sector_table_10deg=table, audit_collar_vertices_inside_tee_solid=int(inside_tee), audit_collar_vertices_inside_mannequin=int(inside_body),
                audit_stand_height_mm_per_30deg_sector=hts,
                PASS=bool(g1 == n and g2 == n and g3 == n))


res = {}
for f in files:
    name = os.path.basename(f).replace('.blend', ''); h0 = sha(f); entry = S[name]
    bpy.ops.wm.open_mainfile(filepath=f, load_ui=False)
    Tee, Man, Old = bpy.data.objects['Tee'], bpy.data.objects['Mannequin'], bpy.data.objects['collar']
    tee_kd = cKDTree(arr(Tee)); tee_bvh = tri_bvh(Tee); body_bvh = tri_bvh(Man)
    row = dict(input_sha256=h0, baked_original=measure(Tee, tee_kd, tee_bvh, body_bvh, Old))
    for variant in ('unseated', 'seated'):
        with bpy.data.libraries.load(entry[variant]['file']) as (src, dst): dst.objects = list(src.objects)
        ob = dst.objects[0]; bpy.context.scene.collection.objects.link(ob)
        row[variant + '_regen'] = dict(collar_file=os.path.relpath(entry[variant]['file']).replace('\\', '/'), collar_file_sha256=sha(entry[variant]['file']), **measure(Tee, tee_kd, tee_bvh, body_bvh, ob))
    assert sha(f) == h0
    row['input_unchanged'] = True; res[name] = row
res['_gate'] = dict(thresholds_mm=dict(vertex_dist_lt=HI, surface_dist_min=LO, surface_dist_max=HI), PASS_ALL_MASTERS_SEATED=all(res[k]['seated_regen']['PASS'] for k in res if not k.startswith('_')))
json.dump(res, open(out_path, 'w'), indent=1, sort_keys=True)
for k in res:
    if k.startswith('_'): continue
    for v in ('baked_original', 'unseated_regen', 'seated_regen'):
        m = res[k][v]; print('%-22s %-15s ring=%5d  G1=%6.2f%% G2=%6.2f%% G3=%6.2f%%  vertex_mm min/med/max=%s/%s/%s  inside_tee=%d inside_body=%d  PASS=%s' % (
            k, v, m['ring_vertices'], m['G1_vertex_dist_lt_1mm']['pct'], m['G2_surface_dist_in_0p1_to_1p0mm']['pct'], m['G3_ring_outside_shell']['pct'],
            m['vertex_mm']['min'], m['vertex_mm']['median'], m['vertex_mm']['max'], m['audit_collar_vertices_inside_tee_solid'], m['audit_collar_vertices_inside_mannequin'], m['PASS']))
print('GATE', res['_gate'])
