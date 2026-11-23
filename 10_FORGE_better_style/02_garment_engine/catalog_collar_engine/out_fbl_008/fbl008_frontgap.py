"""FBL-008 front-opening audit for a built master.  Read-only (never saves).

  python -I -B fbl008_frontgap.py <out.json> <label> <master.blend> [<label> <master.blend>] ...

Per master, on object `collar`:
  opening_chord_m       straight distance between the two front ends of the collar: the smallest distance between a collar vertex of the +x half
                        and one of the -x half, among the vertices in FRONT of the pure-stand centroid (y < cy).  0.16 m is the configured
                        front_gap_m of the camp shirt.
  bridging_edges        collar edges that join a +x-half front vertex to a -x-half front vertex (any such edge spans the opening).  Gate: 0.
  bridging_faces        collar faces with front vertices on both sides.  Gate: 0.
  max_edge_m            longest collar edge anywhere.
  vertices_in_opening   collar vertices inside the baseline opening corridor |x| < chord_ref / 2 - 5 mm, in front of the centroid.
"""
import sys, os, json
import numpy as np
import bpy
from scipy.spatial import cKDTree


def measure(path):
    bpy.ops.wm.open_mainfile(filepath=path, load_ui=False)
    C = bpy.data.objects['collar']; me = C.data; n = len(me.vertices); V = np.empty(n * 3); me.vertices.foreach_get('co', V); V = V.reshape(-1, 3)
    gs = C.vertex_groups['collar_stand'].index; gl = C.vertex_groups['collar_leaf'].index; ws = np.zeros(n); wl = np.zeros(n)
    for v in me.vertices:
        for g in v.groups:
            if g.group == gs: ws[v.index] = g.weight
            elif g.group == gl: wl[v.index] = g.weight
    pure = (ws > 0) & (wl == 0); cen = V[pure][:, :2].mean(0); front = V[:, 1] < cen[1]
    Lm = front & (V[:, 0] > cen[0]); Rm = front & (V[:, 0] <= cen[0]); L = np.flatnonzero(Lm); R = np.flatnonzero(Rm)
    d, j = cKDTree(V[R]).query(V[L]); k = int(np.argmin(d))
    E = np.empty(len(me.edges) * 2, dtype=np.int32); me.edges.foreach_get('vertices', E); E = E.reshape(-1, 2)
    el = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    bridge_e = int(((Lm[E[:, 0]] & Rm[E[:, 1]]) | (Rm[E[:, 0]] & Lm[E[:, 1]])).sum())
    bridge_f = sum(1 for p in me.polygons if Lm[list(p.vertices)].any() and Rm[list(p.vertices)].any())
    th = np.degrees(np.arctan2(V[pure][:, 0] - cen[0], -(V[pure][:, 1] - cen[1])))
    return dict(file=os.path.relpath(path).replace('\\', '/'), opening_chord_m=round(float(d[k]), 5), chord_ends=[np.round(V[L[k]], 4).tolist(), np.round(V[R[j[k]]], 4).tolist()],
                bridging_edges=bridge_e, bridging_faces=int(bridge_f), max_edge_m=round(float(el.max()), 5),
                stand_theta_abs_min_deg=round(float(np.abs(th).min()), 2), stand_theta_pos_min_deg=round(float(th[th > 0].min()), 2), stand_theta_neg_max_deg=round(float(th[th < 0].max()), 2),
                _V=V, _front=front, _cen=cen)


out = os.path.abspath(sys.argv[1]); a = sys.argv[2:]; res = {}
for i in range(0, len(a), 2): res[a[i]] = measure(os.path.abspath(a[i + 1]))
ref = min(m['opening_chord_m'] for m in res.values())
for lab, m in res.items():
    V, front, cen = m.pop('_V'), m.pop('_front'), m.pop('_cen'); half = max(ref / 2 - 0.005, 0.0)
    m['vertices_in_opening'] = int((front & (np.abs(V[:, 0] - cen[0]) < half)).sum()); m['opening_corridor_half_width_m'] = round(half, 5)
    print('%-22s chord=%.4f m  bridging edges=%d faces=%d  max edge=%.4f m  verts in opening corridor(+-%.4f)=%d  stand ends theta %+.1f / %+.1f deg' % (
        lab, m['opening_chord_m'], m['bridging_edges'], m['bridging_faces'], m['max_edge_m'], half, m['vertices_in_opening'], m['stand_theta_neg_max_deg'], m['stand_theta_pos_min_deg']), flush=True)
json.dump(res, open(out, 'w'), indent=1, sort_keys=True)
