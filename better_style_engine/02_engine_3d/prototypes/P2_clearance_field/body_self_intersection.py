"""body_self_intersection.py  does the posed tracksuit body (M2_Baked_ThetaFit in the g0 master) pass through itself where the winding rule and the nearest-face rules disagree?
Counts intersecting triangle pairs that share no vertex (mathutils BVHTree.overlap, the same test REQ-04 used), on the position-welded body.  python -I body_self_intersection.py"""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); import clearance_field as CF
import bpy  # noqa: F401
from mathutils.bvhtree import BVHTree
b = np.load(HERE.parent / "_inputs/arrays/g0_master__M2_Baked_ThetaFit.npz"); V, T, _ = CF.weld(b["V"], b["tris"]); t = BVHTree.FromPolygons([tuple(p) for p in V.tolist()], [tuple(int(i) for i in x) for x in T.tolist()], all_triangles=True, epsilon=0.0)
sets = [set(x) for x in T.tolist()]; pairs = [(i, j) for i, j in t.overlap(t) if i < j and not (sets[i] & sets[j])]; cen = V[T].mean(1); pc = np.array([(cen[i] + cen[j]) / 2 for i, j in pairs]) if pairs else np.zeros((0, 3))
def box(m): return dict(pairs=int(m.sum()), z=[float(pc[m][:, 2].min()), float(pc[m][:, 2].max())] if m.any() else None, abs_x=[float(np.abs(pc[m][:, 0]).min()), float(np.abs(pc[m][:, 0]).max())] if m.any() else None)
z = pc[:, 2]; ax = np.abs(pc[:, 0]); res = dict(vertices=int(len(V)), triangles=int(len(T)), intersecting_pairs_no_shared_vertex=len(pairs), all=box(np.ones(len(pc), bool)),
    armpit_band_z_1p25_to_1p40=box((z > 1.25) & (z < 1.40)), crotch_band_z_0p85_to_1p00_absx_lt_0p05=box((z > 0.85) & (z < 1.0) & (ax < 0.05)),
    z_histogram_0p1m={f"{k:.1f}": int(((z >= k) & (z < k + 0.1)).sum()) for k in np.arange(0, 1.8, 0.1) if ((z >= k) & (z < k + 0.1)).any()})
json.dump(res, open(HERE / "out/body_self_intersection.json", "w"), indent=1); print(json.dumps(res))
