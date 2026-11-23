"""Forensic probe (read-only): body clearance of the CURRENT BS-TRACK-001 G0 master, using the same metric as
engineering/PHASE3_2026-11-09/tracksuit_relax/eval_g0.py (nylon shell = first n_nylon_faces polygons; signed nearest distance
to M2_Baked_ThetaFit). Adds: location of inside-body vertices, and the split by Solidify layer (outer vs inner sheet).
  python probe_track_clearance.py <master.blend> <out.json>
"""
import sys, json, hashlib
import numpy as np, bpy, bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree

path, out = sys.argv[1], sys.argv[2]
sha0 = hashlib.sha256(open(path, 'rb').read()).hexdigest()
bpy.ops.wm.open_mainfile(filepath=path, load_ui=False)
body = bpy.data.objects['M2_Baked_ThetaFit']
bm = bmesh.new(); bm.from_mesh(body.data); bm.transform(body.matrix_world); bvh = BVHTree.FromBMesh(bm)
res = dict(file=path, sha256_before=sha0)
for name in ('G0_jacket_bracket', 'G0_jacket_orbit', 'G0_pants'):
    o = bpy.data.objects[name]; G = np.array([(o.matrix_world @ v.co)[:] for v in o.data.vertices])
    nf = int(o['n_nylon_faces']); idx = sorted({i for p in o.data.polygons[:nf] for i in p.vertices}); S = G[idx]
    d = []; nd = []
    for p in S:
        loc, nrm, _, dist = bvh.find_nearest(Vector(p)); s = -dist if (Vector(p) - loc).dot(nrm) < 0 else dist; d.append(s)
    d = np.array(d) * 1000
    # Solidify (offset -1) writes the original sheet first, then the inner copy: split the shell vertex list in halves
    half = len(idx) // 2; layer = np.array(['outer'] * half + ['inner'] * (len(idx) - half))
    ins = S[d < 0]
    r = dict(n_nylon_faces=nf, shell_verts=len(S), clear_mm={k: round(float(np.percentile(d, q)), 2) for k, q in (('p1', 1), ('p10', 10), ('p25', 25), ('p50', 50), ('p90', 90))},
             inside_body=int((d < 0).sum()), within_7mm=int((d < 7).sum()), frac_within_7mm=round(float((d < 7).mean()), 4),
             by_layer_assumed_half_split={L: dict(n=int((layer == L).sum()), within_7mm=int(((d < 7) & (layer == L)).sum()), inside=int(((d < 0) & (layer == L)).sum()),
                                                  p1_mm=round(float(np.percentile(d[layer == L], 1)), 2)) for L in ('outer', 'inner')},
             inside_bbox_m=[ins.min(0).round(3).tolist(), ins.max(0).round(3).tolist()] if len(ins) else None,
             inside_z_hist_m={f'{z0:.2f}': int(((ins[:, 2] >= z0) & (ins[:, 2] < z0 + 0.1)).sum()) for z0 in np.arange(0.0, 1.6, 0.1)} if len(ins) else None)
    res[name] = r
res['sha256_after'] = hashlib.sha256(open(path, 'rb').read()).hexdigest()
json.dump(res, open(out, 'w'), indent=1); print('OK unchanged' if res['sha256_after'] == sha0 else 'CHANGED!')
