"""Forensic probe v2 (read-only): BS-TRACK-001 G0 master clearance, discriminating measurements.

Extends prior_inspection_v1 probe_track_clearance.py (same metric: nylon shell = first n_nylon_faces polygons; signed nearest
distance to M2_Baked_ThetaFit). Adds:
  * |n_z| of the body normal at the nearest point, for inside-body vs other vertices (tests the lock_z mechanism:
    build_garment_g0.assemble passes lock_z=True for ALL pants vertices and |x|<0.15 m for jacket vertices);
  * sheet classification by vertex-normal orientation, (p - loc) . n_vertex > 0 => outer sheet, instead of the v1 half split;
  * arm-vs-torso split by |x| of the nearest body point (heuristic, threshold 0.21 m, stated in the output);
  * whether each inside vertex lies in the lock_z band.
  python -I probe_track_v2.py <master.blend> <out.json>
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
ARM_X = 0.21
res = dict(file=path, sha256_before=sha0, arm_split_rule=f'|x of nearest body point| > {ARM_X} m => arm (heuristic)')
pct = lambda a, q: round(float(np.percentile(a, q)), 3) if len(a) else None
for name in ('G0_jacket_bracket', 'G0_pants'):
    o = bpy.data.objects[name]; M = o.matrix_world; R = M.to_3x3()
    nf = int(o['n_nylon_faces']); idx = sorted({i for p in o.data.polygons[:nf] for i in p.vertices})
    P = np.array([(M @ o.data.vertices[i].co)[:] for i in idx]); VN = np.array([(R @ o.data.vertices[i].normal).normalized()[:] for i in idx])
    d = np.zeros(len(idx)); nz = np.zeros(len(idx)); bx = np.zeros(len(idx)); side = np.zeros(len(idx))
    for k, p in enumerate(P):
        loc, nrm, _, dist = bvh.find_nearest(Vector(p)); v = Vector(p) - loc
        d[k] = (-dist if v.dot(nrm) < 0 else dist) * 1000; nz[k] = abs(nrm.z); bx[k] = abs(loc.x)
        side[k] = np.dot(np.array(v[:]), VN[k])
    outer = side > 0; arm = bx > ARM_X; ins = d < 0; near = d < 7
    lockband = (np.abs(P[:, 0]) < 0.15) if name.startswith('G0_jacket') else np.ones(len(P), bool)
    r = dict(shell_verts=len(P), inside=int(ins.sum()), within_7mm=int(near.sum()), frac_within_7mm=round(float(near.mean()), 4),
             body_normal_abs_nz=dict(inside=dict(p10=pct(nz[ins], 10), median=pct(nz[ins], 50), p90=pct(nz[ins], 90)),
                                     clear_ge_7mm=dict(p10=pct(nz[~near], 10), median=pct(nz[~near], 50), p90=pct(nz[~near], 90)),
                                     inside_share_nz_gt_0p7=round(float((nz[ins] > 0.7).mean()), 4) if ins.any() else None,
                                     clear_share_nz_gt_0p7=round(float((nz[~near] > 0.7).mean()), 4)),
             inside_in_lock_z_band=int((ins & lockband).sum()),
             sheet_by_vertex_normal=dict(outer=dict(n=int(outer.sum()), inside=int((ins & outer).sum()), within_7mm=int((near & outer).sum()), p1_mm=pct(d[outer], 1)),
                                         inner=dict(n=int((~outer).sum()), inside=int((ins & ~outer).sum()), within_7mm=int((near & ~outer).sum()), p1_mm=pct(d[~outer], 1))),
             region_heuristic=dict(arm=dict(n=int(arm.sum()), within_7mm=int((near & arm).sum()), inside=int((ins & arm).sum())),
                                   torso_or_leg=dict(n=int((~arm).sum()), within_7mm=int((near & ~arm).sum()), inside=int((ins & ~arm).sum()))))
    res[name] = r
res['sha256_after'] = hashlib.sha256(open(path, 'rb').read()).hexdigest(); res['input_unchanged'] = res['sha256_after'] == sha0
json.dump(res, open(out, 'w'), indent=1); print('PROBE TRACK V2', 'unchanged' if res['input_unchanged'] else 'CHANGED!')
