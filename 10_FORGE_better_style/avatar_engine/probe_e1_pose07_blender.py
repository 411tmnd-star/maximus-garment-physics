"""Forensic probe (read-only): POSE_07 native 8-weight vs top-4 body, per-vertex displacement, top vertex indices,
and determinant / rotation spread of the LBS blended matrix sum_k w_k M_k for the worst vertices.

Mirrors the setup of engineering/OWNER_DECISIONS_2026-11-09/e1_4weight/tools/e1_top4_scratch_blender.py (same native blend,
same top-4 rule 'bone_index', same PRT.Retargeter path) but never saves a .blend and never touches the GLB.
  blender -b --factory-startup --python-exit-code 1 -P probe_e1_pose07_blender.py -- <out_json>
"""
import os
import json, math, sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix

HE = Path(os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/human_engine')))
GP = HE / "src_garment_pipeline"; CAT = HE / "src_catalog/better_style_catalog"
sys.path.insert(0, str(GP)); sys.path.insert(0, str(CAT / "engine/components"))
import pose_retarget as PRT  # noqa: E402

OUTJ = Path(sys.argv[sys.argv.index('--') + 1])
NATIVE = GP / "human_build/human_engine/mannequin_modes/03_full_rigged/out/03_full_rigged.blend"
bpy.ops.wm.open_mainfile(filepath=str(NATIVE))
arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
body = next(o for o in bpy.data.objects if o.type == 'MESH' and any(m.type == 'ARMATURE' for m in o.modifiers))
bone_index = {b.name: i for i, b in enumerate(arm.data.bones)}
names = [g.name for g in body.vertex_groups]; keep = [i for i, n in enumerate(names) if n in bone_index]; col = {gi: k for k, gi in enumerate(keep)}
nat = [names[i] for i in keep]; NV = len(body.data.vertices)
Wn = np.zeros((NV, len(keep)))
for v in body.data.vertices:
    for g in v.groups:
        if g.group in col and g.weight > 0: Wn[v.index, col[g.group]] = g.weight
bidx = np.array([bone_index[n] for n in nat])
W4 = np.zeros_like(Wn)
for v in range(NV):
    nz = np.nonzero(Wn[v] > 0)[0]; k4 = sorted(nz, key=lambda k: (-Wn[v, k], bidx[k]))[:4]; W4[v, k4] = Wn[v, k4]
W4 /= W4.sum(1, keepdims=True)
W8 = Wn / Wn.sum(1, keepdims=True)          # what Blender's armature modifier effectively does (normalises the vertex's groups)

b4 = body.copy(); b4.data = body.data.copy(); b4.name = 'PROBE_top4'; bpy.context.scene.collection.objects.link(b4)
for n in nat: b4.vertex_groups[n].remove(list(range(NV)))
for k, n in enumerate(nat):
    vg = b4.vertex_groups[n]
    for vi in np.nonzero(W4[:, k] > 0)[0]: vg.add([int(vi)], float(W4[vi, k]), 'REPLACE')


def wv(obj):
    dg = bpy.context.evaluated_depsgraph_get(); ev = obj.evaluated_get(dg); me = ev.to_mesh()
    a = np.empty(len(me.vertices) * 3); me.vertices.foreach_get('co', a); ev.to_mesh_clear()
    W = np.array(obj.matrix_world); return a.reshape(-1, 3) @ W[:3, :3].T + W[:3, 3]


for pb in arm.pose.bones: pb.matrix_basis = Matrix.Identity(4)
bpy.context.view_layer.update(); rest8 = wv(body)
RT = PRT.Retargeter(arm); P = PRT.load_poses(); RT.apply(P['POSE_07'], 1.0); bpy.context.view_layer.update()
v8, v4 = wv(body), wv(b4)
d = np.linalg.norm(v4 - v8, axis=1)
top = np.argsort(-d)[:15]

# skinning matrices in armature space: S_k = pose_matrix @ rest_matrix^-1
S = {}
for k, n in enumerate(nat):
    pb = arm.pose.bones[n]; S[k] = np.array(pb.matrix @ pb.bone.matrix_local.inverted())


def blend_stats(Wrow):
    ks = np.nonzero(Wrow > 0)[0]; A = sum(Wrow[k] * S[k][:3, :3] for k in ks)
    rots = [S[k][:3, :3] for k in ks]
    ang = 0.0
    for i in range(len(rots)):
        for j in range(i + 1, len(rots)):
            c = (np.trace(rots[i].T @ rots[j]) - 1) / 2; ang = max(ang, math.degrees(math.acos(max(-1, min(1, c)))))
    return float(np.linalg.det(A)), ang, [(nat[k], round(float(Wrow[k]), 4)) for k in ks[np.argsort(-Wrow[ks])]]


rows = []
for v in top:
    det8, ang8, inf8 = blend_stats(W8[v]); det4, ang4, inf4 = blend_stats(W4[v])
    rows.append(dict(vertex=int(v), d_8w_vs_4w_mm=round(float(d[v] * 1e3), 3), rest_xyz_m=[round(float(x), 4) for x in rest8[v]],
                     det_8w=round(det8, 5), det_4w=round(det4, 5), max_pair_rot_deg_8w=round(ang8, 2), max_pair_rot_deg_4w=round(ang4, 2),
                     dropped_weight_mass=round(float(W8[v][(W8[v] > 0) & (W4[v] == 0)].sum()), 4), influences_8w=inf8, influences_4w=inf4))
# whole-body determinant distribution, 8w and 4w
dets8 = np.array([blend_stats(W8[v])[0] for v in range(NV)]); dets4 = np.array([blend_stats(W4[v])[0] for v in range(NV)])
q = lambda a: dict(min=round(float(a.min()), 5), p1=round(float(np.percentile(a, 1)), 5), p50=round(float(np.median(a)), 5))
rep = dict(native=str(NATIVE), pose='POSE_07', vertices=NV, max_d_mm=round(float(d.max() * 1e3), 4),
           p99_d_mm=round(float(np.percentile(d, 99) * 1e3), 4), det_8w=q(dets8), det_4w=q(dets4),
           det_lt_0p9_8w=int((dets8 < 0.9).sum()), det_lt_0p9_4w=int((dets4 < 0.9).sum()),
           corr_top_vertices_det4_minus_det8=[round(r['det_4w'] - r['det_8w'], 5) for r in rows], top=rows)
OUTJ.write_text(json.dumps(rep, indent=1)); print('PROBE DONE', rep['max_d_mm'], flush=True)
