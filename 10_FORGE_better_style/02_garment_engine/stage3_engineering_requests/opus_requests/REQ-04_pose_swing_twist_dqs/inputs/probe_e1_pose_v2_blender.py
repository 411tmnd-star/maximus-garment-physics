"""Forensic probe v2 (read-only, snapshot inputs). Extends prior_inspection_v1 probe_e1_pose07_blender.py.

Adds: bone-pair attribution of the max relative rotation; closed-mesh volume rest vs posed (8w / 4w);
per-region triangle area ratio (posed / rest); rest->pose displacement; max 8w-vs-4w displacement for POSE_01..10.
Inputs come from the snapshot copy; their SHA-256 is checked before and after. No file is saved by Blender.
  blender -b --factory-startup --python-exit-code 1 -P probe_e1_pose_v2_blender.py -- <out_json>
"""
import os
import hashlib, json, math, sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix

HE = Path(os.path.normpath(os.path.expanduser('~/Desktop/better_style_execution_workspace/snapshot/ACTIVE/HUMAN_ENGINE')))
GP = HE / "src_garment_pipeline"; CAT = HE / "src_catalog/better_style_catalog"
sys.path.insert(0, str(GP)); sys.path.insert(0, str(CAT / "engine/components"))
import pose_retarget as PRT  # noqa: E402

OUTJ = Path(sys.argv[sys.argv.index('--') + 1])
NATIVE = GP / "human_build/human_engine/mannequin_modes/03_full_rigged/out/03_full_rigged.blend"
POSES = CAT / "garment_registry/poses/canonical_poses.json"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs_before = {str(p.relative_to(HE)): sha(p) for p in (NATIVE, POSES, Path(PRT.__file__))}
bpy.ops.wm.open_mainfile(filepath=str(NATIVE))
arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
body = next(o for o in bpy.data.objects if o.type == 'MESH' and any(m.type == 'ARMATURE' for m in o.modifiers))
mod = next(m for m in body.modifiers if m.type == 'ARMATURE')
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
W8 = Wn / Wn.sum(1, keepdims=True)
infl_count = (Wn > 0).sum(1)

b4 = body.copy(); b4.data = body.data.copy(); b4.name = 'PROBE_top4'; bpy.context.scene.collection.objects.link(b4)
for n in nat: b4.vertex_groups[n].remove(list(range(NV)))
for k, n in enumerate(nat):
    vg = b4.vertex_groups[n]
    for vi in np.nonzero(W4[:, k] > 0)[0]: vg.add([int(vi)], float(W4[vi, k]), 'REPLACE')

body.data.calc_loop_triangles()
TRI = np.array([[lt.vertices[0], lt.vertices[1], lt.vertices[2]] for lt in body.data.loop_triangles])


def wv(obj):
    dg = bpy.context.evaluated_depsgraph_get(); ev = obj.evaluated_get(dg); me = ev.to_mesh()
    a = np.empty(len(me.vertices) * 3); me.vertices.foreach_get('co', a); ev.to_mesh_clear()
    W = np.array(obj.matrix_world); return a.reshape(-1, 3) @ W[:3, :3].T + W[:3, 3]


def volume(X):  # signed volume of a triangle soup (exact for a closed, consistently oriented mesh)
    a, b, c = X[TRI[:, 0]], X[TRI[:, 1]], X[TRI[:, 2]]
    return float(np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6.0)


def tri_area(X):
    return 0.5 * np.linalg.norm(np.cross(X[TRI[:, 1]] - X[TRI[:, 0]], X[TRI[:, 2]] - X[TRI[:, 0]]), axis=1)


# boundary edges => is the body closed? (volume only meaningful if closed)
from collections import Counter
ec = Counter(tuple(sorted(e)) for t in TRI for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])))
boundary_edges = sum(1 for c in ec.values() if c == 1); nonmanifold_edges = sum(1 for c in ec.values() if c > 2)

for pb in arm.pose.bones: pb.matrix_basis = Matrix.Identity(4)
bpy.context.view_layer.update(); rest = wv(body)
A_rest = tri_area(rest); V_rest = volume(rest)
RT = PRT.Retargeter(arm); P = PRT.load_poses()

# --- all poses: max 8w vs 4w displacement (context for POSE_07)
per_pose = {}
for name in sorted(P):
    for pb in arm.pose.bones: pb.matrix_basis = Matrix.Identity(4)
    RT.apply(P[name], 1.0); bpy.context.view_layer.update()
    a8, a4 = wv(body), wv(b4); dd = np.linalg.norm(a4 - a8, axis=1)
    per_pose[name] = dict(max_mm=round(float(dd.max() * 1e3), 3), argmax_vertex=int(dd.argmax()), p99_mm=round(float(np.percentile(dd, 99) * 1e3), 3),
                          volume_rel_change_8w=round(volume(a8) / V_rest - 1, 6), volume_rel_change_4w=round(volume(a4) / V_rest - 1, 6))

# --- POSE_07 detail
for pb in arm.pose.bones: pb.matrix_basis = Matrix.Identity(4)
RT.apply(P['POSE_07'], 1.0); bpy.context.view_layer.update()
v8, v4 = wv(body), wv(b4)
S = {k: np.array(arm.pose.bones[n].matrix @ arm.pose.bones[n].bone.matrix_local.inverted()) for k, n in enumerate(nat)}


def pair_stats(Wrow):
    ks = np.nonzero(Wrow > 0)[0]; A = sum(Wrow[k] * S[k][:3, :3] for k in ks)
    best = (0.0, None, None)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            Ri, Rj = S[ks[i]][:3, :3], S[ks[j]][:3, :3]
            c = (np.trace(Ri.T @ Rj) - 1) / 2; ang = math.degrees(math.acos(max(-1, min(1, c))))
            if ang > best[0]: best = (ang, nat[ks[i]], nat[ks[j]])
    return float(np.linalg.det(A)), best


GROUPS = {
    'neck_shoulder_top_v1': [8165, 8204, 8172, 8164, 8203, 8187, 8221],
    'anterior_axilla_v1': [8052, 8123, 8124],
}
# one-ring triangles of each group
tri_of = {g: np.nonzero(np.isin(TRI, ids).any(1))[0] for g, ids in GROUPS.items()}
rows = {}
for g, ids in GROUPS.items():
    rr = []
    for v in ids:
        d8, (a8, p8a, p8b) = pair_stats(W8[v]); d4, (a4, p4a, p4b) = pair_stats(W4[v])
        rr.append(dict(vertex=v, rest_xyz_m=[round(float(x), 4) for x in rest[v]], influences_native=int(infl_count[v]),
                       rest_to_pose_disp_mm_8w=round(float(np.linalg.norm(v8[v] - rest[v]) * 1e3), 2),
                       rest_to_pose_disp_mm_4w=round(float(np.linalg.norm(v4[v] - rest[v]) * 1e3), 2),
                       d_8w_vs_4w_mm=round(float(np.linalg.norm(v4[v] - v8[v]) * 1e3), 3),
                       det_8w=round(d8, 5), det_4w=round(d4, 5),
                       max_pair_8w=dict(deg=round(a8, 2), bones=[p8a, p8b]), max_pair_4w=dict(deg=round(a4, 2), bones=[p4a, p4b])))
    t = tri_of[g]
    rows[g] = dict(vertices=rr, one_ring_triangles=int(len(t)),
                   area_ratio_8w=dict(min=round(float((tri_area(v8)[t] / A_rest[t]).min()), 4), median=round(float(np.median(tri_area(v8)[t] / A_rest[t])), 4)),
                   area_ratio_4w=dict(min=round(float((tri_area(v4)[t] / A_rest[t]).min()), 4), median=round(float(np.median(tri_area(v4)[t] / A_rest[t])), 4)))

rep = dict(probe='probe_e1_pose_v2_blender.py', blender=bpy.app.version_string, native=str(NATIVE.relative_to(HE)), vertices=NV,
           triangles=int(len(TRI)), body_boundary_edges=boundary_edges, body_nonmanifold_edges=nonmanifold_edges,
           armature_modifier=dict(use_deform_preserve_volume=bool(mod.use_deform_preserve_volume), use_multi_modifier=bool(mod.use_multi_modifier)),
           native_influences_per_vertex=dict(max=int(infl_count.max()), gt4=int((infl_count > 4).sum()), hist={int(k): int(v) for k, v in zip(*np.unique(infl_count, return_counts=True))}),
           rest_volume_m3=round(V_rest, 6), pose07_volume_rel_change_8w=round(volume(v8) / V_rest - 1, 6), pose07_volume_rel_change_4w=round(volume(v4) / V_rest - 1, 6),
           pose07_max_d_mm=round(float(np.linalg.norm(v4 - v8, axis=1).max() * 1e3), 4), groups=rows, per_pose=per_pose,
           inputs_sha256_before=inputs_before)
rep['inputs_sha256_after'] = {str(p.relative_to(HE)): sha(p) for p in (NATIVE, POSES, Path(PRT.__file__))}
rep['inputs_unchanged'] = rep['inputs_sha256_after'] == inputs_before
OUTJ.parent.mkdir(parents=True, exist_ok=True)
OUTJ.write_text(json.dumps(rep, indent=1)); print('PROBE V2 DONE', rep['pose07_max_d_mm'], rep['inputs_unchanged'], flush=True)
