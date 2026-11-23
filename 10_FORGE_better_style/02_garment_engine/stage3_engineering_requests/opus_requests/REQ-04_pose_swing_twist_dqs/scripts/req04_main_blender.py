"""REQ-04 main probe: swing/twist decomposition, LBS vs DQS, twist-redistribution experiment (read-only; works on copies in inputs/).
Pose application is identical to 00_CONTROL/scripts/inspection_tools/probes/probe_e1_pose_v2_blender.py:
    for pb in arm.pose.bones: pb.matrix_basis = Identity;  Retargeter(arm).apply(P[name], 1.0);  view_layer.update()
Nothing is saved.  Usage:
  blender -b --factory-startup --python-exit-code 1 -P req04_main_blender.py -- <inputs_dir> <out_json>

Definitions
  S_b      = (posed armature-space bone matrix) @ (rest armature-space bone matrix)^-1, rotation part (the skinning matrix; armature world = identity)
  D(a,b)   = Rb_rest^T (S_a^T S_b) Rb_rest  = rotation of bone b relative to bone a, compared with rest, expressed in b's REST bone frame.
             For a parent/ancestor a this is exactly (Ra_rest^T Rb_rest)^T (Ra_pose^T Rb_pose).  Its angle equals the angle between S_a and S_b.
  swing/twist: D = Swing * Twist, Twist about b's rest long axis (local +Y); q=(w,x,y,z) of D:
             twist = 2*atan2(y, w) wrapped to (-180,180];  swing = 2*acos(sqrt(w^2+y^2));  total = 2*acos(|w|).
"""
import hashlib, json, math, sys, time
from collections import Counter
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

T0 = time.time()
a = sys.argv[sys.argv.index('--') + 1:]
INP = Path(a[0]); OUTJ = Path(a[1])
sys.dont_write_bytecode = True
sys.path.insert(0, str(INP))
import pose_retarget as PRT  # noqa: E402  (copied from HE, sha256 de677e4b...)
PRT.REGISTRY = INP / 'canonical_poses.json'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs_before = {p.name: sha(p) for p in sorted(INP.iterdir()) if p.is_file()}

bpy.ops.wm.open_mainfile(filepath=str(INP / '03_full_rigged.blend'))
arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
body = next(o for o in bpy.data.objects if o.type == 'MESH' and any(m.type == 'ARMATURE' for m in o.modifiers))
mod = next(m for m in body.modifiers if m.type == 'ARMATURE')
assert np.allclose(np.array(arm.matrix_world), np.eye(4)) and np.allclose(np.array(body.matrix_world), np.eye(4))
bone_index = {b.name: i for i, b in enumerate(arm.data.bones)}
names = [g.name for g in body.vertex_groups]; keep = [i for i, n in enumerate(names) if n in bone_index]; col = {gi: k for k, gi in enumerate(keep)}
nat = [names[i] for i in keep]; NV = len(body.data.vertices)
Wn = np.zeros((NV, len(keep)))
for v in body.data.vertices:
    for g in v.groups:
        if g.group in col and g.weight > 0: Wn[v.index, col[g.group]] = g.weight
W8 = Wn / Wn.sum(1, keepdims=True)
body.data.calc_loop_triangles()
TRI = np.array([[lt.vertices[0], lt.vertices[1], lt.vertices[2]] for lt in body.data.loop_triangles])
NT = len(TRI)
VI = [8165, 8204, 8172, 8052, 8123, 8124]
v_tris = [[] for _ in range(NV)]
for ti, t in enumerate(TRI):
    for v in t: v_tris[v].append(ti)
# triangle dominant bone (max summed normalised weight over its 3 vertices)
tri_dom = (W8[TRI[:, 0]] + W8[TRI[:, 1]] + W8[TRI[:, 2]]).argmax(1)
# adjacency for self-intersection filtering
tri_sets = [set(t) for t in TRI.tolist()]
REST_B = {b.name: np.array(b.matrix_local) for b in arm.data.bones}


def wv(obj):
    dg = bpy.context.evaluated_depsgraph_get(); ev = obj.evaluated_get(dg); me = ev.to_mesh()
    assert len(me.vertices) == NV
    arr = np.empty(len(me.vertices) * 3); me.vertices.foreach_get('co', arr); ev.to_mesh_clear()
    W = np.array(obj.matrix_world); return arr.reshape(-1, 3) @ W[:3, :3].T + W[:3, 3]


def volume(X):
    p, q, r = X[TRI[:, 0]], X[TRI[:, 1]], X[TRI[:, 2]]
    return float(np.einsum('ij,ij->i', p, np.cross(q, r)).sum() / 6.0)


def tri_cross(X):
    return np.cross(X[TRI[:, 1]] - X[TRI[:, 0]], X[TRI[:, 2]] - X[TRI[:, 0]])


def polar(M):
    U, _, Vt = np.linalg.svd(M); R = U @ Vt
    if np.linalg.det(R) < 0: U[:, -1] *= -1; R = U @ Vt
    return R


def skin_mats():
    S, Srot = {}, {}
    for pb in arm.pose.bones:
        M = np.array(pb.matrix) @ np.linalg.inv(REST_B[pb.name]); S[pb.name] = M; Srot[pb.name] = polar(M[:3, :3])
    return S, Srot


def quat_of(R):
    q = Matrix(R.tolist()).to_quaternion(); q.normalize(); return np.array([q.w, q.x, q.y, q.z])


def swing_twist(D):
    w, x, y, z = quat_of(D)
    if w < 0: w, x, y, z = -w, -x, -y, -z
    total = 2 * math.degrees(math.acos(min(1.0, abs(w))))
    twist = 2 * math.degrees(math.atan2(y, w)); twist = (twist + 180) % 360 - 180
    swing = 2 * math.degrees(math.acos(min(1.0, math.sqrt(w * w + y * y))))
    return dict(total_deg=round(total, 2), swing_deg=round(swing, 2), twist_deg=round(twist, 2),
                twist_ill_conditioned=bool(math.sqrt(w * w + y * y) < 0.05))


def rel(Srot, pa, cb):
    Rb = REST_B[cb][:3, :3]; Rb = polar(Rb)
    return Rb.T @ (Srot[pa].T @ Srot[cb]) @ Rb


def head_w(n): return np.array(arm.pose.bones[n].head)
def tail_w(n): return np.array(arm.pose.bones[n].tail)


def elev(d):  # angle from downward vertical
    d = d / np.linalg.norm(d); return round(math.degrees(math.acos(max(-1, min(1, -d[2])))), 2)


PAIRS = []
for s in 'LR':
    PAIRS += [(f'spine01', f'clavicle.{s}'), (f'clavicle.{s}', f'shoulder01.{s}'), (f'shoulder01.{s}', f'upperarm01.{s}'),
              (f'clavicle.{s}', f'upperarm01.{s}'), (f'upperarm01.{s}', f'upperarm02.{s}'), (f'upperarm02.{s}', f'lowerarm01.{s}'),
              (f'lowerarm01.{s}', f'lowerarm02.{s}'), (f'lowerarm02.{s}', f'wrist.{s}'),
              (f'pelvis.{s}', f'upperleg01.{s}'), (f'upperleg01.{s}', f'upperleg02.{s}'), (f'upperleg02.{s}', f'lowerleg01.{s}'),
              (f'lowerleg01.{s}', f'lowerleg02.{s}'), (f'lowerleg02.{s}', f'foot.{s}')]


def joint_report():
    _, Srot = skin_mats(); out = {}
    for pa, cb in PAIRS: out[f'{pa}->{cb}'] = swing_twist(rel(Srot, pa, cb))
    limbs = {}
    for s in 'LR':
        sh, el, wr = head_w(f'upperarm01.{s}'), head_w(f'lowerarm01.{s}'), head_w(f'wrist.{s}')
        hip, kn, an = head_w(f'upperleg01.{s}'), head_w(f'lowerleg01.{s}'), head_w(f'foot.{s}')
        u = (el - sh) / np.linalg.norm(el - sh); f = (wr - el) / np.linalg.norm(wr - el)
        th = (kn - hip) / np.linalg.norm(kn - hip); sn = (an - kn) / np.linalg.norm(an - kn)
        limbs[s] = dict(upperarm_dir_world=[round(x, 4) for x in u], upperarm_elevation_from_down_deg=elev(u),
                        forearm_dir_world=[round(x, 4) for x in f], forearm_elevation_from_down_deg=elev(f),
                        elbow_flexion_deg=round(math.degrees(math.acos(max(-1, min(1, float(u @ f))))), 2),
                        thigh_dir_world=[round(x, 4) for x in th], thigh_elevation_from_down_deg=elev(th),
                        shin_dir_world=[round(x, 4) for x in sn], knee_flexion_deg=round(math.degrees(math.acos(max(-1, min(1, float(th @ sn))))), 2),
                        wrist_head_world_m=[round(x, 5) for x in wr])
    return out, limbs


def reset():
    for pb in arm.pose.bones: pb.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()


# ---------------- rest
reset(); mod.use_deform_preserve_volume = False; bpy.context.view_layer.update()
REST = wv(body); V_rest = volume(REST); CR = tri_cross(REST); A_rest = 0.5 * np.linalg.norm(CR, axis=1)
N_rest = CR / np.maximum(np.linalg.norm(CR, axis=1, keepdims=True), 1e-20)
VA_rest = np.array([A_rest[v_tris[v]].sum() for v in range(NV)])
ec = Counter(tuple(sorted(e)) for t in TRI.tolist() for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])))
mesh_closed = dict(boundary_edges=sum(1 for c in ec.values() if c == 1), nonmanifold_edges=sum(1 for c in ec.values() if c > 2))
rest_joints, rest_limbs = joint_report()


def self_intersections(X):
    t = BVHTree.FromPolygons([tuple(p) for p in X.tolist()], TRI.tolist(), all_triangles=True, epsilon=0.0)
    n = 0
    for i, j in t.overlap(t):
        if i < j and not (tri_sets[i] & tri_sets[j]): n += 1
    return n


SI_REST = self_intersections(REST)


def metrics(X, Srot=None, si=True):
    C = tri_cross(X); A = 0.5 * np.linalg.norm(C, axis=1); r = A / A_rest
    N = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-20)
    VA = np.array([A[v_tris[v]].sum() for v in range(NV)]); vr = VA / VA_rest
    raw_flip = int(((N * N_rest).sum(1) < 0).sum())
    out = dict(volume_rel_change_pct=round((volume(X) / V_rest - 1) * 100, 3),
               vertex_one_ring_area_ratio=dict(min=round(float(vr.min()), 4), argmin_vertex=int(vr.argmin()), p1=round(float(np.percentile(vr, 1)), 4),
                                               at_vertices={str(v): round(float(vr[v]), 4) for v in VI}),
               one_ring_min_triangle_ratio_at_vertices={str(v): round(float(r[v_tris[v]].min()), 4) for v in VI},
               triangle_area_ratio=dict(min=round(float(r.min()), 4), p1=round(float(np.percentile(r, 1)), 4),
                                        n_lt_0p5=int((r < 0.5).sum()), n_lt_0p25=int((r < 0.25).sum())),
               normal_flip_gt90_vs_rest_raw=raw_flip)
    if Srot is not None:  # flip relative to the rest normal carried by the triangle's dominant bone rotation
        Rd = np.stack([Srot[nat[k]] for k in tri_dom]); Nrd = np.einsum('tij,tj->ti', Rd, N_rest)
        out['normal_flip_gt90_vs_dominant_bone_rotated_rest'] = int(((N * Nrd).sum(1) < 0).sum())
    if si: out['self_intersecting_face_pairs'] = self_intersections(X)
    return out


def det_axilla(S):
    rr = {}
    for v in (8052, 8123, 8124, 8165):
        ks = np.nonzero(W8[v] > 0)[0]; A = sum(W8[v, k] * S[nat[k]][:3, :3] for k in ks); rr[str(v)] = round(float(np.linalg.det(A)), 5)
    return rr


RT = PRT.Retargeter(arm); P = PRT.load_poses()
# neutral stance (context: what the retargeter does before every pose)
reset(); RT.neutral(); bpy.context.view_layer.update(); neutral_joints, neutral_limbs = joint_report()

poses = {}
for name in sorted(P):
    reset(); RT.apply(P[name], 1.0); bpy.context.view_layer.update()
    S, Srot = skin_mats(); jr, limbs = joint_report()
    mod.use_deform_preserve_volume = False; bpy.context.view_layer.update(); XL = wv(body)
    mod.use_deform_preserve_volume = True; bpy.context.view_layer.update(); XD = wv(body)
    mod.use_deform_preserve_volume = False; bpy.context.view_layer.update()
    dd = np.linalg.norm(XL - XD, axis=1)
    poses[name] = dict(pose_name=P[name].get('pose_name'), hand_contacts={k: v['target'] for k, v in (P[name].get('hand_contacts') or {}).items()},
                       joints=jr, limbs=limbs, det_blended_linear_LBS=det_axilla(S),
                       LBS=metrics(XL, Srot), DQS=metrics(XD, Srot),
                       LBS_vs_DQS_displacement_mm=dict(max=round(float(dd.max() * 1e3), 3), argmax_vertex=int(dd.argmax()),
                                                       p99=round(float(np.percentile(dd, 99) * 1e3), 3),
                                                       at_vertices={str(v): round(float(dd[v] * 1e3), 3) for v in VI}))
    print(name, 'done', round(time.time() - T0, 1), 's', flush=True)

# ---------------- POSE_07 top bone pairs (item 4)
reset(); RT.apply(P['POSE_07'], 1.0); bpy.context.view_layer.update(); S7, Srot7 = skin_mats()
depth = {b.name: len(b.parent_recursive) for b in arm.data.bones}
pair_cnt = Counter()
for v in range(NV):
    ks = np.nonzero(W8[v] > 0.05)[0]
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)): pair_cnt[(ks[i], ks[j])] += 1
prs = []
for (i, j), c in pair_cnt.items():
    ba, bb = nat[i], nat[j]
    if depth[ba] > depth[bb] or (depth[ba] == depth[bb] and ba > bb): ba, bb = bb, ba  # bb = deeper bone; axis = bb's rest long axis
    st = swing_twist(rel(Srot7, ba, bb))
    prs.append(dict(bone_a=ba, bone_b_axis=bb, b_is_descendant_of_a=ba in [p.name for p in arm.data.bones[bb].parent_recursive],
                    covertices_gt5pct=int(c), **st))
prs.sort(key=lambda d: -d['total_deg'])
top_pairs = prs[:6]

# ---------------- twist redistribution (item 6), POSE_07, LBS
def twist_experiment(f):
    reset(); RT.apply(P['POSE_07'], 1.0); bpy.context.view_layer.update()
    wr0 = {s: head_w(f'wrist.{s}') for s in 'LR'}; el0 = {s: head_w(f'lowerarm01.{s}') for s in 'LR'}
    _, Srot = skin_mats(); info = {}
    for s in 'LR':
        st = swing_twist(rel(Srot, f'clavicle.{s}', f'upperarm01.{s}')); tau = st['twist_deg']
        pb1, pb2 = arm.pose.bones[f'upperarm01.{s}'], arm.pose.bones[f'upperarm02.{s}']
        keep2 = pb2.matrix.copy()
        # remove f*tau about upperarm01's own posed long axis (local +Y, i.e. right-multiply by Ry(-f*tau)), then restore upperarm02 world matrix
        pb1.matrix = pb1.matrix @ Matrix.Rotation(math.radians(-f * tau), 4, 'Y'); bpy.context.view_layer.update()
        pb2.matrix = keep2; bpy.context.view_layer.update()
        info[s] = dict(twist_before_deg=tau, moved_deg=round(f * tau, 2),
                       gap_upperarm01_tail_to_upperarm02_head_mm=round(float(np.linalg.norm(tail_w(f'upperarm01.{s}') - head_w(f'upperarm02.{s}')) * 1e3), 3))
    S, Srot = skin_mats(); jr, limbs = joint_report()
    for s in 'LR':
        info[s]['wrist_residual_mm'] = round(float(np.linalg.norm(head_w(f'wrist.{s}') - wr0[s]) * 1e3), 4)
        info[s]['elbow_residual_mm'] = round(float(np.linalg.norm(head_w(f'lowerarm01.{s}') - el0[s]) * 1e3), 4)
        for k in (f'clavicle.{s}->upperarm01.{s}', f'upperarm01.{s}->upperarm02.{s}', f'upperarm02.{s}->lowerarm01.{s}'): info[s][k] = jr[k]
    mod.use_deform_preserve_volume = False; bpy.context.view_layer.update(); XL = wv(body)
    mod.use_deform_preserve_volume = True; bpy.context.view_layer.update(); XD = wv(body)
    mod.use_deform_preserve_volume = False; bpy.context.view_layer.update()
    return dict(fraction=f, sides=info, det_blended_linear_LBS=det_axilla(S), LBS=metrics(XL, Srot), DQS_for_reference=metrics(XD, Srot, si=False), _X=XL)


tw = {}
for f in (0.5, 1.0):
    r = twist_experiment(f); r.pop('_X'); tw[f'f={f}'] = r
    print('twist f', f, 'done', round(time.time() - T0, 1), flush=True)

# ---------------- reproduce probe's 176.85 deg (angle between S_clavicle.L and S_upperarm01.L, POSE_07)
c = (np.trace(Srot7['clavicle.L'].T @ Srot7['upperarm01.L']) - 1) / 2
rep = dict(script='req04_main_blender.py', blender=bpy.app.version_string, vertices=NV, triangles=NT, mesh_closed=mesh_closed,
           armature_modifier_native=dict(use_deform_preserve_volume=False),
           definitions=dict(D='Rb_rest^T (S_a^T S_b) Rb_rest; S = pose @ rest^-1 rotation (polar)', swing_twist='D = Swing*Twist, twist about child rest local +Y; twist=2atan2(y,w), swing=2acos(sqrt(w^2+y^2))',
                            elevation='angle between limb direction and world -Z', normal_flip_dominant='triangle normal vs rest normal rotated by S of the triangle dominant bone (max summed normalised weight)',
                            vertex_one_ring_area_ratio='sum of posed areas of incident triangles / same at rest',
                            self_intersections='BVHTree.overlap(self), pairs i<j sharing no vertex'),
           rest_volume_m3=round(V_rest, 6), rest_self_intersecting_face_pairs=SI_REST,
           rest=dict(joints=rest_joints, limbs=rest_limbs), neutral_stance=dict(joints=neutral_joints, limbs=neutral_limbs),
           poses=poses, pose07_top6_bone_pairs=top_pairs, pose07_n_copairs_gt5pct=len(prs), pose07_top20_bone_pairs=prs[:20],
           pose07_reproduce_probe_angle_clavL_upperarm01L_deg=round(math.degrees(math.acos(max(-1, min(1, c)))), 2),
           twist_redistribution_pose07=tw, runtime_s=round(time.time() - T0, 1), inputs_sha256_before=inputs_before)
rep['inputs_sha256_after'] = {p.name: sha(p) for p in sorted(INP.iterdir()) if p.is_file()}
rep['inputs_unchanged'] = rep['inputs_sha256_after'] == inputs_before
OUTJ.write_text(json.dumps(rep, indent=1, default=float)); print('REQ04 MAIN DONE', rep['runtime_s'], rep['inputs_unchanged'], flush=True)
