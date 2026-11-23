"""measure_bs_walk.py - measurements on the existing Better Style walk scene (read-only copy of dual_state_scene.blend).
usage: python -I -B measure_bs_walk.py <scene.blend> <out.json>"""
import sys, json, math
import numpy as np, bpy
from mathutils import Vector
bpy.ops.wm.open_mainfile(filepath=sys.argv[1], load_ui=False); sc = bpy.context.scene
rig = next(o for o in bpy.data.objects if o.type == 'ARMATURE'); print('meshes', [(o.name, len(o.data.vertices), [m.type for m in o.modifiers]) for o in bpy.data.objects if o.type == 'MESH'])
body = next(o for o in bpy.data.objects if o.type == 'MESH' and any(m.type == 'ARMATURE' for m in o.modifiers) and 'foot.L' in o.vertex_groups and not o.name.startswith(('G0', 'Float')))
print('rig', rig.name, len(rig.data.bones), 'body', body.name, len(body.data.vertices), 'frames', sc.frame_start, sc.frame_end, 'fps', sc.render.fps)
print('armature modifier preserve volume:', [m.use_deform_preserve_volume for m in body.modifiers if m.type == 'ARMATURE'])
names = ['root', 'spine03', 'head', 'neck01', 'clavicle.L', 'clavicle.R', 'upperarm01.L', 'upperarm01.R', 'upperarm02.L', 'lowerarm01.L', 'lowerarm01.R', 'lowerarm02.L', 'wrist.L', 'wrist.R',
         'upperleg01.L', 'upperleg01.R', 'lowerleg01.L', 'lowerleg01.R', 'foot.L', 'foot.R', 'toe1-1.L', 'toe1-1.R', 'finger3-1.L', 'finger3-3.L', 'finger1-1.L']
names = [n for n in names if n in rig.pose.bones]
animated = sorted({fc.data_path.split('"')[1] for fc in (rig.animation_data.action.fcurves if rig.animation_data and rig.animation_data.action and hasattr(rig.animation_data.action, 'fcurves') else []) if '"' in fc.data_path})
if not animated:
    try:
        act = rig.animation_data.action; animated = sorted({fc.data_path.split('"')[1] for layer in act.layers for strip in layer.strips for cb in strip.channelbags for fc in cb.fcurves if '"' in fc.data_path})
    except Exception as e: print('fcurve read failed', e)
print('animated bones:', animated)
vg = {g.name: g.index for g in body.vertex_groups}
def group_idx(prefixes):
    ids = {vg[n] for n in vg if n.startswith(prefixes)}
    return [v.index for v in body.data.vertices if any(g.group in ids and g.weight > 0.5 for g in v.groups)]
footv = {s: group_idx((f'foot.{s}', f'toe')) for s in 'LR'}
for s in 'LR': footv[s] = [i for i in footv[s] if any(body.vertex_groups[g.group].name.endswith('.' + s) for g in body.data.vertices[i].groups if g.weight > 0.5)]
rec = []; dg = bpy.context.evaluated_depsgraph_get()
for f in range(sc.frame_start, sc.frame_end + 1):
    sc.frame_set(f); dg = bpy.context.evaluated_depsgraph_get(); M = rig.matrix_world
    H = {n: np.array(M @ rig.pose.bones[n].head) for n in names}; Tl = {n: np.array(M @ rig.pose.bones[n].tail) for n in names}
    ev = body.evaluated_get(dg); me = ev.to_mesh(); co = np.zeros(len(me.vertices) * 3); me.vertices.foreach_get('co', co); co = co.reshape(-1, 3)
    Mb = np.array(ev.matrix_world); W = co @ Mb[:3, :3].T + Mb[:3, 3]
    r = dict(frame=f, t=(f - sc.frame_start) / sc.render.fps, heads={n: H[n].tolist() for n in names}, tails={n: Tl[n].tolist() for n in names},
             foot_min_z={s: float(W[footv[s], 2].min()) for s in 'LR'}, foot_centroid={s: W[footv[s]].mean(0).tolist() for s in 'LR'},
             body_min_z=float(W[:, 2].min()), walker=[list(o.matrix_world.translation) for o in bpy.data.objects if o.name == 'Walker'],
             yaw=[math.degrees(o.matrix_world.to_euler().z) for o in bpy.data.objects if o.name == 'Walker'])
    ev.to_mesh_clear(); rec.append(r)
cam = sc.camera; floor_z = max([o.matrix_world.translation.z + o.dimensions.z for o in bpy.data.objects if o.name == 'Baseplate'] + [0.0])
json.dump(dict(floor_z=floor_z, frames=rec, names=names, animated=animated), open(sys.argv[2], 'w'))

# ---- summary
A = lambda k, n: np.array([r[k][n] for r in rec]); t = np.array([r['t'] for r in rec])
def ang(v, w): return math.degrees(math.acos(np.clip(v @ w / (np.linalg.norm(v) * np.linalg.norm(w)), -1, 1)))
walk = t < 8.0
for s in 'LR':
    ua = A('tails', f'upperarm01.{s}') - A('heads', f'upperarm01.{s}') if f'upperarm01.{s}' in names else None
    sh = A('heads', f'upperarm01.{s}'); wr = A('heads', f'wrist.{s}'); arm = wr - sh
    ab = [ang(a, np.array([0, 0, -1.0])) for a in arm]
    print(f'arm.{s}: shoulder->wrist angle from vertical: min {min(ab):.1f} max {max(ab):.1f} deg; wrist lateral offset from shoulder (body frame not removed) mean |dx| {np.abs(arm[:,0]).mean()*100:.1f} cm')
    el = A('heads', f'lowerarm01.{s}'); fl = [180 - ang(e - a, w - e) for a, e, w in zip(sh, el, wr)]
    print(f'elbow.{s}: included flexion min {180-max(fl):.1f} max {180-min(fl):.1f} deg (0 = straight)')
ank = {s: A('heads', f'foot.{s}') for s in 'LR'}; hip = {s: A('heads', f'upperleg01.{s}') for s in 'LR'}
yaw = np.radians(np.array([r['yaw'][0] if r['yaw'] else 0.0 for r in rec]))
lat = np.stack([np.cos(yaw), np.sin(yaw), 0 * yaw], 1); fwd = np.stack([np.sin(yaw), -np.cos(yaw), 0 * yaw], 1)
sep = np.einsum('ij,ij->i', ank['L'] - ank['R'], lat); print(f'step width (ankle separation along the body lateral axis): mean {sep[walk].mean()*100:.1f} cm, min {sep[walk].min()*100:.1f}, max {sep[walk].max()*100:.1f}; hip joint separation {np.einsum("ij,ij->i", hip["L"]-hip["R"], lat).mean()*100:.1f} cm')
for s in 'LR':
    rel = ank[s] - hip[s]; f_ = np.einsum('ij,ij->i', rel, fwd); l_ = np.einsum('ij,ij->i', rel, lat)
    print(f'leg.{s}: ankle excursion relative to hip: fore-aft {f_[walk].min()*100:.1f}..{f_[walk].max()*100:.1f} cm, lateral {l_[walk].min()*100:.1f}..{l_[walk].max()*100:.1f} cm (lateral swing {np.ptp(l_[walk])*100:.1f} cm)')
    # swing plane: principal direction of the ankle path relative to the hip in the horizontal plane, vs the body forward axis
    P = np.stack([f_[walk], l_[walk]], 1); P = P - P.mean(0); d = np.linalg.svd(P)[2][0]; print(f'   swing-plane deviation from the sagittal plane: {abs(math.degrees(math.atan2(d[1], d[0]))) % 180:.1f} deg')
    z = np.array([r['foot_min_z'][s] for r in rec]) - floor_z; print(f'foot.{s}: lowest sole point above floor: min {z[walk].min()*1000:.1f} mm, max {z[walk].max()*1000:.1f} mm')
    c = np.array([r['foot_centroid'][s] for r in rec]); stance = z < z[walk].min() + 0.004; v = np.linalg.norm(np.diff(c[:, :2], axis=0), axis=1) * sc.render.fps
    print(f'   while lowest (stance, {int((stance&walk).sum())} frames): horizontal foot speed mean {v[(stance&walk)[:-1]].mean()*100:.1f} cm/s, max {v[(stance&walk)[:-1]].max()*100:.1f} cm/s')
zmin = np.minimum(*[np.array([r['foot_min_z'][s] for r in rec]) for s in 'LR']) - floor_z
print(f'lowest foot point above floor over the walk: min {zmin[walk].min()*1000:.1f} mm max {zmin[walk].max()*1000:.1f} mm (0 = contact; >0 = floating; <0 = through the floor)')
hd = A('heads', 'head'); print(f'head: vertical travel {np.ptp(hd[walk,2])*1000:.1f} mm, lateral travel {np.ptp(np.einsum("ij,ij->i", hd, lat)[walk])*1000:.1f} mm; pelvis (root) vertical travel {np.ptp(A("heads","root")[walk,2])*1000:.1f} mm')
wk = np.array([r['walker'][0] for r in rec]) if rec[0]['walker'] else None
if wk is not None: print(f'walker translation over the clip: x {np.ptp(wk[:,0])*100:.1f} cm, y {np.ptp(wk[:,1])*100:.1f} cm (forward travel), yaw {math.degrees(yaw[0]):.1f} -> {math.degrees(yaw[-1]):.1f} deg')
toe = {s: A('heads', f'toe1-1.{s}') if f'toe1-1.{s}' in names else None for s in 'LR'}
for s in 'LR':
    if toe[s] is not None:
        d = toe[s] - ank[s]; fa = np.degrees(np.arctan2(np.einsum('ij,ij->i', d, lat), np.einsum('ij,ij->i', d, fwd))); pit = np.degrees(np.arcsin(-d[:, 2] / np.linalg.norm(d, axis=1)))
        print(f'foot.{s}: toe-out angle mean {fa[walk].mean():.1f} deg; foot pitch (toe down +) {pit[walk].min():.1f}..{pit[walk].max():.1f} deg')
cl = [n for n in animated if 'clav' in n or 'shoulder' in n or 'finger' in n or 'neck' in n or 'head' in n]; print('clavicle/shoulder/finger/neck/head bones animated:', cl or 'none')
if cam:
    print('camera lens', cam.data.lens, 'location f1..', [tuple(round(x, 2) for x in cam.matrix_world.translation)])
