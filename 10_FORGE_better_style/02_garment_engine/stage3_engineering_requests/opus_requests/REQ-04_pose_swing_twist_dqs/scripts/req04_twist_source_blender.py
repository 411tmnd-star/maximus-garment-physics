"""REQ-04 step C: where does the upperarm01 twist come from?  Stage-wise swing/twist of clavicle->upperarm01 for every pose:
  stage 'anatomical' = Retargeter.apply with hand_contacts removed (Euler-composed anatomical rotations only)
  stage 'full'       = Retargeter.apply as shipped (anatomical, then 2-bone IK _align for hand contacts)
Also for poses with contacts: the swing angle of the IK _align step on upperarm01 (angle between upperarm01 direction before and after IK).
Read-only, nothing saved.  blender -b --factory-startup --python-exit-code 1 -P req04_twist_source_blender.py -- <inputs_dir> <out_json>
"""
import copy, json, math, sys
from pathlib import Path
sys.dont_write_bytecode = True
import bpy
import numpy as np
from mathutils import Matrix

a = sys.argv[sys.argv.index('--') + 1:]; INP = Path(a[0]); OUTJ = Path(a[1])
sys.path.insert(0, str(INP)); import pose_retarget as PRT  # noqa: E402
PRT.REGISTRY = INP / 'canonical_poses.json'
bpy.ops.wm.open_mainfile(filepath=str(INP / '03_full_rigged.blend'))
arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
REST_B = {b.name: np.array(b.matrix_local)[:3, :3] for b in arm.data.bones}


def Srot(n): return np.array(arm.pose.bones[n].matrix)[:3, :3] @ REST_B[n].T


def st(pa, cb):
    D = REST_B[cb].T @ Srot(pa).T @ Srot(cb) @ REST_B[cb]
    q = Matrix(D.tolist()).to_quaternion(); w, y = q.w, q.y
    if w < 0: w, y = -w, -y
    tw = 2 * math.degrees(math.atan2(y, w)); tw = (tw + 180) % 360 - 180
    return dict(total_deg=round(2 * math.degrees(math.acos(min(1, abs(w)))), 2), swing_deg=round(2 * math.degrees(math.acos(min(1, math.sqrt(w * w + y * y)))), 2), twist_deg=round(tw, 2))


def ydir(n):
    pb = arm.pose.bones[n]; d = np.array(pb.tail - pb.head); return d / np.linalg.norm(d)


def reset():
    for pb in arm.pose.bones: pb.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()


RT = PRT.Retargeter(arm); P = PRT.load_poses(); out = {}
for name in sorted(P):
    spec = P[name]; r = {}
    for s in 'LR':
        r[s] = {}
    noc = copy.deepcopy(spec); noc.pop('hand_contacts', None)
    reset(); RT.apply(noc, 1.0); bpy.context.view_layer.update()
    d_anat = {s: ydir(f'upperarm01.{s}') for s in 'LR'}
    for s in 'LR': r[s]['anatomical_only'] = st(f'clavicle.{s}', f'upperarm01.{s}')
    reset(); RT.apply(spec, 1.0); bpy.context.view_layer.update()
    for s in 'LR':
        r[s]['full'] = st(f'clavicle.{s}', f'upperarm01.{s}')
        r[s]['has_contact'] = s in (spec.get('hand_contacts') or {})
        r[s]['ik_align_swing_of_upperarm01_deg'] = round(math.degrees(math.acos(max(-1, min(1, float(d_anat[s] @ ydir(f'upperarm01.{s}')))))), 2)
    out[name] = r
OUTJ.write_text(json.dumps(out, indent=1)); print('TWIST SOURCE DONE', flush=True)
