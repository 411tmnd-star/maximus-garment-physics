"""capture_walk_blender.py  Paper 01 step 1: capture everything the gates need from a walk scene, in one pass. Read-only on the scene file.

usage:
  blender -b --factory-startup --python-exit-code 1 -P capture_walk_blender.py -- <scene.blend> <out.npz> [--clearance-frames 61,69,...]

What is recorded (all lengths metres, Blender Z-up):
  per frame   world matrices of the Walker empty and the rig object; armature-space matrix of every pose bone; world heads of every bone;
              the evaluated body (mannequin) vertices in world space; the number of self-intersecting body triangle pairs (P2/REQ-04 test:
              mathutils BVHTree.overlap on the welded body, pairs sharing no vertex), split head / armpit band / other
  rest        the same body quantities with every pose bone at identity (the scene's bind pose), as the interim reference until theta_fit exists
  clearance   evaluated garment (G0_*) vertices in world space at the listed frames only (winding number is O(points x triangles))
  static      bone names, parents, rest matrices (matrix_local), body triangles, foot vertex sets (same rule as evidence/measure_bs_walk.py),
              animated bone names, constraint count, Armature-modifier preserve-volume flags, floor height (Baseplate top), fps, frame range
Nothing is written to the .blend. Its sha256 is recorded before and after; the run fails if they differ."""
import sys, json, hashlib, time
from pathlib import Path
import numpy as np
import bpy
from mathutils import Matrix
from mathutils.bvhtree import BVHTree

argv = sys.argv[sys.argv.index("--") + 1:]
SCENE, OUT = Path(argv[0]), Path(argv[1])
CLEAR_FRAMES = [int(x) for x in argv[argv.index("--clearance-frames") + 1].split(",")] if "--clearance-frames" in argv else []
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
H0 = sha(SCENE); T0 = time.time()

bpy.ops.wm.open_mainfile(filepath=str(SCENE), load_ui=False); sc = bpy.context.scene
rig = next(o for o in bpy.data.objects if o.type == "ARMATURE")
body = next(o for o in bpy.data.objects if o.type == "MESH" and any(m.type == "ARMATURE" for m in o.modifiers) and "foot.L" in o.vertex_groups and not o.name.startswith(("G0", "Float")))
garments = [o for o in bpy.data.objects if o.type == "MESH" and o.name.startswith("G0") and any(m.type == "ARMATURE" for m in o.modifiers)]
walker = bpy.data.objects.get("Walker")
pbs = rig.pose.bones; bones = rig.data.bones; NAMES = [b.name for b in bones]
PARENT = np.array([NAMES.index(b.parent.name) if b.parent else -1 for b in bones])
REST_LOCAL = np.stack([np.array(b.matrix_local) for b in bones])

# animated bones (Blender 4.4+ layered actions, with the legacy path as fallback)
animated = set()
act = rig.animation_data.action if rig.animation_data else None
if act is not None:
    try:
        for layer in act.layers:
            for strip in layer.strips:
                for cb in strip.channelbags:
                    for fc in cb.fcurves:
                        if '"' in fc.data_path: animated.add(fc.data_path.split('"')[1])
    except AttributeError:
        for fc in act.fcurves:
            if '"' in fc.data_path: animated.add(fc.data_path.split('"')[1])
constraints = sum(len(pb.constraints) for pb in pbs)
preserve = {o.name: [m.use_deform_preserve_volume for m in o.modifiers if m.type == "ARMATURE"] for o in [body] + garments}

# foot vertex sets: identical rule to 08_KNOWLEDGE/stage3_engineering/mannequin_solution/evidence/measure_bs_walk.py
vg = {g.name: g.index for g in body.vertex_groups}
def group_idx(prefixes):
    ids = {vg[n] for n in vg if n.startswith(prefixes)}
    return [v.index for v in body.data.vertices if any(g.group in ids and g.weight > 0.5 for g in v.groups)]
footv = {}
for s in "LR":
    cand = group_idx((f"foot.{s}", "toe"))
    footv[s] = np.array([i for i in cand if any(body.vertex_groups[g.group].name.endswith("." + s) for g in body.data.vertices[i].groups if g.weight > 0.5)], np.int64)

body.data.calc_loop_triangles(); TRI = np.array([lt.vertices[:] for lt in body.data.loop_triangles], np.int64); 
# self-intersection runs on the position-welded body, as P2 did (split vertices at seams would otherwise count neighbouring triangles as intersecting)
_co = np.array([v.co[:] for v in body.data.vertices]); _, WELD = np.unique(np.round(_co / 1e-6).astype(np.int64), axis=0, return_inverse=True); WELD = WELD.ravel()
TRI_SETS = [set(t) for t in WELD[TRI].tolist()]
floor_z = max([o.matrix_world.translation.z + o.dimensions.z for o in bpy.data.objects if o.name == "Baseplate"] + [0.0])

def eval_world(obj):
    dg = bpy.context.evaluated_depsgraph_get(); ev = obj.evaluated_get(dg); me = ev.to_mesh(); a = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", a); ev.to_mesh_clear()
    M = np.array(ev.matrix_world); X = a.reshape(-1, 3); return X @ M[:3, :3].T + M[:3, 3]
# region of a pair, from the bind-pose z of its two triangle centroids (pose-independent): 0 head (z > 1.45 m), 1 armpit band (1.25 to 1.40 m, P2's band), 2 other
_cz = _co[TRI].mean(1)[:, 2]
def self_pairs(X):
    t = BVHTree.FromPolygons([tuple(p) for p in X.tolist()], TRI.tolist(), all_triangles=True, epsilon=0.0); c = np.zeros(3, np.int64)
    for i, j in t.overlap(t):
        if i < j and not (TRI_SETS[i] & TRI_SETS[j]):
            z = (_cz[i] + _cz[j]) / 2; c[0 if z > 1.45 else (1 if 1.25 < z < 1.40 else 2)] += 1
    return c
def bone_state():
    return np.stack([np.array(pb.matrix) for pb in pbs]), np.stack([np.array(rig.matrix_world @ pb.head) for pb in pbs])

frames = list(range(sc.frame_start, sc.frame_end + 1)); F = len(frames)
POSE = np.empty((F, len(pbs), 4, 4)); HEADW = np.empty((F, len(pbs), 3)); BODY = np.empty((F, len(body.data.vertices), 3), np.float32)
SELF = np.empty((F, 3), np.int64); WALK = np.empty((F, 4, 4)); RIGW = np.empty((F, 4, 4)); GARM = {g.name: [] for g in garments}
for k, f in enumerate(frames):
    sc.frame_set(f)
    POSE[k], HEADW[k] = bone_state(); X = eval_world(body); BODY[k] = X; SELF[k] = self_pairs(X)
    WALK[k] = np.array(walker.matrix_world) if walker else np.eye(4); RIGW[k] = np.array(rig.matrix_world)
    if f in CLEAR_FRAMES:
        for g in garments: GARM[g.name].append(eval_world(g).astype(np.float32))
    if k % 50 == 0: print(f"frame {f}  self-pairs head/armpit/other {SELF[k].tolist()}  {time.time() - T0:.0f}s", flush=True)

# rest: every pose bone at identity, at frame 1's object transforms
sc.frame_set(frames[0])
for pb in pbs: pb.matrix_basis = Matrix.Identity(4)
bpy.context.view_layer.update(); REST_POSE, REST_HEADW = bone_state(); REST_BODY = eval_world(body); REST_SELF = self_pairs(REST_BODY)

H1 = sha(SCENE)
if H1 != H0: raise SystemExit("scene file changed during capture")
meta = dict(script="capture_walk_blender.py", blender=bpy.app.version_string, scene=str(SCENE), scene_sha256=H0, rig=rig.name, body=body.name, body_vertices=len(body.data.vertices),
            body_triangles=int(len(TRI)), garments={g.name: len(g.data.vertices) for g in garments}, fps=sc.render.fps, frame_start=sc.frame_start, frame_end=sc.frame_end,
            floor_z=floor_z, animated_bones=sorted(animated), constraints=constraints, armature_preserve_volume=preserve, clearance_frames=CLEAR_FRAMES,
            body_unique_positions=int(WELD.max() + 1), walker=bool(walker), seconds=round(time.time() - T0, 1))
np.savez_compressed(OUT, names=np.array(NAMES), parent=PARENT, rest_local=REST_LOCAL, frames=np.array(frames), pose=POSE, headw=HEADW, body=BODY, self_pairs=SELF,
                    walker=WALK, rigw=RIGW, rest_pose=REST_POSE, rest_headw=REST_HEADW, rest_body=REST_BODY, rest_self_pairs=REST_SELF, tri=TRI,
                    foot_L=footv["L"], foot_R=footv["R"], meta=json.dumps(meta),
                    **{f"garment__{n}": np.stack(v) for n, v in GARM.items() if v})
print("WROTE", OUT, json.dumps(meta)[:600], flush=True)
