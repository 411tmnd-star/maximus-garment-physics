"""FBL-003 step 1 (bpy): evaluate the walk scene at REST and at frame 61, in memory, and write the arrays the analysis needs.

REST  = frame 1 object transforms, every pose bone matrix_basis = identity (exactly what capture_walk_blender.py does for rest_body).
FRAME 61 = scene.frame_set(61).
Evaluation = depsgraph-evaluated mesh in world space (same eval_world as the capture).
The scene file is never saved; its sha256 is checked before and after. Output: states.npz in this folder only.
Run:  python -I -B eval_states.py
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
import bpy
from mathutils import Matrix

HERE = Path(__file__).resolve().parent
AE = HERE.parents[1]
WS = AE.parents[1]
SCENE = WS / "08_KNOWLEDGE/stage3_engineering/mannequin_solution/evidence/dual_state_scene_copy.blend"
CAP = AE / "baseline/capture_current_walk.npz"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
H0 = sha(SCENE)
assert H0.startswith("ae24fe62"), H0

bpy.ops.wm.open_mainfile(filepath=str(SCENE), load_ui=False); sc = bpy.context.scene
rig = next(o for o in bpy.data.objects if o.type == "ARMATURE")
body = next(o for o in bpy.data.objects if o.type == "MESH" and any(m.type == "ARMATURE" for m in o.modifiers) and "foot.L" in o.vertex_groups and not o.name.startswith(("G0", "Float")))
garments = sorted([o for o in bpy.data.objects if o.type == "MESH" and o.name.startswith("G0") and any(m.type == "ARMATURE" for m in o.modifiers)], key=lambda o: o.name)
assert [g.name for g in garments] == ["G0_jacket", "G0_pants"], [g.name for g in garments]
for o in [body] + garments:
    assert len(o.vertex_groups) > 0, f"{o.name} has no vertex groups"          # packet stop condition

def eval_world(obj):
    dg = bpy.context.evaluated_depsgraph_get(); ev = obj.evaluated_get(dg); me = ev.to_mesh(); a = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", a)
    ed = np.empty(len(me.edges) * 2, np.int64); me.edges.foreach_get("vertices", ed); ev.to_mesh_clear()
    M = np.array(ev.matrix_world); X = a.reshape(-1, 3); return X @ M[:3, :3].T + M[:3, 3], ed.reshape(-1, 2)

def dominant_groups(obj):
    names = [g.name for g in obj.vertex_groups]; out = []
    for v in obj.data.vertices:
        out.append(max(((g.weight, g.group) for g in v.groups), default=(0.0, -1))[1])
    return np.array(out, np.int64), np.array(names)

arrays = {}
arrays["body_bind_co"] = np.array([v.co[:] for v in body.data.vertices])                    # bind-pose coordinates, as the capture's _co
body.data.calc_loop_triangles(); arrays["body_tri"] = np.array([lt.vertices[:] for lt in body.data.loop_triangles], np.int64)
arrays["body_group_idx"], arrays["body_group_names"] = dominant_groups(body)
for g in garments:
    arrays[f"{g.name}_group_idx"], arrays[f"{g.name}_group_names"] = dominant_groups(g)
arrays["armature_preserve_volume"] = np.array(json.dumps({o.name: [m.use_deform_preserve_volume for m in o.modifiers if m.type == "ARMATURE"] for o in [body] + garments}))
arrays["modifiers"] = np.array(json.dumps({o.name: [(m.name, m.type) for m in o.modifiers] for o in [body] + garments}))

sc.frame_set(61)
Xb, _ = eval_world(body); arrays["f061_body"] = Xb
for g in garments:
    X, E = eval_world(g); arrays[f"f061_{g.name}"] = X; arrays[f"{g.name}_edges"] = E

sc.frame_set(1)
for pb in rig.pose.bones: pb.matrix_basis = Matrix.Identity(4)
bpy.context.view_layer.update()
Xb, _ = eval_world(body); arrays["rest_body"] = Xb
for g in garments: arrays[f"rest_{g.name}"] = eval_world(g)[0]

# cross-check against the capture, which was made by the same evaluation
Z = np.load(CAP); chk = {}
chk["f061_body_max_abs_diff_m"] = float(np.abs(arrays["f061_body"].astype(np.float32).astype(float) - Z["body"][60].astype(float)).max())
chk["rest_body_max_abs_diff_m"] = float(np.abs(arrays["rest_body"] - Z["rest_body"]).max())
for i, g in enumerate(garments):
    chk[f"f061_{g.name}_max_abs_diff_m"] = float(np.abs(arrays[f"f061_{g.name}"].astype(np.float32).astype(float) - Z[f"garment__{g.name}"][0].astype(float)).max())
    assert len(arrays[f"f061_{g.name}"]) == len(g.data.vertices)
assert np.array_equal(arrays["body_tri"], Z["tri"])
arrays["capture_crosscheck"] = np.array(json.dumps(chk))
H1 = sha(SCENE); assert H1 == H0, "scene file changed"
arrays["scene_sha256_before_after"] = np.array(json.dumps([H0, H1]))
np.savez(HERE / "states.npz", **arrays)
print(json.dumps(chk, indent=1)); print("scene sha256 unchanged:", H0)
