"""FBL-005 step 1 (bpy): evaluate body and both garments at frames 61 and 76 under three preserve-volume variants, in memory only.

  A  as built : body off, garments on          (checked against the capture)
  B  all off  : body off, garments off
  C  all on   : body on,  garments on
Only Armature modifier .use_deform_preserve_volume is changed, in memory. The scene file is never saved; its sha256 is checked before and after.
Output: evals.npz in this folder only.
Run:  python -I -B eval_variants.py
"""
import hashlib, json
from pathlib import Path
import numpy as np
import bpy

HERE = Path(__file__).resolve().parent
AE = HERE.parents[1]
WS = AE.parents[1]
SCENE = WS / "stage3_engineering/mannequin_solution/evidence/dual_state_scene_copy.blend"
CAP = AE / "baseline/capture_current_walk.npz"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
H0 = sha(SCENE); assert H0.startswith("ae24fe62"), H0

bpy.ops.wm.open_mainfile(filepath=str(SCENE), load_ui=False); sc = bpy.context.scene
body = next(o for o in bpy.data.objects if o.type == "MESH" and any(m.type == "ARMATURE" for m in o.modifiers) and "foot.L" in o.vertex_groups and not o.name.startswith(("G0", "Float")))
garments = sorted([o for o in bpy.data.objects if o.type == "MESH" and o.name.startswith("G0") and any(m.type == "ARMATURE" for m in o.modifiers)], key=lambda o: o.name)
assert [g.name for g in garments] == ["G0_jacket", "G0_pants"]

def flags(o): return [m.use_deform_preserve_volume for m in o.modifiers if m.type == "ARMATURE"]
def setflag(o, v):
    for m in o.modifiers:
        if m.type == "ARMATURE": m.use_deform_preserve_volume = v

def eval_world(obj):
    dg = bpy.context.evaluated_depsgraph_get(); ev = obj.evaluated_get(dg); me = ev.to_mesh(); a = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", a); ev.to_mesh_clear()
    M = np.array(ev.matrix_world); X = a.reshape(-1, 3); return X @ M[:3, :3].T + M[:3, 3]

as_built = {o.name: flags(o) for o in [body] + garments}
assert as_built == {"mannequin": [False], "G0_jacket": [True], "G0_pants": [True]}, as_built      # the capture metadata
VARIANTS = {"A": (False, True), "B": (False, False), "C": (True, True)}                           # (body, garments)
arrays = {}; log = {}
for fr in (61, 76):
    sc.frame_set(fr)
    for name, (vb, vg) in VARIANTS.items():
        setflag(body, vb)
        for g in garments: setflag(g, vg)
        bpy.context.view_layer.update()
        assert (flags(body), [flags(g) for g in garments]) == ([vb], [[vg], [vg]])
        arrays[f"{name}_{fr}_body"] = eval_world(body)
        for g in garments: arrays[f"{name}_{fr}_{g.name}"] = eval_world(g)
Z = np.load(CAP); chk = {}
for fr, ci in ((61, 0), (76, 2)):
    k = int(np.nonzero(Z["frames"] == fr)[0][0]); assert int(json.loads(str(Z["meta"]))["clearance_frames"][ci]) == fr
    chk[f"A_{fr}_body_max_abs_diff_m"] = float(np.abs(arrays[f"A_{fr}_body"].astype(np.float32).astype(float) - Z["body"][k].astype(float)).max())
    for g in garments:
        chk[f"A_{fr}_{g.name}_max_abs_diff_m"] = float(np.abs(arrays[f"A_{fr}_{g.name}"].astype(np.float32).astype(float) - Z[f"garment__{g.name}"][ci].astype(float)).max())
arrays["capture_crosscheck"] = np.array(json.dumps(chk)); arrays["as_built_flags"] = np.array(json.dumps(as_built))
H1 = sha(SCENE); assert H1 == H0, "scene file changed"
arrays["scene_sha256_before_after"] = np.array(json.dumps([H0, H1]))
np.savez(HERE / "evals.npz", **arrays)
print(json.dumps(chk, indent=1)); print("scene sha256 unchanged:", H0)
