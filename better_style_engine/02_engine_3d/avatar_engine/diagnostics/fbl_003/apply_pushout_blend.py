"""Write rest_fixed.npz garment positions into a DERIVED copy of the scene (never the original).
  blender -b --factory-startup --python-exit-code 1 -P apply_pushout_blend.py -- <orig.blend> <rest_fixed.npz> <derived.blend> [<rest_meshes.npz>]
  (or with the bpy module: python apply_pushout_blend.py -- ...). rest_meshes.npz defaults to the one next to rest_fixed.npz.
Safe because each garment has only an Armature modifier and no shape keys: mesh.vertices are the bind pose.
"""
import sys, hashlib
from pathlib import Path
import bpy, numpy as np
ARGS = [Path(x) for x in sys.argv[sys.argv.index("--") + 1:]]
ORIG, FIXED, OUT = ARGS[:3]; RESTREF = ARGS[3] if len(ARGS) > 3 else FIXED.parent / "rest_meshes.npz"
if OUT.resolve() == ORIG.resolve(): raise SystemExit("refusing to overwrite the original scene")
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
H0 = sha(ORIG)
bpy.ops.wm.open_mainfile(filepath=str(ORIG), load_ui=False)
bpy.context.scene.frame_set(1)       # extract_rest.py read matrix_world at frame 1; the scene is saved at frame 300 and the garments are parented to the animated rig
F = np.load(FIXED)
for name in ["G0_jacket", "G0_pants"]:
    o = bpy.data.objects[name]; me = o.data
    assert len(me.vertices) == len(F[f"rest_{name}"]) and not me.shape_keys and [m.type for m in o.modifiers] == ["ARMATURE"], name
    M = np.array(o.matrix_world, dtype=np.float64); Mi = np.linalg.inv(M)
    cur = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", cur); cur = cur.reshape(-1, 3)
    cur_w = cur @ M[:3, :3].T + M[:3, 3]
    # the rest extraction used world = co @ M3.T + t, so the original must reproduce rest_meshes.npz
    R0 = np.load(RESTREF)[f"rest_{name}"]
    err = np.abs(cur_w - R0).max() * 1000
    assert err < 1e-3, f"{name}: mesh data differs from extracted rest by {err} mm"
    new_local = (F[f"rest_{name}"] - M[:3, 3]) @ Mi[:3, :3].T
    me.vertices.foreach_set("co", new_local.astype(np.float32).ravel()); me.update()
    print(f"{name}: wrote {len(cur)} verts, max move {np.abs(F[f'rest_{name}'] - R0).max() * 1000:.2f} mm (baseline match {err:.5f} mm)")
OUT.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=str(OUT), copy=True)
assert sha(ORIG) == H0, "original changed"
print("saved", OUT, "| original untouched:", H0[:12])
# read-back: re-open the derived file and export what it really contains (armature muted = bind/rest pose, world space)
bpy.ops.wm.open_mainfile(filepath=str(OUT), load_ui=False)
bpy.context.scene.frame_set(1)       # same frame as the extraction (see above)
out = {}
for name in ["mannequin", "G0_jacket", "G0_pants"]:
    o = bpy.data.objects[name]
    for m in o.modifiers: m.show_viewport = m.type != "ARMATURE"
    dg = bpy.context.evaluated_depsgraph_get(); ev = o.evaluated_get(dg); me = ev.to_mesh()
    co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
    M = np.array(o.matrix_world, dtype=np.float64); out[f"rest_{name}"] = co @ M[:3, :3].T + M[:3, 3]; ev.to_mesh_clear()
np.savez(OUT.with_suffix(".rest.npz"), **out)
print("read-back rest meshes:", OUT.with_suffix(".rest.npz"), {k: v.shape for k, v in out.items()})
