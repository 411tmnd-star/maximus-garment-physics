import os
import bpy, numpy as np

bpy.ops.wm.open_mainfile(filepath=os.path.normpath(os.path.expanduser('~/Desktop/better_style_execution_workspace/08_KNOWLEDGE/stage3_engineering/mannequin_solution/evidence/dual_state_scene_copy.blend')))
bpy.context.scene.frame_set(1)

out = {}
for name in ["mannequin", "G0_jacket", "G0_pants"]:
    obj = bpy.data.objects[name]
    orig_states = []
    for m in obj.modifiers:
        orig_states.append((m, m.show_viewport))
        if m.type in {"ARMATURE", "CLOTH", "SURFACE_DEFORM", "MESH_DEFORM"}:
            m.show_viewport = False
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    M = np.array(obj.matrix_world, dtype=np.float64)
    verts = np.array([v.co for v in mesh.vertices], dtype=np.float64)
    verts_world = (verts @ M[:3, :3].T) + M[:3, 3]
    out[f"rest_{name}"] = verts_world
    print(f"Extracted rest_{name}: shape={verts_world.shape}, z=[{verts_world[:,2].min():.3f}, {verts_world[:,2].max():.3f}]")
    eval_obj.to_mesh_clear()
    for m, st in orig_states:
        m.show_viewport = st

np.savez(r"02_engine_3d\avatar_engine\diagnostics\fbl_003\rest_meshes.npz", **out)
