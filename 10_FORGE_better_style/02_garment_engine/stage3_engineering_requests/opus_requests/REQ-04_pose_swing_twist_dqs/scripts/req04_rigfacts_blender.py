"""REQ-04 step A: rig facts (read-only, works on the copied blend in inputs/). Never saves.
  blender -b --factory-startup --python-exit-code 1 -P req04_rigfacts_blender.py -- <inputs_dir> <out_json>
"""
import json, sys
from pathlib import Path
import bpy
import numpy as np

a = sys.argv[sys.argv.index('--') + 1:]
INP = Path(a[0]); OUTJ = Path(a[1])
bpy.ops.wm.open_mainfile(filepath=str(INP / '03_full_rigged.blend'))
arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
body = next(o for o in bpy.data.objects if o.type == 'MESH' and any(m.type == 'ARMATURE' for m in o.modifiers))
mod = next(m for m in body.modifiers if m.type == 'ARMATURE')
NV = len(body.data.vertices)
gname = {g.index: g.name for g in body.vertex_groups}
cnt, tot, cnt05 = {}, {}, {}
for v in body.data.vertices:
    for g in v.groups:
        if g.weight > 0:
            n = gname[g.group]; cnt[n] = cnt.get(n, 0) + 1; tot[n] = tot.get(n, 0.0) + g.weight
            if g.weight > 0.05: cnt05[n] = cnt05.get(n, 0) + 1
KEYS = ('upperarm', 'lowerarm', 'clavicle', 'shoulder', 'upperleg', 'lowerleg')
bones = []
for b in arm.data.bones:
    if any(k in b.name for k in KEYS) or b.name.startswith(('wrist', 'pelvis', 'foot', 'root', 'spine0')):
        bones.append(dict(name=b.name, parent=b.parent.name if b.parent else None, length_m=round(b.length, 5), use_deform=b.use_deform,
                          use_connect=b.use_connect, inherit_rotation=b.use_inherit_rotation, inherit_scale=b.inherit_scale,
                          children=[c.name for c in b.children],
                          head_local=[round(x, 4) for x in b.head_local], tail_local=[round(x, 4) for x in b.tail_local],
                          vgroup_exists=b.name in body.vertex_groups, weighted_vertices=cnt.get(b.name, 0),
                          weighted_vertices_gt5pct=cnt05.get(b.name, 0), total_weight=round(tot.get(b.name, 0.0), 3)))
rep = dict(blender=bpy.app.version_string, armature=arm.name, armature_matrix_world=[list(r) for r in arm.matrix_world], body=body.name,
           body_matrix_world=[list(r) for r in body.matrix_world], n_bones=len(arm.data.bones), n_vertices=NV,
           body_modifiers=[dict(name=m.name, type=m.type, show_viewport=m.show_viewport) for m in body.modifiers],
           shape_keys=(len(body.data.shape_keys.key_blocks) if body.data.shape_keys else 0),
           armature_modifier=dict(use_deform_preserve_volume=mod.use_deform_preserve_volume, use_vertex_groups=mod.use_vertex_groups,
                                  use_bone_envelopes=mod.use_bone_envelopes, use_multi_modifier=mod.use_multi_modifier),
           bones=bones, all_bone_names=[b.name for b in arm.data.bones])
OUTJ.write_text(json.dumps(rep, indent=1)); print('RIGFACTS DONE', flush=True)
