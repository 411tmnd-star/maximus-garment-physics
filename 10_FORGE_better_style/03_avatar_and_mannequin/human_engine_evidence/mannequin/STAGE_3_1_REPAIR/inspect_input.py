import os
import bpy
bpy.ops.import_scene.gltf(filepath=os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/better_style_engine/03_governance_and_evidence/411_01_BETTER_STYLE_ACTIVE/05_3D_ASSETS/glb/mannequin_full_rigged_m01b_v1.glb')))
for o in bpy.context.scene.objects:
 print('OBJECT',o.name,o.type, len(o.data.vertices) if o.type=='MESH' else '')
 if o.type=='ARMATURE':
  print('BONES',len(o.data.bones),'CUSTOM_SHAPES',[(p.name,p.custom_shape.name if p.custom_shape else None) for p in o.pose.bones if p.custom_shape])
 if o.type=='MESH':print('MESH',len(o.data.vertices),len(o.data.edges),len(o.data.polygons),'custom normals',o.data.has_custom_normals,'groups',len(o.vertex_groups),'keys',len(o.data.shape_keys.key_blocks) if o.data.shape_keys else 0)
