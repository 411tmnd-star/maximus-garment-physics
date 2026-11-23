import os
import bpy
p=os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/better_style_engine/03_governance_and_evidence/411_01_BETTER_STYLE_ACTIVE/03_DOCS_AND_EVIDENCE/evidence/mannequin/STAGE_3_1_REPAIR/STAGE_3_1_EXPORT_VALIDATION/mannequin_M1_122276_walk_v1.glb'));bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.import_scene.gltf(filepath=p,merge_vertices=False);s=bpy.context.scene;r=next(o for o in s.objects if o.type=='ARMATURE');
for a in bpy.data.actions:
 r.animation_data.action=a;r.animation_data.action_slot=a.slots[0] if a.slots else None
 for f in (a.frame_range[0],sum(a.frame_range)/2,a.frame_range[1]):s.frame_set(round(f));bpy.context.view_layer.update();print(a.name,f,tuple(round(x,4) for x in r.pose.bones['upperarm01.L'].rotation_euler),tuple(round(x,4) for x in r.pose.bones['upperarm01.R'].rotation_euler))
