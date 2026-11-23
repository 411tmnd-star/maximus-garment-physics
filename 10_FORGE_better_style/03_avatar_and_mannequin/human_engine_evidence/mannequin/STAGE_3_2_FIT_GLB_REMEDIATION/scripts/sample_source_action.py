import os
import bpy
p=os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/better_style_engine/03_governance_and_evidence/411_01_BETTER_STYLE_ACTIVE/03_DOCS_AND_EVIDENCE/evidence/mannequin/STAGE_3_1_REPAIR/STAGE_3_1_EXPORT_VALIDATION/stage31_final_scene.blend'));bpy.ops.wm.open_mainfile(filepath=p);s=bpy.context.scene;r=bpy.data.objects['Human.rig'];
for tr in r.animation_data.nla_tracks:tr.mute=True
for a in bpy.data.actions:
 r.animation_data.action=a;r.animation_data.action_slot=a.slots[0];
 for f in (a.frame_range[0],sum(a.frame_range)/2,a.frame_range[1]):s.frame_set(round(f));bpy.context.view_layer.update();print(a.name,f,tuple(round(x,4) for x in r.pose.bones['upperarm01.L'].rotation_euler),tuple(round(x,4) for x in r.pose.bones['upperarm01.R'].rotation_euler))
