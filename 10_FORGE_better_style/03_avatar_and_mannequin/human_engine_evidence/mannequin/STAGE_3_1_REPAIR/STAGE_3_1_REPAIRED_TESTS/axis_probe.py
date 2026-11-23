import os
import bpy,math,json
from mathutils import Euler
p=os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/better_style_engine/03_governance_and_evidence/411_01_BETTER_STYLE_ACTIVE/03_DOCS_AND_EVIDENCE/evidence/mannequin/STAGE_3_1_REPAIR/stage31_repaired_working.blend'));bpy.ops.wm.open_mainfile(filepath=p);rig=bpy.data.objects['Human.rig'];sc=bpy.context.scene
for side in ('L','R'):
 print('SIDE',side)
 for ax in range(3):
  for sign in (-1,1):
   for pb in rig.pose.bones:pb.rotation_mode='XYZ';pb.rotation_euler=(0,0,0)
   bpy.context.view_layer.update();base=rig.matrix_world @ rig.pose.bones['wrist.'+side].head
   e=[0.,0.,0.];e[ax]=math.radians(22*sign);rig.pose.bones['upperarm01.'+side].rotation_euler=Euler(e,'XYZ');bpy.context.view_layer.update();pos=rig.matrix_world @ rig.pose.bones['wrist.'+side].head
   print(ax,sign,'base',tuple(round(x,4) for x in base),'new',tuple(round(x,4) for x in pos),'delta',tuple(round(x,4) for x in pos-base))
