import bpy,json,os
p=os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/better_style_engine/03_governance_and_evidence/411_01_BETTER_STYLE_ACTIVE/03_DOCS_AND_EVIDENCE/evidence/mannequin/STAGE_3_1_REPAIR/STAGE_3_1_EXPORT_VALIDATION/mannequin_M1_122276_walk_v1.glb'));out=os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/better_style_engine/03_governance_and_evidence/411_01_BETTER_STYLE_ACTIVE/03_DOCS_AND_EVIDENCE/evidence/mannequin/STAGE_3_1_REPAIR/STAGE_3_1_EXPORT_VALIDATION'))
bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.import_scene.gltf(filepath=p,merge_vertices=True)
sc=bpy.context.scene;meshes=sorted((o.name,len(o.data.vertices)) for o in sc.objects if o.type=='MESH');rig=next((o for o in sc.objects if o.type=='ARMATURE'),None);acts=sorted(a.name for a in bpy.data.actions);play={}
if rig and rig.animation_data:
 for a in bpy.data.actions:
  rig.animation_data.action=a;sc.frame_set(int((a.frame_range[0]+a.frame_range[1])/2));bpy.context.view_layer.update();play[a.name]={'frame_range':list(a.frame_range),'evaluated':bool(rig.pose.bones),'sample_bone_head':list(rig.pose.bones['head'].matrix.translation)}
r={'meshes':meshes,'bones':len(rig.data.bones) if rig else 0,'actions':acts,'playback_samples':play}
json.dump(r,open(os.path.join(out,'blender_reimport_inventory.json'),'w'),indent=2);print(json.dumps(r))
