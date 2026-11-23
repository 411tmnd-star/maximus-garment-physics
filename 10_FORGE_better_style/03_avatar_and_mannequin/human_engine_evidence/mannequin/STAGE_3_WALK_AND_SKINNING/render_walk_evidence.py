import bpy,os,math
from mathutils import Vector
root=os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/better_style_engine/03_governance_and_evidence/411_01_BETTER_STYLE_ACTIVE/03_DOCS_AND_EVIDENCE/evidence/mannequin/STAGE_3_WALK_AND_SKINNING'));blend=os.path.join(root,'stage3c_walk_working.blend');out=os.path.join(root,'walk_evidence');os.makedirs(out,exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=blend);s=bpy.context.scene;body=bpy.data.objects['mannequin'];gar=bpy.data.objects['garment_torso'];rig=next(o for o in s.objects if o.type=='ARMATURE')
# Isolate the walk loop for diagnostic renders; the saved deliverable retains all three actions.
for tr in rig.animation_data.nla_tracks: tr.mute=True
rig.animation_data.action=bpy.data.actions['walk_editorial_loop']
for o in s.objects:o.hide_render=o not in (body,gar)
# use temporary neutral gray material for stable body/garment readability
for obj in (body,gar):
 m=bpy.data.materials.new('TEMP_walk_'+obj.name);m.use_nodes=True;bs=m.node_tree.nodes.get('Principled BSDF');bs.inputs['Base Color'].default_value=(.18,.18,.17,1);bs.inputs['Roughness'].default_value=.65;obj.data.materials.clear();obj.data.materials.append(m)
s.render.engine='BLENDER_EEVEE';s.eevee.taa_render_samples=8;s.render.resolution_x=480;s.render.resolution_y=640;s.render.image_settings.file_format='PNG';s.view_settings.view_transform='Standard';s.view_settings.exposure=0;s.world.color=(.2,.2,.2)
for nm,loc,pow in [('Key',(-3,-4,4),250),('Fill',(3,-3,2.5),120),('Rim',(0,2,4),160)]:
 ld=bpy.data.lights.new('TEMP_'+nm,'AREA');ld.energy=pow;ld.size=4;o=bpy.data.objects.new('TEMP_'+nm,ld);s.collection.objects.link(o);o.location=loc;o.rotation_euler=(Vector((0,0,.85))-o.location).to_track_quat('-Z','Y').to_euler()
cd=bpy.data.cameras.new('TEMP_walkcam');cam=bpy.data.objects.new('TEMP_walkcam',cd);s.collection.objects.link(cam);s.camera=cam;cd.type='ORTHO';cd.ortho_scale=1.95
views={'front':(0,-3,0),'profile':(-3,0,0),'three_quarter':(-2.15,-2.15,0)};frames=[0,8,15,23,30,38,45,53]
for vn,off in views.items():
 cam.location=Vector((0,0,.86))+Vector(off);cam.rotation_euler=(Vector((0,0,.86))-cam.location).to_track_quat('-Z','Y').to_euler()
 for f in frames:
  s.frame_set(f);s.render.filepath=os.path.join(out,f'{vn}_{f:02d}.png');bpy.ops.render.render(write_still=True)

