import bpy, math, os
from mathutils import Vector
blend=os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/better_style_engine/03_governance_and_evidence/411_01_BETTER_STYLE_ACTIVE/03_DOCS_AND_EVIDENCE/evidence/mannequin/STAGE_2_IMPLEMENTATION/M01B_experimental_prototype.blend')); out=os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/better_style_engine/03_governance_and_evidence/411_01_BETTER_STYLE_ACTIVE/03_DOCS_AND_EVIDENCE/evidence/mannequin/STAGE_2_IMPLEMENTATION/VISUAL_EVIDENCE'))
bpy.ops.wm.open_mainfile(filepath=blend); s=bpy.context.scene; b=bpy.data.objects['mannequin_M01B_EXPERIMENTAL'];
for o in s.objects:o.hide_render=o not in [b]
# isolate render with temporary neutral clay override
m=bpy.data.materials.new('TEMP_M0');m.use_nodes=True;m.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value=(.497,.497,.471,1);m.node_tree.nodes.get('Principled BSDF').inputs['Roughness'].default_value=.78;b.data.materials.clear();b.data.materials.append(m)
s.render.engine='BLENDER_EEVEE';s.eevee.taa_render_samples=32;s.view_settings.view_transform='Standard';s.view_settings.exposure=0;s.view_settings.gamma=1;s.world.color=(.2,.2,.2)
# keep rigged rest pose (zero transforms). Set studio lights/camera temporary.
for name,loc,pow,size in [('key',(-3,-4,4),150,4),('fill',(3,-3,2.5),85,4),('rim',(0,2,4),110,3)]:
 ld=bpy.data.lights.new('TEMP_'+name,'AREA');ld.energy=pow;ld.shape='DISK';ld.size=size;o=bpy.data.objects.new('TEMP_'+name,ld);s.collection.objects.link(o);o.location=loc;o.rotation_euler=(Vector((0,0,1.59))-o.location).to_track_quat('-Z','Y').to_euler()
cd=bpy.data.cameras.new('TEMP_camera');c=bpy.data.objects.new('TEMP_camera',cd);s.collection.objects.link(c);s.camera=c;cd.type='ORTHO';cd.ortho_scale=.40
for name,off in [('front',(0,-1,0)),('profile',(-1,0,0)),('three_quarter',(-.707,-.707,0))]:
 tgt=Vector((0,0,1.59));c.location=tgt+Vector(off)*.6;c.rotation_euler=(tgt-c.location).to_track_quat('-Z','Y').to_euler();s.render.resolution_x=s.render.resolution_y=800;s.render.filepath=os.path.join(out,'G01_after_'+name+'.png');bpy.ops.render.render(write_still=True)
