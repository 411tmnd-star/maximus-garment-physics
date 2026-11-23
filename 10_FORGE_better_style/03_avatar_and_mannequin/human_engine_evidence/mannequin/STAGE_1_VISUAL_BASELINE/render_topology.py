import os
import bpy
from mathutils import Vector
src=os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/better_style_engine/03_governance_and_evidence/411_01_BETTER_STYLE_ACTIVE/05_3D_ASSETS/glb/mannequin_full_rigged_122276_morphed.glb'))
out=os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/better_style_engine/03_governance_and_evidence/411_01_BETTER_STYLE_ACTIVE/03_DOCS_AND_EVIDENCE/evidence/mannequin/STAGE_1_VISUAL_BASELINE/VISUAL_EVIDENCE/topology_front.png'))
bpy.ops.import_scene.gltf(filepath=src); s=bpy.context.scene
b=next(o for o in s.objects if o.type=='MESH' and o.name=='mannequin'); gar=[o for o in s.objects if o.type=='MESH' and o!=b]
for o in s.objects:o.hide_render=o not in [b]
b.data.materials.clear(); m=bpy.data.materials.new('TEMP_wire_dark');m.diffuse_color=(.02,.02,.02,1);m.use_nodes=True;m.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value=(.018,.018,.018,1);m.node_tree.nodes.get('Principled BSDF').inputs['Roughness'].default_value=1;b.data.materials.append(m)
wf=b.modifiers.new('TEMP_topology_wireframe','WIREFRAME');wf.thickness=.0012;wf.use_even_offset=True;wf.offset=0
floor_z=0;bpy.ops.mesh.primitive_plane_add(size=200,location=(0,0,floor_z-.03));f=bpy.context.object;f.hide_render=False;fm=bpy.data.materials.new('TEMP_floor');fm.use_nodes=True;fm.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value=(.12,.12,.12,1);fm.node_tree.nodes.get('Principled BSDF').inputs['Roughness'].default_value=1;f.data.materials.append(fm)
d=bpy.data.cameras.new('TEMP_camera');c=bpy.data.objects.new('TEMP_camera',d);s.collection.objects.link(c);s.camera=c; target=Vector((0,0,.8645)); c.location=(0,-2.85,.8645);c.rotation_euler=(target-c.location).to_track_quat('-Z','Y').to_euler();d.type='ORTHO';d.ortho_scale=1.936
ld=bpy.data.lights.new('TEMP_light','AREA');ld.energy=250;ld.shape='DISK';ld.size=4;lo=bpy.data.objects.new('TEMP_light',ld);s.collection.objects.link(lo);lo.location=(-3,-4,4);lo.rotation_euler=(target-lo.location).to_track_quat('-Z','Y').to_euler()
s.render.engine='BLENDER_EEVEE';s.eevee.taa_render_samples=16;s.render.resolution_x=720;s.render.resolution_y=960;s.render.resolution_percentage=100;s.view_settings.view_transform='Standard';s.view_settings.exposure=0;s.render.filepath=out;s.render.image_settings.file_format='PNG';bpy.ops.render.render(write_still=True)
