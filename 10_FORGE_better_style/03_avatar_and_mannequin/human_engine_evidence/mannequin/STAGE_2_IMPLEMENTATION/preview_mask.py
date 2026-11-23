import os
import bpy,math
from mathutils import Vector
src=os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/better_style_engine/03_governance_and_evidence/411_01_BETTER_STYLE_ACTIVE/05_3D_ASSETS/glb/mannequin_full_rigged_122276_morphed.glb')); out=os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/better_style_engine/03_governance_and_evidence/411_01_BETTER_STYLE_ACTIVE/03_DOCS_AND_EVIDENCE/evidence/mannequin/STAGE_2_IMPLEMENTATION/VISUAL_EVIDENCE'))
bpy.ops.import_scene.gltf(filepath=src); s=bpy.context.scene; b=next(o for o in s.objects if o.type=='MESH' and o.name=='mannequin')
# hide all other imported objects
for o in s.objects:o.hide_render=o!=b
# Red material on candidate vertices, gray on remainder; material assignment only temporary in memory.
red=bpy.data.materials.new('TEMP_candidate_mask'); red.diffuse_color=(.8,.025,.015,1); red.use_nodes=True; red.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value=(.8,.025,.015,1); red.node_tree.nodes.get('Principled BSDF').inputs['Roughness'].default_value=1
gray=bpy.data.materials.new('TEMP_unmasked'); gray.use_nodes=True; gray.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value=(.48,.48,.45,1); gray.node_tree.nodes.get('Principled BSDF').inputs['Roughness'].default_value=.85
b.data.materials.clear();b.data.materials.append(gray);b.data.materials.append(red)
idx={g.name:g.index for g in b.vertex_groups}; head=idx['head']; jaw=idx['jaw']
weights=[{} for v in b.data.vertices]
for v in b.data.vertices:
 for a in v.groups:
  nm=b.vertex_groups[a.group].name; weights[v.index][nm]=a.weight
# explicit candidate weights: anterior face and ear transition zones, tapered before neck and at lateral edge
mask=[]
def sm(a,b,x):
 t=max(0,min(1,(x-a)/(b-a)));return t*t*(3-2*t)
for i,v in enumerate(b.data.vertices):
 x,y,z=v.co; w=weights[i]; h=max(w.get('head',0),w.get('jaw',0)); zf=sm(1.475,1.515,z)*(1-sm(1.675,1.705,z)); xf=1-sm(.068,.085,abs(x));
 front=1-sm(-.105,-.045,y) # 0 behind -0.045, full anterior once y<=-.105
 face=zf*xf*front
 ear=0
 if 1.545<z<1.655 and abs(x)>.066 and y<.025 and y>-.11:
  ear=max(0,min(1,h*2))*sm(.062,.078,abs(x))*(1-sm(.084,.091,abs(x)))*sm(1.545,1.57,z)*(1-sm(1.635,1.66,z))
 mask.append(max(face,ear))
# evaluate mask coverage
def assign_weighted_faces():
 for p in b.data.polygons:
  avg=sum(mask[i] for i in p.vertices)/len(p.vertices); p.material_index=1 if avg>=.28 else 0
assign_weighted_faces()
# render setup
s.render.engine='BLENDER_EEVEE';s.eevee.taa_render_samples=16;s.render.resolution_x=720;s.render.resolution_y=720;s.render.resolution_percentage=100;s.view_settings.view_transform='Standard';s.view_settings.exposure=0;s.view_settings.gamma=1
s.world.color=(.2,.2,.2)
def light(n,loc,power,size):
 ld=bpy.data.lights.new(n,'AREA');ld.energy=power;ld.shape='DISK';ld.size=size;o=bpy.data.objects.new(n,ld);s.collection.objects.link(o);o.location=loc;o.rotation_euler=(Vector((0,0,1.6))-o.location).to_track_quat('-Z','Y').to_euler()
light('TEMP_key',(-3,-4,4),150,4);light('TEMP_fill',(3,-3,2.5),85,4);light('TEMP_rim',(0,2,4),110,3)
cd=bpy.data.cameras.new('TEMP_cam');c=bpy.data.objects.new('TEMP_cam',cd);s.collection.objects.link(c);s.camera=c;cd.type='ORTHO';cd.ortho_scale=.42
for name,off in [('front',(0,-1,0)),('profile',(-1,0,0)),('three_quarter',(-.707,-.707,0))]:
 tgt=Vector((0,0,1.59));c.location=tgt+Vector(off)*.65;c.rotation_euler=(tgt-c.location).to_track_quat('-Z','Y').to_euler();s.render.filepath=out+f'\\G01_candidate_mask_{name}.png';bpy.ops.render.render(write_still=True)
print('mask_vertices_gt_0.5',sum(x>.5 for x in mask),'mask_vertices_gt_0.1',sum(x>.1 for x in mask),'mask_max',max(mask))
