import bpy, os, json, math
import numpy as np
from mathutils import Vector, Euler
from mathutils.bvhtree import BVHTree
ROOT=os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/better_style_engine/03_governance_and_evidence/411_01_BETTER_STYLE_ACTIVE/03_DOCS_AND_EVIDENCE/evidence/mannequin/STAGE_3_1_REPAIR'))
BLEND=os.path.join(ROOT,'stage31_repaired_working.blend'); OUT=os.path.join(ROOT,'STAGE_3_1_EXPORT_VALIDATION'); EV=os.path.join(ROOT,'STAGE_3_1_VISUAL_EVIDENCE')
bpy.ops.wm.open_mainfile(filepath=BLEND); sc=bpy.context.scene; rig=bpy.data.objects['Human.rig']; body=bpy.data.objects['mannequin']; gar=bpy.data.objects['garment_torso']
def reset():
 rig.location=(0,0,0)
 for pb in rig.pose.bones: pb.rotation_mode='XYZ';pb.rotation_euler=(0,0,0);pb.location=(0,0,0);pb.scale=(1,1,1)
 for k in body.data.shape_keys.key_blocks:k.value=0
 bpy.context.view_layer.update()
def pose(deg=31.5,clav=2.875):
 reset()
 rig.pose.bones['upperarm01.L'].rotation_euler=Euler((0,math.radians(-deg),0),'XYZ');rig.pose.bones['upperarm01.R'].rotation_euler=Euler((0,math.radians(deg),0),'XYZ')
 rig.pose.bones['clavicle.L'].rotation_euler=Euler((0,math.radians(clav),0),'XYZ');rig.pose.bones['clavicle.R'].rotation_euler=Euler((0,math.radians(-clav),0),'XYZ')
 bpy.context.view_layer.update()
def evalmesh(o):
 bpy.context.view_layer.update(); e=o.evaluated_get(bpy.context.evaluated_depsgraph_get());m=e.to_mesh(); vs=[e.matrix_world@v.co for v in m.vertices]; ns=[(e.matrix_world.to_3x3()@v.normal).normalized() for v in m.vertices]; ps=[tuple(p.vertices) for p in m.polygons]; es=[tuple(x.vertices) for x in m.edges];e.to_mesh_clear();return vs,ns,ps,es
def selfover(v,p):
 t=BVHTree.FromPolygons(v,p,all_triangles=False); raw=t.overlap(t); ss=[set(x) for x in p];return {(a,b) for a,b in raw if a<b and ss[a].isdisjoint(ss[b])}
# Candidate wrist positions and posed fit proxy
pose(); wrists={s:list(rig.matrix_world@rig.pose.bones['wrist.'+s].head) for s in ('L','R')}
bv,bn,bp,be=evalmesh(body);gv,gn,gp,ge=evalmesh(gar);bt=BVHTree.FromPolygons(bv,bp,all_triangles=False);pen=[];clear=[]
for v in gv:
 hit=bt.find_nearest(v)
 if hit and hit[0] is not None:
  sd=(v-hit[0]).dot(hit[1]);pen.append(max(0,-sd)*1000);clear.append(sd*1000)
pairs=selfover(gv,gp);cross=BVHTree.FromPolygons(gv,gp,all_triangles=False).overlap(bt)
# natural pose & mesh metrics recorded, not equated with anatomy approval
fit={'wrist_world_m':wrists,'max_abs_wrist_x_m':max(abs(wrists['L'][0]),abs(wrists['R'][0])),'wrist_symmetry_delta_m':float(np.linalg.norm(np.array(wrists['L'])*[-1,1,1]-np.array(wrists['R']))),'garment_clearance_min_signed_mm':float(min(clear)) if clear else None,'p99_penetration_mm':float(np.percentile(pen,99)) if pen else None,'max_penetration_mm':float(max(pen)) if pen else None,'self_overlap_pairs':len(pairs),'body_garment_intersecting_triangle_pairs':len(cross),'mannequin_vertices':len(body.data.vertices),'garment_vertices':len(gar.data.vertices),'garment_groups':len(gar.vertex_groups)}
# create stable simple three-action animation in separate working scene
for a in list(bpy.data.actions):bpy.data.actions.remove(a)
rig.animation_data_clear();ad=rig.animation_data_create();sc.render.fps=30
frames_walk=[1,16,31,46,61]
def make_action(name,frames,fn):
 act=bpy.data.actions.new(name);rig.animation_data.action=act
 for f in frames:
  fn(f);sc.frame_set(f);bpy.context.view_layer.update()
  for pb in rig.pose.bones:pb.keyframe_insert(data_path='rotation_euler',frame=f,group=pb.name);pb.keyframe_insert(data_path='location',frame=f,group=pb.name)
 tr=ad.nla_tracks.new();tr.name=name;strip=tr.strips.new(name,frames[0],act);strip.blend_type='REPLACE';return act,tr

def neutral(f):pose()
def walk(f):
 t=(f-1)/60*2*math.pi;pose();rig.pose.bones['upperleg01.L'].rotation_euler.x=math.radians(12*math.sin(t));rig.pose.bones['upperleg01.R'].rotation_euler.x=math.radians(-12*math.sin(t));rig.pose.bones['lowerleg01.L'].rotation_euler.x=math.radians(16*max(0,-math.sin(t)));rig.pose.bones['lowerleg01.R'].rotation_euler.x=math.radians(16*max(0,math.sin(t)));rig.pose.bones['upperarm01.L'].rotation_euler.x=math.radians(-5*math.sin(t));rig.pose.bones['upperarm01.R'].rotation_euler.x=math.radians(5*math.sin(t))
def present(f):
 t=max(0,min(1,(f-1)/89));pose();rig.pose.bones['head'].rotation_euler.z=math.radians(18*math.sin(math.pi*t));rig.pose.bones['lowerarm01.R'].rotation_euler.x=math.radians(40*math.sin(math.pi*t))
a1,_=make_action('neutral_stand',[1,30],neutral);a2,_=make_action('walk_editorial_loop',frames_walk,walk);a3,_=make_action('walk_and_present_turn',[1,30,60,90],present)
for tr in ad.nla_tracks:tr.mute=(tr.name!='walk_editorial_loop')
rig.animation_data.action=None;sc.frame_start=1;sc.frame_end=90
# animation pose floor and foot movement probe on sampled loop frames
floor=[];feet=[]
for f in range(1,61):
 sc.frame_set(f);bpy.context.view_layer.update();v,_,_,_=evalmesh(body);floor.append(min(x.z for x in v));feet.append([list(rig.matrix_world@rig.pose.bones['foot.'+s].head) for s in ('L','R')])
fit['walk_floor_min_mm']=[min(floor)*1000,max(floor)*1000];fit['stance_foot_slip_sampled_mm']=None;fit['stance_foot_slip_reason']='No approved contact-phase/stance classifier in source; foot locks not implemented.'
# save a derived scene only
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT,'stage31_final_scene.blend'))
# export animated artifact
bpy.ops.object.select_all(action='SELECT');bpy.context.view_layer.objects.active=rig
outglb=os.path.join(OUT,'mannequin_M1_122276_walk_v1.glb')
bpy.ops.export_scene.gltf(filepath=outglb,export_format='GLB',export_animations=True,export_nla_strips=True,export_skins=True,export_morph=True,export_normals=True)
# multi-view render, 8 frames x 3 views
for o in list(sc.objects):
 if o.type in ('LIGHT','CAMERA'):bpy.data.objects.remove(o,do_unlink=True)
# materials
for o,col in ((body,(.32,.32,.30,1)),(gar,(.48,.48,.46,1))):
 m=bpy.data.materials.new('TEMP_'+o.name);m.diffuse_color=col;m.use_nodes=True;m.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value=col;m.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value=.7;o.data.materials.clear();o.data.materials.append(m)
sc.render.engine='BLENDER_EEVEE';sc.eevee.taa_render_samples=8;sc.render.resolution_x=360;sc.render.resolution_y=540;sc.render.resolution_percentage=100;sc.render.image_settings.file_format='PNG';sc.render.film_transparent=False
world=bpy.data.worlds.new('ValidationWorld');world.use_nodes=True;world.node_tree.nodes['Background'].inputs[0].default_value=(.24,.24,.24,1);world.node_tree.nodes['Background'].inputs[1].default_value=.6;sc.world=world
for name,loc,power in [('K',(-3,-4,4),500),('F',(3,-3,2.5),300),('R',(0,3,4),400)]:
 ld=bpy.data.lights.new(name,'AREA');ld.energy=power;ld.shape='DISK';ld.size=4;ob=bpy.data.objects.new(name,ld);sc.collection.objects.link(ob);ob.location=loc;ob.rotation_euler=(Vector((0,0,.9))-ob.location).to_track_quat('-Z','Y').to_euler()
views={'front':Vector((0,-3.5,0)),'three_quarter':Vector((2.5,-3,0)),'side':Vector((3.5,0,0))};frames=[1,8,16,24,32,40,48,56];imgs={};lumas={}
for vn,off in views.items():
 cd=bpy.data.cameras.new('Cam_'+vn);cam=bpy.data.objects.new('Cam_'+vn,cd);sc.collection.objects.link(cam);cam.location=Vector((0,0,.9))+off;cam.rotation_euler=(Vector((0,0,.9))-cam.location).to_track_quat('-Z','Y').to_euler();cd.type='ORTHO';cd.ortho_scale=2.0;sc.camera=cam
 for f in frames:
  sc.frame_set(f);bpy.context.view_layer.update();sc.render.filepath=os.path.join(OUT,'tiles',f'{vn}_{f:02d}.png');bpy.ops.render.render(write_still=True);imgs[(vn,f)]=sc.render.filepath
# compose contact sheet externally by Pillow
with open(os.path.join(OUT,'render_manifest.json'),'w') as fp:json.dump({'frames':frames,'views':list(views),'tiles':imgs,'fit_metrics':fit},fp,indent=2)
with open(os.path.join(OUT,'pose_fit_metrics.json'),'w') as fp:json.dump(fit,fp,indent=2)
print('FIT_METRICS',json.dumps(fit))

