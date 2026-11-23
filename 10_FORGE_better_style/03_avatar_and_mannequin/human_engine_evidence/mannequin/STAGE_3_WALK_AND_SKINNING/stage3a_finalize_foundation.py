import bpy, json, math, os
import numpy as np
from mathutils import Euler
root=os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/better_style_engine/03_governance_and_evidence/411_01_BETTER_STYLE_ACTIVE/03_DOCS_AND_EVIDENCE/evidence/mannequin/STAGE_3_WALK_AND_SKINNING'))
src=os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/better_style_engine/03_governance_and_evidence/411_01_BETTER_STYLE_ACTIVE/05_3D_ASSETS/glb/mannequin_full_rigged_m01b_v1.glb'))
bpy.ops.import_scene.gltf(filepath=src); sc=bpy.context.scene
body=next(o for o in sc.objects if o.type=='MESH' and o.name.lower().startswith('mannequin'))
rig=next(o for o in sc.objects if o.type=='ARMATURE')
keys=list(body.data.shape_keys.key_blocks)
finite_tests=[]
for key in keys[1:]:
 for k in keys:k.value=0
 key.value=1.0;bpy.context.view_layer.update();ev=body.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh(); arr=np.array([v.co[:] for v in mesh.vertices]);finite=bool(np.isfinite(arr).all());ev.to_mesh_clear();finite_tests.append({'shape_key':key.name,'value':1.0,'finite':finite,'vertices':len(arr)})
for k in keys:k.value=0
# facial probes numeric only
pose_tests=[]
for pb in rig.pose.bones:pb.rotation_mode='XYZ';pb.rotation_euler=(0,0,0);pb.location=(0,0,0)
for bn,rot,loc in [('jaw',(math.radians(15),0,0),(0,0,0)),('oris01',(0,0,0),(0,0,.005))]:
 pb=rig.pose.bones.get(bn);exists=pb is not None
 if exists:pb.rotation_mode='XYZ';pb.rotation_euler=rot;pb.location=loc
 bpy.context.view_layer.update(); ev=body.evaluated_get(bpy.context.evaluated_depsgraph_get()); me=ev.to_mesh(); a=np.array([v.co[:] for v in me.vertices]);finite=bool(np.isfinite(a).all());ev.to_mesh_clear();pose_tests.append({'control':bn,'present':exists,'finite':finite})
for pb in rig.pose.bones:pb.rotation_euler=(0,0,0);pb.location=(0,0,0)
for k in keys:k.value=0
bpy.context.view_layer.update()
# seam gap approximate from split head shell boundary vertices near lower z against torso/head other component
me=body.data;coords=np.array([v.co[:] for v in me.vertices]);
# component labels
a=[[] for _ in me.vertices]
for e in me.edges:u,v=e.vertices;a[u].append(v);a[v].append(u)
seen=set();components=[]
for i in range(len(a)):
 if i in seen:continue
 q=[i];seen.add(i);c=[]
 while q:
  u=q.pop();c.append(u)
  for v in a[u]:
   if v not in seen:seen.add(v);q.append(v)
 components.append(c)
head=next((c for c in components if len(c)==3203),[]);rest=[i for c in components if c is not head for i in c if coords[i,2]>1.45]
seam=[i for i in head if coords[i,2]<1.555];gap_mm=0.0
if seam and rest:gap_mm=float(np.min(np.linalg.norm(coords[np.array(seam)][:,None,:]-coords[np.array(rest)][None,:,:],axis=2))*1000)
# wire-free finish materials, M1 assigned by default; keep M2/M3 in the object material slots for finish switching
colors={'M1_Obsidian_Noir':'#1A1A1D','M2_Alabaster_Chalk':'#E2E4E8','M3_Travertine_Warm':'#D8CFC2'}
def lin(h):
 vals=[int(h[i:i+2],16)/255 for i in (1,3,5)];return tuple(c/12.92 if c<=.04045 else ((c+.055)/1.055)**2.4 for c in vals)+(1,)
body.data.materials.clear()
for name,h in colors.items():
 m=bpy.data.materials.new(name);m.diffuse_color=lin(h);m.use_nodes=True;bs=m.node_tree.nodes.get('Principled BSDF');bs.inputs['Base Color'].default_value=lin(h);bs.inputs['Roughness'].default_value=.48;body.data.materials.append(m)
# explicitly make M1 the active/default material on all polygons
for p in body.data.polygons:p.material_index=0
# save project native derivative, original promoted GLB remains unchanged
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root,'stage3a_working.blend'))
# neutral_stand Action: rest transforms keyed on armature bones
sc.frame_start=0;sc.frame_end=1
for pb in rig.pose.bones:
 pb.rotation_mode='XYZ';pb.keyframe_insert(data_path='rotation_euler',frame=0,group=pb.name);pb.keyframe_insert(data_path='location',frame=0,group=pb.name)
act=rig.animation_data.action if rig.animation_data else None
if act:act.name='neutral_stand'
# diagnostics and material metadata
report={'shape_key_tests':finite_tests,'facial_control_probes':pose_tests,'seam_gap_mm':gap_mm,'custom_normals':bool(me.has_custom_normals),'mesh':{'vertices':len(me.vertices),'edges':len(me.edges),'faces':len(me.polygons)},'bones':len(rig.data.bones),'materials':colors,'wire_texture_bypassed':True,'pose_action':act.name if act else None}
open(os.path.join(root,'stage3a_foundation_report.json'),'w').write(json.dumps(report,indent=2))
print('STAGE3A',json.dumps(report))
