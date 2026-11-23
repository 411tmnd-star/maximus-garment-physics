import bpy,json,math,os
from mathutils import Euler
root=os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/better_style_engine/03_governance_and_evidence/411_01_BETTER_STYLE_ACTIVE/03_DOCS_AND_EVIDENCE/evidence/mannequin/STAGE_2_IMPLEMENTATION')); glb=os.path.join(root,'M01B_experimental_prototype.glb'); out=os.path.join(root,'VALIDATION','prototype_validation.json')
bpy.ops.import_scene.gltf(filepath=glb); s=bpy.context.scene
meshes=[o for o in s.objects if o.type=='MESH']; rigs=[o for o in s.objects if o.type=='ARMATURE']; b=next(o for o in meshes if "M01B_EXPERIMENTAL" in o.name); r=rigs[0]; keys=list(b.data.shape_keys.key_blocks) if b.data.shape_keys else []
checks={'reimport':{'mesh_count':len(meshes),'rig_count':len(rigs),'vertices':len(b.data.vertices),'edges':len(b.data.edges),'faces':len(b.data.polygons),'shape_keys':[k.name for k in keys],'bone_count':len(r.data.bones),'uv_layers':[u.name for u in b.data.uv_layers],'materials':[m.name if m else None for m in b.data.materials],'armature_modifiers':[m.object.name if m.type=='ARMATURE' and m.object else None for m in b.modifiers if m.type=='ARMATURE']}}
# mesh integrity and weight integrity
edgeface={tuple(sorted(e.vertices)):0 for e in b.data.edges}
for p in b.data.polygons:
 for ek in p.edge_keys:edgeface[tuple(sorted(ek))]=edgeface.get(tuple(sorted(ek)),0)+1
checks['mesh_integrity']={'boundary_edges':sum(v==1 for v in edgeface.values()),'nonmanifold_edges':sum(v>2 for v in edgeface.values()),'nonfinite_coordinates':sum(not all(math.isfinite(c) for c in v.co) for v in b.data.vertices),'verts':len(b.data.vertices),'edges':len(b.data.edges),'faces':len(b.data.polygons)}
groups={g.index:g.name for g in b.vertex_groups}; sums=[]
for v in b.data.vertices: sums.append(sum(a.weight for a in v.groups))
checks['weights']={'group_count':len(groups),'vertices_with_weight_sum_gt_1.001':sum(x>1.001 for x in sums),'min_nonzero_sum':min((x for x in sums if x>0),default=0),'max_sum':max(sums,default=0)}
# shape key activation calculations; reset after each
shape_results=[]
for k in keys[1:]:
 for q in keys:q.value=0
 k.value=1.0;s.view_layers.update(); dg=bpy.context.evaluated_depsgraph_get(); ev=b.evaluated_get(dg); me=ev.to_mesh(); vals=[tuple(v.co) for v in me.vertices]; finite=all(math.isfinite(c) for v in vals for c in v); ev.to_mesh_clear(); shape_results.append({'name':k.name,'evaluated_vertices':len(vals),'finite':finite})
for q in keys:q.value=0
# rig motion probe samples; native pose controls in memory, then restore
pose_tests=[]
for bn,axis,angle in [('head','Y',20),('neck02','X',12),('neck02','Y',12),('jaw','X',15),('eye.L','Z',10),('oris01','X',10)]:
 pb=r.pose.bones.get(bn)
 if not pb: pose_tests.append({'bone':bn,'available':False});continue
 pb.rotation_mode='XYZ';pb.rotation_euler=(0,0,0); idx='XYZ'.index(axis);pb.rotation_euler[idx]=math.radians(angle);s.view_layers.update();dg=bpy.context.evaluated_depsgraph_get();ev=b.evaluated_get(dg);me=ev.to_mesh();finite=all(math.isfinite(c) for v in me.vertices for c in v.co);ev.to_mesh_clear();pose_tests.append({'bone':bn,'available':True,'finite':finite,'visual_deformation_review':'NOT AUTOMATED'})
for pb in r.pose.bones:pb.rotation_euler=(0,0,0)
s.view_layers.update();checks['shape_key_activation']=shape_results;checks['pose_probes']=pose_tests
open(out,'w').write(json.dumps(checks,indent=2))
