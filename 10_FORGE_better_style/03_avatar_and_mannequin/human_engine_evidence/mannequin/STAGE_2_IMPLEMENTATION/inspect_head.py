import os
import bpy, json, math
from collections import Counter
from mathutils import Vector
src=os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/better_style_engine/03_governance_and_evidence/411_01_BETTER_STYLE_ACTIVE/05_3D_ASSETS/glb/mannequin_full_rigged_122276_morphed.glb')); out=os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/better_style_engine/03_governance_and_evidence/411_01_BETTER_STYLE_ACTIVE/03_DOCS_AND_EVIDENCE/evidence/mannequin/STAGE_2_IMPLEMENTATION/VALIDATION/head_mesh_inspection.json'))
bpy.ops.import_scene.gltf(filepath=src); ob=next(o for o in bpy.context.scene.objects if o.type=='MESH' and o.name=='mannequin'); me=ob.data
# topology counts
faceuse=Counter(); edge_faces=Counter()
for p in me.polygons:
 for vi in p.vertices: faceuse[vi]+=1
 for ek in p.edge_keys: edge_faces[tuple(sorted(ek))]+=1
boundary=[e for e,n in edge_faces.items() if n==1]; nonman=[e for e,n in edge_faces.items() if n>2]
# group map, weights & per-region bbox
groups={g.index:g.name for g in ob.vertex_groups}; gw={}
for v in me.vertices:
 for a in v.groups: gw.setdefault(v.index,{})[groups.get(a.group,str(a.group))]=a.weight
head_groups=[n for n in groups.values() if any(s in n.lower() for s in ('head','jaw','eye','oris','oculi','ear','face'))]
neck_groups=[n for n in groups.values() if 'neck' in n.lower()]
# head vertices where head/face weights > .5 OR spatial head domain z >= 1.48, broad head base z ~1.36
head_ids=set(i for i,w in gw.items() if any(w.get(n,0)>0.05 for n in head_groups))
# include geometric envelope head above jaw/neck approximate, report both
coords=[v.co for v in me.vertices]
keyinfo=[]
if me.shape_keys:
 basis=me.shape_keys.key_blocks[0]
 for k in list(me.shape_keys.key_blocks)[1:]:
  ds=[(k.data[i].co-basis.data[i].co).length for i in range(len(coords))]
  affected=[i for i,d in enumerate(ds) if d>1e-6]
  keyinfo.append({'name':k.name,'affected_vertices_over_1e-6':len(affected),'max_displacement':max(ds,default=0),'head_group_affected':sum(i in head_ids for i in affected),'head_group_max_displacement':max((ds[i] for i in head_ids),default=0)})
# group stats bbox, influence
stats={}
for n in head_groups+neck_groups:
 vals=[(i,w.get(n,0)) for i,w in gw.items() if w.get(n,0)>0]
 if vals:
  ids=[i for i,w in vals]; stats[n]={'vertex_count':len(ids),'weight_min':min(w for i,w in vals),'weight_max':max(w for i,w in vals),'z_min':min(coords[i].z for i in ids),'z_max':max(coords[i].z for i in ids)}
# head group bounds and interface edges where face vertices mix head-dom and nonhead
bound_edges=[]
for e in me.edges:
 a,b=e.vertices
 if (a in head_ids)!=(b in head_ids):bound_edges.append([a,b])
# UV info
uv=[]
for layer in me.uv_layers:
 vs=[l.uv for l in layer.data]; uv.append({'name':layer.name,'loops':len(vs),'u_min':min(x.x for x in vs),'u_max':max(x.x for x in vs),'v_min':min(x.y for x in vs),'v_max':max(x.y for x in vs),'out_of_unit_loops':sum(x.x<0 or x.x>1 or x.y<0 or x.y>1 for x in vs)})
# duplicate exact coordinates
pos=Counter(tuple(round(c,7) for c in v.co) for v in me.vertices); dup=sum(n-1 for n in pos.values() if n>1)
# face normals check: polygons recalc consistent? report min area, zero area and orientation mix via normal z distribution only
areas=[p.area for p in me.polygons]
report={'mesh':{'name':ob.name,'vertices':len(me.vertices),'edges':len(me.edges),'faces':len(me.polygons),'boundary_edges':len(boundary),'nonmanifold_edges_over2_faces':len(nonman),'vertices_with_no_faces':sum(1 for i in range(len(me.vertices)) if faceuse[i]==0),'duplicate_positions_rounded_1e-7':dup,'zero_or_tiny_faces_area_lt_1e-10':sum(a<1e-10 for a in areas),'min_face_area':min(areas),'normal_z_min':min(p.normal.z for p in me.polygons),'normal_z_max':max(p.normal.z for p in me.polygons)},'head_vertex_group_names':head_groups,'neck_vertex_group_names':neck_groups,'group_stats':stats,'head_vertex_count_union_groups':len(head_ids),'head_union_bounds':{ax:[min(coords[i][j] for i in head_ids),max(coords[i][j] for i in head_ids)] for j,ax in enumerate('xyz')} if head_ids else {},'head_neck_interface_edges_by_group_membership':len(bound_edges),'shape_keys':keyinfo,'uv_layers':uv,'armature_modifiers':[{'name':m.name,'target':m.object.name if m.object else None,'vertex_groups':m.use_vertex_groups,'preserve_volume':m.use_deform_preserve_volume} for m in ob.modifiers if m.type=='ARMATURE'],'rig':{'armatures':[{'name':o.name,'bones':len(o.data.bones)} for o in bpy.context.scene.objects if o.type=='ARMATURE']}}
open(out,'w').write(json.dumps(report,indent=2))
