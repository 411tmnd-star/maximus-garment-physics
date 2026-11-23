import os
import bpy,json
from collections import defaultdict
from mathutils import Vector
p=os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/better_style_engine/03_governance_and_evidence/411_01_BETTER_STYLE_ACTIVE/05_3D_ASSETS/glb/mannequin_full_rigged_122276_morphed.glb')); out=os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/better_style_engine/03_governance_and_evidence/411_01_BETTER_STYLE_ACTIVE/03_DOCS_AND_EVIDENCE/evidence/mannequin/STAGE_2_IMPLEMENTATION/VALIDATION/connected_components.json'))
bpy.ops.import_scene.gltf(filepath=p);o=next(x for x in bpy.context.scene.objects if x.type=='MESH' and x.name=='mannequin'); me=o.data
adj=[[] for _ in me.vertices]
for e in me.edges:
 a,b=e.vertices;adj[a].append(b);adj[b].append(a)
seen=set(); comps=[]
for i in range(len(adj)):
 if i in seen:continue
 st=[i];seen.add(i); ids=[]
 while st:
  v=st.pop();ids.append(v)
  for n in adj[v]:
   if n not in seen:seen.add(n);st.append(n)
 if len(ids)>2:
  vs=[me.vertices[j].co for j in ids]; groups=defaultdict(list)
  for j in ids:
   for a in me.vertices[j].groups: groups[o.vertex_groups[a.group].name].append(a.weight)
  comps.append({'vertices':len(ids),'edges':sum(len(adj[j]) for j in ids)//2,'bounds':{'x':[min(v.x for v in vs),max(v.x for v in vs)],'y':[min(v.y for v in vs),max(v.y for v in vs)],'z':[min(v.z for v in vs),max(v.z for v in vs)]},'groups':{k:{'vertices':len(v),'max':max(v)} for k,v in groups.items() if any(t in k.lower() for t in ['head','jaw','eye','oculi','oris','neck'])}})
comps.sort(key=lambda c:c['vertices'],reverse=True);open(out,'w').write(json.dumps(comps,indent=2));print('component_count',len(comps));print(json.dumps(comps[:20],indent=2))
