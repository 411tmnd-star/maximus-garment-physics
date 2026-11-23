import sys, json, numpy as np, collections
R='/home/user/human-engine/better_style_catalog/rhinestone_lab'
for q in (R+'/py', R+'/blender_addon'): sys.path.insert(0,q)
import bpy, mathutils
from mathutils.bvhtree import BVHTree
import real_garment as RG
bpy.ops.wm.open_mainfile(filepath=str(RG.BLEND))
B=json.load(open(R+'/out/step4/real_authoring/out/review.review.json')); con=json.load(open(R+'/out/step4/real_authoring/surface.json'))
dg=bpy.context.evaluated_depsgraph_get(); ob={n:bpy.data.objects[n] for n in ('Tee','collar','placket_overlap','Mannequin')}
tree={n:BVHTree.FromObject(o,dg) for n,o in ob.items()}
def vg_weights(o):
    names={g.index:g.name for g in o.vertex_groups}; W=collections.defaultdict(dict)
    for v in o.data.vertices:
        for g in v.groups: W[v.index][names[g.group]]=g.weight
    return W
Wt=vg_weights(ob['Tee']); Wc=vg_weights(ob['collar'])
pobj={p['panel_id']:p['surface_reference']['object'] for p in con['panels']}
pgroup={p['panel_id']:(p['surface_reference'].get('vertex_group') or '') for p in con['panels']}
yb=lambda v: np.array([v[0],-v[2],v[1]],float)           # glTF Y-up -> Blender Z-up
rows=[]
for s in B['stones']:
    if not s['transform'] or s['outcome']=='REJECTED': continue
    t=s['transform']; base=yb(t['base']); cax=yb(t['normal']); o=pobj[s['panel_id']]
    h=tree[o].find_nearest(mathutils.Vector(base)); 
    if h[0] is None: continue
    nl=np.array(h[1]); m=tree['Mannequin'].find_nearest(mathutils.Vector(h[0])); dec=np.array(h[0])-np.array(m[0]); dec/=np.linalg.norm(dec)
    if nl@dec<0: nl=-nl                                    # outer-layer normal sense (nearest face could be either layer): orient with the declared reference ONLY for the angle calc; flag if it needed flipping
    flipped = bool(np.array(h[1])@dec<0)
    me=bpy.data.objects[o].data; poly=me.polygons[h[2]]
    wsrc=Wt if o=='Tee' else (Wc if o=='collar' else {})
    grp=collections.Counter()
    for vi in poly.vertices:
        if vi in wsrc:
            for k,w in wsrc[vi].items():
                if w>0.5: grp[k]+=1
    intended=pgroup[s['panel_id']]; in_panel = (intended=='' ) or grp.get(intended,0)>=2
    ang=lambda a,b: float(np.degrees(np.arccos(np.clip(a@b,-1,1))))
    rows.append(dict(id=s['id'],curve=s['curve_id'],panel=s['panel_id'],outcome=s['outcome'],snap=s['snap_mm'] or 0,sample=s['sample'],dec_vs_local=ang(dec,nl),crystal_vs_local=ang(cax,nl),crystal_vs_dec=ang(cax,dec),
                     engine_dev_local=s['measurement']['normal_dev_local_deg'],engine_dev_out=s['measurement']['normal_dev_outward_deg'],nearest_flipped=flipped,in_intended_panel=bool(in_panel),dist_to_surface_mm=h[3]*1e3,groups=dict(grp)))
json.dump(rows,open('/tmp/claude-0/-home-user-human-engine/fe403360-08ed-5551-930f-a67b8d68b9ee/scratchpad/orient_rows.json','w'))
print(len(rows),'non-rejected stones analysed')
