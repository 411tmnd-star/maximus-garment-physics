import sys, json, numpy as np
R='/home/user/human-engine/better_style_catalog/rhinestone_lab'
for q in (R+'/py', R+'/blender_addon'): sys.path.insert(0,q)
import bpy, mathutils
from mathutils.bvhtree import BVHTree
import design_intent as DI, authoring as AU, contracts as CT, real_garment as RG, surfaces as SF
bpy.ops.wm.open_mainfile(filepath=str(RG.BLEND))
import bs_rhinestone_authoring as ADDON
objs = ADDON._objects()
con = json.load(open(R+'/out/step4/real_authoring/surface.json')); des = json.load(open(R+'/out/step4/real_authoring/design.json'))
gs = SF.GarmentSurfaces(con, objs, None, ('placket_buttons',))
cc = DI.to_curve_contract(des); c = [x for x in cc['curves'] if x['curve_id']=='chest_probe'][0]
path = AU.path_from_source(c['source']); conf, st = AU.conform_path(gs, 'body_front_outer', path)
print('curve length %.1f mm; conform stats %s' % (path.length*1e3, st))
panel = gs.panels['body_front_outer']
trees = {n: BVHTree.FromObject(objs[n], bpy.context.evaluated_depsgraph_get()) for n in ('Tee','placket_overlap','collar')}
n = 61; s = np.linspace(0, path.length, 122)  # same count as stones (122 samples)
rows=[]
for i, t in enumerate(s):
    p = path.point(np.array([t]))[0]; d_panel = panel.bvh.find_nearest(mathutils.Vector(p))
    d = {k: (v.find_nearest(mathutils.Vector(p))[3]) for k, v in trees.items()}
    # signed side relative to the declared outward direction at the panel hit
    hit = np.array(d_panel[0]); nrm = np.array(d_panel[1])
    rows.append((i, d_panel[3]*1e3, d['Tee']*1e3, d['placket_overlap']*1e3, d['collar']*1e3, float((p-hit)@nrm)*1e3))
rows=np.array(rows)
print('dist(raw curve point -> panel body_front_outer) mm: percentiles 50/90/99/max', np.round(np.percentile(rows[:,1],[50,90,99,100]),2))
print('dist -> any Tee face (all panels, inner+outer) mm: 50/90/99/max', np.round(np.percentile(rows[:,2],[50,90,99,100]),2))
print('samples with panel distance > 2 mm:', int((rows[:,1]>2).sum()), '> 6 mm:', int((rows[:,1]>6).sum()))
print('idx  d_panel  d_Tee(any)  d_placket  d_collar  signed_height_over_panel')
for r in rows:
    if r[1] > 2.0: print('%3d %7.2f %8.2f %8.2f %8.2f %8.2f' % tuple(r))
json.dump(rows.tolist(), open('/tmp/claude-0/-home-user-human-engine/fe403360-08ed-5551-930f-a67b8d68b9ee/scratchpad/chest_rows.json','w'))
