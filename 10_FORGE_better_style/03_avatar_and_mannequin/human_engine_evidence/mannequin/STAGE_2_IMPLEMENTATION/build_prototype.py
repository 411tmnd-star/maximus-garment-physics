import bpy, math, json, os
from mathutils import Vector
src=os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/better_style_engine/03_governance_and_evidence/411_01_BETTER_STYLE_ACTIVE/05_3D_ASSETS/glb/mannequin_full_rigged_122276_morphed.glb'))
root=os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/better_style_engine/03_governance_and_evidence/411_01_BETTER_STYLE_ACTIVE/03_DOCS_AND_EVIDENCE/evidence/mannequin/STAGE_2_IMPLEMENTATION'))
vis=os.path.join(root,'VISUAL_EVIDENCE'); outblend=os.path.join(root,'M01B_experimental_pre_edit.blend'); outglb=os.path.join(root,'M01B_experimental_prototype.glb')
bpy.ops.import_scene.gltf(filepath=src); scene=bpy.context.scene
body=next(o for o in scene.objects if o.type=='MESH' and o.name=='mannequin'); rig=next(o for o in scene.objects if o.type=='ARMATURE' and o.name=='Human.rig')
# Keep a hidden, untouched in-memory reference object. Working object owns copied mesh data and modifiers/groups.
ref=body.copy(); ref.data=body.data.copy(); ref.name='mannequin_CANONICAL_REFERENCE'; scene.collection.objects.link(ref); ref.hide_render=True; ref.hide_viewport=True
work=body.copy(); work.data=body.data.copy(); work.name='mannequin_M01B_EXPERIMENTAL'; scene.collection.objects.link(work)
for o in scene.objects:
 if o.type=='MESH' and o not in (ref,work): o.hide_render=True; o.hide_viewport=True
# recoverable checkpoint before any coordinate changes
bpy.ops.wm.save_as_mainfile(filepath=outblend)
# Actual mask: facial influence groups + anterior-facing position; neck feather starts above head/neck overlap.
blocks=list(work.data.shape_keys.key_blocks); basis=blocks[0]; orig=[v.co.copy() for v in basis.data]
gnames=[g.name.lower() for g in work.vertex_groups]; fg=[i for i,n in enumerate(gnames) if any(s in n for s in ['head','jaw','eye','oculi','oris'])]
vg={}
for v in work.data.vertices:
 for a in v.groups: vg.setdefault(v.index,{})[a.group]=a.weight

def sm(a,b,x):
 t=max(0,min(1,(x-a)/(b-a))); return t*t*(3-2*t)
def interp(z, pts):
 for (z0,y0),(z1,y1) in zip(pts,pts[1:]):
  if z<=z1:return y0+(y1-y0)*max(0,min(1,(z-z0)/(z1-z0)))
 return pts[-1][1]
profile=[(1.47,-.083),(1.50,-.105),(1.54,-.116),(1.58,-.119),(1.62,-.119),(1.66,-.114),(1.69,-.103),(1.705,-.090)]
mask=[]; earweights=[]; moves=[]
for i,p in enumerate(orig):
 x,y,z=p; weights=vg.get(i,{}); hw=max((weights.get(j,0) for j in fg),default=0)
 zf=sm(1.455,1.495,z)*(1-sm(1.685,1.715,z)); xf=1-sm(.074,.087,abs(x)); front=1-sm(-.105,-.045,y)
 fw=zf*xf*front
 ew=sm(.066,.078,abs(x))*(1-sm(.085,.093,abs(x)))*sm(1.535,1.565,z)*(1-sm(1.64,1.67,z))
 mask.append(fw); earweights.append(ew)
 # set y to a smooth visor surface, retaining lateral head curvature
 yt=interp(z,profile)+.022*(abs(x)/.085)**2
 dy=(yt-y)*fw
 dx=-(1 if x>=0 else -1)*max(0,abs(x)-.065)*ew
 moves.append(Vector((dx,dy,0)))
# Apply identical displacement vector to Basis and every shape key, preserving each existing key delta exactly.
for block in blocks:
 for i in range(len(orig)): block.data[i].co += moves[i]
# evaluated material remains canonical; remove temp mask objects from render
for o in scene.objects:
 if o.type=='MESH': o.hide_render=o not in (work,)
# metadata custom props on experimental object
work['prototype']='M-01B experimental'; work['source_sha256']='28b487ce520caf596943834ba9ddec721a1c3b4359322edd7a74b56fb357c143'; work['geometry_edit']='feathered anterior-face y projection + ear x inset; same delta on Basis/all five keys'
# save experimental native project
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root,'M01B_experimental_prototype.blend'))
# export mesh/rig only; selected export avoids garment edits/contents
bpy.ops.object.select_all(action='DESELECT'); work.select_set(True); rig.select_set(True); bpy.context.view_layer.objects.active=work
bpy.ops.export_scene.gltf(filepath=outglb,export_format='GLB',use_selection=True,export_apply=False)
print(json.dumps({'mask_gt_0.5':sum(v>.5 for v in mask),'mask_gt_0.1':sum(v>.1 for v in mask),'ear_gt_0.5':sum(v>.5 for v in earweights),'max_displacement':max(v.length for v in moves),'mesh':{'vertices':len(work.data.vertices),'edges':len(work.data.edges),'faces':len(work.data.polygons)},'keys':[k.name for k in blocks],'rig_bones':len(rig.data.bones),'glb':outglb}))
