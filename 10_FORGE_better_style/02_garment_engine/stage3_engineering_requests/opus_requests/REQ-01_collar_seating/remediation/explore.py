import bpy, numpy as np, sys, bmesh
from scipy.spatial import cKDTree
f=sys.argv[-1]
bpy.ops.wm.open_mainfile(filepath=f, load_ui=False)
T=bpy.data.objects['Tee']; C=bpy.data.objects['collar']
me=T.data; n=len(me.vertices); co=np.empty(n*3); me.vertices.foreach_get('co',co); co=co.reshape(-1,3)
ed=np.empty(len(me.edges)*2,int); me.edges.foreach_get('vertices',ed); ed=ed.reshape(-1,2)
cnt=np.zeros(len(me.edges),int)
for p in me.polygons:
    for k in p.edge_keys: pass
# boundary edges via loops
el=np.empty(len(me.loops),int); me.loops.foreach_get('edge_index',el); bc=np.bincount(el,minlength=len(me.edges))
b=np.flatnonzero(bc==1); print('boundary edges',len(b))
bv=np.unique(ed[b].ravel()); print('boundary verts',len(bv)); B=co[bv]
print('z range of boundary', B[:,2].min(),B[:,2].max())
# cluster by z
import collections
top=B[B[:,2]>np.percentile(B[:,2],70)]
print('top cluster n',len(top), top.min(0), top.max(0))
cm=C.data; m=len(cm.vertices); cc=np.empty(m*3); cm.vertices.foreach_get('co',cc); cc=cc.reshape(-1,3)
g=C.vertex_groups['collar_stand'].index; gl=C.vertex_groups['collar_leaf'].index
ws=np.array([next((x.weight for x in v.groups if x.group==g),0) for v in cm.vertices]); wl=np.array([next((x.weight for x in v.groups if x.group==gl),0) for v in cm.vertices])
ps=(ws>0)&(wl==0); print('pure stand',ps.sum(), cc[ps].min(0),cc[ps].max(0))
kd=cKDTree(B); d,_=kd.query(cc[ps]); print('pure stand to boundary verts mm',np.percentile(d,[0,5,50,95,100])*1e3)
kdT=cKDTree(co); d2,_=kdT.query(cc[ps]); print('to all Tee verts',np.percentile(d2,[0,5,50,95,100])*1e3)
