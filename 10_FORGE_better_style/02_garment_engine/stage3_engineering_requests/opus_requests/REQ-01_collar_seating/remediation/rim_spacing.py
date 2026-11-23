import bpy, numpy as np, sys, json
from scipy.spatial import cKDTree
bpy.ops.wm.open_mainfile(filepath=sys.argv[-2], load_ui=False)
T=bpy.data.objects['Tee']; m=T.data; a=np.empty(len(m.vertices)*3); m.vertices.foreach_get('co',a); a=a.reshape(-1,3)
S=json.load(open(sys.argv[-1])); nm=[k for k in S][0]
with bpy.data.libraries.load(S[nm]['seated']['file']) as (s,d): d.objects=list(s.objects)
c=d.objects[0]; cm=c.data; b=np.empty(len(cm.vertices)*3); cm.vertices.foreach_get('co',b); b=b.reshape(-1,3)
kd=cKDTree(a); dist,i=kd.query(b); low=b[:,2]<np.percentile(b[:,2],3)
near=np.unique(i[dist<0.004]); P=a[near]; k2=cKDTree(P); nn=k2.query(P,k=2)[0][:,1]*1e3
print('Tee verts near collar base:',len(near),'nn spacing mm pct10/50/90/max',np.percentile(nn,[10,50,90,100]).round(2))
