import sys, bpy, numpy as np
sys.dont_write_bytecode=True
bpy.ops.wm.open_mainfile(filepath=sys.argv[1], load_ui=False)
o=bpy.data.objects['Tee']; me=o.data
V=np.array([v.co[:] for v in me.vertices])
# boundary edges
import collections
ec=collections.Counter()
for p in me.polygons:
    vs=list(p.vertices)
    for i in range(len(vs)):
        a,b=vs[i],vs[(i+1)%len(vs)]; ec[(min(a,b),max(a,b))]+=1
bd=[e for e,c in ec.items() if c==1]
print('boundary edges',len(bd), 'nonmanifold', sum(1 for c in ec.values() if c>2))
bv=np.unique(np.array(bd).ravel()); B=V[bv]
print('boundary verts z>1.35:', (B[:,2]>1.35).sum())
nk=B[B[:,2]>1.35]
print('neck bbox',nk.min(0).round(4),nk.max(0).round(4))
# vertex groups extents
for g in o.vertex_groups:
    idx=[v.index for v in me.vertices if any(gg.group==g.index and gg.weight>0.5 for gg in v.groups)]
    P=V[idx]
    if len(idx): print(g.name,len(idx),P.min(0).round(3),P.max(0).round(3))
    else: print(g.name,0)
# neck opening ring: verts z>1.40 & |x|<0.12 radius profile
m=(V[:,2]>1.38)&(np.abs(V[:,0])<0.13)
P=V[m]
for z0 in np.arange(1.38,1.49,0.01):
    s=P[np.abs(P[:,2]-z0)<0.004]
    if len(s): print('z',round(z0,2),'x',s[:,0].min().round(3),s[:,0].max().round(3),'y',s[:,1].min().round(3),s[:,1].max().round(3),len(s))
# front centre line profile x~0, front
f=V[(np.abs(V[:,0])<0.004)&(V[:,1]<-0.02)]
for z0 in np.arange(0.86,1.46,0.04):
    s=f[np.abs(f[:,2]-z0)<0.01]
    if len(s): print('front cl z',round(z0,2),'y',s[:,1].min().round(4),s[:,1].max().round(4))
# sleeve hems
for side in 'LR':
    gi=o.vertex_groups['set_in_sleeve_'+side].index
    idx=[v.index for v in me.vertices if any(gg.group==gi and gg.weight>0.5 for gg in v.groups)]
    P=V[idx]; k=np.argmax(np.abs(P[:,0])); print(side,'tip',P[k].round(3), 'n',len(P))
col=bpy.data.objects['collar']; C=np.array([v.co[:] for v in col.data.vertices]); print('collar',C.min(0),C.max(0))
ma=bpy.data.objects['Mannequin']; M=np.array([v.co[:] for v in ma.data.vertices])
for z0 in (1.40,1.44,1.48,1.52):
    s=M[(np.abs(M[:,2]-z0)<0.006)&(np.abs(M[:,0])<0.2)]
    print('mann z',z0,'x',s[:,0].min().round(3),s[:,0].max().round(3),'y',s[:,1].min().round(3),s[:,1].max().round(3))
print('attrs',[a.name for a in me.attributes])
