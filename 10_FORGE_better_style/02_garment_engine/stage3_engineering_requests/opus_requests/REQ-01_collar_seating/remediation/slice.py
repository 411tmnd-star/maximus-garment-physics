import bpy, numpy as np, sys
f=sys.argv[-2]; ang=float(sys.argv[-1])
bpy.ops.wm.open_mainfile(filepath=f, load_ui=False)
T=bpy.data.objects['Tee']; C=bpy.data.objects['collar']
def V(o):
    m=o.data; a=np.empty(len(m.vertices)*3); m.vertices.foreach_get('co',a); return a.reshape(-1,3)
t=V(T); c=V(C)
g=C.vertex_groups['collar_stand'].index; gl=C.vertex_groups['collar_leaf'].index
ws=np.array([next((x.weight for x in v.groups if x.group==g),0) for v in C.data.vertices]); wl=np.array([next((x.weight for x in v.groups if x.group==gl),0) for v in C.data.vertices])
ps=(ws>0)&(wl==0); cen=c[ps][:,:2].mean(0)
def th(p): return np.degrees(np.arctan2(p[:,0]-cen[0], -(p[:,1]-cen[1])))
def rad(p): return np.hypot(p[:,0]-cen[0],p[:,1]-cen[1])
for name,P in (('stand',c[ps]),('Tee',t)):
    m=np.abs(((th(P)-ang+180)%360)-180)<1.0
    if name=='Tee': m&=(P[:,2]>1.35)&(rad(P)<0.16)
    Q=P[m]; r=rad(Q); z=Q[:,2]
    print(name,m.sum())
    # print coarse histogram of (r,z) extremes
    o=np.argsort(z)[::-1]
    sel=o[:40] if name=="Tee" else o[::6]
    for i in sel[:25]: print('  r=%.1f z=%.1f'%(r[i]*1e3,z[i]*1e3))
