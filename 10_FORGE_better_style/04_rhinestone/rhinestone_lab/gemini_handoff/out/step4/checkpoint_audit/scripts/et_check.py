import json, sys, numpy as np
sys.path.insert(0,'/home/user/human-engine/better_style_catalog/rhinestone_lab/py'); import chaton_geometry as CG
B=json.load(open('/home/user/human-engine/better_style_catalog/rhinestone_lab/out/step4/real_authoring/out/review.review.json')); St={s['id']:s for s in B['stones']}
P=json.load(open(sys.argv[1])); V=CG.V; T=CG.TRI
EDG=sorted({tuple(sorted(e)) for a,b,c in T for e in ((a,b),(b,c),(c,a))})
def hw(t):
    n=np.array(t['normal']); tg=np.array(t['tangent']); z=np.cross(tg,n); return np.array(t['base'])+t['diameter']*(V[:,0:1]*tg+V[:,1:2]*n+V[:,2:3]*z)
def seg_tri(p,q,a,b,c,eps=1e-12):                 # segment p->q vs triangle abc (Moller-Trumbore, segment-limited)
    d=q-p; e1=b-a; e2=c-a; h=np.cross(d,e2); det=e1@h
    if abs(det)<eps: return False
    f=1/det; s=p-a; u=f*(s@h)
    if u<-1e-9 or u>1+1e-9: return False
    qv=np.cross(s,e1); v=f*(d@qv)
    if v<-1e-9 or u+v>1+1e-9: return False
    t=f*(e2@qv); return -1e-9<=t<=1+1e-9
def intersects(HA,HB):
    for HX,HY in ((HA,HB),(HB,HA)):
        for i,j in EDG:
            for a,b,c in T:
                if seg_tri(HX[i],HX[j],HY[a],HY[b],HY[c]): return True
    return False
def contains(H,pt):                                # centre containment (full overlap)
    return False
# validate on constructed cases: two stones with identical frame, shifted by dx along the tangent
t0=dict(base=[0,0,0],normal=[0,1,0],tangent=[1,0,0],diameter=0.0028)
for dx in (0.0010,0.0027,0.0029,0.0035):
    t1=dict(t0,base=[dx,0,0]); print('constructed shift %.1f mm -> intersects %s'%(dx*1e3,intersects(hw(t0),hw(t1))))
out=[]
for p in P:
    HA=hw(St[p['a']]['transform']); HB=hw(St[p['b']]['transform']); out.append(intersects(HA,HB))
import collections
print(collections.Counter((p['exact_gap_mm']<=-0.001, o) for p,o in zip(P,out)))
dis=[(p['a'],p['b'],round(p['exact_gap_mm'],4),p['engine_sev']) for p,o in zip(P,out) if (p['exact_gap_mm']<=-0.001)!=o]; print(dis)
json.dump(out,open(sys.argv[2],'w'))
