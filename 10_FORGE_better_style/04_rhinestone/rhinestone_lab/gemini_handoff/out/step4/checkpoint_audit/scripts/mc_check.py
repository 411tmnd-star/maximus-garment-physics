import json, sys, numpy as np
sys.path.insert(0,'/home/user/human-engine/better_style_catalog/rhinestone_lab/py'); import chaton_geometry as CG
B=json.load(open('/home/user/human-engine/better_style_catalog/rhinestone_lab/out/step4/real_authoring/out/review.review.json')); St={s['id']:s for s in B['stones']}
P=json.load(open(sys.argv[1])); V=CG.V; T=CG.TRI; N=CG.NRM; rng=np.random.default_rng(1)
def hw(s):
    t=s['transform']; n=np.array(t['normal']); tg=np.array(t['tangent']); z=np.cross(tg,n); return np.array(t['base'])+t['diameter']*(V[:,0:1]*tg+V[:,1:2]*n+V[:,2:3]*z), (tg,n,z), t
def inside_pts(Q, s):                      # points Q (m,3) inside hull of stone s? (convex: all face half-spaces)
    H,(tg,n,z),t=hw(s); R=np.stack([tg,n,z],1)               # columns: local axes in world
    L=((Q-np.array(t['base']))/t['diameter'])@R               # to unit-hull local coordinates
    ok=np.ones(len(Q),bool)
    for k,(a,b,c) in enumerate(T): ok&=((L-V[a])@N[k])<=1e-12
    return ok
def sample(s, m=4000):
    H,_,_=hw(s); w=rng.dirichlet(np.ones(len(H))*0.4,size=m); return w@H
res=[]
for p in P:
    a,b=St[p['a']],St[p['b']]; qa=sample(a); qb=sample(b)
    # deepest point-in-other: fraction of A's sampled volume inside B and vice versa
    fa=inside_pts(qa,b).mean(); fb=inside_pts(qb,a).mean(); res.append((p['a'],p['b'],fa,fb))
agree=0; dis=[]
for p,(a,b,fa,fb) in zip(P,res):
    mc = (fa>0 or fb>0); ex = p['exact_gap_mm']<=-0.001
    if mc==ex: agree+=1
    else: dis.append((a,b,round(p['exact_gap_mm'],4),fa,fb))
print('pairs',len(P),'agree',agree,'disagree',len(dis)); print(dis[:12])
# validate exact SAT on synthetic known cases
