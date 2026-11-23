"""FBL-006a sanity check of the occlusion ray method on the bind body (read-only). Inward rays must hit the far side of the body; outward rays from the flagged triangles are reported in REPORT.md.
Run: python -I -B occlusion_sanity.py"""
import os
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.normpath(os.path.expanduser('~/Desktop/better_style_execution_workspace/03_ENGINE/avatar_engine/poses')))
sys.dont_write_bytecode = True
import posekit as K
from mathutils.bvhtree import BVHTree
rig = K.Rig(); rig.reset(); X = rig.eval_body(); TRI = rig.TRI
cen = X[TRI].mean(1); n = np.cross(X[TRI[:,1]]-X[TRI[:,0]], X[TRI[:,2]]-X[TRI[:,0]]); n = n/np.linalg.norm(n,axis=1)[:,None]
bvh = BVHTree.FromPolygons([tuple(p) for p in X.tolist()], TRI.tolist(), all_triangles=True, epsilon=0.0)
top = int(np.argmax(cen[:,2])); bot = int(np.argmin(cen[:,2]))
print("top tri normal z", n[top,2].round(3), "centroid z", cen[top,2].round(3))
print("bottom tri normal z", n[bot,2].round(3))
# outward check: fraction of triangles whose normal points away from the body centre
c0 = X.mean(0); print("frac outward (normal . (centroid-c0) > 0):", float(((n*(cen-c0)).sum(1)>0).mean()).__round__(3))
# inward rays 0.3 m should hit the other side for a closed body
hits = 0
for t in range(0, len(TRI), 500):
    h = bvh.ray_cast(tuple(cen[t]+n[t]*1e-6*0+(-n[t])*1e-6), tuple(-n[t]), 0.3)
    hits += h[2] is not None and h[2] != t
print("inward 0.3 m hits among sampled:", hits, "of", len(range(0,len(TRI),500)))
for t in (2607, 7120, 15844, 20357):
    print(t, "outward hit 20mm:", bvh.ray_cast(tuple(cen[t]+n[t]*1e-6), tuple(n[t]), 0.02)[2], "inward 20mm:", bvh.ray_cast(tuple(cen[t]-n[t]*1e-6), tuple(-n[t]), 0.02)[2], "inward 0.3m:", bvh.ray_cast(tuple(cen[t]-n[t]*1e-6), tuple(-n[t]), 0.3)[2:4])
