"""Compare scratch-rebuilt master vs copy of original master: counts, per-vertex deltas, probe_v2/eval_g0 inside metric.
blender -b --factory-startup --python compare_repro.py -- <orig.blend> <new.blend> <out.json>"""
import sys, json
import numpy as np, bpy, bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree
a = sys.argv[sys.argv.index('--') + 1:]; A, B, OUT = a[:3]

def load(path):
    bpy.ops.wm.open_mainfile(filepath=path, load_ui=False)
    body = bpy.data.objects['M2_Baked_ThetaFit']
    bm = bmesh.new(); bm.from_mesh(body.data); bm.transform(body.matrix_world); bvh = BVHTree.FromBMesh(bm)
    BV = np.array([v.co[:] for v in bm.verts]); bm.free()
    d = {'body_verts': BV}
    for name in ('G0_jacket_bracket', 'G0_jacket_orbit', 'G0_pants'):
        o = bpy.data.objects[name]; M = o.matrix_world
        G = np.array([(M @ v.co)[:] for v in o.data.vertices]); nf = int(o['n_nylon_faces'])
        idx = sorted({i for p in o.data.polygons[:nf] for i in p.vertices}); S = G[idx]
        sd = []
        for p in S:
            loc, nrm, _, dist = bvh.find_nearest(Vector(p)); sd.append(-dist if (Vector(p) - loc).dot(nrm) < 0 else dist)
        d[name] = dict(G=G, nf=nf, nverts=len(G), npolys=len(o.data.polygons), shell=len(idx), inside=int((np.array(sd) < 0).sum()),
                       props={k: (float(o[k]) if isinstance(o[k], (int, float)) else str(o[k])) for k in o.keys() if not k.startswith('_')})
    return d

dA = load(A); dB = load(B)
res = {'orig': A, 'new': B, 'body_verts_equal_count': len(dA['body_verts']) == len(dB['body_verts']),
       'body_max_abs_diff_m': float(np.abs(dA['body_verts'] - dB['body_verts']).max()) if len(dA['body_verts']) == len(dB['body_verts']) else None}
for n in ('G0_jacket_bracket', 'G0_jacket_orbit', 'G0_pants'):
    x, y = dA[n], dB[n]
    r = {k: [x[k], y[k]] for k in ('nverts', 'npolys', 'nf', 'shell', 'inside')}
    r['props'] = [x['props'], y['props']]
    if x['nverts'] == y['nverts']: r['max_abs_vertex_diff_m'] = float(np.abs(x['G'] - y['G']).max())
    res[n] = r
json.dump(res, open(OUT, 'w'), indent=1); print(json.dumps(res, indent=1))
