"""REQ-01 inventory (read-only). Run: blender -b --factory-startup --python req01_inventory.py -- <out.json> <blend> [...]
Opens each COPY, lists every object (type, collections, modifiers), and for Tee / collar / placket_overlap / placket_buttons the vertex, face,
material and vertex-group member counts (weight > 0). Never saves."""
import bpy, sys, json, hashlib
import numpy as np

argv = sys.argv[sys.argv.index('--') + 1:]
out_path, files = argv[0], argv[1:]


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''): h.update(ch)
    return h.hexdigest()


res = {}
for f in files:
    h0 = sha(f)
    bpy.ops.wm.open_mainfile(filepath=f, load_ui=False)
    objs = []
    for ob in bpy.data.objects:
        objs.append(dict(name=ob.name, type=ob.type, collections=[c.name for c in ob.users_collection],
                         modifiers=[(m.name, m.type) for m in ob.modifiers], parent=ob.parent.name if ob.parent else None,
                         verts=len(ob.data.vertices) if ob.type == 'MESH' else None,
                         identity_matrix=bool(np.allclose(np.array(ob.matrix_world), np.eye(4)))))
    detail = {}
    for nm in ('Tee', 'collar', 'placket_overlap', 'placket_buttons'):
        ob = bpy.data.objects.get(nm)
        if ob is None: detail[nm] = None; continue
        me = ob.data
        cnt = {g.name: 0 for g in ob.vertex_groups}; gi = {g.index: g.name for g in ob.vertex_groups}
        for v in me.vertices:
            for x in v.groups:
                if x.weight > 0 and x.group in gi: cnt[gi[x.group]] += 1
        mi = np.zeros(len(me.polygons), int); me.polygons.foreach_get('material_index', mi)
        be = 0
        me.calc_loop_triangles()
        ec = np.zeros(len(me.edges), int)
        for p in me.polygons:
            for k in p.edge_keys: pass
        # boundary edges via edge->face count
        import bmesh
        bm = bmesh.new(); bm.from_mesh(me); be = sum(1 for e in bm.edges if len(e.link_faces) == 1); nm_e = sum(1 for e in bm.edges if len(e.link_faces) > 2); bm.free()
        detail[nm] = dict(verts=len(me.vertices), edges=len(me.edges), faces=len(me.polygons), tris=len(me.loop_triangles),
                          materials=[m.name if m else None for m in me.materials],
                          faces_per_material_index={int(k): int(v) for k, v in zip(*np.unique(mi, return_counts=True))},
                          vertex_groups=cnt, boundary_edges=be, nonmanifold_edges_gt2faces=nm_e,
                          modifiers=[(m.name, m.type) for m in ob.modifiers])
    res[f] = dict(sha256=h0, objects=objs, detail=detail, unchanged=(sha(f) == h0))
    print('INV', f.split('\\')[-1].split('/')[-1], json.dumps(detail)[:600])
json.dump(res, open(out_path, 'w'), indent=1)
