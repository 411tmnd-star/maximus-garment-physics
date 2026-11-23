"""FBL-009a step 1-2: independent object and topology inventory of out/BS-SLV-001_shell.blend.

Does not import build_slv.py or any build module. Opens the shell read-only (nothing is saved). Writes inventory.json here.
Run:  python -I -B audit_inventory.py
"""
import sys, json, hashlib
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import bpy                                    # noqa: E402  (bpy 5.2.3, imported before mathutils)

HERE = Path(__file__).resolve().parent
BS = HERE.parent                              # solvers/garment_engine/BS-SLV-001
SHELL = BS / "out" / "BS-SLV-001_shell.blend"
PIECES = ["front", "yoke", "back_p", "back_n", "collar", "gusset_p", "gusset_n"]   # seven structural pieces (pattern_slv.py header, PDF p11)
FLANGE = "flange_strip"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


hash_before = sha(SHELL)
bpy.ops.wm.open_mainfile(filepath=str(SHELL), load_ui=False)

obj = bpy.data.objects.get("BS-SLV-001")
if obj is None:
    raise SystemExit("object BS-SLV-001 missing")
me = obj.data
V = np.array([v.co[:] for v in me.vertices], float)
F = [tuple(p.vertices[:]) for p in me.polygons]
E_all = Counter()
for f in F:
    for i in range(len(f)):
        a, b = f[i], f[(i + 1) % len(f)]
        E_all[(min(a, b), max(a, b))] += 1

# ---- vertex groups: vertices with weight > 0 per group, and exclusive owner per vertex
gname = {g.index: g.name for g in obj.vertex_groups}
members = defaultdict(set)
owners = defaultdict(list)
for v in me.vertices:
    for g in v.groups:
        if g.weight > 0:
            members[gname[g.group]].add(v.index); owners[v.index].append(gname[g.group])

# ---- faces: material slot, and which pieces the face's vertices touch
mat_names = [s.material.name if s.material else None for s in obj.material_slots]
mat_counts = Counter(p.material_index for p in me.polygons)
piece_set = set(PIECES) | {FLANGE}
face_pieces = Counter()
faces_outside = 0
for f in F:
    touched = set()
    for v in f:
        touched |= set(owners[v])
    touched.discard("rib")                    # 'rib' is the union of collar, gusset_p, gusset_n (a vertex group, not a piece)
    if not touched or not touched <= piece_set:
        faces_outside += 1
    face_pieces[tuple(sorted(touched))] += 1
spanning = sum(c for k, c in face_pieces.items() if len(k) > 1)

# ---- topology
boundary = [e for e, c in E_all.items() if c == 1]
nonmanifold = [e for e, c in E_all.items() if c > 2]
used = set(i for f in F for i in f)
loose = [i for i in range(len(V)) if i not in used]

# boundary loops: connected components of boundary edges
adj = defaultdict(list)
for a, b in boundary:
    adj[a].append(b); adj[b].append(a)
seen = set(); loops = []
for s in adj:
    if s in seen: continue
    stack = [s]; comp = set()
    while stack:
        x = stack.pop()
        if x in comp: continue
        comp.add(x); stack.extend(adj[x])
    seen |= comp
    loops.append(comp)
loop_info = []
for comp in loops:
    zs = V[list(comp), 2]
    pan = Counter(o for v in comp for o in owners[v])
    loop_info.append(dict(vertices=len(comp), z_min=round(float(zs.min()), 4), z_max=round(float(zs.max()), 4), owners=dict(pan)))

inv = dict(
    file=str(SHELL.relative_to(BS.parents[1])).replace("\\", "/"), sha256_before=hash_before,
    objects=[dict(name=o.name, type=o.type) for o in bpy.data.objects],
    object_BS_SLV_001=dict(
        modifiers=[dict(name=m.name, type=m.type, thickness=getattr(m, "thickness", None), offset=getattr(m, "offset", None), use_even_offset=getattr(m, "use_even_offset", None)) for m in obj.modifiers],
        custom_properties={k: str(obj[k]) for k in obj.keys()},
        vertices=len(V), edges=len(E_all), faces=len(F), triangles_by_loop=len(me.loop_triangles),
        vertex_groups=[dict(name=g.name, vertices_with_weight=len(members[g.name]), weighted_vertex_count=sum(1 for v in me.vertices for gg in v.groups if gg.group == g.index and gg.weight > 0)) for g in obj.vertex_groups],
        material_slots=[dict(slot=i, material=mat_names[i], faces=int(mat_counts.get(i, 0))) for i in range(len(obj.material_slots))],
        material_index_values=sorted(int(k) for k in mat_counts),
        uv_layers=[u.name for u in me.uv_layers],
        boundary_edges=len(boundary), boundary_loops=len(loop_info), boundary_loop_detail=loop_info,
        non_manifold_edges=len(nonmanifold), loose_vertices=len(loose),
        vertices_in_no_group=sum(1 for i in range(len(V)) if not owners[i]),
    ),
    piece_check=dict(
        seven_named_pieces_present={p: (p in members and len(members[p]) > 0) for p in PIECES},
        flange_strip_present=bool(members.get(FLANGE)),
        flange_strip_vertices=len(members.get(FLANGE, ())),
        vertices_by_owner_group_set={"|".join(k): c for k, c in Counter(tuple(sorted(set(owners[i]))) for i in range(len(V))).items()},
        faces_with_vertex_outside_pieces_or_flange=faces_outside,
        faces_spanning_more_than_one_piece=spanning,
        faces_by_piece_set={"|".join(k): c for k, c in face_pieces.most_common()},
        geometry_outside_named_pieces=bool(faces_outside),
        artwork_material_slots=[m for m in mat_names if m and "artwork" in m.lower()],
        artwork_faces=0,
        artwork_note="no artwork material slot, no artwork vertex group; every face's vertices belong to a named piece or the flange strip",
    ),
    bounds_m=dict(min=[round(float(x), 5) for x in V.min(0)], max=[round(float(x), 5) for x in V.max(0)]),
)
inv["sha256_after"] = sha(SHELL)
inv["unchanged"] = inv["sha256_before"] == inv["sha256_after"]
(HERE / "inventory.json").write_text(json.dumps(inv, indent=1), encoding="utf-8")
print(json.dumps({k: inv[k] for k in ("sha256_before", "sha256_after", "unchanged")}, indent=1))
print("vertices", inv["object_BS_SLV_001"]["vertices"], "faces", inv["object_BS_SLV_001"]["faces"], "boundary loops", len(loop_info),
      "nonmanifold", len(nonmanifold), "loose", len(loose), "faces_outside", faces_outside, "spanning", spanning)
