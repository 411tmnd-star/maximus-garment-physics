"""FBL-009a step 3-4: independent re-measurement of POM-04, POM-09, POM-02 and POM-03 on out/BS-SLV-001_shell.blend.

Does not import build_slv.py, pattern_slv.py, or the check script. Reads the base mesh (before the Solidify modifier), positions only,
and the vertex-group ownership. Nothing is saved. Writes poms_independent.json here.
Run:  python -I -B audit_poms.py
"""
import json, hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
import bpy                                   # noqa: E402  (bpy 5.2.3, imported before mathutils)

HERE = Path(__file__).resolve().parent
BS = HERE.parent
SHELL = BS / "out" / "BS-SLV-001_shell.blend"
SPEC = BS / "spec" / "bs_slv_001_spec.json"
CHECK = BS / "out" / "BS-SLV-001_stage1-3_check.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


sha_before = sha(SHELL)
bpy.ops.wm.open_mainfile(filepath=str(SHELL), load_ui=False)
obj = bpy.data.objects["BS-SLV-001"]; me = obj.data
V = np.array([v.co[:] for v in me.vertices], float)
F = [tuple(p.vertices[:]) for p in me.polygons]
gname = {g.index: g.name for g in obj.vertex_groups}
owners = defaultdict(set)
for v in me.vertices:
    for g in v.groups:
        if g.weight > 0: owners[v.index].add(gname[g.group])
N = len(V)

# edges and adjacency (base mesh)
E = set()
for f in F:
    for i in range(3):
        a, b = f[i], f[(i + 1) % 3]; E.add((min(a, b), max(a, b)))
E = np.array(sorted(E))
elen = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
nbr = defaultdict(set)
for a, b in E: nbr[int(a)].add(int(b)); nbr[int(b)].add(int(a))


def in_(v, name): return name in owners[v]


def edges_between(g1, g2):
    out = []
    for a, b in E:
        a, b = int(a), int(b)
        if (in_(a, g1) and in_(b, g2)) or (in_(a, g2) and in_(b, g1)): out.append((a, b))
    return out


def boundary_loop_vertices(group_pred):
    """Vertices on the mesh boundary (edges used by one face) whose owner set satisfies group_pred."""
    from collections import Counter
    ec = Counter()
    for f in F:
        for i in range(3):
            a, b = f[i], f[(i + 1) % 3]; ec[(min(a, b), max(a, b))] += 1
    bd = [e for e, c in ec.items() if c == 1]
    return sorted({v for e in bd for v in e if group_pred(v)}), bd


def graph_path(allowed, src, targets):
    """Shortest path length (3D edge lengths) from src to any target, using only edges with both ends in `allowed`."""
    allowed = sorted(allowed); idx = {v: i for i, v in enumerate(allowed)}
    rows, cols, w = [], [], []
    for a, b in E:
        a, b = int(a), int(b)
        if a in idx and b in idx:
            l = float(np.linalg.norm(V[a] - V[b])); rows += [idx[a], idx[b]]; cols += [idx[b], idx[a]]; w += [l, l]
    G = coo_matrix((w, (rows, cols)), shape=(len(allowed), len(allowed))).tocsr()
    d = dijkstra(G, indices=idx[src])
    tv = [t for t in targets if t in idx]
    best = min(tv, key=lambda t: d[idx[t]])
    return float(d[idx[best]]), best


def refine_mesh(Vr, Fr):
    """One 1-to-4 midpoint subdivision (numpy). Returns new vertex array and face array."""
    Fr = np.asarray(Fr, int)
    ed = np.sort(np.vstack([Fr[:, [0, 1]], Fr[:, [1, 2]], Fr[:, [2, 0]]]), 1)
    ue, inv = np.unique(ed, axis=0, return_inverse=True); inv = inv.reshape(3, -1).T
    mid = (Vr[ue[:, 0]] + Vr[ue[:, 1]]) / 2
    base = len(Vr); Vn = np.vstack([Vr, mid])
    m01, m12, m20 = base + inv[:, 0], base + inv[:, 1], base + inv[:, 2]
    a, b, c = Fr[:, 0], Fr[:, 1], Fr[:, 2]
    Fn = np.vstack([np.c_[a, m01, m20], np.c_[m01, b, m12], np.c_[m20, m12, c], np.c_[m01, m12, m20]])
    return Vn, Fn

def geodesic_refined(allowed, src, targets, levels):
    """Shortest path on the 1-to-4-refined submesh made of faces whose three vertices are all in `allowed`. Returns (length, target vertex id in original numbering)."""
    allowed_arr = np.zeros(N, bool); allowed_arr[list(allowed)] = True
    Fa = np.array([f for f in F if allowed_arr[list(f)].all()], int)
    Vr = V.copy(); Fr = Fa.copy(); ids = np.arange(N)  # original vertex ids for the first level; new vertices map to -1
    src_pos = None
    for _ in range(levels):
        Vr, Fr = refine_mesh(Vr, Fr)
    # the source and targets are original vertices, which keep their index in the refined mesh
    ed = np.sort(np.vstack([Fr[:, [0, 1]], Fr[:, [1, 2]], Fr[:, [2, 0]]]), 1); ue = np.unique(ed, axis=0)
    w = np.linalg.norm(Vr[ue[:, 0]] - Vr[ue[:, 1]], axis=1)
    used = np.unique(ue)
    remap = -np.ones(len(Vr), int); remap[used] = np.arange(len(used))
    G = coo_matrix((np.r_[w, w], (np.r_[remap[ue[:, 0]], remap[ue[:, 1]]], np.r_[remap[ue[:, 1]], remap[ue[:, 0]]])), shape=(len(used), len(used))).tocsr()
    d = dijkstra(G, indices=remap[src])
    tv = [t for t in targets if remap[t] >= 0]
    best = min(tv, key=lambda t: d[remap[t]])
    return float(d[remap[best]]), int(best), int(len(Fr))

# ------------------------------------------------------------------ landmark sets
shoulder_edges = edges_between("front", "yoke") + edges_between("yoke", "back_p") + edges_between("yoke", "back_n")
shoulder_seam = sorted({v for e in shoulder_edges for v in e})
flange = [v for v in range(N) if in_(v, "flange_strip")]
flange_on_shoulder = sorted({u for v in shoulder_seam for u in nbr[v] if in_(u, "flange_strip")})
collar_top, _ = boundary_loop_vertices(lambda v: in_(v, "collar") or in_(v, "rib"))
collar_top = [v for v in collar_top if in_(v, "collar") or (in_(v, "rib") and not in_(v, "gusset_p") and not in_(v, "gusset_n"))]
# neck opening = collar top loop (boundary of the collar piece). Use boundary edges of the collar owner only.
collar_bd = set()
from collections import Counter
ec = Counter()
for f in F:
    for i in range(3):
        a, b = f[i], f[(i + 1) % 3]; ec[(min(a, b), max(a, b))] += 1
bd_edges = [e for e, c in ec.items() if c == 1]
neck_loop = sorted({v for a, b in bd_edges for v in (a, b) if in_(v, "collar")})
# shoulder-seam join points on the neck loop: neck-loop vertices with a neighbour in front or yoke
join = [v for v in neck_loop if any(in_(u, "front") or in_(u, "yoke") for u in nbr[v])]
def boundary_components():
    from collections import Counter as _C
    ec2 = _C()
    for f in F:
        for i in range(3):
            a, b = f[i], f[(i + 1) % 3]; ec2[(min(a, b), max(a, b))] += 1
    bdl = [e for e, c in ec2.items() if c == 1]
    nbb = defaultdict(set)
    for a, b in bdl: nbb[a].add(b); nbb[b].add(a)
    seen_, comps_ = set(), []
    for s0 in list(nbb):
        if s0 in seen_: continue
        stack, comp = [s0], set()
        while stack:
            x = stack.pop()
            if x in comp: continue
            comp.add(x); stack.extend(nbb[x])
        seen_ |= comp; comps_.append(comp)
    return comps_
_comps = boundary_components()
hem_comp = max(_comps, key=lambda c: sum(1 for v in c if in_(v, "front")))     # the hem: the loop carrying the front hem
hem_front = sorted(v for v in hem_comp if in_(v, "front"))
hem_back = sorted(v for v in hem_comp if in_(v, "back_p") or in_(v, "back_n"))

out = dict(
    file=str(SHELL.relative_to(BS.parents[1])).replace("\\", "/"), sha256_before=sha_before,
    sets=dict(shoulder_seam_vertices=len(shoulder_seam), flange_on_shoulder=len(flange_on_shoulder), neck_loop_vertices=len(neck_loop),
              neck_join_vertices=len(join), hem_front_vertices=len(hem_front), hem_back_vertices=len(hem_back)),
)

# ------------------------------------------------------------------ POM-04 across shoulders
# Outer shoulder points = the highest point of each armhole loop (boundary loop carrying the flange strip): the flange raw edge at the shoulder seam.
arm_loops = [c for c in _comps if any(in_(v, "flange_strip") for v in c)]
arm_top = [max(c, key=lambda v: V[v, 2]) for c in arm_loops]
arm_top.sort(key=lambda v: V[v, 0])
iL, iR = arm_top[0], arm_top[-1]
pom04 = float(np.linalg.norm(V[iR] - V[iL]))
flange_x = [v for v in range(N) if in_(v, "flange_strip")]
jL_f = min(flange_x, key=lambda v: V[v, 0]); jR_f = max(flange_x, key=lambda v: V[v, 0])
out["POM-04_across_shoulders"] = dict(
    method="straight line between the two outer shoulder points: the highest vertex of each armhole boundary loop (flange raw edge at the shoulder seam)",
    left_vertex=iL, right_vertex=iR, left_xyz_mm=[round(float(c) * 1e3, 2) for c in V[iL]], right_xyz_mm=[round(float(c) * 1e3, 2) for c in V[iR]],
    value_m=round(pom04, 5), armhole_loops=len(arm_loops),
    sensitivity_flange_max_abs_x_m=round(float(np.linalg.norm(V[jR_f] - V[jL_f])), 5))

# ------------------------------------------------------------------ POM-09 neck opening, inside width at the shoulder seam
# The front shoulder seam (front-yoke edges) runs from the neck to the outer shoulder. Its neck end on each side is the seam vertex
# nearest the neck opening (collar top loop) on that side: the collar join at the high point of the shoulder.
front_yoke_seam = sorted({v for e in edges_between("front", "yoke") for v in e})
neck_xyz = V[neck_loop]
def neck_end(side):
    cand = [v for v in front_yoke_seam if (V[v, 0] < 0 if side == "L" else V[v, 0] > 0)]
    dd = {v: float(np.min(np.linalg.norm(neck_xyz - V[v], axis=1))) for v in cand}
    return min(cand, key=lambda v: dd[v])
nL, nR = neck_end("L"), neck_end("R")
pom09 = float(np.linalg.norm(V[nR] - V[nL]))
out["POM-09_neck_opening_width_inside"] = dict(
    method="straight line between the two shoulder-seam points where the front-yoke shoulder seam meets the collar inner edge (neck end of each front-yoke seam, nearest the collar top loop)",
    left_vertex=nL, right_vertex=nR, left_xyz_mm=[round(float(c) * 1e3, 2) for c in V[nL]], right_xyz_mm=[round(float(c) * 1e3, 2) for c in V[nR]],
    value_m=round(pom09, 5), neck_loop_vertices=len(neck_loop))

# ------------------------------------------------------------------ POM-02 / POM-03 surface path lengths from the high-point collar join
# High point of shoulder = the neck end of the shoulder seam on each side (the shoulder seam is highest there). Path lengths are
# computed from each side's HPS and then averaged; the two sides are reported separately.
# centre-front and centre-back hem points: the hem vertex with the smallest |x| (ties: the most front / most back)
CF_FRONT = min(hem_front, key=lambda v: (abs(V[v, 0]), V[v, 1]))
CF_BACK = min(hem_back, key=lambda v: (abs(V[v, 0]), -V[v, 1]))
res = {}; LEVELS = (0, 1, 2)
for side, HPS in (("L", nL), ("R", nR)):
    front_allowed = {v for v in range(N) if in_(v, "front") or in_(v, "collar")} | {HPS}
    back_allowed = {v for v in range(N) if in_(v, "back_p") or in_(v, "back_n") or in_(v, "yoke") or in_(v, "collar")} | {HPS}
    res[side] = {}
    for lv in LEVELS:
        lf_, tf_, nf_ = geodesic_refined(front_allowed, HPS, [CF_FRONT], lv)
        lb_, tb_, nb_ = geodesic_refined(back_allowed, HPS, [CF_BACK], lv)
        res[side][lv] = dict(front=(lf_, tf_, nf_), back=(lb_, tb_, nb_))
        nlf, ntf, _ = geodesic_refined(front_allowed, HPS, hem_front, lv)
        nlb, ntb, _ = geodesic_refined(back_allowed, HPS, hem_back, lv)
        res[side][("nearest", lv)] = dict(front=(nlf, ntf, 0), back=(nlb, ntb, 0))
conv = {lv: dict(front_mean_m=round((res["L"][lv]["front"][0] + res["R"][lv]["front"][0]) / 2, 5),
                 back_mean_m=round((res["L"][lv]["back"][0] + res["R"][lv]["back"][0]) / 2, 5)) for lv in LEVELS}
near = {lv: dict(front_mean_m=round((res["L"][("nearest", lv)]["front"][0] + res["R"][("nearest", lv)]["front"][0]) / 2, 5),
                 back_mean_m=round((res["L"][("nearest", lv)]["back"][0] + res["R"][("nearest", lv)]["back"][0]) / 2, 5)) for lv in LEVELS}
LV = LEVELS[-1]
lf = (res["L"][LV]["front"][0] + res["R"][LV]["front"][0]) / 2; lb = (res["L"][LV]["back"][0] + res["R"][LV]["back"][0]) / 2
hps_desc = {side: dict(hps_vertex=HPS, hps_xyz_mm=[round(float(c) * 1e3, 2) for c in V[HPS]],
                       front_path_m=round(res[side][LV]["front"][0], 5), front_hem_vertex=res[side][LV]["front"][1],
                       back_path_m=round(res[side][LV]["back"][0], 5), back_hem_vertex=res[side][LV]["back"][1]) for side, HPS in (("L", nL), ("R", nR))}
out["POM-02_front_length_hps"] = dict(
    method=f"shortest surface path (Dijkstra, 3D lengths) on the front panel plus collar after {LV} midpoint subdivision levels, from each side's high-point collar join to the centre-front hem vertex (smallest |x| on the front hem); value = mean of the two sides",
    convergence=conv, nearest_hem_sensitivity=near[LV],
    sides=hps_desc, value_m=round(lf, 5), side_difference_mm=round(abs(res["L"][LV]["front"][0] - res["R"][LV]["front"][0]) * 1e3, 2))
out["POM-03_back_length_hps"] = dict(
    method=f"shortest surface path (Dijkstra, 3D lengths) on the back panels, yoke and collar after {LV} midpoint subdivision levels, from the same high-point collar joins to the centre-back hem vertex (smallest |x| on the back hem); value = mean of the two sides",
    convergence=conv, nearest_hem_sensitivity=near[LV],
    sides=hps_desc, value_m=round(lb, 5), side_difference_mm=round(abs(res["L"][LV]["back"][0] - res["R"][LV]["back"][0]) * 1e3, 2))

# ------------------------------------------------------------------ compare with spec and with the build's check file
spec = json.loads(SPEC.read_text(encoding="utf-8"))["poms"]
chk = json.loads(CHECK.read_text(encoding="utf-8"))["built_poms"]
for key, sk, ck in (("POM-04_across_shoulders", "POM-04_across_shoulders", "POM-04_across_shoulders"),
                    ("POM-09_neck_opening_width_inside", "POM-09_neck_opening_width_inside", "POM-09_neck_opening_width_inside"),
                    ("POM-02_front_length_hps", "POM-02_front_length_hps", "POM-02_front_length_hps"),
                    ("POM-03_back_length_hps", "POM-03_back_length_hps", "POM-03_back_length_hps")):
    s = spec[sk]; spec_v = s["value"]; tol = s["tol"]
    v = out[key]["value_m"]
    out[key].update(spec_m=spec_v, tol_m=tol, indep_minus_spec_mm=round((v - spec_v) * 1e3, 2), indep_within_tol=abs(v - spec_v) <= tol + 1e-12,
                    build_check_value_m=chk[ck]["built_3d"], build_check_method=chk[ck]["method"],
                    build_check_diff_mm=chk[ck]["diff_mm"], build_check_tol_mm=chk[ck]["tol_mm"],
                    indep_minus_build_check_mm=round((v - chk[ck]["built_3d"]) * 1e3, 2))
out["sha256_after"] = sha(SHELL)
out["unchanged"] = out["sha256_before"] == out["sha256_after"]
(HERE / "poms_independent.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
for k in ("POM-04_across_shoulders", "POM-09_neck_opening_width_inside", "POM-02_front_length_hps", "POM-03_back_length_hps"):
    print(k, out[k]["value_m"], "spec", out[k]["spec_m"], "indep-spec mm", out[k]["indep_minus_spec_mm"], "build", out[k]["build_check_value_m"], "indep-build mm", out[k]["indep_minus_build_check_mm"])
print("sets", out["sets"], "unchanged", out["unchanged"])
