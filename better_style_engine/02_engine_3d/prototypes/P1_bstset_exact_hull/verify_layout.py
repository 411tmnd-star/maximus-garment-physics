"""verify_layout.py  independent deterministic verification of the recorded BS-BSTSET-001 rhinestone layout. Read-only. PROTOTYPE.
  python -I verify_layout.py            (about 3 minutes)
Inputs (all copies made by REQ-03; the run that produced them reproduced the engine's recorded outputs bit for bit):
  assembled_meta.npz   final 1,499 stones            req03_prune.npz   all 1,729 candidates with keep / occluded flags and prune order
  req03_track_curves.npz  dense track curves         chaton_v1.params.json   stone profile          camp_master__*.npz   garment arrays (P4-style extraction)
Policy values (EPS_MIN 0.05 mm, EPS_MAX 0.10 mm, glue offset 0.010 mm) are the engine's own constants, read here as reference numbers. Nothing is approved or changed."""
import sys, json, time, hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
HERE = Path(__file__).resolve().parent; PROT = HERE.parent; WS = PROT.parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(PROT / "P2_clearance_field")); import convex_gap as G, clearance_field as CF
REQ = WS / "stage3_engineering/opus_requests/REQ-03_bst_factor_isolation"; LAB = REQ / "scratch/rl/py"
F = dict(final=REQ / "scratch/out_orig/assembled_meta.npz", prune=REQ / "scratch/runs/F0_instrumented/req03_prune.npz", curves=REQ / "scratch/runs/F0_instrumented/req03_track_curves.npz",
         params=REQ / "scratch/rl/assets/chaton_v1.params.json", lab_exact_hull=LAB / "exact_hull.py", lab_chaton=LAB / "chaton_geometry.py", bst_core=REQ / "scratch/tools_pristine/bst_core.py",
         p3_assemble=REQ / "scratch/tools_pristine/p3_assemble.py", req03_part3=REQ / "part3_metric.json", audit=REQ / "scratch/out_orig/phase3_audit.json")
ARR = PROT / "_inputs/arrays"; OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
MM = 1e-3; EPS_MIN = 0.05 * MM; EPS_MAX = 0.10 * MM; GLUE = 0.010 * MM; T0 = time.time()
def q(a, ps=(0, 1, 5, 50, 95, 99, 100)):
    a = np.asarray(a, float); return {("min" if p == 0 else "max" if p == 100 else f"p{p}"): float(np.percentile(a, p)) for p in ps} | {"n": int(len(a))} if len(a) else {"n": 0}

res = dict(script="verify_layout.py", units="metres in arrays; *_mm fields in millimetres", frame="Blender world, Z-up", inputs={k: dict(path=str(v.relative_to(WS)), sha256=sha(v)) for k, v in F.items()},
           engine_constants_used_as_reference=dict(EPS_MIN_mm=0.05, EPS_MAX_mm=0.10, GLUE_OFFSET_mm=0.010, source="bst_core.py lines 13-16"))
params = json.load(open(F["params"])); VU, TRI, EDG = G.chaton_unit(params)
fin = np.load(F["final"]); pz = np.load(F["prune"]); P, N, T, R = pz["P"], pz["N"], pz["T"], pz["R"]; keep, occl, order = pz["keep"], pz["occl"], pz["order"]
zone = pz["zone"].astype(str); curve = pz["curve"].astype(str); kind = pz["kind"].astype(str); s = pz["s"]; n = len(P)

# ---------------------------------------------------------------- 1. stone model facts
res["stone_model"] = dict(profile=params["name"], octagon_vertices_on_girdle_circle=True, sizes={f"R={r * 1e3:.2f}mm": int(np.isclose(R, r).sum()) for r in np.unique(np.round(R, 9))},
    octagon_apothem_over_radius=float(np.cos(np.pi / 8)), octagon_flat_shortfall_mm={f"R={r * 1e3:.2f}": float(r * (1 - np.cos(np.pi / 8)) * 1e3) for r in np.unique(np.round(R, 9))},
    height_over_diameter=params["total_height"], frames=dict(N_unit_max_err=float(np.abs(np.linalg.norm(N, axis=1) - 1).max()), T_unit_max_err=float(np.abs(np.linalg.norm(T, axis=1) - 1).max()), T_dot_N_max=float(np.abs((N * T).sum(1)).max())))

# ---------------------------------------------------------------- 2. final layout identity: prune dump (keep) == recorded final layout
kd = cKDTree(fin["P_blender"]); dmatch, imatch = kd.query(P[keep])
res["identity"] = dict(candidates=int(n), keep=int(keep.sum()), occluded=int(occl.sum()), rejected_clearance=int((~keep & ~occl).sum()), final_recorded=int(len(fin["R"])),
    keep_positions_match_recorded_final_max_m=float(dmatch.max()), bijection=bool(len(np.unique(imatch)) == keep.sum() == len(fin["R"])),
    normals_match_max=float(np.abs(N[keep] - fin["N_blender"][imatch]).max()), radii_match=bool(np.array_equal(R[keep], fin["R"][imatch])))

# ---------------------------------------------------------------- 3. replay of the engine's prune rule (centre distance minus radii >= EPS_MIN - 1e-9), same order
def replay(gapfun, eps=EPS_MIN - 1e-9, reach=5.0 * MM):
    k = np.zeros(n, bool); tree = cKDTree(P); nb = tree.query_ball_point(P, reach); blockers = {}
    for i in order:
        if occl[i]: continue
        c = [j for j in nb[i] if k[j]]
        g = gapfun(i, c) if c else np.array([np.inf])
        if g.min() >= eps: k[i] = True
        else: blockers[int(i)] = float(g.min())
    return k, blockers
centre_gap = lambda i, c: np.linalg.norm(P[c] - P[i], axis=1) - R[c] - R[i]
k_centre, _ = replay(centre_gap)
res["prune_replay_centre_metric"] = dict(kept=int(k_centre.sum()), rejected=int((~k_centre & ~occl).sum()), identical_to_recorded_keep=bool(np.array_equal(k_centre, keep)),
    rejected_by_zone={z: int((~k_centre & ~occl & (zone == z)).sum()) for z in "ABC"})

# ---------------------------------------------------------------- 4. exact gaps
HULL = [G.hull_world(VU, P[i], N[i], T[i], 2 * R[i]) for i in range(n)]; SUP_H = [G.PolytopeSupport(h) for h in HULL]; SUP_R = [G.round_support(params, P[i], N[i], 2 * R[i]) for i in range(n)]
_cache = {}
def gaps(i, j):
    """(hull upper, hull lower, hull hit, hull feature-pair, round upper, round lower, round hit) for one pair; cached; symmetric."""
    key = (i, j) if i < j else (j, i)
    if key not in _cache:
        a, b = key; hu, hl, _, hh = G.gjk_distance(SUP_H[a], SUP_H[b]); fp = np.nan if hh else G.polytope_distance(HULL[a], HULL[b], TRI, EDG); ru, rl, _, rh = G.gjk_distance(SUP_R[a], SUP_R[b], rel_tol=1e-11)
        _cache[key] = (hu, hl, hh, fp, ru, rl, rh)
    return _cache[key]

fi = np.nonzero(keep)[0]; pr = cKDTree(P[fi]).query_pairs(2 * 2.0 * MM + 0.5 * MM, output_type="ndarray"); pr = fi[pr]              # the same neighbour set the engine audit uses (centre distance < 4.5 mm)
metric = np.linalg.norm(P[pr[:, 0]] - P[pr[:, 1]], axis=1) - R[pr[:, 0]] - R[pr[:, 1]]
Gp = np.array([gaps(i, j) for i, j in pr]); hull_gap = Gp[:, 0]; hull_hit = Gp[:, 2].astype(bool); fp = Gp[:, 3]; round_gap = Gp[:, 4]; round_hit = Gp[:, 6].astype(bool)
ang = np.degrees(np.arccos(np.clip((N[pr[:, 0]] * N[pr[:, 1]]).sum(1), -1, 1)))
def model_block(g, hit):
    return dict(gap_mm=q(g[~hit] / MM), overlapping_pairs=int(hit.sum()), pairs_below_EPS_MIN=int((hit | (g < EPS_MIN - 1e-12)).sum()), pairs_below_0=int(hit.sum()), min_gap_mm=float(0.0 if hit.any() else g.min() / MM),
                stones_in_a_pair_below_EPS_MIN=int(len(np.unique(pr[hit | (g < EPS_MIN - 1e-12)]))), exact_minus_centre_metric_mm=q((g - metric)[~hit] / MM), pairs_where_exact_lt_metric=int(((g < metric - 1e-12) | hit).sum()))
res["final_layout_pairs"] = dict(pairs=int(len(pr)), neighbour_rule="centre distance < 4.5 mm (engine audit rule)", centre_metric=dict(gap_mm=q(metric / MM), pairs_below_EPS_MIN=int((metric < EPS_MIN - 1e-9).sum()), min_gap_mm=float(metric.min() / MM)),
    octagonal_hull=model_block(hull_gap, hull_hit), round_stone=model_block(round_gap, round_hit),
    numerical=dict(gjk_bracket_width_max_m=dict(hull=float((Gp[:, 0] - Gp[:, 1])[~hull_hit].max()), round=float((Gp[:, 4] - Gp[:, 5])[~round_hit].max())),
                   hull_gjk_minus_feature_pair_max_abs_m=float(np.abs(hull_gap - fp)[~hull_hit].max())),
    corr_hull_minus_metric_vs_normal_angle=float(np.corrcoef((hull_gap - metric)[~hull_hit], ang[~hull_hit])[0, 1]), corr_round_minus_metric_vs_normal_angle=float(np.corrcoef((round_gap - metric)[~round_hit], ang[~round_hit])[0, 1]),
    normal_angle_deg=q(ang))
below = np.nonzero(~hull_hit & (hull_gap < EPS_MIN - 1e-12))[0]
res["final_layout_pairs"]["hull_pairs_below_EPS_MIN"] = [dict(i=int(pr[k, 0]), j=int(pr[k, 1]), zones=zone[pr[k, 0]] + zone[pr[k, 1]], curves=[curve[pr[k, 0]], curve[pr[k, 1]]], metric_mm=float(metric[k] / MM), hull_mm=float(hull_gap[k] / MM), round_mm=float(round_gap[k] / MM), normal_angle_deg=float(ang[k])) for k in below]
# tolerance sensitivity: how many final pairs fall below EPS_MIN / touch if every stone is larger in diameter by delta (round model; a uniform radial growth of delta/2 each reduces a gap by about delta)
res["final_layout_pairs"]["diameter_oversize_sensitivity_round_model"] = [dict(oversize_mm=d, pairs_below_EPS_MIN=int((round_hit | (round_gap - d * MM < EPS_MIN - 1e-12)).sum()), pairs_touching_or_overlapping=int((round_hit | (round_gap - d * MM <= 0)).sum())) for d in (0.0, 0.01, 0.025, 0.05, 0.075, 0.10)]
res["final_layout_pairs"]["oversize_note"] = "first-order: gap reduced by the diameter oversize; exact only for coplanar stones. No supplier tolerance is on record (owner decision)."

# ---------------------------------------------------------------- 5. lab module comparison (existing implementation): SAT gap and edge-triangle intersection
sys.path.insert(0, str(LAB)); sys.dont_write_bytecode = True
import exact_hull as EH, chaton_geometry as CGL
st = lambda i: dict(base=P[i], normal=N[i], tangent=T[i], diameter=2 * R[i])
lab_hull_max = max(float(np.abs(np.sort(EH.hull_world(P[i], N[i], T[i], 2 * R[i]).round(12), 0) - np.sort(HULL[i].round(12), 0)).max()) for i in fi[:200])
sat = np.array([EH.separation_exact(st(i), st(j))["gap_m"] for i, j in pr]); inter = np.array([EH.intersects_edges(st(i), st(j)) for i, j in pr])
res["lab_exact_hull_comparison"] = dict(lab_hull_vertices_equal_ours_max_m=lab_hull_max, sat_minus_gjk_hull_mm=q((sat - hull_gap)[~hull_hit] / MM), sat_exceeds_exact_count=int(((sat - hull_gap) > 1e-12)[~hull_hit].sum()),
    intersection_verdicts_agree=bool(np.array_equal(inter, hull_hit)), lab_intersections=int(inter.sum()))

# ---------------------------------------------------------------- 6. the 222 clearance rejects
rank = np.empty(n, int); rank[order] = np.arange(n); rej = np.nonzero(~keep & ~occl)[0]; tree = cKDTree(P); rows = []
for i in rej:
    nb = [j for j in tree.query_ball_point(P[i], 4.5 * MM) if j != i and keep[j] and rank[j] < rank[i]]; gg = np.array([gaps(i, j) for j in nb]); m = np.linalg.norm(P[nb] - P[i], axis=1) - R[nb] - R[i]
    hh = gg[:, 2].astype(bool); rh = gg[:, 6].astype(bool)
    # also against EVERY final stone, including those accepted later in the order
    nb_all = [j for j in tree.query_ball_point(P[i], 4.5 * MM) if j != i and keep[j]]; ga = np.array([gaps(i, j) for j in nb_all])
    rows.append(dict(i=int(i), zone=zone[i], curve=curve[i], kind=kind[i], metric_min_mm=float(m.min() / MM), hull_overlap=bool(hh.any()), hull_min_mm=float(0 if hh.any() else gg[:, 0].min() / MM),
                     round_overlap=bool(rh.any()), round_min_mm=float(0 if rh.any() else gg[:, 4].min() / MM),
                     hull_min_vs_all_final_mm=float(0 if ga[:, 2].any() else ga[:, 0].min() / MM), round_min_vs_all_final_mm=float(0 if ga[:, 6].any() else ga[:, 4].min() / MM)))
def cls(rows, key_o, key_g):
    fine = [r for r in rows if not r[key_o] and r[key_g] >= 0.05 - 1e-9]
    return dict(fine_ge_EPS_MIN=len(fine), fine_by_zone={z: sum(r["zone"] == z for r in fine) for z in "ABC"}, overlap=sum(r[key_o] for r in rows), between_0_and_EPS_MIN=sum((not r[key_o]) and r[key_g] < 0.05 - 1e-9 for r in rows))
res["rejected_222"] = dict(n=len(rows), by_zone={z: sum(r["zone"] == z for r in rows) for z in "ABC"}, definition="each rejected stone against the stones accepted BEFORE it in the engine's prune order (REQ-03 definition)",
    octagonal_hull=cls(rows, "hull_overlap", "hull_min_mm"), round_stone=cls(rows, "round_overlap", "round_min_mm"),
    fine_vs_every_final_stone=dict(octagonal_hull=sum(r["hull_min_vs_all_final_mm"] >= 0.05 - 1e-9 for r in rows), round_stone=sum(r["round_min_vs_all_final_mm"] >= 0.05 - 1e-9 for r in rows)),
    metric_min_mm=q([r["metric_min_mm"] for r in rows]), req03_reference=dict(fine=105, overlap=76, between=41, by_zone=dict(A=40, B=48, C=17)))
res["rejected_rows"] = rows

# ---------------------------------------------------------------- 7. greedy replay with the exact criteria (does the per-stone count survive as a layout?)
def exact_gapfun(col_gap, col_hit):
    def f(i, c):
        gg = np.array([gaps(i, j) for j in c]); return np.where(gg[:, col_hit].astype(bool), -1.0, gg[:, col_gap])
    return f
def voids(k):
    out = {}
    for cv in sorted(set(curve)):
        if "track" not in cv: continue
        idx = np.nonzero((curve == cv) & k)[0]; idx = idx[np.argsort(s[idx], kind="stable")]
        if len(idx) < 2: continue
        g = np.linalg.norm(P[idx[1:]] - P[idx[:-1]], axis=1) - R[idx[1:]] - R[idx[:-1]]                       # open chain; wrap pair reported separately
        out[cv] = dict(stones=int(len(idx)), largest_gap_mm=float(g.max() / MM), gaps_over_EPS_MAX=int((g > EPS_MAX + 1e-9).sum()), gaps_over_1_stone_mm=int((g > 2 * R[idx[0]]).sum()), wrap_gap_mm=float((np.linalg.norm(P[idx[0]] - P[idx[-1]]) - R[idx[0]] - R[idx[-1]]) / MM))
    return out
k_hull, _ = replay(exact_gapfun(0, 2), eps=EPS_MIN - 1e-12); k_round, _ = replay(exact_gapfun(4, 6), eps=EPS_MIN - 1e-12)
def rb(k): return dict(kept=int(k.sum()), rejected=int((~k & ~occl).sum()), rejected_by_zone={z: int((~k & ~occl & (zone == z)).sum()) for z in "ABC"}, recovered_vs_recorded=int((k & ~keep).sum()), lost_vs_recorded=int((keep & ~k).sum()),
                   by_curve_track2={cv: int(((curve == cv) & k).sum()) for cv in sorted(set(curve)) if "track2" in cv})
res["greedy_replay"] = dict(note="same candidates, same order, same EPS_MIN; only the gap measure changes. Occluded stones stay excluded.", recorded_centre_metric=rb(keep) | dict(voids=voids(keep)),
    octagonal_hull=rb(k_hull) | dict(voids=voids(k_hull)), round_stone=rb(k_round) | dict(voids=voids(k_round)))

# ---------------------------------------------------------------- 8. along-track spacing and track-to-track separation (final layout)
at = {}
for cv in sorted(set(curve)):
    idx = np.nonzero((curve == cv) & keep)[0]; idx = idx[np.argsort(s[idx], kind="stable")]
    if len(idx) < 2 or cv == "C_crown_pave": continue
    m = np.linalg.norm(P[idx[1:]] - P[idx[:-1]], axis=1) - R[idx[1:]] - R[idx[:-1]]; pitch = np.linalg.norm(P[idx[1:]] - P[idx[:-1]], axis=1)
    at[cv] = dict(stones=int(len(idx)), consecutive_pairs=int(len(m)), gap_mm=q(m / MM, (0, 50, 100)), in_window_EPS_MIN_EPS_MAX=int(((m >= EPS_MIN - 1e-9) & (m <= EPS_MAX + 1e-9)).sum()), below_EPS_MIN=int((m < EPS_MIN - 1e-9).sum()), above_EPS_MAX=int((m > EPS_MAX + 1e-9).sum()),
                  pitch_mm_p50=float(np.median(pitch) / MM))
res["along_track"] = dict(note="consecutive final stones in s order on each curve, open chain (no wrap pair); gap = centre distance minus radii", curves=at,
    totals=dict(pairs=int(sum(v["consecutive_pairs"] for v in at.values())), in_window=int(sum(v["in_window_EPS_MIN_EPS_MAX"] for v in at.values())), above_EPS_MAX=int(sum(v["above_EPS_MAX"] for v in at.values())), below_EPS_MIN=int(sum(v["below_EPS_MIN"] for v in at.values()))))
tc = np.load(F["curves"]); sep = {}
def d_poly(X, Q):
    A, B = Q[:-1], Q[1:]; t = cKDTree(A); _, idx = t.query(X, k=16); cand = np.concatenate([idx, np.maximum(idx - 1, 0)], 1); a = A[cand]; e = B[cand] - a; w = X[:, None] - a
    tt = np.clip((w * e).sum(-1) / np.maximum((e * e).sum(-1), 1e-30), 0, 1); return np.linalg.norm(w - tt[..., None] * e, axis=-1).min(1)
for lab in ("A_collar", "B_cuff_L", "B_cuff_R"):
    d = d_poly(tc[f"{lab}_track2__Q"], tc[f"{lab}_track1__Q"]); need = 2 * 1.4 * MM + EPS_MIN
    sep[lab] = dict(separation_mm=q(d / MM, (0, 5, 50, 95, 100)), nominal_mm=2.875, minimum_for_EPS_MIN_between_SS10_mm=2.85, share_below_2p85=float((d < need).mean()), curve_crossing_or_touching=bool(d.min() < 1e-6))
res["track_to_track"] = dict(note="dense track-2 curve samples to the dense track-1 polyline (the curves stones are placed on)", zones=sep)

# ---------------------------------------------------------------- 9. stone-to-garment: nearest surface (P2 library), ray layers under the footprint, exterior side
gar = {}; GV = []; GT = []; lab_obj = []; off = 0
for o in ("Tee", "collar", "placket_overlap", "placket_buttons"):
    z = np.load(ARR / f"camp_master__{o}.npz"); GV.append(z["V"]); GT.append(z["tris"] + off); lab_obj += [o] * len(z["tris"]); off += len(z["V"])
GV = np.vstack(GV); GT = np.vstack(GT); lab_obj = np.array(lab_obj)
near = np.zeros(len(GT), bool); tcen = GV[GT].mean(1); hits = cKDTree(tcen).query_ball_point(P, 8.0 * MM); near[np.unique(np.concatenate([np.array(h, int) for h in hits]))] = True      # crop: triangles within 8 mm of any candidate
S = CF.TriSurface(GV, GT[near], "camp shirt master, 4 objects, triangles within 8 mm of a stone"); obj_of = lab_obj[near]
ext = json.load(open(ARR / "extraction_record.json"))["camp_master"]; res["inputs"]["garment_master"] = dict(path=ext["path"], sha256=ext["sha256_before"], unchanged_after_extraction=ext["unchanged"])
cp = S.closest_point(P); fn = S.FN[cp["tri"]]; side = ((P - cp["point"]) * fn).sum(1)
def ray_layers(i, pts):
    """For rays from pts + 2 mm * N along -N (4 mm long): list of (depth above base plane, facing) per hit, per ray. Candidates: triangles near the stone."""
    cand = np.array(S.tree.query_ball_point(P[i], R[i] + 2.5 * MM + S.rmax), int)
    if not len(cand): return [[] for _ in pts]
    a, b, c = S.A[cand], S.B[cand], S.C[cand]; d = -N[i]; o = pts + 2 * MM * N[i]; e1 = b - a; e2 = c - a; h = np.cross(d, e2); det = (e1 * h).sum(1); ok = np.abs(det) > 1e-18; inv = np.where(ok, 1 / np.where(ok, det, 1), 0)
    sv = o[:, None] - a[None]; u = (sv * h[None]).sum(-1) * inv; qv = np.cross(sv, e1[None]); v = (qv @ d) * inv; t = (qv * e2[None]).sum(-1) * inv
    hit = ok[None] & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t >= 0) & (t <= 4 * MM); out = []
    for r in range(len(pts)):
        k = np.nonzero(hit[r])[0]; k = k[np.argsort(t[r, k])]; out.append([(2 * MM - t[r, kk], float(S.FN[cand[kk]] @ N[i]), obj_of[cand[kk]]) for kk in k])
    return out
depth = np.full(n, -np.inf); first_face = np.zeros(n); nlayer = np.zeros(n, int); miss = np.zeros(n, int); first_obj = np.array(["none"] * n, dtype="<U16"); thick = np.full(n, np.nan)
for i in range(n):
    t, nn, z = G.stone_frame(N[i], T[i]); th = np.arange(32) * 2 * np.pi / 32
    pts = np.vstack([P[i][None]] + [P[i] + rr * R[i] * (np.cos(th)[:, None] * t + np.sin(th)[:, None] * z) for rr in (1 / 3, 2 / 3, 1.0)])            # the engine's 97-point footprint grid
    L = ray_layers(i, pts); miss[i] = sum(1 for l in L if not l)
    depth[i] = max((l[0][0] for l in L if l), default=-np.inf)                                                                                       # highest surface above the base plane under the footprint
    c0 = L[0]
    if c0:
        first_face[i] = c0[0][1]; first_obj[i] = c0[0][2]; dz = [x[0] for x in c0]; uniq = [dz[0]] + [b for a, b in zip(dz, dz[1:]) if a - b > 1e-6]; nlayer[i] = len(uniq)
        if len(uniq) > 1: thick[i] = uniq[0] - uniq[1]
def garment_block(m):
    return dict(stones=int(m.sum()), base_centre_to_nearest_surface_mm=q(cp["distance"][m] / MM), base_centre_on_normal_side_of_nearest_face=int((side[m] > 0).sum()),
        angle_stone_normal_to_nearest_face_normal_deg=q(np.degrees(np.arccos(np.clip(np.abs((fn[m] * N[m]).sum(1)), -1, 1))), (50, 95, 100)),
        nearest_object={o: int((obj_of[cp["tri"][m]] == o).sum()) for o in np.unique(obj_of)},
        footprint_97_rays=dict(highest_surface_above_base_plane_mm=q(depth[m][np.isfinite(depth[m])] / MM), stones_with_surface_above_base=int((depth[m] > 0).sum()), stones_with_surface_above_base_gt_0p2mm=int((depth[m] > 0.2 * MM).sum()),
                               stones_with_missed_rays=int((miss[m] > 0).sum()), missed_rays=int(miss[m].sum())),
        centre_ray=dict(first_surface_faces_the_stone=int((first_face[m] > 0).sum()), first_surface_faces_away=int((first_face[m] < 0).sum()), no_hit=int((first_obj[m] == "none").sum()),
                        first_object={o: int((first_obj[m] == o).sum()) for o in np.unique(first_obj[m])}, layers_within_4mm={str(k): int((nlayer[m] == k).sum()) for k in np.unique(nlayer[m])},
                        first_to_second_layer_distance_mm=q(thick[m][np.isfinite(thick[m])] / MM, (0, 5, 50, 95, 100))))
res["stone_to_garment"] = dict(surface=S.facts | dict(objects={o: int((obj_of == o).sum()) for o in np.unique(obj_of)}),
    note="garment = Tee + collar + placket_overlap + placket_buttons as stored in the camp shirt master (open shells; no winding sign is used). 'above base plane' reproduces the engine's dense footprint test: rays from 2 mm above the base along -N.",
    final_1499=garment_block(keep), rejected_clearance_222=garment_block(~keep & ~occl), occluded_8=garment_block(occl),
    engine_recorded_final=json.load(open(F["audit"]))["penetration"], by_zone_final={z: garment_block(keep & (zone == z)) for z in "ABC"},
    compare_engine_dense_depth=dict(note="req03_prune.npz 'dd' is the engine's dense depth BEFORE the re-lift; positions in the dump are AFTER the re-lift, so only the re-lift arithmetic can be checked: expected depth after = dd - (dd + glue) = -glue for lifted stones",
        lifted=int(pz["lift"].sum()), ours_after_lift_mm=q(depth[pz["lift"] & np.isfinite(depth)] / MM, (0, 50, 100)), expected_mm=-0.010))

res["runtime_s"] = round(time.time() - T0, 1); res["pairs_evaluated"] = len(_cache)
res["inputs_sha256_after"] = {k: sha(v) for k, v in F.items()}; res["inputs_unchanged"] = all(res["inputs"][k]["sha256"] == res["inputs_sha256_after"][k] for k in F)
json.dump(res, open(OUT / "verify_layout.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print(json.dumps({k: v for k, v in res.items() if k not in ("rejected_rows", "inputs", "inputs_sha256_after")}, indent=1, default=str))
