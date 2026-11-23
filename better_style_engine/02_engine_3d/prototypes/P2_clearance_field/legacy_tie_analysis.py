"""legacy_tie_analysis.py  why the legacy sign test (nearest-face normal) gives 142 / 412 in the engine and 141 / 548 in clearance_field's 'face_normal' rule.
Hypothesis: when the closest point lies on an edge or vertex, several faces are equally near and their normals can disagree in sign; the count then depends on which face the
search returns. Test: (1) reproduce the engine's numbers with the engine's own search (mathutils BVHTree.find_nearest) on the same arrays; (2) for every query, evaluate the sign
under EVERY face that ties for nearest, and report the range of counts any tie-break could produce.
  python -I legacy_tie_analysis.py"""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); import clearance_field as CF
import bpy  # noqa: F401  (the bpy module provides mathutils)
from mathutils import Vector
from mathutils.bvhtree import BVHTree
ARR = HERE.parent / "_inputs/arrays"; OUT = HERE / "out"; TIE = 1e-7                       # metres; float32 resolution at 1 m is 6e-8
b = np.load(ARR / "g0_master__M2_Baked_ThetaFit.npz"); V, T = b["V"], b["tris"]; S = CF.TriSurface(V, T, "body as stored"); Vw, Tw, _ = CF.weld(V, T); SW = CF.TriSurface(Vw, Tw, "welded")
bvh = BVHTree.FromPolygons([tuple(p) for p in V.tolist()], [tuple(int(i) for i in t) for t in T.tolist()], all_triangles=True)
res = dict(tie_tolerance_m=TIE, reference=dict(jacket=142, pants=412))
for name in ("G0_jacket_bracket", "G0_pants"):
    g = np.load(ARR / f"g0_master__{name}.npz"); nf = int(g["prop_n_nylon_faces"]); idx = np.unique(g["loops"][:int(g["face_sizes"][:nf].sum())]); P = g["V"][idx]; n = len(P)
    ins_bvh = np.zeros(n, bool); d_bvh = np.zeros(n)
    for k in range(n):
        loc, nrm, fi, dist = bvh.find_nearest(Vector(P[k])); ins_bvh[k] = (Vector(P[k]) - loc).dot(nrm) < 0; d_bvh[k] = dist
    cp = S.closest_point(P); lists = S.tree.query_ball_point(P, cp["distance"] + TIE + S.rmax); n_tie = np.zeros(n, int); any_in = np.zeros(n, bool); all_in = np.zeros(n, bool)
    for k in range(n):
        c = np.array(lists[k], int); q, _ = CF._closest_on_triangles(np.repeat(P[k][None], len(c), 0), S.A[c], S.B[c], S.C[c]); d = np.linalg.norm(q - P[k], axis=1); tie = d <= cp["distance"][k] + TIE
        sg = ((P[k] - q[tie]) * S.FN[c[tie]]).sum(1) < 0; n_tie[k] = int(tie.sum()); any_in[k] = sg.any(); all_in[k] = sg.all()
    ours = S.signed_distance(P, "face_normal", cp=cp)["inside"]; pn = SW.signed_distance(P, "pseudonormal")["inside"]; amb = any_in & ~all_in
    res[name] = dict(points=n, engine_search_bvh_find_nearest_inside=int(ins_bvh.sum()), max_abs_distance_bvh_vs_exact_m=float(np.abs(d_bvh - cp["distance"]).max()),
        clearance_field_face_normal_inside=int(ours.sum()), points_with_tied_nearest_faces=int((n_tie > 1).sum()), sign_ambiguous_points=int(amb.sum()),
        inside_under_every_tied_face=int(all_in.sum()), inside_under_any_tied_face=int(any_in.sum()), pseudonormal_inside=int(pn.sum()),
        bvh_vs_ours_disagreements=int((ins_bvh != ours).sum()), disagreements_that_are_sign_ambiguous=int(((ins_bvh != ours) & amb).sum()),
        bvh_inside_within_range=bool(all_in.sum() <= ins_bvh.sum() <= any_in.sum()), pseudonormal_resolves_ambiguous=dict(inside=int(pn[amb].sum()), outside=int((~pn[amb]).sum())),
        feature_of_ambiguous={k: int((cp["feature"][amb] == i).sum()) for i, k in enumerate(("face", "edge", "vertex"))})
    print(name, json.dumps(res[name]), flush=True)
json.dump(res, open(OUT / "legacy_tie_analysis.json", "w"), indent=1)
