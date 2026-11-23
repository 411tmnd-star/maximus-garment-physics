"""FBL-003 step 2 (numpy only): inside sets, depths, regions, clusters for REST and frame 61, from states.npz.

Rule: inside = generalised winding number > 0.5 on the position-welded body (fbl_004/fast_clearance.py, exact, float64).
Depth = exact unsigned distance to the body surface (TriSurface.closest_point) for inside vertices only.
Body region = dominant vertex group (largest weight) of the FIRST vertex of the nearest body triangle.
Garment group = dominant vertex group of the garment vertex.  Near-surface = depth < 0.1 mm (kept in all counts, flagged).
Armpit band = nearest triangle's bind-pose centroid z in (1.25, 1.40) m.
Run:  python -I -B analyze.py [--workers 3]
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
FBL4 = HERE.parent / "fbl_004"
sys.path.insert(0, str(FBL4))
import fast_clearance as FC                          # noqa: E402
CF = FC.CF
AE = HERE.parents[1]
GARMENTS = ["G0_jacket", "G0_pants"]
EXPECT_61 = {"G0_jacket": 424, "G0_pants": 600}
NEAR_MM = 0.1


def state_surface(body, tri):
    V, T, inv = CF.weld(body.astype(float), tri)
    T2 = inv[tri]; ok = (T2[:, 0] != T2[:, 1]) & (T2[:, 1] != T2[:, 2]) & (T2[:, 0] != T2[:, 2])
    surf = CF.TriSurface(V, T, "body")
    assert surf.n_degenerate == 0                    # so surf triangle j == original triangle np.nonzero(ok)[0][j]
    return V, T, surf, np.nonzero(ok)[0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=3); a = ap.parse_args()
    S = np.load(HERE / "states.npz", allow_pickle=False)
    tri = S["body_tri"]; bind = S["body_bind_co"]; gnames = S["body_group_names"]; bgi = S["body_group_idx"]
    cz = bind[tri].mean(1)[:, 2]
    Zc = np.load(AE / "baseline/capture_current_walk.npz")
    out = {}; all_records = {}
    for state in ["rest", "f061"]:
        body = S[f"{state}_body"]; V, T, surf, orig = state_surface(body, tri)
        facts = {k: surf.facts[k] for k in ("closed", "manifold", "boundary_edges", "non_manifold_edges", "consistently_oriented")}
        if not facts["closed"]:
            raise SystemExit(f"STOP: body not closed in state {state}")
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(a.workers, initializer=FC._wk_init, initargs=(V, T)) if a.workers > 1 else None
        st = dict(body_facts=facts, garments={})
        for g in GARMENTS:
            P = S[f"{state}_{g}"].astype(float); ck = HERE / f"winding_{state}_{g}.npz"
            if ck.exists():
                wn = np.load(ck)["wn"]
            else:
                t0 = time.time()
                if pool is not None: wn, _, _ = FC.winding_parallel(pool, a.workers, P, FC.DEFAULT_CHUNK_PAIRS, len(surf.T))
                else: wn, _ = FC.winding_chunked(surf, P, FC.DEFAULT_CHUNK_PAIRS, time.time() + 1e9, [])
                np.savez(ck, wn=wn); print(state, g, f"{time.time() - t0:.0f}s", flush=True)
            inside = wn > 0.5; idx = np.nonzero(inside)[0]
            cp = surf.closest_point(P[idx]); depth_mm = cp["distance"] * 1e3
            otri = orig[cp["tri"]]                                         # index into the body's original triangle list
            reg_i = bgi[tri[otri, 0]]; assert (reg_i >= 0).all(), "inside vertex with empty body region"
            region = gnames[reg_i]; ggi = S[f"{g}_group_idx"]; gg = S[f"{g}_group_names"]
            gv = np.where(ggi[idx] >= 0, gg[np.clip(ggi[idx], 0, None)], "NONE")
            band = (cz[otri] > 1.25) & (cz[otri] < 1.40)
            E = S[f"{g}_edges"]; keep = inside[E[:, 0]] & inside[E[:, 1]]
            n = len(P); m = coo_matrix((np.ones(keep.sum()), (E[keep, 0], E[keep, 1])), shape=(n, n))
            ncomp, lab = connected_components(m, directed=False); lab_in = lab[idx]
            comp_ids, comp_inv = np.unique(lab_in, return_inverse=True)
            clusters = []
            for c in range(len(comp_ids)):
                s = comp_inv == c; d = depth_mm[s]; pos = P[idx][s]
                reg = {str(k): int(v) for k, v in zip(*np.unique(region[s], return_counts=True))}; grp = {str(k): int(v) for k, v in zip(*np.unique(gv[s], return_counts=True))}
                clusters.append(dict(garment=g, state=state, size=int(s.sum()), max_depth_mm=round(float(d.max()), 3), median_depth_mm=round(float(np.median(d)), 3),
                                     body_regions=reg, garment_groups=grp, centroid_m=[round(float(x), 4) for x in pos.mean(0)],
                                     near_surface=int((d < NEAR_MM).sum()), armpit_band=int(band[s].sum()), vertices=[int(x) for x in idx[s]]))
            clusters.sort(key=lambda c: -c["size"])
            recs = [dict(garment=g, vertex=int(idx[j]), depth_mm=round(float(depth_mm[j]), 4), body_triangle=int(otri[j]), body_region=str(region[j]), garment_group=str(gv[j]),
                         world_m=[round(float(x), 5) for x in P[idx[j]]], near_surface=bool(depth_mm[j] < NEAR_MM), nearest_in_armpit_band=bool(band[j]), cluster=int(comp_inv[j])) for j in range(len(idx))]
            all_records.setdefault(state, []).extend(recs)
            if state == "f061":
                assert len(idx) == EXPECT_61[g], f"STOP: frame 61 {g} inside {len(idx)} != {EXPECT_61[g]}"
            st["garments"][g] = dict(points=n, inside=int(len(idx)), inside_excluding_near_surface=int((depth_mm >= NEAR_MM).sum()), near_surface=int((depth_mm < NEAR_MM).sum()),
                                     depth_mm=dict(min=round(float(depth_mm.min()), 4), median=round(float(np.median(depth_mm)), 4), p90=round(float(np.percentile(depth_mm, 90)), 3), max=round(float(depth_mm.max()), 3)) if len(idx) else None,
                                     clusters=len(clusters), clusters_sum=int(sum(c["size"] for c in clusters)), nearest_in_armpit_band=int(band.sum()), _clusters=clusters)
            assert st["garments"][g]["clusters_sum"] == len(idx)
        if pool is not None: pool.close(); pool.join()
        out[state] = st
    cmp = {}
    for g in GARMENTS:
        r = {x["vertex"] for x in all_records["rest"] if x["garment"] == g}; f = {x["vertex"] for x in all_records["f061"] if x["garment"] == g}
        cmp[g] = dict(inside_in_both=len(r & f), only_at_rest=len(r - f), only_at_61=len(f - r), rest_total=len(r), f061_total=len(f))
    tables = {}
    for state in out:
        t = {}
        for x in all_records[state]:
            d = t.setdefault(x["garment"], {}).setdefault(x["garment_group"], {}); d[x["body_region"]] = d.get(x["body_region"], 0) + 1
        tables[state] = t
    f61 = all_records["f061"]; rest_set = {(x["garment"], x["vertex"]) for x in all_records["rest"]}
    for x in f61: x["inside_at_rest_too"] = (x["garment"], x["vertex"]) in rest_set
    cats = {}
    for g in GARMENTS:
        xs = [x for x in f61 if x["garment"] == g]
        cats[g] = dict(total=len(xs), inside_at_rest_and_61=sum(x["inside_at_rest_too"] for x in xs), only_at_61=sum(not x["inside_at_rest_too"] for x in xs),
                       near_surface=sum(x["near_surface"] for x in xs), nearest_in_armpit_band=sum(x["nearest_in_armpit_band"] for x in xs),
                       only_at_61_and_not_near_surface=sum((not x["inside_at_rest_too"]) and not x["near_surface"] for x in xs))
    selfp = dict(f061_self_pairs_head_armpit_other=[int(x) for x in Zc["self_pairs"][60]], rest_self_pairs_head_armpit_other=[int(x) for x in Zc["rest_self_pairs"]])
    clusters = {s: [c for g in GARMENTS for c in out[s]["garments"][g].pop("_clusters")] for s in out}
    (HERE / "inside_rest.json").write_text(json.dumps(dict(state="rest", body_facts=out["rest"]["body_facts"], garments=out["rest"]["garments"], records=all_records["rest"]), indent=1))
    (HERE / "inside_f061.json").write_text(json.dumps(dict(state="frame 61", body_facts=out["f061"]["body_facts"], garments=out["f061"]["garments"], records=f61), indent=1))
    (HERE / "clusters.json").write_text(json.dumps(clusters, indent=1))
    (HERE / "summary.json").write_text(json.dumps(dict(totals={s: {g: out[s]["garments"][g]["inside"] for g in GARMENTS} for s in out}, rest_vs_f061=cmp, categories_f061=cats,
                                                       self_intersection_pairs=selfp, body_facts={s: out[s]["body_facts"] for s in out}, tables_garment_group_x_body_region=tables,
                                                       capture_crosscheck=json.loads(str(S["capture_crosscheck"])), preserve_volume=json.loads(str(S["armature_preserve_volume"]))), indent=1))
    print(json.dumps(dict(totals={s: {g: out[s]["garments"][g]["inside"] for g in GARMENTS} for s in out}, rest_vs_f061=cmp, categories_f061=cats), indent=1))


if __name__ == "__main__":
    main()
