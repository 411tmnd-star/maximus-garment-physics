"""compare_existing.py  run the clearance library on the tracksuit master and compare with the existing metric (probe_track_v2 / eval_g0 / REQ-02).
Existing reference numbers: sign test (legacy) 142 jacket / 412 pants inside; REQ-02 winding-number test 213 / 578.
  python -I compare_existing.py"""
import sys, json, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); import clearance_field as CF
ARR = HERE.parent / "_inputs/arrays"; OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
ext = json.load(open(ARR / "extraction_record.json"))["g0_master"]
res = dict(input=dict(path=ext["path"], artifact_sha256=ext["sha256_before"], unchanged_after_read=ext["unchanged"]), units="metres", frame="Blender world, Z-up",
           reference=dict(legacy_sign_test=dict(jacket=142, pants=412), req02_winding=dict(jacket=213, pants=578)))
b = np.load(ARR / "g0_master__M2_Baked_ThetaFit.npz"); raw = CF.TriSurface(b["V"], b["tris"], "body as stored (14,517 vertices)")
Vw, Tw, _ = CF.weld(b["V"], b["tris"]); welded = CF.TriSurface(Vw, Tw, "body welded by exact position")
res["body_facts"] = dict(raw=raw.facts, welded=welded.facts)
t = time.time(); sc = welded.self_contact(0.002); sc5 = welded.self_contact(0.005); res["body_self_contact"] = dict(eps_2mm_vertices=sc["n_flagged"], eps_5mm_vertices=sc5["n_flagged"], seconds=round(time.time() - t, 1),
    z_range_2mm=[float(Vw[sc["flag"]][:, 2].min()), float(Vw[sc["flag"]][:, 2].max())] if sc["n_flagged"] else None)
tri_contact = sc5["flag"][welded.T].any(1)
for name in ("G0_jacket_bracket", "G0_pants"):
    g = np.load(ARR / f"g0_master__{name}.npz"); nf = int(g["prop_n_nylon_faces"]); nloops = int(g["face_sizes"][:nf].sum()); idx = np.unique(g["loops"][:nloops]); P = g["V"][idx]; r = dict(shell_vertices=int(len(P)), n_nylon_faces=nf); T0 = time.time()
    cp_raw = raw.closest_point(P); r["t_closest_raw_s"] = round(time.time() - T0, 2)
    leg_raw = raw.signed_distance(P, "face_normal", cp=cp_raw); r["legacy_on_raw_body"] = CF.clearance_report(leg_raw["signed"] * 1e3, limits=(7.0,))["all"]
    T0 = time.time(); cp_w = welded.closest_point(P); r["t_closest_welded_s"] = round(time.time() - T0, 2); r["max_abs_distance_raw_vs_welded_m"] = float(np.abs(cp_raw["distance"] - cp_w["distance"]).max())
    leg = welded.signed_distance(P, "face_normal", cp=cp_w); pn = welded.signed_distance(P, "pseudonormal", cp=cp_w)
    T0 = time.time(); win = welded.signed_distance(P, "winding", cp=cp_w, allow_open=not welded.facts["closed"]); r["t_winding_s"] = round(time.time() - T0, 1); w = win["winding"]
    r["inside_counts"] = dict(legacy_face_normal_raw=int(leg_raw["inside"].sum()), legacy_face_normal_welded=int(leg["inside"].sum()), pseudonormal_welded=int(pn["inside"].sum()), winding_welded=int(win["inside"].sum()))
    r["winding_values"] = dict(min=float(w.min()), max=float(w.max()), frac_integer_within_1e_6=float((np.abs(w - np.rint(w)) < 1e-6).mean()), count_ge_1p5=int((w >= 1.5).sum()),
                               histogram_rounded={str(int(k)): int(c) for k, c in zip(*np.unique(np.rint(w), return_counts=True))})
    dis = win["inside"] != leg["inside"]; r["winding_vs_legacy_disagreements"] = dict(n=int(dis.sum()), winding_only=int((win["inside"] & ~leg["inside"]).sum()), legacy_only=int((leg["inside"] & ~win["inside"]).sum()),
        z_range=[float(P[dis][:, 2].min()), float(P[dis][:, 2].max())] if dis.any() else None, abs_x_range=[float(np.abs(P[dis][:, 0]).min()), float(np.abs(P[dis][:, 0]).max())] if dis.any() else None,
        nearest_body_triangle_in_self_contact_zone_5mm=int(tri_contact[cp_w["tri"]][dis].sum()))
    zone = tri_contact[cp_w["tri"]]; r["by_body_zone_winding_rule"] = CF.clearance_report(win["signed"] * 1e3, labels=np.where(zone, "nearest body point in self-contact zone (5 mm)", "elsewhere"))["by_label"]
    r["feature_of_closest_point"] = {k: int((cp_w["feature"] == i).sum()) for i, k in enumerate(("face", "edge", "vertex"))}
    res[name] = r; print(name, json.dumps(r["inside_counts"]), "winding s", r["t_winding_s"], flush=True)
json.dump(res, open(OUT / "compare_existing_g0.json", "w"), indent=1); print(json.dumps(res["body_facts"], indent=0)[:900]); print(res["body_self_contact"])
