import os
import sys, json
import numpy as np
sys.path.insert(0, "02_engine_3d/prototypes/P2_clearance_field")
import clearance_field as CF

Z = np.load("02_engine_3d/avatar_engine/baseline/capture_current_walk.npz")
RM = np.load("02_engine_3d/avatar_engine/diagnostics/fbl_003/rest_meshes.npz")
R = os.path.normpath(os.path.expanduser('~/Desktop/COLAB_RESULTS-20261114T105006Z-1-001/COLAB_RESULTS/clearance'))

rest_body = RM["rest_mannequin"]
print("rest_mannequin vs capture rest_body max diff (mm):", np.abs(rest_body - Z["rest_body"]).max() * 1000.0)

V, T, _ = CF.weld(rest_body, Z["tri"])
S = CF.TriSurface(V, T, "rest_body")

def chunked_closest(surf, pts, step=2000):
    return np.concatenate([surf.closest_point(pts[i:i+step])["distance"] for i in range(0, len(pts), step)])

def chunked_winding(surf, pts, step=2000):
    return np.concatenate([surf.winding_number(pts[i:i+step]) for i in range(0, len(pts), step)])

report = {"garments": {}}
for g in ["G0_jacket", "G0_pants"]:
    P_rest = RM[f"rest_{g}"]
    f61_idx = np.load(f"{R}/inside_f061_gpu.npz")[f"{g}_index"]

    d_all = chunked_closest(S, P_rest)
    cand = np.where(d_all < 0.15)[0]
    w_cand = chunked_winding(S, P_rest[cand])
    ins_idx = cand[w_cand > 0.5]

    d_ins = d_all[ins_idx] * 1000.0 if len(ins_idx) else np.array([0.0])
    q = np.quantile(d_ins, [0.5, 0.9, 1.0])
    shared = len(set(ins_idx.tolist()) & set(f61_idx.tolist()))
    walk_only = len(set(f61_idx.tolist()) - set(ins_idx.tolist()))

    print(f"\n=== {g} AT REST ===")
    print(f"  Rest inside count: {len(ins_idx)} | Frame 61 inside: {len(f61_idx)} | Shared static: {shared} | Walk-induced only: {walk_only}")
    print(f"  Rest depth (mm): med={q[0]:.2f}, p90={q[1]:.2f}, max={q[2]:.2f} | >5mm={int((d_ins>5).sum())}, >10mm={int((d_ins>10).sum())}")
    if len(ins_idx):
        j = d_ins.argmax()
        print(f"  Deepest rest point: {np.round(P_rest[ins_idx[j]], 3).tolist()} ({d_ins[j]:.1f} mm)")

    report["garments"][g] = {
        "rest_inside_count": int(len(ins_idx)),
        "f61_inside_count": int(len(f61_idx)),
        "shared_static_count": int(shared),
        "walk_induced_only_count": int(walk_only),
        "rest_depth_mm": {"median": float(q[0]), "p90": float(q[1]), "max": float(q[2])},
        "gt_5mm": int((d_ins > 5).sum()),
        "gt_10mm": int((d_ins > 10).sum()),
        "rest_inside_indices": ins_idx.tolist()
    }

with open("02_engine_3d/avatar_engine/diagnostics/fbl_003/rest_report.json", "w") as fp:
    json.dump(report, fp, indent=2)
print("\nSaved 02_engine_3d/avatar_engine/diagnostics/fbl_003/rest_report.json")
