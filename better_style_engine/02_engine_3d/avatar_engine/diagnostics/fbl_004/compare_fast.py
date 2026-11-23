"""FBL-004 check: fast_clearance checkpoints against the verbatim reference and the packet counts.

Writes compare_fast.json. Run: python -I -B compare_fast.py
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EXPECT = {61: (424, 600), 69: (425, 591), 76: (425, 587), 84: (425, 596), 91: (424, 599)}

ref = np.load(HERE / "ref_inside_f061.npz")
out = {"reference": "ref_inside_f061.npz (verbatim path)", "frames": {}}
ok_all = True
for fr, (ej, ep) in EXPECT.items():
    f = HERE / f"inside_f{fr:03d}.npz"
    if not f.exists():
        out["frames"][str(fr)] = {"status": "not run"}; ok_all = False; continue
    z = np.load(f, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    cj, cp = int(z["G0_jacket_inside"].sum()), int(z["G0_pants_inside"].sum())
    row = dict(jacket=cj, pants=cp, expected=[ej, ep], counts_match=(cj == ej and cp == ep),
               wall_seconds=meta["wall_seconds"], peak_working_set_mb=meta["peak_working_set_mb"],
               garment_chunk_means_by_quarter={g: meta["garments"][g]["chunk_seconds_mean_by_quarter"] for g in meta["garments"]},
               body_facts=meta["body_facts"])
    if fr == 61:
        same_j = np.array_equal(z["G0_jacket_index"], ref["jacket_index"])
        same_p = np.array_equal(z["G0_pants_index"], ref["pants_index"])
        row.update(index_set_identical_to_reference=bool(same_j and same_p),
                   max_abs_winding_diff_vs_reference=float(max(np.abs(z["G0_jacket_winding"] - ref["jacket_winding"]).max(),
                                                               np.abs(z["G0_pants_winding"] - ref["pants_winding"]).max())))
        ok_all = ok_all and same_j and same_p
    ok_all = ok_all and row["counts_match"]
    out["frames"][str(fr)] = row
out["all_pass"] = bool(ok_all)
(HERE / "compare_fast.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k != "frames"}, indent=1))
for fr, r in out["frames"].items(): print(fr, r.get("jacket"), r.get("pants"), r.get("counts_match"), r.get("wall_seconds"), r.get("index_set_identical_to_reference"))
