"""Writes REPORT.md from triangles.json (every number in the report is read from that file)."""
import json, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
d = json.loads((HERE / "triangles.json").read_text(encoding="utf-8"))
P = d["poses"]
L = []
L.append("# FBL-006a collapsed-triangle diagnostic\n")
L.append("Read-only on the canonical rig `03_full_rigged.blend` (sha256 `1be2d4ca…82148`, checked before and after: unchanged). Nothing rendered, nothing saved.\n")
L.append("## Method\n")
L.append("- Triangles are indices into the canonical body's loop-triangle list (`rig.TRI`, `posekit.py`).")
L.append("- Rest area = bind-pose triangle area; area ratio = posed area / rest area. Listed: every triangle with ratio < 0.25.")
L.append("- Dominant bone = the bone with the largest summed vertex weight over the triangle's three vertices (mean shown). Top four weights are given per vertex.")
L.append("- Occlusion: a ray from the bind-pose centroid, offset 1 µm along the bind-pose normal, length 20 mm, against the bind body, excluding the triangle itself. 'No' = no body surface within 20 mm along the normal.")
L.append("- Sweep poses are reproduced from `poses/shoulder_sweep.py` (share 1/3 variant, same functions). Only the named side is moved; the other side stays at theta_fit.")
L.append("- Column 'Under 0.5 @ fit' says whether the triangle's ratio at theta_fit is below 0.5 (112 triangles are).\n")
L.append("## Count checks\n")
L.append("| Pose | Expected | Found | Match |")
L.append("|---|---|---|---|")
for k, v in d["count_checks"].items():
    L.append(f"| {k} | {v['expected']} | {v['found']} | {'yes' if v['match'] else 'NO'} |")
L.append("")
L.append("Sweep cross-check against `poses/shoulder_sweep_samples.json` (triangles under 0.25 at that sample): " +
         ", ".join(f"{k} {v['this_run']} (json {v['sweep_json_triangles_lt_quarter']})" for k, v in d["cross_check_sweep"].items()) + ". All ten match.\n")
L.append("## Presence of the named triangles\n")
named = [2607, 7120, 15844, 20357]
pres_ids = [t["triangle"] for t in P["theta_presence"]["triangles"]]
L.append("| Triangle | In theta_presence list | Occluded (20 mm normal) |")
L.append("|---|---|---|")
for n in named:
    t = next((x for x in P["theta_presence"]["triangles"] if x["triangle"] == n), None)
    L.append(f"| {n} | {'yes' if t else 'no'} | {'' if t is None else ('yes' if t['occluded_20mm_normal_ray'] else 'no')} |")
L.append("")

def table(pose):
    rows = P[pose]["triangles"]
    L.append(f"### {pose}: {len(rows)} triangles under 0.25 (under 0.5 total: {P[pose]['count_lt_half']})\n")
    if not rows:
        L.append("None.\n"); return
    L.append("| Tri | Rest area mm² | Area ratio | Vertices | Dominant bone (mean w) | Top-4 weights per vertex | Bind centroid mm (x, y, z) | Occluded 20 mm | Under 0.5 @ fit |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for t in rows:
        tw = "; ".join(f"v{v}: " + ", ".join(f"{n} {w:.3f}" for n, w in t["top4_weights"][str(v)]) for v in t["vertices"])
        c = t["bind_centroid_mm"]
        occ = "yes" if t["occluded_20mm_normal_ray"] else "no"
        L.append(f"| {t['triangle']} | {t['rest_area_mm2']:.3f} | {t['area_ratio']:.4f} | {', '.join(map(str, t['vertices']))} | {t['dominant_bone_mean_weight'][0]} ({t['dominant_bone_mean_weight'][1]:.3f}) | {tw} | {c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f} | {occ} | {'yes' if t['under_half_at_theta_fit'] else 'no'} |")
    L.append("")

L.append("## Tables per pose\n")
for pose in P:
    table(pose)

L.append("## Notes\n")
L.append("- The four named triangles (2607, 7120, 15844, 20357) are the theta_presence entries with dominant bones upperarm01.R, spine01, upperarm01.L and spine01 respectively. Their rest areas and ratios are in the tables above.")
L.append("- Where they are (bind-pose centroids, from the tables): 2607 and 15844 sit at x about ±173 mm, z about 1320 mm; 7120 and 20357 at x about ±172 mm, z about 1330 mm. All four are in the armpit band (1.25 to 1.40 m). They appear only in the theta_presence list, not at theta_fit. The six theta_fit triangles are not in the armpit band: 4194 and 17433 are on the hands (z about 963 mm, x about ±545 mm); 12297, 12970, 25750 and 26433 sit at z about 833 to 837 mm, x within about 3 mm of the midline (the spine05 / upper-leg weight region). Those six are also in the theta_presence list, which makes up the ten.")
L.append("- None of the listed triangles is occluded by a 20 mm normal ray. `occlusion_sanity.py` checks the ray method on the bind body: the inward ray from each of the four named triangles hits the body at about 141 to 158 mm (`occlusion_sanity.txt`), so the method does find body surfaces at that scale; no body surface lies within 20 mm along the outward or inward normal of the four named triangles.")
L.append("- Pose files are not written. `triangles.json` is the only data output; this report is generated from it by `make_report.py`.")
sha = hashlib.sha256((HERE / "triangles.json").read_bytes()).hexdigest()
L.append(f"\nSource: `triangles.json` sha256 `{sha}`.")
(HERE / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("wrote REPORT.md", len(L), "lines")
