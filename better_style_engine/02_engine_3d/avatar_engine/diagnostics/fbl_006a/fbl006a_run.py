"""FBL-006a: collapsed-triangle diagnostic on the canonical rig (read-only; nothing is saved).

Poses: bind, theta_fit, theta_presence, and the shoulder-sweep samples forward 5, sideways 5, shrug 5 mm, walk +5 and +12
(share_1_3 variant, the published envelope variant). The sweep motions are reproduced from poses/shoulder_sweep.py (same
functions and order); its counts are cross-checked against poses/shoulder_sweep_samples.json.

Triangles are indices into the loop-triangle list of the canonical body (rig.TRI in posekit.py). Rest area = bind-pose area.
Occlusion: ray from the bind-pose centroid, offset 1 micrometre along the bind-pose normal, length 20 mm, against the bind body.

Run:  python -I -B fbl006a_run.py
"""
import sys, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
POSES = HERE.parents[1] / "poses"
sys.path.insert(0, str(POSES)); sys.dont_write_bytecode = True
import posekit as K                                   # noqa: E402  imports bpy first, then mathutils
from mathutils.bvhtree import BVHTree                  # noqa: E402

SHARE = 1 / 3                                          # shoulder01 share, variant share_1_3
OCC_MM = 20.0
LT_Q, LT_H = 0.25, 0.5

rig = K.Rig()                                          # opens the canonical rig read-only, checks sha
RIG_HASH_BEFORE = K.sha(K.RIG_BLEND)
rig.reset()
X_BIND = rig.eval_body()
A0 = rig.A0
TRI = rig.TRI
FIT = K.read_pose(POSES / "theta_fit.json"); PRES = K.read_pose(POSES / "theta_presence.json")
SAMPLES = json.loads((POSES / "shoulder_sweep_samples.json").read_text())["samples"]["share_1_3"]

me = rig.body.data
GROUP_NAME = {g.index: g.name for g in rig.body.vertex_groups}
def top4(vi):
    gs = sorted(me.vertices[vi].groups, key=lambda g: -g.weight)[:4]
    return [[GROUP_NAME[g.group], round(float(g.weight), 4)] for g in gs]
def weights_full(vi):
    return {GROUP_NAME[g.group]: float(g.weight) for g in me.vertices[vi].groups}

# bind-pose geometry for centroids, normals and occlusion
cen_bind = X_BIND[TRI].mean(1)
nrm_bind = np.cross(X_BIND[TRI[:, 1]] - X_BIND[TRI[:, 0]], X_BIND[TRI[:, 2]] - X_BIND[TRI[:, 0]])
nrm_bind = nrm_bind / np.linalg.norm(nrm_bind, axis=1)[:, None]
bvh_bind = BVHTree.FromPolygons([tuple(p) for p in X_BIND.tolist()], TRI.tolist(), all_triangles=True, epsilon=0.0)

def occlusion(t):
    o = cen_bind[t] + nrm_bind[t] * 1e-6
    loc, nor, idx, dist = bvh_bind.ray_cast(tuple(o), tuple(nrm_bind[t]), OCC_MM / 1000)
    if idx is None or idx == t:
        return False, None, None
    return True, round(float(dist) * 1000, 3), int(idx)

def dominant_bone(t):
    tot = {}
    for vi in TRI[t]:
        for n, w in weights_full(int(vi)).items():
            tot[n] = tot.get(n, 0.0) + w
    if not tot: return "", 0.0
    n = max(tot, key=tot.get); return n, round(tot[n] / 3, 4)

def state_tris(X):
    ratio = rig.tri_area(X) / A0
    return ratio

def record(X, ratio, t):
    occ, occ_mm, occ_tri = occlusion(t)
    dom, dom_w = dominant_bone(t)
    return dict(
        triangle=int(t), rest_area_mm2=round(float(A0[t]) * 1e6, 4), area_ratio=round(float(ratio[t]), 5),
        vertices=[int(v) for v in TRI[t]], dominant_bone_mean_weight=[dom, dom_w],
        top4_weights={str(int(v)): top4(int(v)) for v in TRI[t]},
        bind_centroid_mm=[round(float(c) * 1000, 3) for c in cen_bind[t]],
        bind_normal=[round(float(c), 5) for c in nrm_bind[t]],
        occluded_20mm_normal_ray=occ, occluder_hit_mm=occ_mm, occluder_triangle=occ_tri,
    )

# ---------------------------------------------------------------- sweep motions (reproduced from shoulder_sweep.py)
def frame(s):
    S_, E_ = rig.H(f"upperarm01.{s}"), rig.H(f"lowerarm01.{s}"); l = K.unit(E_ - S_); f = K.unit(K.X_ - (K.X_ @ l) * l); t = K.unit(np.cross(l, f))
    return S_, E_, l, f, t

def oriented(axis, s, want):
    S_, E_ = rig.H(f"upperarm01.{s}"), rig.H(f"lowerarm01.{s}"); e1 = S_ + K.rot(axis, 1.0) @ (E_ - S_)
    return axis if (e1 - E_) @ want > 0 else -axis

def swing(s, axis, deg, share):
    if share: rig.rot_world(f"shoulder01.{s}", K.rot(axis, deg * share))
    rig.rot_world(f"upperarm01.{s}", K.rot(axis, deg * (1 - share)))

def set_elbow(s, target):
    S_, E_, W_ = rig.H(f"upperarm01.{s}"), rig.H(f"lowerarm01.{s}"), rig.H(f"wrist.{s}"); u = K.unit(E_ - S_); n = K.unit(np.cross(E_ - S_, W_ - E_))
    rig.aim(f"lowerarm01.{s}", f"lowerarm01.{s}", f"wrist.{s}", K.rot(n, target) @ u)

def shrug(s, mm):
    c = f"clavicle.{s}"; P = rig.H(c); A = rig.H(f"upperarm01.{s}"); r = A - P
    ax = K.Y_ if np.cross(K.Y_, r)[2] > 0 else -K.Y_
    lo, hi = 0.0, 30.0
    for _ in range(40):
        mdeg = (lo + hi) / 2; rise = (P + K.rot(ax, mdeg) @ r)[2] - A[2]
        lo, hi = (mdeg, hi) if rise < mm / 1000 else (lo, mdeg)
    d0 = K.unit(rig.H(f"wrist.{s}") - rig.H(f"upperarm01.{s}")); rig.rot_world(c, K.rot(ax, lo))
    d1 = K.unit(rig.H(f"wrist.{s}") - rig.H(f"upperarm01.{s}")); rig.rot_world(f"upperarm01.{s}", K.ST.min_arc(d1, d0))
    return round(lo, 3), round((rig.H(f"upperarm01.{s}")[2] - A[2]) * 1e3, 3)

def sweep_pose(motion, s, value):
    rig.apply(FIT["bones"]); S_, E_, l, f, t = frame(s); lat = K.X_ if s == "L" else -K.X_
    fwd_ax = oriented(f, s, K.FWD); side_ax = oriented(t, s, lat)
    if motion == "forward": swing(s, fwd_ax, value, SHARE)
    elif motion == "sideways": swing(s, side_ax, value, SHARE)
    elif motion == "shrug_mm": return shrug(s, value)
    elif motion == "walk":
        swing(s, fwd_ax, value, SHARE); set_elbow(s, 8 + (value + 8) * (20 - 8) / 20)
    rig.update()

# ---------------------------------------------------------------- poses
POSE_DEFS = [("bind", None), ("theta_fit", FIT), ("theta_presence", PRES)]
SWEEP_DEFS = []
for s in "LR":
    SWEEP_DEFS += [(f"sweep_forward_5_{s}", ("forward", s, 5)), (f"sweep_sideways_5_{s}", ("sideways", s, 5)),
                   (f"sweep_shrug_5mm_{s}", ("shrug_mm", s, 5)), (f"sweep_walk_+5_{s}", ("walk", s, 5)), (f"sweep_walk_+12_{s}", ("walk", s, 12))]

out = dict(rig_sha256_before=RIG_HASH_BEFORE, A0_triangles=int(len(TRI)), body_vertices=int(len(X_BIND)), poses={}, cross_check_sweep={})
expected = dict(bind=0, theta_fit=6, theta_presence=10)
report_rows = {}

def add_pose(name, X, extra=None):
    ratio = state_tris(X)
    low = np.nonzero(ratio < LT_Q)[0]
    entries = [record(X, ratio, int(t)) for t in low]
    out["poses"][name] = dict(count_lt_quarter=int(len(low)), count_lt_half=int((ratio < LT_H).sum()),
                              triangles=entries, extra=extra or {})
    print(name, "lt_quarter", len(low), "lt_half", int((ratio < LT_H).sum()), flush=True)

rig.reset(); add_pose("bind", rig.eval_body())
rig.apply(FIT["bones"]); X_FIT = rig.eval_body(); add_pose("theta_fit", X_FIT)
rig.apply(PRES["bones"]); add_pose("theta_presence", rig.eval_body())

for name, d in SWEEP_DEFS:
    motion, s, value = d
    rig.apply(FIT["bones"]); extra = sweep_pose(motion, s, value)
    X = rig.eval_body()
    samp = next((r for r in SAMPLES[s][motion] if r["value"] == value), None)
    add_pose(name, X, extra=dict(motion=motion, side=s, value=value, shrug_clavicle_deg_and_rise_mm=extra if motion == "shrug_mm" else None))
    out["cross_check_sweep"][name] = dict(sweep_json_triangles_lt_quarter=None if samp is None else samp["triangles_lt_quarter"],
                                          this_run=out["poses"][name]["count_lt_quarter"])

# theta_fit ratio < 0.5 marker and fit list for comparison
fit_low = set(t["triangle"] for t in out["poses"]["theta_fit"]["triangles"])
fit_half = set(np.nonzero(state_tris(X_FIT) < LT_H)[0].tolist())
for name, p in out["poses"].items():
    for t in p["triangles"]:
        t["under_half_at_theta_fit"] = t["triangle"] in fit_half

out["count_checks"] = {k: dict(expected=v, found=out["poses"][k]["count_lt_quarter"], match=out["poses"][k]["count_lt_quarter"] == v) for k, v in expected.items()}
rig.reset()
out["rig_sha256_after"] = K.sha(K.RIG_BLEND)
out["fit_sha256_file"] = FIT["sha256"]; out["presence_sha256_file"] = PRES["sha256"]
(HERE / "triangles.json").write_text(json.dumps(out, indent=1, default=K._np), encoding="utf-8")
print("count_checks", out["count_checks"])
print("rig sha before/after", RIG_HASH_BEFORE, out["rig_sha256_after"])
