"""shoulder_sweep.py  Paper 01 step 2b (section 9): sweep each arm from theta_fit through six motions, measure every 5 deg (5 mm for the shrug),
publish the passing range as shoulder_envelope.json.

usage:  python -I -B shoulder_sweep.py          (bpy 5.2 module; a few minutes)

Joint frame (base layer 1), computed once at theta_fit per side: long axis l = shoulder (upperarm01 head) -> elbow (lowerarm01 head);
flexion axis = the body's left-right axis made perpendicular to l; third axis = their cross product.
Motions (angle > 0 in the named direction; the sign of each axis is chosen by moving the elbow the named way):
  forward   0-90 deg   flexion about the flexion axis, elbow forward (-Y)
  sideways  0-90 deg   abduction about the third axis, elbow away from the midline
  back      0-30 deg   extension about the flexion axis, elbow backward (+Y)
  across    0-40 deg   INTERPRETATION (the paper gives only "across the chest 0 to 40"): horizontal adduction toward the midline about the world
                       vertical, starting from forward 90 (the standard cross-body test; a hanging arm cannot cross the chest without flexion)
  shrug     0-30 mm    clavicle turned about the fore-aft axis through its head until the shoulder joint rises by the amount; the upper arm is
                       turned back so the arm keeps its hanging direction
  walk      -8..+12    the walk's own range (paper 01 4.4): flexion 8 back to 12 forward, elbow 8 deg at the back to 20 deg at the front (linear)
Shoulder-girdle share: variant "share_1_3" gives shoulder01 one third of every swing (base layer 3 rule 5, P3 C3; paper 01 4.4 "clavicle: one
third"), variant "share_0" moves the upper arm alone. The envelope is published from share_1_3, the variant the walk uses; share_0 is reference.
Gates at every sample (base layer 6; reference = theta_fit, which step 2 made the reference): body volume change against theta_fit within
+-1.5 %; triangles under a quarter of rest area no more than at theta_fit; non-head self-intersecting pairs no more than at theta_fit.
Passing range = from 0 up to the last sample before the first failure."""
import sys, json, math, time
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import posekit as K  # noqa: E402

T0 = time.time(); rig = K.Rig(); FIT = K.read_pose(HERE / "theta_fit.json")
rig.apply(FIT["bones"]); X = rig.eval_body()
REF = dict(volume=rig.volume(X), tri_q=int((rig.tri_area(X) / rig.A0 < 0.25).sum()), self=rig.self_pairs(X)); REF["self"].pop("_pairs")
REF_Q = set(np.nonzero(rig.tri_area(X) / rig.A0 < 0.25)[0].tolist())
VOL_LIM = 1.5
print("theta_fit reference:", {k: v for k, v in REF.items() if k != "volume"}, flush=True)


def measure(tag):
    X = rig.eval_body(); sp = rig.self_pairs(X); sp.pop("_pairs"); vol = (rig.volume(X) / REF["volume"] - 1) * 100; q = int((rig.tri_area(X) / rig.A0 < 0.25).sum())
    fails = []
    if abs(vol) > VOL_LIM: fails.append("volume")
    if q > REF["tri_q"]: fails.append("triangles")
    if sp["non_head"] > REF["self"]["non_head"]: fails.append("self_intersection")
    r = rig.tri_area(X) / rig.A0; newq = sorted(set(np.nonzero(r < 0.25)[0].tolist()) - REF_Q)
    where = {}
    for t in newq: b = str(rig.dominant[rig.TRI[t][0]]); where[b] = where.get(b, 0) + 1
    return dict(volume_pct_vs_fit=round(float(vol), 4), triangles_lt_quarter=q, new_collapsed_vs_fit=len(newq), new_collapsed_by_bone=where,
                min_area_ratio=round(float(r.min()), 4), self_non_head=int(sp["non_head"]), self_armpit_band=int(sp["armpit_band"]), self_other=int(sp["other"]),
                pass_=bool(not fails), fails=fails)


def frame(s):
    S_, E_ = rig.H(f"upperarm01.{s}"), rig.H(f"lowerarm01.{s}"); l = K.unit(E_ - S_); f = K.unit(K.X_ - (K.X_ @ l) * l); t = K.unit(np.cross(l, f))
    return S_, E_, l, f, t


def oriented(axis, s, want):
    """sign the axis so a small positive turn of the elbow about the shoulder moves it along `want`"""
    S_, E_ = rig.H(f"upperarm01.{s}"), rig.H(f"lowerarm01.{s}"); e1 = S_ + K.rot(axis, 1.0) @ (E_ - S_)
    return axis if (e1 - E_) @ want > 0 else -axis


def swing(s, axis, deg, share):
    if share: rig.rot_world(f"shoulder01.{s}", K.rot(axis, deg * share))
    rig.rot_world(f"upperarm01.{s}", K.rot(axis, deg * (1 - share)))


def set_elbow(s, target):
    S_, E_, W_ = rig.H(f"upperarm01.{s}"), rig.H(f"lowerarm01.{s}"), rig.H(f"wrist.{s}"); u = K.unit(E_ - S_); n = K.unit(np.cross(E_ - S_, W_ - E_))
    rig.aim(f"lowerarm01.{s}", f"lowerarm01.{s}", f"wrist.{s}", K.rot(n, target) @ u)


results = {}; frames = {}
for variant, share in (("share_1_3", 1 / 3), ("share_0", 0.0)):
    results[variant] = {}
    for s in "LR":
        rig.apply(FIT["bones"]); S_, E_, l, f, t = frame(s); lat = K.X_ if s == "L" else -K.X_
        fwd_ax = oriented(f, s, K.FWD); side_ax = oriented(t, s, lat)
        frames[s] = dict(shoulder=S_.round(5).tolist(), long_axis=l.round(5).tolist(), flexion_axis_forward_positive=fwd_ax.round(5).tolist(), third_axis_abduction_positive=side_ax.round(5).tolist())
        R = {}
        for motion, axis, top in (("forward", fwd_ax, 90), ("sideways", side_ax, 90), ("back", -fwd_ax, 30)):
            rows = []
            for deg in range(0, top + 1, 5):
                rig.apply(FIT["bones"]); swing(s, axis, deg, share); rows.append(dict(value=deg, **measure(motion)))
            R[motion] = rows
        rows = []
        mid = oriented(K.Z_, s, -lat)                                    # horizontal adduction: elbow toward the midline
        for deg in range(0, 41, 5):
            rig.apply(FIT["bones"]); swing(s, fwd_ax, 90, share); swing(s, mid, deg, share); rows.append(dict(value=deg, **measure("across")))
        R["across"] = rows
        rows = []
        for mm in range(0, 31, 5):
            rig.apply(FIT["bones"]); c = f"clavicle.{s}"
            P = rig.H(c); A = rig.H(f"upperarm01.{s}"); r = A - P
            ax = K.Y_ if np.cross(K.Y_, r)[2] > 0 else -K.Y_                  # turning about this axis lifts the shoulder joint
            lo, hi = 0.0, 30.0
            for _ in range(40):                                              # bisection on the clavicle angle for the requested rise
                mdeg = (lo + hi) / 2; rise = (P + K.rot(ax, mdeg) @ r)[2] - A[2]
                lo, hi = (mdeg, hi) if rise < mm / 1000 else (lo, mdeg)
            d0 = K.unit(rig.H(f"wrist.{s}") - rig.H(f"upperarm01.{s}")); rig.rot_world(c, K.rot(ax, lo))
            d1 = K.unit(rig.H(f"wrist.{s}") - rig.H(f"upperarm01.{s}")); rig.rot_world(f"upperarm01.{s}", K.ST.min_arc(d1, d0))
            rows.append(dict(value=mm, clavicle_deg=round(lo, 3), rise_mm=round((rig.H(f"upperarm01.{s}")[2] - A[2]) * 1e3, 3), **measure("shrug")))
        R["shrug_mm"] = rows
        rows = []
        for deg in (-8, -5, 0, 5, 10, 12):
            rig.apply(FIT["bones"]); swing(s, fwd_ax, deg, share); set_elbow(s, 8 + (deg + 8) * (20 - 8) / 20); rows.append(dict(value=deg, elbow_deg=round(8 + (deg + 8) * 12 / 20, 2), **measure("walk")))
        R["walk"] = rows
        results[variant][s] = R
        print(variant, s, {m: [r["value"] for r in rows_ if not r["pass_"]][:1] for m, rows_ in R.items()}, "%.0fs" % (time.time() - T0), flush=True)


def passing(rows):
    """contiguous passing range from 0 (walk: around 0 in both directions)."""
    vals = [r["value"] for r in rows]; ok = {r["value"]: r["pass_"] for r in rows}
    if 0 not in ok or not ok[0]: return None
    hi = 0
    for v in sorted(x for x in vals if x >= 0):
        if not ok[v]: break
        hi = v
    lo = 0
    for v in sorted((x for x in vals if x <= 0), reverse=True):
        if not ok[v]: break
        lo = v
    return [lo, hi]


env = {}
for s in "LR":
    R = results["share_1_3"][s]
    env[s] = {m: dict(passing_range=passing(rows), sampled=[rows[0]["value"], rows[-1]["value"]], first_failure=next(({"value": r["value"], "fails": r["fails"]} for r in rows if not r["pass_"]), None))
              for m, rows in R.items()}
    if env[s]["across"]["passing_range"] is not None and not next(r for r in results["share_1_3"][s]["forward"] if r["value"] == 90)["pass_"]:
        env[s]["across"]["note"] = "start pose (forward 90) is itself outside the forward envelope"
construction = dict(paper="paper 01 step 2b, section 9", theta_fit_sha256=FIT["sha256"], frames=frames, variant_published="share_1_3",
                    shoulder_girdle_share=dict(bone="shoulder01", fraction=round(1 / 3, 6), source="base layer 3 rule 5; P3 variant C3; paper 01 4.4"),
                    motions=dict(forward="flexion 0-90, 5 deg", sideways="abduction 0-90, 5 deg", back="extension 0-30, 5 deg",
                                 across="INTERPRETATION: horizontal adduction 0-40 about world vertical from forward 90, 5 deg",
                                 shrug_mm="clavicle elevation, shoulder joint rise 0-30 mm, 5 mm; arm direction kept", walk="flexion -8..+12 with elbow 8..20 (paper 01 4.4)"),
                    gates=dict(volume_pct_vs_theta_fit=VOL_LIM, triangles_lt_quarter_max=REF["tri_q"], self_intersection_non_head_max=REF["self"]["non_head"]))
h = K.write_pose(HERE / "shoulder_envelope.json", "shoulder_envelope", "1.0.0", {}, construction, dict(envelope=env), extra=dict(envelope=env))
(HERE / "shoulder_sweep_samples.json").write_text(json.dumps(dict(theta_fit_sha256=FIT["sha256"], envelope_sha256=h, reference=dict(tri_q=REF["tri_q"], self=REF["self"]), samples=results), indent=1, default=K._np))
print(json.dumps(env, indent=1, default=K._np)); print("WROTE shoulder_envelope.json", h, "%.0fs" % (time.time() - T0))
