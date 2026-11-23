"""build_theta_presence.py  Paper 01 step 3 (section 3): presentation posture offsets on top of theta_fit; static gates at the result.
Garment clearance at this pose is a separate script (presence_clearance.py), run once a garment built on theta_fit exists.

usage:  python -I -B build_theta_presence.py

Offsets (paper 01 section 3; midpoint of every range):
  spine02, spine03   4.0 deg extension in total (3-5), 2.0 deg each, about the body's left-right axis through each bone's head.
                     NOTE: the paper calls these the "upper spine". In this rig the chain is root > spine05 > ... > spine01 > neck01, so spine01 is the
                     top segment and spine02 / spine03 span 1.08 to 1.28 m (mid back). Applied to the bones as NAMED; flagged in the report.
  clavicles          5.0 deg back (4-6) and 2.5 deg down (2-3), about the vertical and fore-aft axes through each clavicle head
  neck01-03          2.5 deg extension in total (2-3), shared equally
  head               level gaze: world orientation returned to its theta_fit (= rest) orientation, then 1.0 deg chin down (0-2)
  pelvis             neutral: no change
  arms               arm line re-aimed to theta_fit's 9 deg from vertical in its own vertical plane (the clavicle offsets move it), then the upper arm is
                     turned about the shoulder->elbow line so the palm faces the thigh (palm normal's horizontal part points to the midline)
  hands              theta_fit curl kept. "Thumb resting toward the index finger" has no number in the paper: NOT applied, flagged
Every rotation is applied after theta_fit and composed into the bone's local matrix; the result is stored as absolute bone values plus the offsets."""
import sys, json, math, time
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import posekit as K  # noqa: E402

ATTEMPT = "low" if "--low" in sys.argv else "mid"                    # stop rule: two attempts. mid = midpoint of every range; low = lower bound of every range
SPINE_EXT, CLAV_BACK, CLAV_DOWN, NECK_EXT, CHIN_DOWN = (4.0, 5.0, 2.5, 2.5, 1.0) if ATTEMPT == "mid" else (3.0, 4.0, 2.0, 2.0, 1.0)
OUTF = HERE / ("theta_presence.json" if ATTEMPT == "mid" else "theta_presence_attempt2_low.json")
rig = K.Rig(); FIT = K.read_pose(HERE / "theta_fit.json"); rig.apply(FIT["bones"]); fit = rig.metrics(label="theta_fit")
Rhead_fit = rig.Rw("head")


def ext_axis(n):
    """unit axis along +-X such that a positive turn about the bone's head moves the bone's tail backward (+Y): extension"""
    p, t = rig.H(n), rig.T(n); return K.X_ if (K.rot(K.X_, 1.0) @ (t - p))[1] > (t - p)[1] else -K.X_


steps = []
for n in ("spine03", "spine02"):                                   # lower first, so each turn carries the bones above it
    ax = ext_axis(n); rig.rot_world(n, K.rot(ax, SPINE_EXT / 2)); steps.append(dict(bone=n, extension_deg=SPINE_EXT / 2, axis=ax.tolist()))
for s in "LR":
    c = f"clavicle.{s}"; lat = rig.T(c) - rig.H(c)
    back = K.Z_ if (K.rot(K.Z_, 1.0) @ lat)[1] > lat[1] else -K.Z_
    rig.rot_world(c, K.rot(back, CLAV_BACK)); lat = rig.T(c) - rig.H(c)
    down = K.Y_ if (K.rot(K.Y_, 1.0) @ lat)[2] < lat[2] else -K.Y_
    rig.rot_world(c, K.rot(down, CLAV_DOWN)); steps.append(dict(bone=c, back_deg=CLAV_BACK, down_deg=CLAV_DOWN, back_axis=back.tolist(), down_axis=down.tolist()))
for n in ("neck01", "neck02", "neck03"):
    ax = ext_axis(n); rig.rot_world(n, K.rot(ax, NECK_EXT / 3)); steps.append(dict(bone=n, extension_deg=round(NECK_EXT / 3, 6)))
rig.rot_world("head", Rhead_fit @ rig.Rw("head").T)                    # level gaze: back to the theta_fit / rest world orientation
fl = -ext_axis("head"); rig.rot_world("head", K.rot(fl, CHIN_DOWN)); steps.append(dict(bone="head", level_then_chin_down_deg=CHIN_DOWN))
for s in "LR":                                                       # palm turn first, then the arm line is re-aimed (one minimal arc), so the arm line ends exactly on target
    S_, E_ = rig.H(f"upperarm01.{s}"), rig.H(f"lowerarm01.{s}"); l = K.unit(E_ - S_); med = -K.X_ if s == "L" else K.X_
    def palm_err(deg):                                               # angle between the palm normal's horizontal part and the medial direction after a turn
        n = K.rot(l, deg) @ rig.palm_normal(s); return K.ang(np.array([n[0], n[1], 0.0]), med)
    grid = np.arange(-30, 30.01, 0.05); tw = float(grid[np.argmin([palm_err(d) for d in grid])])
    before = palm_err(0.0); rig.rot_world(f"upperarm01.{s}", K.rot(l, tw))
    a0 = rig.H(f"wrist.{s}") - rig.H(f"upperarm01.{s}"); h = K.unit(np.array([a0[0], a0[1], 0.0])); A = fit["arm"][s]["arm_line_from_vertical_deg"]
    arc = rig.aim(f"upperarm01.{s}", f"upperarm01.{s}", f"wrist.{s}", np.sin(np.radians(A)) * h + np.cos(np.radians(A)) * K.DOWN)
    n_ = rig.palm_normal(s); after = K.ang(np.array([n_[0], n_[1], 0.0]), med)
    steps.append(dict(bone=f"upperarm01.{s}", palm_to_thigh_turn_deg=round(tw, 3), arm_line_reaim_arc_deg=round(arc, 3), palm_error_before_deg=round(before, 3), palm_error_after_deg=round(after, 3)))
pres = rig.metrics(label="theta_presence")
bones = rig.pose_dict()

# chest lift: change of the spine03-head -> neck01-head line's sagittal angle (positive = backward), which carries spine02 + spine03 extension
def sag(v): return math.degrees(math.atan2(v[1], v[2]))
rig.apply(FIT["bones"]); FIT_H = {n: rig.H(n) for n in ("spine03", "neck01", "head")}; FIT_CL = {s: (rig.H(f"clavicle.{s}"), rig.T(f"clavicle.{s}")) for s in "LR"}
FIT_HEADR = rig.Rw("head")
rig.apply(bones)
posture = dict(chest_lift_deg=round(sag(rig.H("neck01") - rig.H("spine03")) - sag(FIT_H["neck01"] - FIT_H["spine03"]), 3))
for s in "LR":
    a0 = FIT_CL[s][1] - FIT_CL[s][0]; a1 = rig.T(f"clavicle.{s}") - rig.H(f"clavicle.{s}")
    posture[f"clavicle_{s}_back_deg"] = round(math.degrees(math.atan2(a1[1], abs(a1[0]))) - math.degrees(math.atan2(a0[1], abs(a0[0]))), 3)
    posture[f"clavicle_{s}_down_deg"] = round(math.degrees(math.atan2(-a1[2], np.hypot(a1[0], a1[1]))) - math.degrees(math.atan2(-a0[2], np.hypot(a0[0], a0[1]))), 3)
Rh = rig.Rw("head") @ FIT_HEADR.T; fwd = Rh @ K.FWD; latv = Rh @ K.X_
posture["head_pitch_deg_chin_down_positive"] = round(-math.degrees(math.asin(np.clip(fwd @ K.Z_, -1, 1))), 3)
posture["head_roll_deg"] = round(math.degrees(math.asin(np.clip(latv @ K.Z_, -1, 1))), 3); posture["head_yaw_deg"] = round(math.degrees(math.atan2(fwd @ K.X_, fwd @ K.FWD)), 3)

def g(name, ok, value, limit, note=""): return dict(gate=name, status="PASS" if ok else "FAIL", value=value, limit=limit, note=note)
sp_f, sp_p = fit["self_intersection"], pres["self_intersection"]
gates = [
    g("floor_contact (both soles)", all(-1 <= v <= 1 for v in pres["foot_lowest_mm"].values()), pres["foot_lowest_mm"], [-1, 1]),
    g("no_penetration", pres["body_lowest_mm"] >= -1, pres["body_lowest_mm"], -1),
    g("twist", all(abs(v["twist_deg"]) <= 15 for v in pres["twist"].values()), {k: v["twist_deg"] for k, v in pres["twist"].items()}, 15.0),
    g("volume vs theta_fit", abs(pres["volume_pct_vs_bind"] - fit["volume_pct_vs_bind"]) <= 1.5 and abs(pres["volume_pct_vs_bind"]) <= 1.5,
      dict(vs_theta_fit_pct=round((pres["_volume"] / fit["_volume"] - 1) * 100, 4), vs_bind_pct=pres["volume_pct_vs_bind"]), 1.5),
    g("triangles < 1/4 rest area", pres["triangles_lt_quarter"] <= fit["triangles_lt_quarter"], dict(theta_fit=fit["triangles_lt_quarter"], theta_presence=pres["triangles_lt_quarter"]), "no more than at theta_fit"),
    g("self-intersection (non-head pairs)", sp_p["non_head"] <= sp_f["non_head"], dict(theta_fit=sp_f, theta_presence=sp_p), "no more than at theta_fit"),
    g("arm line", all(8 <= a["arm_line_from_vertical_deg"] <= 10 for a in pres["arm"].values()), {s: a["arm_line_from_vertical_deg"] for s, a in pres["arm"].items()}, [8, 10]),
    g("S6 posture (static check)", abs(posture["chest_lift_deg"] - SPINE_EXT) <= 1 and all(abs(posture[f"clavicle_{s}_back_deg"] - CLAV_BACK) <= 1 and abs(posture[f"clavicle_{s}_down_deg"] - CLAV_DOWN) <= 1 for s in "LR"),
      {k: v for k, v in posture.items() if k.startswith(("chest", "clavicle"))}, "within 1 deg of the offsets", "at the static pose S6 checks the construction; over a stride it is checked in step 6"),
    g("S7 head (static check)", all(abs(posture[k]) <= 2 for k in ("head_pitch_deg_chin_down_positive", "head_roll_deg", "head_yaw_deg")),
      {k: posture[k] for k in ("head_pitch_deg_chin_down_positive", "head_roll_deg", "head_yaw_deg")}, 2.0, "head orientation against level (theta_fit = rest head orientation)"),
    dict(gate="garment clearance", status="PENDING", value=None, limit=0, note="presence_clearance.py, after a garment built on theta_fit exists (paper 03 BS-SLV-001 shell)"),
]
offsets = dict(spine02_spine03_extension_total_deg=SPINE_EXT, clavicle_back_deg=CLAV_BACK, clavicle_down_deg=CLAV_DOWN, neck_extension_total_deg=NECK_EXT, head="level gaze, chin down %.1f deg" % CHIN_DOWN,
               pelvis="neutral (unchanged)", arms="arm line re-aimed to theta_fit's value, upper arm turned so the palm faces the thigh", hands="theta_fit curl; thumb-to-index NOT applied (no number in paper 01)")
construction = dict(paper="paper 01 step 3, section 3", attempt=ATTEMPT, base_pose=dict(file="theta_fit.json", sha256=FIT["sha256"]), offsets=offsets, value_sources=("midpoint" if ATTEMPT == "mid" else "lower bound") + " of each paper 01 section 3 range",
                    flags=["paper 01 names spine02/spine03 as 'upper spine'; in this rig spine01 is the top segment, spine02/03 are mid back (1.08-1.28 m). Applied as named.",
                           "'thumb resting toward the index finger': no value given; not applied",
                           "'level gaze' taken as the rest (= theta_fit) head orientation; the rig carries no gaze landmark"],
                    steps=steps, use="presentation only; garments are built on theta_fit and checked for clearance here")
h = K.write_pose(OUTF, "theta_presence", "1.0.0", bones, construction, dict(theta_fit=K.clean(fit), theta_presence=K.clean(pres), posture=posture, static_gates=gates))
for gg in gates: print(f"{gg['status']:8s} {gg['gate']:36s} {gg['value']}")
print("steps:", json.dumps(steps, default=K._np)); print("arm-to-torso mm fit %.1f presence %.1f" % (fit["arm_to_torso_min_mm"], pres["arm_to_torso_min_mm"]))
print("self-int by bone (presence):", pres["self_intersection_non_head_by_dominant_bone"]); print("WROTE", OUTF.name, h)
