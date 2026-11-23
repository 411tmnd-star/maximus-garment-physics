"""build_theta_fit.py  Paper 01 step 2 (base document layer 0): build theta_fit.json by limb-line construction and measure the static gates.

usage:  python -I -B build_theta_fit.py            (bpy 5.2 module; about 30 s)

Construction, in this order, each limb rotation ONE minimal arc from the parent-carried rest aimed along the joint-to-joint line:
  1. legs     upperleg01: the hip->ankle line is turned so the ankle centres are ANKLE_SEP apart; the ankle's fore-aft offset from the hip and the
              hip->ankle length are kept (base L0: "bring each leg inward at the hip")
  2. feet     foot: world orientation set back to rest (sole flat), then turned about the vertical through the ankle to TOE_OUT
  3. floor    root: translated vertically so the lowest sole vertex is on the floor (z = 0)
  4. elbows   lowerarm01: the elbow->wrist line is turned in the rest arm plane (about its normal, the P3 hinge axis) to ELBOW flexion
  5. arms     upperarm01: the shoulder->wrist arm line is turned in its own vertical plane to ARM_LINE from vertical
  6. hands    finger joints flexed toward the palm (fingers 2-5) and in the thumb's own rest curl plane (thumb), by FINGER / THUMB
Values: base layer 0 and paper 01 section 3 / 4.1. Where a range is given the midpoint is used and recorded as such.
"""
import sys, json, time
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import posekit as K  # noqa: E402

T0 = time.time()
ANKLE_SEP = 0.10            # base L0: "until the ankle centres are 10 cm apart (proposed range 8 to 12 cm)"; paper 01 4.1: 8 to 10 cm, "unchanged" -> 10 cm is in both
TOE_OUT = 6.0               # base L0 and paper 01 4.1: 5 to 7 deg -> midpoint
ARM_LINE = 9.0              # base L0: 8 to 10 deg from vertical, "do not go lower" -> midpoint
ELBOW = 13.5                # base L0: 12 to 15 deg -> midpoint
FINGER = (20.0, 30.0, 15.0)  # base L0: knuckle to tip
THUMB = (10.0, 15.0)        # base L0: thumb 10, 15 (applied to finger1-2 and finger1-3; finger1-1 is the thumb's metacarpal joint)

rig = K.Rig(); bind = rig.metrics(label="bind (canonical rest)")
steps = []

# 1. legs
for s in "LR":
    hip, ank = rig.H(f"upperleg01.{s}"), rig.H(f"foot.{s}"); d0 = ank - hip; L = np.linalg.norm(d0)
    x1 = (ANKLE_SEP / 2) * (1 if s == "L" else -1) - hip[0]; y1 = d0[1]; z1 = -np.sqrt(L * L - x1 * x1 - y1 * y1)
    a = rig.aim(f"upperleg01.{s}", f"upperleg01.{s}", f"foot.{s}", np.array([x1, y1, z1]))
    steps.append(dict(step="leg inward at the hip", bone=f"upperleg01.{s}", arc_deg=round(a, 3)))
# 2. feet: back to the rest world orientation, then toe-out about the vertical through the ankle
for s in "LR":
    rig.rot_world(f"foot.{s}", rig.rel(f"foot.{s}").T)
    cur = rig.foot_axis(s); turn = (TOE_OUT - cur) * (1 if s == "L" else -1)
    rig.rot_world(f"foot.{s}", K.rot(K.Z_, turn))
    steps.append(dict(step="foot flat, toe-out", bone=f"foot.{s}", rest_axis_deg=round(cur, 3), turn_about_vertical_deg=round(turn, 3)))
# 3. floor
X = rig.eval_body(); low = min(X[rig.footv[s], 2].min() for s in "LR") - K.FLOOR_Z
rig.translate_world("root", [0, 0, -low]); steps.append(dict(step="root to floor", bone="root", dz_mm=round(-low * 1e3, 3)))
# 4-5. arms
for s in "LR":
    S_, E_, W_ = rig.H(f"upperarm01.{s}"), rig.H(f"lowerarm01.{s}"), rig.H(f"wrist.{s}")
    u = K.unit(E_ - S_); n = K.unit(np.cross(E_ - S_, W_ - E_))           # rest arm plane normal (P3 hinge axis)
    f1 = K.rot(n, ELBOW) @ u
    a = rig.aim(f"lowerarm01.{s}", f"lowerarm01.{s}", f"wrist.{s}", f1); steps.append(dict(step="elbow in the rest arm plane", bone=f"lowerarm01.{s}", arc_deg=round(a, 3)))
    a0 = rig.H(f"wrist.{s}") - rig.H(f"upperarm01.{s}"); h = K.unit(np.array([a0[0], a0[1], 0.0]))
    a1 = np.sin(np.radians(ARM_LINE)) * h + np.cos(np.radians(ARM_LINE)) * K.DOWN
    a = rig.aim(f"upperarm01.{s}", f"upperarm01.{s}", f"wrist.{s}", a1); steps.append(dict(step="arm line in its own vertical plane", bone=f"upperarm01.{s}", arc_deg=round(a, 3)))
# 6. hands
for s in "LR":
    pn = rig.palm_normal(s)
    for f in (2, 3, 4, 5):
        for k, deg in zip((1, 2, 3), FINGER):
            b = f"finger{f}-{k}.{s}"; d = K.unit(rig.T(b) - rig.H(b)); rig.rot_world(b, K.rot(np.cross(d, pn), deg))
    t1, t3 = K.unit(rig.T(f"finger1-1.{s}") - rig.H(f"finger1-1.{s}")), K.unit(rig.T(f"finger1-3.{s}") - rig.H(f"finger1-3.{s}"))
    tc = K.unit(t3 - (t3 @ t1) * t1)                                          # thumb's own rest curl direction
    for k, deg in zip((2, 3), THUMB):
        b = f"finger1-{k}.{s}"; d = K.unit(rig.T(b) - rig.H(b)); rig.rot_world(b, K.rot(np.cross(d, tc), deg))
    steps.append(dict(step="hand curl", side=s, palm_normal=[round(float(x), 4) for x in pn], fingers_deg=FINGER, thumb_deg=THUMB))

fit = rig.metrics(label="theta_fit")
bones = rig.pose_dict()
# static gates (base layer 6 at a static pose). Limits from harness/gate_limits.json and base L0; "no more than at theta_fit" gates have theta_fit as their reference.
LIM = json.loads((K.AE / "harness/gate_limits.json").read_text())
def g(name, ok, value, limit, note=""): return dict(gate=name, status="PASS" if ok else "FAIL", value=value, limit=limit, note=note)
gates = [
    g("floor_contact (both soles)", all(-1 <= v <= 1 for v in fit["foot_lowest_mm"].values()), fit["foot_lowest_mm"], [-1, 1]),
    g("no_penetration", fit["body_lowest_mm"] >= -1, fit["body_lowest_mm"], -1),
    g("standing ankle separation", 8 <= fit["ankle_separation_cm"] <= 10, fit["ankle_separation_cm"], [8, 10], "paper 01 4.1 standing 8 to 10 cm; base L0 8 to 12"),
    g("toe-out", all(5 <= v <= 7 for v in fit["toe_out_deg"].values()), fit["toe_out_deg"], [5, 7], "ankle -> toe2-1 axis"),
    g("sole flat", all(v <= 0.5 for v in fit["foot_tilt_from_rest_deg"].values()), fit["foot_tilt_from_rest_deg"], "<= 0.5 deg tilt from rest", "construction check, not a paper gate"),
    g("arm line", all(8 <= a["arm_line_from_vertical_deg"] <= 10 for a in fit["arm"].values()), {s: a["arm_line_from_vertical_deg"] for s, a in fit["arm"].items()}, [8, 10]),
    g("elbow flexion", all(12 <= a["elbow_flexion_deg"] <= 15 for a in fit["arm"].values()), {s: a["elbow_flexion_deg"] for s, a in fit["arm"].items()}, [12, 15]),
    g("twist", all(abs(v["twist_deg"]) <= LIM["base_gates"]["twist_deg"]["limit_abs"] for v in fit["twist"].values()), {k: v["twist_deg"] for k, v in fit["twist"].items()}, 15.0),
    g("volume vs bind", abs(fit["volume_pct_vs_bind"]) <= LIM["base_gates"]["volume_pct"]["limit_abs"], fit["volume_pct_vs_bind"], 1.5),
    dict(gate="triangles < 1/4 rest area", status="REFERENCE", value=dict(bind=bind["triangles_lt_quarter"], theta_fit=fit["triangles_lt_quarter"]), limit="no more than at theta_fit",
         note="theta_fit IS the reference for this gate from here on"),
    dict(gate="self-intersection", status="REFERENCE", value=dict(bind=bind["self_intersection"], theta_fit=fit["self_intersection"]), limit="no more than at theta_fit",
         note="theta_fit IS the reference for this gate from here on"),
]
construction = dict(paper="paper 01 step 2 / base layer 0", values=dict(ankle_separation_m=ANKLE_SEP, toe_out_deg=TOE_OUT, arm_line_deg=ARM_LINE, elbow_flexion_deg=ELBOW, finger_deg=FINGER, thumb_deg=THUMB),
                    value_sources=dict(ankle_separation_m="base L0 stated value 10 cm (range 8-12); inside paper 01 4.1's 8-10", toe_out_deg="midpoint of 5-7", arm_line_deg="midpoint of 8-10",
                                       elbow_flexion_deg="midpoint of 12-15", finger_deg="base L0 as stated", thumb_deg="base L0 as stated; applied to finger1-2, finger1-3"),
                    rule="one minimal arc per limb from the parent-carried rest, aimed along the joint-to-joint line (P3 'zero'); rest stays the bind pose", steps=steps)
h = K.write_pose(HERE / "theta_fit.json", "theta_fit", "1.0.0", bones, construction, dict(bind=K.clean(bind), theta_fit=K.clean(fit), static_gates=gates))
print(json.dumps(dict(steps=steps), indent=0)[:3000])
for gg in gates: print(f"{gg['status']:9s} {gg['gate']:28s} {gg['value']}")
print("arm-to-torso min mm  bind %.1f  fit %.1f | thigh-to-thigh mm bind %.1f fit %.1f" % (bind["arm_to_torso_min_mm"], fit["arm_to_torso_min_mm"], bind["thigh_to_thigh_min_mm"], fit["thigh_to_thigh_min_mm"]))
print("self-intersection non-head by dominant bone (fit):", fit["self_intersection_non_head_by_dominant_bone"])
print("arm detail:", json.dumps(fit["arm"])); print("WROTE theta_fit.json sha256", h, "bones", len(bones), "%.1fs" % (time.time() - T0))
