"""gates.py  Paper 01 step 1: measurement harness. Base-document layer 6 gates plus paper 01 gates S1 to S9, evaluated on a capture from capture_walk_blender.py.

usage:  python -I -B gates.py <capture.npz> <out_dir> [--no-clearance]

Outputs  <out_dir>/gate_report.json   every measured value, the limit it was compared with, and PASS / FAIL / NOT_APPLICABLE / REFERENCE_PENDING
         <out_dir>/section2_check.json the base document's section 2 numbers recomputed from the capture, for the step 1 done-criterion

Limits come from gate_limits.json beside this file and are never adjusted here (paper 01 stop rule).
Definitions are written next to each measure. Frames: the body frame is world space turned by minus the Walker empty's heading (forward -Y, figure's left +X, up +Z). It removes the scene's
35 -> 0 degree turn to camera but keeps the Walker's translation, which in this scene carries the vertical bob and sideways sway. Floor quantities use world Z against the floor height recorded in the capture.
Metric definitions reused, not re-derived: volume and triangle-area ratio from P3/REQ-04 (run_pose_construction.py), self-intersection from P2/REQ-04,
clearance by generalised winding number from P2 (clearance_field.py), swing/twist from P3 (swing_twist.py)."""
import sys, json, math, hashlib, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; WS = HERE.parents[2]
sys.path.insert(0, str(WS / "solvers/prototypes/P3_pose_swing_twist")); sys.path.insert(0, str(WS / "solvers/prototypes/P2_clearance_field"))
import swing_twist as ST  # noqa: E402

CAP, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True); DO_CLEAR = "--no-clearance" not in sys.argv
PRESET = sys.argv[sys.argv.index("--preset") + 1] if "--preset" in sys.argv else "BASELINE_M5"
LIM = json.loads((HERE / "gate_limits.json").read_text()); T0 = time.time()
PRESET_CFG = LIM.get("presets", {}).get(PRESET, {})
for _sec in ("base_gates", "p01_gates", "targets_not_gates"):
    if _sec in PRESET_CFG:
        LIM[_sec] = {**LIM[_sec], **PRESET_CFG[_sec]}
CLAVICLE_LOCK_REQUIRED = bool(PRESET_CFG.get("clavicle_lock", False))
Z = np.load(CAP); meta = json.loads(str(Z["meta"])); NAMES = [str(n) for n in Z["names"]]; IDX = {n: i for i, n in enumerate(NAMES)}; PAR = Z["parent"]
fps = meta["fps"]; frames = Z["frames"]; F = len(frames); t = (frames - frames[0]) / fps; floor = meta["floor_z"]
WALKW = t < 8.0                                    # walking part of the clip, as in the base document section 2 (settle fade starts at 8.5 s)
POSE = Z["pose"]; RL = Z["rest_local"]; HW = Z["headw"]; BODY = Z["body"].astype(float); TRI = Z["tri"]; W = Z["walker"]; RW = Z["rigw"]
footv = {"L": Z["foot_L"], "R": Z["foot_R"]}


def polar(M):
    U, _, Vt = np.linalg.svd(M); R = U @ Vt
    if np.linalg.det(R) < 0: U[:, -1] *= -1; R = U @ Vt
    return R


def rot_angle(R): return np.degrees(np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


def logmap(R):
    """rotation vectors (radians) of a stack of rotations"""
    a = np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)); v = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    s = np.sin(a); k = np.where(s > 1e-9, a / (2 * np.where(s > 1e-9, s, 1)), 0.5); return v * k[..., None]


# ---------------------------------------------------------------- frames and joint data in the body frame
# body frame: undo the Walker's heading only. The scene puts the vertical bob and sideways sway on the Walker itself, so its translation must NOT be removed.
_yaw = np.arctan2(W[:, 1, 0], W[:, 0, 0]); Winv = np.zeros((F, 4, 4)); Winv[:, 3, 3] = 1; Winv[:, 2, 2] = 1
Winv[:, 0, 0] = np.cos(_yaw); Winv[:, 0, 1] = np.sin(_yaw); Winv[:, 1, 0] = -np.sin(_yaw); Winv[:, 1, 1] = np.cos(_yaw)
def to_body(P): return np.einsum("fij,f...j->f...i", Winv[:, :3, :3], P) + Winv[:, None, :3, 3] if P.ndim == 3 else np.einsum("fij,fj->fi", Winv[:, :3, :3], P) + Winv[:, :3, 3]
HB = to_body(HW)                                                                       # (F, bones, 3) heads in the body frame
C = np.einsum("fij,fjk->fik", Winv[:, :3, :3], RW[:, :3, :3])                           # armature space -> body frame
PR = np.stack([[polar(POSE[f, b, :3, :3]) for b in range(len(NAMES))] for f in range(F)]); R0 = np.stack([polar(RL[b, :3, :3]) for b in range(len(NAMES))])
def body_rel(n):
    """rest-relative rotation of bone n in the body frame: B_t B_rest^T"""
    b = IDX[n]; return np.einsum("fij,fjk,lk,fml->fim", C, PR[:, b], R0[b], C)
def local_rel(n):
    """parent-relative rotation of bone n against its rest parent-relative rotation"""
    b = IDX[n]; p = PAR[b]
    if p < 0: return np.einsum("fij,kj->fik", PR[:, b], R0[b])
    L = np.einsum("fji,fjk->fik", PR[:, p], PR[:, b]); L0 = R0[p].T @ R0[b]; return np.einsum("fij,kj->fik", L, L0)
H = lambda n: HB[:, IDX[n]]
FWD, LAT, UP = np.array([0, -1.0, 0]), np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
left_sign = np.sign((H("upperleg01.L") - H("upperleg01.R")) @ LAT).mean(); assert left_sign > 0, "figure's left is not +X in the Walker frame"


def yaw_pitch_roll(R):
    f = R @ FWD; l = R @ LAT
    return np.degrees(np.arctan2(f @ LAT, f @ FWD)), np.degrees(np.arcsin(np.clip(f @ UP, -1, 1))), np.degrees(np.arcsin(np.clip(l @ UP, -1, 1)))


def sag_angle(a, b):
    """sagittal angle of segment a->b: 0 pointing down, positive forward"""
    d = H(b) - H(a); return np.degrees(np.arctan2(d @ FWD, -(d @ UP)))


def flex(a, b, c):
    u = H(b) - H(a); v = H(c) - H(b); return np.degrees(np.arccos(np.clip(np.einsum("fi,fi->f", u, v) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)), -1, 1)))


# foot heights and centroids (world, as the floor is a world object)
fmin = {s: BODY[:, footv[s], 2].min(1) - floor for s in "LR"}; fcen = {s: BODY[:, footv[s]].mean(1) for s in "LR"}

# ---------------------------------------------------------------- stride period
anc = {s: H(f"foot.{s}") - H(f"upperleg01.{s}") for s in "LR"}
def period_frames(x):
    x = x - x.mean(); ac = np.correlate(x, x, "full")[len(x) - 1:]; ac /= ac[0]; lo = int(0.6 * fps); k = lo + int(np.argmax(ac[lo:int(4 * fps)]))
    if 0 < k < len(ac) - 1:                                                             # parabolic refinement
        a, b, c = ac[k - 1], ac[k], ac[k + 1]; k = k + 0.5 * (a - c) / (a - 2 * b + c)
    return float(k)
Tf = float(np.mean([period_frames((anc[s] @ FWD)[WALKW]) for s in "LR"])); T_s = Tf / fps
nstr = int(WALKW.sum() // Tf); seg = np.arange(int(round(nstr * Tf)))                   # whole strides from the first frame
def stride_resample(x, n=64):
    out = []
    for k in range(nstr):
        q = k * Tf + np.arange(n) * Tf / n; out.append(np.interp(q, np.arange(len(x)), x))
    return np.array(out)


def result(gate, status, value, limit, note=""):
    return dict(gate=gate, status=status, value=value, limit=limit, note=note)


R = []; facts = {}
facts["stride"] = dict(period_s_measured=round(T_s, 3), whole_strides_in_walk=nstr, target_s=LIM["targets_not_gates"]["stride_period_s"],
                       method="autocorrelation of ankle fore-aft position relative to the hip, body frame, first 8 s; mean of both legs")

# ---------------------------------------------------------------- base gates
# stance foot = the lower foot in each frame (the current walk has no footstep plan, so no planned stance exists)
low = np.where(fmin["L"] <= fmin["R"], 0, 1); stz = np.where(low == 0, fmin["L"], fmin["R"]) * 1e3
w = WALKW
ok = (stz[w] >= -1) & (stz[w] <= 1)
R.append(result("base.floor_contact", "PASS" if ok.all() else "FAIL", dict(min_mm=round(float(stz[w].min()), 1), max_mm=round(float(stz[w].max()), 1), frames_within=int(ok.sum()), frames=int(w.sum())),
                LIM["base_gates"]["floor_contact_mm"]["limit"], "stance foot taken as the lower foot in each frame"))
anyz = np.minimum(fmin["L"], fmin["R"]) * 1e3
R.append(result("base.no_penetration", "PASS" if anyz.min() >= -1 else "FAIL", dict(min_mm=round(float(anyz.min()), 1), frames_below_limit=int((anyz < -1).sum()), frames=F),
                LIM["base_gates"]["no_penetration_mm"]["limit_min"], "all frames"))
cen = np.where(low[:, None] == 0, fcen["L"], fcen["R"]); same = (low[1:] == low[:-1]) & w[1:] & w[:-1]
slide = np.linalg.norm(np.diff(cen[:, :2], axis=0), axis=1)[same] * 1e3
R.append(result("base.no_slide", "PASS" if slide.max() <= 1 else "FAIL", dict(max_mm_per_frame=round(float(slide.max()), 2), median_mm_per_frame=round(float(np.median(slide)), 2), frame_pairs=int(same.sum())),
                LIM["base_gates"]["no_slide_mm_per_frame"]["limit_max"], "world-frame movement of the lower foot's vertex centroid between consecutive frames with the same lower foot"))
hd = H("head"); hv = float(np.ptp(hd[w] @ UP) * 1e3); hl = float(np.ptp(hd[w] @ LAT) * 1e3)
R.append(result("base.head_travel", "PASS" if hv < 45 and hl < 30 else "FAIL", dict(vertical_mm=round(hv, 1), sideways_mm=round(hl, 1)), dict(vertical_max=45, sideways_max=30), "head bone head, body frame, first 8 s"))
R.append(result("base.reach", "NOT_APPLICABLE", dict(constraints=meta["constraints"], ik_solves=0), 0, "the current walk is forward kinematics with no IK; there are no solves to count"))

# twist, P3 definition: D = Rb^T (Srot_a^T Srot_b) Rb, twist about the bone's own Y axis
def twist_series(a, b):
    ia, ib = IDX[a], IDX[b]; Rb = R0[ib]; out = []
    for f in range(F):
        Sa = polar(PR[f, ia] @ R0[ia].T); Sb = polar(PR[f, ib] @ R0[ib].T); D = Rb.T @ (Sa.T @ Sb) @ Rb; tw, sw, _, _, ill = ST.swing_twist(D); out.append((math.degrees(tw), math.degrees(sw), ill))
    return np.array(out)
tw = {f"{a}->{b}": twist_series(a, b) for s in "LR" for a, b in ((f"clavicle.{s}", f"upperarm01.{s}"), (f"pelvis.{s}", f"upperleg01.{s}"))}
twmax = {k: round(float(np.abs(v[:, 0]).max()), 2) for k, v in tw.items()}
R.append(result("base.twist", "PASS" if max(twmax.values()) <= 15 else "FAIL", dict(max_abs_twist_deg=twmax, ill_conditioned_frames={k: int(v[:, 2].sum()) for k, v in tw.items()}), 15.0, "all frames; P3 swing_twist"))

# volume / triangles / self-intersection (P3 and P2 definitions), reference = bind pose until theta_fit exists
def volume(X): return np.einsum("ij,ij->i", X[TRI[:, 0]], np.cross(X[TRI[:, 1]], X[TRI[:, 2]])).sum() / 6
def tri_area(X): return 0.5 * np.linalg.norm(np.cross(X[TRI[:, 1]] - X[TRI[:, 0]], X[TRI[:, 2]] - X[TRI[:, 0]]), axis=1)
RB = Z["rest_body"]; Vr = volume(RB); Ar = tri_area(RB)
vol = np.array([(volume(BODY[f]) / Vr - 1) * 100 for f in range(F)]); q = np.array([int((tri_area(BODY[f]) / Ar < 0.25).sum()) for f in range(F)]); sp = Z["self_pairs"]
R.append(result("base.volume", "PASS" if np.abs(vol).max() <= 1.5 else "FAIL", dict(min_pct=round(float(vol.min()), 3), max_pct=round(float(vol.max()), 3), frame_of_max_abs=int(frames[np.abs(vol).argmax()])), 1.5,
                "reference = bind pose (all pose bones identity). Will be re-run against theta_fit once it exists"))
if "max_G5" in PRESET_CFG:
    max_g5_lim = int(PRESET_CFG["max_G5"])
    R.append(result("base.triangles_lt_quarter", "PASS" if int(q.max()) <= max_g5_lim else "FAIL", dict(bind=0, walk_max=int(q.max()), frame_of_max=int(frames[q.argmax()])), max_g5_lim, f"preset={PRESET} max_G5 <= {max_g5_lim}"))
else:
    R.append(result("base.triangles_lt_quarter", "REFERENCE_PENDING", dict(bind=0, walk_max=int(q.max()), frame_of_max=int(frames[q.argmax()])), "no more than at theta_fit", "theta_fit.json does not exist yet (paper 01 step 2)"))
rsp = Z["rest_self_pairs"]; reg = ("head", "armpit_band", "other")
R.append(result("base.self_intersection", "REFERENCE_PENDING", {r: dict(bind=int(rsp[i]), walk_min=int(sp[:, i].min()), walk_max=int(sp[:, i].max()), frame_of_max=int(frames[sp[:, i].argmax()])) for i, r in enumerate(reg)},
                "no more than at theta_fit", "theta_fit.json does not exist yet (paper 01 step 2). Regions by bind-pose height: head z > 1.45 m, armpit band 1.25 to 1.40 m (P2), other"))

# ---------------------------------------------------------------- paper 01 gates
# S1 stance share: a foot is "on the floor" when its lowest vertex is no more than 2 mm above the floor (penetration counts as contact)
CONTACT = 0.002
on = {s: fmin[s] <= CONTACT for s in "LR"}; segm = np.zeros(F, bool); segm[seg] = True
share = {s: round(float(on[s][segm].mean()), 3) for s in "LR"}
lo_, hi_ = LIM["p01_gates"]["S1_stance_share"]["limit"]
R.append(result("S1.stance_share", "PASS" if all(lo_ <= v <= hi_ for v in share.values()) else "FAIL", share, [lo_, hi_], f"contact = lowest foot vertex <= {CONTACT*1e3:.0f} mm above floor, over {nstr} whole strides"))

# S2 harmonic energy per stride (stride-resampled, 64 samples), energy in harmonics >= 2 over harmonics >= 1, mean over strides
sig = {f"hip.{s}": sag_angle(f"upperleg01.{s}", f"lowerleg01.{s}") for s in "LR"}
sig.update({f"knee.{s}": flex(f"upperleg01.{s}", f"lowerleg01.{s}", f"foot.{s}") for s in "LR"})
sig.update({f"shoulder.{s}": sag_angle(f"upperarm01.{s}", f"lowerarm01.{s}") for s in "LR"})
sig.update({f"elbow.{s}": flex(f"upperarm01.{s}", f"lowerarm01.{s}", f"wrist.{s}") for s in "LR"})
def harm(x):
    S = stride_resample(x); E = np.abs(np.fft.rfft(S - S.mean(1, keepdims=True), axis=1)) ** 2; return float(np.mean(E[:, 2:].sum(1) / np.maximum(E[:, 1:].sum(1), 1e-30)))
h2 = {k: round(harm(sig[k]), 4) for k in sig if not k.startswith("elbow")}; mins = LIM["p01_gates"]["S2_harmonic_energy_min"]
s2ok = all(h2[k] >= mins[k.split(".")[0]] for k in h2)
R.append(result("S2.not_a_pendulum", "PASS" if s2ok else "FAIL", h2, mins, "hip = sagittal thigh angle, knee = flexion, shoulder = sagittal upper-arm angle; joint positions in the body frame"))

# rotations used by S3, S6, S7 and the targets
yaw = {}; pit = {}; rol = {}
for n in ("root", "spine01", "spine02", "spine03", "head"):
    Rr = body_rel(n); yp = np.array([yaw_pitch_roll(Rr[f]) for f in range(F)]); yaw[n], pit[n], rol[n] = yp[:, 0], yp[:, 1], yp[:, 2]
def principal_angle(n):
    """signed angle of a bone's parent-relative rotation, projected on its dominant rotation axis over the clip"""
    v = logmap(local_rel(n)); v = v - v.mean(0); ax = np.linalg.svd(v, full_matrices=False)[2][0]; return np.degrees(v @ ax)
sig["chest_rot"] = yaw["spine01"]; sig["pelvis_rot"] = yaw["root"]; sig["head_yaw"] = yaw["head"]
sig["wrist.L"] = principal_angle("wrist.L"); sig["fingers.L"] = principal_angle("finger3-1.L")

# S3 delays: circular cross-correlation over whole strides; lag of the follower behind the driver, as a fraction of the stride
def delay(driver, follower, sign=None):
    d = driver[seg] - driver[seg].mean(); f = follower[seg] - follower[seg].mean()
    if np.ptp(d) < 0.5 or np.ptp(f) < 0.5: return None, dict(driver_range_deg=round(float(np.ptp(d)), 3), follower_range_deg=round(float(np.ptp(f)), 3))
    n = len(d); xc = np.array([np.dot(np.roll(d, L), f) for L in range(n)]); lags = (np.arange(n) + n // 2) % n - n // 2
    if sign is None: m = np.abs(lags) <= Tf / 4; k = np.argmax(np.abs(xc) * m); sg = int(np.sign(xc[k]))
    else: k = int(np.argmax(sign * xc)); sg = sign
    return round(float(lags[k] / Tf), 4), dict(sign=sg, corr=round(float(xc[k] / (np.linalg.norm(d) * np.linalg.norm(f))), 3))
pairs = [("chest_rot<-pelvis_rot(opposite)", "pelvis_rot", "chest_rot", -1), ("shoulder_swing<-chest_rot", "chest_rot", "shoulder.L", None),
         ("elbow<-shoulder", "shoulder.L", "elbow.L", None), ("wrist<-elbow", "elbow.L", "wrist.L", None), ("fingers<-wrist", "wrist.L", "fingers.L", None), ("head<-chest", "chest_rot", "head_yaw", 1)]
s3 = {}; s3ok = True
for name, a, b, sgn in pairs:
    dly, info = delay(sig[a], sig[b], sgn); lim = LIM["p01_gates"]["S3_delay_stride_fraction"][name]; ok = dly is not None and lim[0] - 1e-9 <= dly <= lim[1] + 1e-9; s3ok &= ok
    s3[name] = dict(delay=dly, **info, limit=lim, status="PASS" if ok else ("FAIL (no motion)" if dly is None else "FAIL"))
R.append(result("S3.overlap", "PASS" if s3ok else "FAIL", s3, LIM["p01_gates"]["S3_delay_stride_fraction"], "left side; a delay is undefined (FAIL) when either part moves less than 0.5 deg"))

# S4 asymmetry of peak-to-peak amplitudes
asym = {}
for j in ("hip", "knee", "shoulder", "elbow"):
    a, b = np.ptp(sig[f"{j}.L"][w]), np.ptp(sig[f"{j}.R"][w]); asym[j] = round(float(abs(a - b) / ((a + b) / 2) * 100), 3)
lo_, hi_ = LIM["p01_gates"]["S4_asymmetry_pct"]["limit"]
R.append(result("S4.asymmetry", "PASS" if all(lo_ <= v <= hi_ for v in asym.values()) else "FAIL", asym, [lo_, hi_], "|L-R| / mean of peak-to-peak amplitude over the first 8 s, percent"))

# S5 no frozen parts: largest pairwise angle between the bone's parent-relative rotations over the clip
_s5_limb_bones = ("upperarm01", "lowerarm01", "wrist", "upperleg01", "lowerleg01", "foot") if CLAVICLE_LOCK_REQUIRED else ("clavicle", "upperarm01", "lowerarm01", "wrist", "upperleg01", "lowerleg01", "foot")
S5_BONES = ["root", "spine03", "spine02", "spine01", "neck01", "head"] + [f"{b}.{s}" for s in "LR" for b in _s5_limb_bones] \
    + [f"finger{i}-1.{s}" for s in "LR" for i in range(1, 6)] + [f"toe{i}-1.{s}" for s in "LR" for i in range(1, 6)]
S5_INFO = ["spine05", "spine04", "neck02", "neck03", "shoulder01.L", "shoulder01.R"] + (["clavicle.L", "clavicle.R"] if CLAVICLE_LOCK_REQUIRED else [])
sub = np.arange(0, F, 2)
def range_deg(n):
    Lr = local_rel(n)[sub]; return float(rot_angle(np.einsum("aij,bkj->abik", Lr, Lr)).max())
rng = {n: round(range_deg(n), 3) for n in S5_BONES + S5_INFO if n in IDX}
frozen = [n for n in S5_BONES if rng.get(n, 0) <= 0.5]
clav_locked_ok = all(rng.get(f"clavicle.{s}", 0.0) <= 1e-4 for s in "LR") if CLAVICLE_LOCK_REQUIRED else True
R.append(result("S5.no_frozen_parts", "PASS" if (not frozen and clav_locked_ok) else "FAIL", dict(frozen=frozen, frozen_count=len(frozen), checked=len(S5_BONES), clavicle_lock_required=CLAVICLE_LOCK_REQUIRED, clavicle_locked_ok=clav_locked_ok, range_deg=rng), 0.5,
                "gate list: pelvis (root), spine01-03, neck01, head, clavicles (locked in RUNWAY_CONFIDENT_V2), shoulders (upperarm01), elbows (lowerarm01), wrists, first finger and toe joints, hips, knees, ankles. "
                "spine04/05, neck02/03, shoulder01 reported for information only"))

# S6 posture held: needs theta_presence.json
chest_ext = pit["spine02"] + pit["spine03"]
R.append(result("S6.posture_held", "NOT_APPLICABLE", dict(chest_pitch_spine02_plus_spine03_deg_mean=round(float(chest_ext[w].mean()), 3)), 1.0, "theta_presence.json does not exist yet (paper 01 step 3); recorded for later comparison"))

# S7 head: rest-relative orientation in the body frame (bind pose head = 0)
hy, hp, hr = yaw["head"][w], pit["head"][w], rol["head"][w]
s7 = dict(yaw=[round(float(hy.min()), 2), round(float(hy.max()), 2)], pitch=[round(float(hp.min()), 2), round(float(hp.max()), 2)], roll=[round(float(hr.min()), 2), round(float(hr.max()), 2)])
R.append(result("S7.head", "PASS" if max(np.abs(hy).max(), np.abs(hp).max(), np.abs(hr).max()) <= 2 else "FAIL", s7, 2.0, "head orientation relative to its bind orientation, body frame, first 8 s"))

# S8 step line: heel strikes = onsets of contact (S1 rule) in the walk; lateral ankle position at each strike; successive opposite-foot strikes
strikes = sorted([(int(i), s) for s in "LR" for i in np.nonzero(on[s][1:] & ~on[s][:-1] & w[1:])[0] + 1])
steps = []
for (i, a), (j, b) in zip(strikes, strikes[1:]):
    if a == b: continue
    lat_a = float(H(f"foot.{a}")[i] @ LAT); lat_b = float(H(f"foot.{b}")[j] @ LAT); left, right = (lat_a, lat_b) if a == "L" else (lat_b, lat_a)
    steps.append(dict(frames=[int(frames[i]), int(frames[j])], feet=a + b, line_cm=round((left - right) * 100, 2)))
lo_, hi_ = LIM["p01_gates"]["S8_step_line_cm"]["limit"]; vals = [s["line_cm"] for s in steps]
R.append(result("S8.step_line", "PASS" if vals and all(lo_ <= v <= hi_ for v in vals) else "FAIL", dict(strikes=len(strikes), steps=steps, min_cm=min(vals) if vals else None, max_cm=max(vals) if vals else None),
                [lo_, hi_], "also stands for the base step-width gate (paper 01 4.1 CHANGE). Lateral = left ankle minus right ankle; negative = crossover"))

# S9 smooth: angular jerk of each moving joint's parent-relative rotation; per stride, peak over median
def jerk(n):
    Lr = local_rel(n); dR = np.einsum("fij,fkj->fik", Lr[1:], Lr[:-1]); om = logmap(dR) * fps; al = np.diff(om, axis=0) * fps; j = np.linalg.norm(np.diff(al, axis=0) * fps, axis=1); return j
s9 = {}
for n in S5_BONES:
    if n in frozen: continue
    j = jerk(n); ratios = []
    for k in range(nstr):
        a, b = int(round(k * Tf)), int(round((k + 1) * Tf)); x = j[a:min(b, len(j))]
        if len(x) and np.median(x) > 1e-9: ratios.append(float(x.max() / np.median(x)))
    if ratios: s9[n] = round(max(ratios), 2)
bad = {k: v for k, v in s9.items() if v > 4}
R.append(result("S9.smooth", "PASS" if s9 and not bad else "FAIL", dict(worst_ratio_by_joint=s9, over_limit=sorted(bad)), 4.0, "also stands for the base smoothness gate. Frozen joints are reported by S5, not here"))

# ---------------------------------------------------------------- clearance (P2 winding number), sampled frames
if DO_CLEAR:
    import clearance_field as CF
    cl = {}; cf_frames = meta["clearance_frames"]; gnames = [k for k in Z.files if k.startswith("garment__")]
    for i, fr in enumerate(cf_frames):
        k = int(np.nonzero(frames == fr)[0][0]); V, T, _ = CF.weld(BODY[k], TRI); surf = CF.TriSurface(V, T, "body")
        for g in gnames:
            P = Z[g][i].astype(float); wn = surf.winding_number(P); cl.setdefault(g[9:], {})[str(fr)] = int((wn > 0.5).sum())
        print("clearance frame", fr, {g: cl[g][str(fr)] for g in cl}, f"{time.time() - T0:.0f}s", flush=True)
    worst = max(v for g in cl.values() for v in g.values())
    R.append(result("base.clearance", "PASS" if worst == 0 else "FAIL", dict(inside_by_garment_and_frame=cl, body_closed=surf.facts["closed"], body_boundary_edges=surf.facts["boundary_edges"]), 0,
                    f"winding number > 0.5 on the position-welded body; frames {cf_frames} only (8 phases of stride 2 plus the settled frame). Garments use the scene's skinning as built (preserve volume on)"))
else:
    R.append(result("base.clearance", "NOT_RUN", None, 0, "--no-clearance"))

# ---------------------------------------------------------------- targets (not gates) for the walk paper 01 asks for
pel = H("root"); facts["targets_measured"] = dict(
    pelvis_vertical_travel_mm=round(float(np.ptp(pel[w] @ UP) * 1e3), 1), pelvis_rotation_deg_amplitude=round(float(np.ptp(yaw["root"][w]) / 2), 2),
    pelvis_tilt_deg_amplitude=round(float(np.ptp(rol["root"][w]) / 2), 2), chest_rotation_vs_pelvis_deg_amplitude=round(float(np.ptp((yaw["spine01"] - yaw["root"])[w]) / 2), 2),
    pelvis_sideways_shift_mm_each_way=round(float(np.ptp(pel[w] @ LAT) / 2 * 1e3), 1), targets=LIM["targets_not_gates"])

# ---------------------------------------------------------------- section 2 reproduction (step 1 done-criterion)
def ang(v, u): return math.degrees(math.acos(np.clip(v @ u / (np.linalg.norm(v) * np.linalg.norm(u)), -1, 1)))
lat_w = np.einsum("fij,j->fi", W[:, :3, :3], LAT); fwd_w = np.einsum("fij,j->fi", W[:, :3, :3], FWD)
ankw = {s: HW[:, IDX[f"foot.{s}"]] for s in "LR"}; hipw = {s: HW[:, IDX[f"upperleg01.{s}"]] for s in "LR"}
sep = np.einsum("ij,ij->i", ankw["L"] - ankw["R"], lat_w) * 100
arm = {s: HW[:, IDX[f"wrist.{s}"]] - HW[:, IDX[f"upperarm01.{s}"]] for s in "LR"}; armang = {s: np.array([ang(a, np.array([0, 0, -1.0])) for a in arm[s]]) for s in "LR"}
elb = {s: np.array([ang(e - a, wv - e) for a, e, wv in zip(HW[:, IDX[f"upperarm01.{s}"]], HW[:, IDX[f"lowerarm01.{s}"]], HW[:, IDX[f"wrist.{s}"]])]) for s in "LR"}
speeds = []
for s in "LR":
    zz = fmin[s]; st = zz < zz[w].min() + 0.004; v = np.linalg.norm(np.diff(fcen[s][:, :2], axis=0), axis=1) * fps; speeds.append(float(v[(st & w)[:-1]].mean() * 100))
pitch = []; toeout = []
for s in "LR":
    d = HW[:, IDX[f"toe1-1.{s}"]] - ankw[s]; toeout.append(abs(float(np.degrees(np.arctan2(np.einsum("ij,ij->i", d, lat_w), np.einsum("ij,ij->i", d, fwd_w)))[w].mean())))
    pitch += list(np.degrees(np.arcsin(-d[:, 2] / np.linalg.norm(d, axis=1)))[w])
zmin = np.minimum(fmin["L"], fmin["R"]) * 1e3; hdw = HW[:, IDX["head"]]
got = dict(ankle_sep_mean_cm=sep[w].mean(), ankle_sep_min_cm=sep[w].min(), ankle_sep_max_cm=sep[w].max(), ankle_sep_settled_cm=sep[-1],
           hip_sep_cm=np.einsum("ij,ij->i", hipw["L"] - hipw["R"], lat_w).mean() * 100, lowest_foot_min_mm=zmin[w].min(), lowest_foot_max_mm=zmin[w].max(),
           frames_foot_more_than_5mm_below=int((zmin < -5).sum()), foot_speed_min_cm_s=min(speeds), foot_speed_max_cm_s=max(speeds), foot_pitch_min_deg=min(pitch), foot_pitch_max_deg=max(pitch),
           toe_out_deg=float(np.mean(toeout)), arm_line_min_deg=min(armang["L"].min(), armang["R"].min()), arm_line_max_deg=max(armang["L"].max(), armang["R"].max()),
           arm_line_settled_deg=armang["L"][-1], elbow_min_deg=min(elb["L"].min(), elb["R"].min()), elbow_max_deg=max(elb["L"].max(), elb["R"].max()), elbow_settled_deg=elb["L"][-1],
           head_vertical_mm=np.ptp(hdw[w, 2]) * 1e3, head_lateral_mm=np.ptp(np.einsum("ij,ij->i", hdw, lat_w)[w]) * 1e3, pelvis_vertical_mm=np.ptp(HW[w, IDX["root"], 2]) * 1e3,
           bones_animated=len(meta["animated_bones"]), constraints=meta["constraints"])
doc = dict(ankle_sep_mean_cm=42.2, ankle_sep_min_cm=39.5, ankle_sep_max_cm=44.8, ankle_sep_settled_cm=39.6, hip_sep_cm=21.7, lowest_foot_min_mm=-107, lowest_foot_max_mm=4.8,
           frames_foot_more_than_5mm_below=175, foot_speed_min_cm_s=118, foot_speed_max_cm_s=121, foot_pitch_min_deg=-6.8, foot_pitch_max_deg=72.4, toe_out_deg=16.5,
           arm_line_min_deg=12.9, arm_line_max_deg=40.7, arm_line_settled_deg=21.8, elbow_min_deg=37.6, elbow_max_deg=49.3, elbow_settled_deg=37.6,
           head_vertical_mm=73, head_lateral_mm=95, pelvis_vertical_mm=73, bones_animated=12, constraints=0)
tol = lambda v: 0.5 if float(v).is_integer() else 0.051                                  # the document rounds to whole units or one decimal
s2 = {k: dict(document=doc[k], recomputed=round(float(got[k]), 2), match=abs(abs(float(got[k])) - abs(doc[k])) <= tol(doc[k]) or abs(float(got[k]) - doc[k]) <= tol(doc[k])) for k in doc}
s2["_note"] = ("Recomputed with the rules of evidence/measure_bs_walk.py. That script takes arm-line and elbow ranges over all 300 frames, not the first 8 s; "
              "section 2's elbow minimum 37.6 deg is the settled pose (first 8 s alone: 41.5 deg). Toe-out is the per-foot magnitude (feet are mirrored)")
s2["armature_preserve_volume"] = dict(document="body off; jacket and pants on", recomputed=meta["armature_preserve_volume"])
reproduced = all(v["match"] for k, v in s2.items() if isinstance(v, dict) and "match" in v)

summary = {r["gate"]: r["status"] for r in R}
rep = dict(harness="solvers/avatar_engine/harness/gates.py", gate_limits_version=LIM["version"], capture=str(CAP), capture_sha256=hashlib.sha256(CAP.read_bytes()).hexdigest(), capture_meta=meta,
           summary=summary, counts={s: sum(1 for v in summary.values() if v == s) for s in sorted(set(summary.values()))}, results=R, facts=facts, seconds=round(time.time() - T0, 1))
(OUT / "gate_report.json").write_text(json.dumps(rep, indent=1))
(OUT / "section2_check.json").write_text(json.dumps(dict(reproduced=reproduced, rows=s2), indent=1))
print(json.dumps(summary, indent=1)); print("section 2 reproduced:", reproduced); print(json.dumps({k: v for k, v in s2.items() if isinstance(v, dict) and not v.get("match", True)}, indent=1))
