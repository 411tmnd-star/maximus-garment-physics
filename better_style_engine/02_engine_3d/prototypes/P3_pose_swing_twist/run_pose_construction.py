"""run_pose_construction.py  PROTOTYPE alternative construction of the hand-contact arm, measured with the REQ-04 metrics. Read-only: nothing is saved.
  python -I run_pose_construction.py        (bpy module 5.2; about 5 minutes)
Inputs are the REQ-04 copies of the rig, the retargeter and the pose registry (hashes equal the live engine files, re-checked here).

What the engine does (pose_retarget.py): authored anatomical rotations, then for a hand contact a two-bone IK whose result is written with two
MINIMAL-ARC re-aims starting from the anatomically rotated arm (L148). Composing non-coaxial swings leaves twist (REQ-04: POSE_07 L 173.66 deg).

What this prototype does for the contact arm only, with the SAME landmark, pole, bone lengths and IK formula:
  1. upper arm and forearm go back to their rest orientation carried by the parent (matrix_basis identity);
  2. the limb LINE shoulder->elbow (not the bone's own axis, which is about 5 deg off it) is aimed at the IK elbow;
  3. the axial rotation is set EXPLICITLY by a named rule:
        zero    none: one minimal arc from the parent-carried rest (twist about the limb line = 0 by construction)
        hinge   the elbow's rest hinge axis (normal of the rest shoulder-elbow-wrist plane) is put on the normal of the IK arm plane,
                so the forearm bends about the hinge it was modelled and weighted with
  4. optionally a fraction f of the resulting shoulder twist is carried by upperarm02 (measured, not assumed to help: REQ-04 found it harmful at 174 deg);
  5. optionally a share of the arm swing is given to shoulder01 before the IK (scapulo-humeral share; experimental).
Metric definitions are REQ-04's (volume, one-ring area ratio, triangle area ratio, dominant-bone normal flips, self-intersecting face pairs, det of the blended linear part)."""
import os
import sys, json, math, time, hashlib, copy
from pathlib import Path
import numpy as np
import bpy
from mathutils import Matrix
from mathutils.bvhtree import BVHTree
HERE = Path(__file__).resolve().parent; WS = HERE.parents[2]; sys.dont_write_bytecode = True
INP = WS / "stage3_engineering/opus_requests/REQ-04_pose_swing_twist_dqs/inputs"
HE = Path(os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/human_engine')))
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(INP)); import swing_twist as ST, pose_retarget as PRT
PRT.REGISTRY = INP / "canonical_poses.json"; OUT = HERE / "out"; OUT.mkdir(exist_ok=True); T0 = time.time()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
FILES = ["03_full_rigged.blend", "pose_retarget.py", "canonical_poses.json"]; h_before = {f: sha(INP / f) for f in FILES}
QUICK = "--quick" in sys.argv

bpy.ops.wm.open_mainfile(filepath=str(INP / "03_full_rigged.blend"), load_ui=False)
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE"); body = next(o for o in bpy.data.objects if o.type == "MESH" and any(m.type == "ARMATURE" for m in o.modifiers))
mod = next(m for m in body.modifiers if m.type == "ARMATURE"); assert np.allclose(np.array(arm.matrix_world), np.eye(4)) and np.allclose(np.array(body.matrix_world), np.eye(4))
pbs = arm.pose.bones; upd = bpy.context.view_layer.update
bone_index = {b.name: i for i, b in enumerate(arm.data.bones)}; names = [g.name for g in body.vertex_groups]; keepg = [i for i, nme in enumerate(names) if nme in bone_index]; col = {gi: k for k, gi in enumerate(keepg)}
nat = [names[i] for i in keepg]; NV = len(body.data.vertices); Wn = np.zeros((NV, len(keepg)))
for v in body.data.vertices:
    for g in v.groups:
        if g.group in col and g.weight > 0: Wn[v.index, col[g.group]] = g.weight
W8 = Wn / Wn.sum(1, keepdims=True)
W4 = W8.copy(); drop = np.argsort(-W4, axis=1)[:, 4:]; np.put_along_axis(W4, drop, 0.0, 1); W4 /= W4.sum(1, keepdims=True)                      # top-4 by weight, renormalised (the glTF single-set limit)
body.data.calc_loop_triangles(); TRI = np.array([lt.vertices[:] for lt in body.data.loop_triangles]); v_tris = [[] for _ in range(NV)]
for ti, t in enumerate(TRI):
    for v in t: v_tris[v].append(ti)
tri_dom = (W8[TRI[:, 0]] + W8[TRI[:, 1]] + W8[TRI[:, 2]]).argmax(1); tri_sets = [set(t) for t in TRI.tolist()]; REST_B = {b.name: np.array(b.matrix_local) for b in arm.data.bones}
VI = [8165, 8204, 8172, 8052, 8123, 8124]

def wv():
    dg = bpy.context.evaluated_depsgraph_get(); ev = body.evaluated_get(dg); me = ev.to_mesh(); a = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", a); ev.to_mesh_clear(); return a.reshape(-1, 3)
def polar(M):
    U, _, Vt = np.linalg.svd(M); R = U @ Vt
    if np.linalg.det(R) < 0: U[:, -1] *= -1; R = U @ Vt
    return R
def skin_mats():
    S = {pb.name: np.array(pb.matrix) @ np.linalg.inv(REST_B[pb.name]) for pb in pbs}; return S, {k: polar(v[:3, :3]) for k, v in S.items()}
def rel(Srot, a, b):
    Rb = polar(REST_B[b][:3, :3]); return Rb.T @ (Srot[a].T @ Srot[b]) @ Rb
def st_deg(D):
    tw, sw, _, _, ill = ST.swing_twist(D); return dict(total_deg=round(math.degrees(ST.angle_of(D)), 2), swing_deg=round(math.degrees(sw), 2), twist_deg=round(math.degrees(tw), 2), ill_conditioned=ill)
head = lambda nme: np.array(pbs[nme].head); tail = lambda nme: np.array(pbs[nme].tail); rot = lambda nme: np.array(pbs[nme].matrix.to_3x3())
def reset():
    for pb in pbs: pb.matrix_basis = Matrix.Identity(4)
    upd()
def volume(X): return float(np.einsum("ij,ij->i", X[TRI[:, 0]], np.cross(X[TRI[:, 1]], X[TRI[:, 2]])).sum() / 6)
def tcross(X): return np.cross(X[TRI[:, 1]] - X[TRI[:, 0]], X[TRI[:, 2]] - X[TRI[:, 0]])
reset(); mod.use_deform_preserve_volume = False; upd(); REST = wv(); V_rest = volume(REST); CR = tcross(REST); A_rest = 0.5 * np.linalg.norm(CR, axis=1); N_rest = CR / np.linalg.norm(CR, axis=1, keepdims=True)
VA_rest = np.array([A_rest[v_tris[v]].sum() for v in range(NV)])
ARM_L = np.nonzero(W8[:, [nat.index(b) for b in ("clavicle.L", "shoulder01.L", "upperarm01.L", "upperarm02.L")]].sum(1) > 0.05)[0]; TRI_ARM_L = np.nonzero(np.isin(TRI, ARM_L).any(1))[0]

def self_intersections(X):
    t = BVHTree.FromPolygons([tuple(p) for p in X.tolist()], TRI.tolist(), all_triangles=True, epsilon=0.0); return sum(1 for i, j in t.overlap(t) if i < j and not (tri_sets[i] & tri_sets[j]))
def metrics(X, Srot, S=None, si=True):
    C = tcross(X); A = 0.5 * np.linalg.norm(C, axis=1); r = A / A_rest; Nn = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-20); vr = np.array([A[v_tris[v]].sum() for v in range(NV)]) / VA_rest
    Rd = np.stack([Srot[nat[k]] for k in tri_dom]); flips = (Nn * np.einsum("tij,tj->ti", Rd, N_rest)).sum(1) < 0
    o = dict(volume_pct=round((volume(X) / V_rest - 1) * 100, 3), ring_min=round(float(vr.min()), 4), ring_argmin=int(vr.argmin()), ring_p1=round(float(np.percentile(vr, 1)), 4), tri_lt_0p5=int((r < 0.5).sum()), tri_lt_0p25=int((r < 0.25).sum()),
             flips_dom=int(flips.sum()), left_shoulder_region=dict(triangles=int(len(TRI_ARM_L)), tri_lt_0p5=int((r[TRI_ARM_L] < 0.5).sum()), tri_lt_0p25=int((r[TRI_ARM_L] < 0.25).sum()), tri_gt_2=int((r[TRI_ARM_L] > 2).sum()), flips_dom=int(flips[TRI_ARM_L].sum()),
                                        ring_min=round(float(vr[ARM_L].min()), 4), ring_max=round(float(vr[ARM_L].max()), 4)),
             ring_at={str(v): round(float(vr[v]), 3) for v in VI})
    if S is not None: o["det_blend"] = {str(v): round(float(np.linalg.det(sum(W8[v, k] * S[nat[k]][:3, :3] for k in np.nonzero(W8[v] > 0)[0]))), 4) for v in (8052, 8123, 8124, 8165)}
    if si: o["self_intersections"] = self_intersections(X)
    return o
def lbs_numpy(S, W):
    X = np.zeros((NV, 3)); R1 = np.c_[REST, np.ones(NV)]
    for k, nme in enumerate(nat):
        m = W[:, k] > 0
        if m.any(): X[m] += W[m, k, None] * (R1[m] @ S[nme][:3].T)
    return X

RT = PRT.Retargeter(arm); POSES = PRT.load_poses()
# ---------------- rest facts used by the constructions
reset(); REST_ARM = {}
for s in "LR":
    S_, E_, Wr_ = head(f"upperarm01.{s}"), head(f"lowerarm01.{s}"), head(f"wrist.{s}"); u = ST.unit(E_ - S_); f = ST.unit(Wr_ - E_); Ru = rot(f"upperarm01.{s}"); Rf = rot(f"lowerarm01.{s}")
    hinge_w = ST.unit(np.cross(u, f)); flex = math.degrees(math.acos(np.clip(u @ f, -1, 1)))
    REST_ARM[s] = dict(a=float(np.linalg.norm(E_ - S_)), b=float(np.linalg.norm(Wr_ - E_)), e_loc=Ru.T @ u, h_loc=Ru.T @ hinge_w, f_loc=Rf.T @ f, rest_elbow_flexion_deg=flex,
                       limb_line_vs_bone_axis_deg=math.degrees(math.acos(np.clip(u @ Ru[:, 1], -1, 1))), forearm_line_vs_bone_axis_deg=math.degrees(math.acos(np.clip(f @ Rf[:, 1], -1, 1))))
RT.neutral(); upd(); NEUT = {}
for s in "LR":
    S_, E_, Wr_ = head(f"upperarm01.{s}"), head(f"lowerarm01.{s}"), head(f"wrist.{s}"); hn = ST.unit(np.cross(E_ - S_, Wr_ - E_)); hr = rot(f"upperarm01.{s}") @ REST_ARM[s]["h_loc"]
    NEUT[s] = dict(hinge_world=hn.round(4).tolist(), angle_rest_hinge_vs_neutral_plane_normal_deg=round(math.degrees(math.acos(np.clip(abs(hn @ hr), -1, 1))), 2))

def rebuild_arm(side, contact, t, W0, mode, f02=0.0, share=0.0, swivel_deg=0.0):
    """Rebuild one contact arm. Returns a record of what was done."""
    A = REST_ARM[side]; U, U2, F, SH = pbs[f"upperarm01.{side}"], pbs[f"upperarm02.{side}"], pbs[f"lowerarm01.{side}"], pbs[f"shoulder01.{side}"]
    U.matrix_basis = Matrix.Identity(4); F.matrix_basis = Matrix.Identity(4); upd()                         # parent-carried rest
    sx = 1.0 if side == "L" else -1.0; chest = RT.acc.get("spine_03", np.eye(3)); pole = chest @ (np.array(contact["pole"], float) * np.array([sx, 1.0, 1.0]))
    Tf = RT.landmark(contact["target"], side); Tg = W0 + (Tf - W0) * t; rec = dict(mode=mode, f02=f02, share=share, swivel_deg=swivel_deg)
    def solve():
        S_ = head(U.name); pl = pole if not swivel_deg else ST.axis_angle(Tg - S_, math.radians(swivel_deg)) @ (pole - ST.unit(Tg - S_) * (pole @ ST.unit(Tg - S_)))   # swivel: turn the elbow about the shoulder->target axis
        E, Wt, cl = ST.two_bone_ik(S_, Tg, A["a"], A["b"], pl); return S_, E, Wt, cl
    S_, E, Wt, cl = solve()
    if share > 0:                                                                                           # give part of the arm swing to shoulder01, then solve again from the moved shoulder
        d0 = rot(U.name) @ A["e_loc"]; u0 = ST.unit(E - S_); ax = np.cross(d0, u0); ang = math.atan2(np.linalg.norm(ax), d0 @ u0)
        RT._rot_about_head(SH, ST.axis_angle(ax, share * ang)); S_, E, Wt, cl = solve(); rec["shoulder01_rotation_deg"] = round(math.degrees(share * ang), 2)
    Rc = rot(U.name); u = ST.unit(E - S_); fdir = ST.unit(Wt - E)
    if mode == "zero": Rn = ST.min_arc(Rc @ A["e_loc"], u) @ Rc
    elif mode == "hinge": Rn = ST.frame_from_two(A["e_loc"], A["h_loc"], u, np.cross(u, fdir))
    else: raise ValueError(mode)
    RT._rot_about_head(U, Rn @ Rc.T)
    g0 = rot(F.name) @ A["f_loc"]; Rarc = ST.min_arc(g0, ST.unit(Wt - head(F.name))); RT._rot_about_head(F, Rarc)
    ax = np.cross(g0, fdir); rec["forearm_arc_axis_vs_arm_plane_normal_deg"] = round(math.degrees(math.acos(np.clip(abs(ST.unit(ax) @ ST.unit(np.cross(u, fdir))), -1, 1))), 2) if np.linalg.norm(ax) > 1e-9 else None
    if f02:
        _, Srot = skin_mats(); tau = st_deg(rel(Srot, f"clavicle.{side}", U.name))["twist_deg"]; keep2 = U2.matrix.copy()
        U.matrix = U.matrix @ Matrix.Rotation(math.radians(-f02 * tau), 4, "Y"); upd(); U2.matrix = keep2; upd(); rec["twist_moved_to_upperarm02_deg"] = round(f02 * tau, 2)
    rec.update(elbow_world_m=E.round(5).tolist(), reach_clamped=cl, target=contact["target"], wrist_to_target_mm=round(float(np.linalg.norm(head(f"wrist.{side}") - Tg) * 1e3), 3), elbow_to_ik_elbow_mm=round(float(np.linalg.norm(head(F.name) - E) * 1e3), 3))
    return rec

def apply_variant(pose, variant, t=1.0):
    """variant: 'baseline' | 'authored_only' | dict(mode, f02, share). Returns info dict."""
    spec = POSES[pose]; contacts = spec.get("hand_contacts") or {}; info = dict(contacts={})
    nc = copy.deepcopy(spec); nc.pop("hand_contacts", None); reset(); RT.apply(nc, t); upd(); W0 = {s: head(f"wrist.{s}") for s in contacts}
    if variant == "authored_only":
        for s, c in contacts.items(): info["contacts"][s] = dict(target=c["target"], wrist_to_target_mm=round(float(np.linalg.norm(head(f"wrist.{s}") - RT.landmark(c["target"], s)) * 1e3), 3))
    elif variant == "baseline":
        reset(); RT.apply(spec, t); upd(); acc_keep = RT.acc
        for s, c in contacts.items():                                                                       # note: for a two-arm pose the landmark is evaluated after both arms are solved (reported, not used)
            Tg = W0[s] + (RT.landmark(c["target"], s) - W0[s]) * t; info["contacts"][s] = dict(target=c["target"], wrist_to_target_mm=round(float(np.linalg.norm(head(f"wrist.{s}") - Tg) * 1e3), 3))
    else:
        for s, c in contacts.items(): info["contacts"][s] = rebuild_arm(s, c, t, W0[s], **variant)
    upd(); return info

def measure(pose, variant, si=True, dqs=True, weights4=False):
    info = apply_variant(pose, variant); S, Srot = skin_mats(); out = dict(variant=variant, **info, joints={})
    for s in "LR":
        out["joints"][s] = dict(clav_to_ua01=st_deg(rel(Srot, f"clavicle.{s}", f"upperarm01.{s}")), ua01_to_ua02=st_deg(rel(Srot, f"upperarm01.{s}", f"upperarm02.{s}")), elbow=st_deg(rel(Srot, f"upperarm02.{s}", f"lowerarm01.{s}")),
                                wrist_joint=st_deg(rel(Srot, f"lowerarm02.{s}", f"wrist.{s}")), clav_to_shoulder01=st_deg(rel(Srot, f"clavicle.{s}", f"shoulder01.{s}")),
                                wrist_head_m=head(f"wrist.{s}").round(5).tolist(), hand_rot=rot(f"wrist.{s}").round(6).tolist())
    mod.use_deform_preserve_volume = False; upd(); XL = wv(); out["LBS"] = metrics(XL, Srot, S, si)
    if dqs: mod.use_deform_preserve_volume = True; upd(); XD = wv(); mod.use_deform_preserve_volume = False; upd(); out["DQS"] = metrics(XD, Srot, None, si); d = np.linalg.norm(XL - XD, axis=1); out["LBS_vs_DQS_mm"] = dict(max=round(float(d.max() * 1e3), 2), p99=round(float(np.percentile(d, 99) * 1e3), 2))
    if weights4:
        X8 = lbs_numpy(S, W8); X4 = lbs_numpy(S, W4); d = np.linalg.norm(X8 - X4, axis=1)
        out["weights_8_vs_top4"] = dict(numpy_lbs_vs_blender_lbs_max_mm=round(float(np.linalg.norm(X8 - XL, axis=1).max() * 1e3), 6), max_mm=round(float(d.max() * 1e3), 4), argmax_vertex=int(d.argmax()), at_8165_mm=round(float(d[8165] * 1e3), 4),
                                        p99_mm=round(float(np.percentile(d, 99) * 1e3), 4), vertices_over_1mm=int((d > 1e-3).sum()), vertices_with_more_than_4_weights=int(((W8 > 0).sum(1) > 4).sum()))
        out["_X4_volume_pct"] = round((volume(X4) / V_rest - 1) * 100, 3)
    # rig / glTF compatibility of the pose that was built: every bone's local matrix must be a rigid rotation with its rest offset (TRS, S = 1, no shear, no stretch)
    worst = dict(orth=0.0, det=0.0, trans=0.0)
    for pb in pbs:
        Mb = np.array(pb.matrix_basis); Rb = Mb[:3, :3]; worst["orth"] = max(worst["orth"], float(np.abs(Rb.T @ Rb - np.eye(3)).max())); worst["det"] = max(worst["det"], abs(float(np.linalg.det(Rb)) - 1))
        if pb.name != "root": worst["trans"] = max(worst["trans"], float(np.abs(Mb[:3, 3]).max()))
    out["rig_compat"] = dict(max_basis_orthonormality_error=worst["orth"], max_basis_det_minus_1=worst["det"], max_basis_translation_non_root_m=worst["trans"], bones=len(pbs))
    return out

VARIANTS = {"B0_baseline": "baseline", "B1_authored_only_no_contact": "authored_only", "C1_zero_twist": dict(mode="zero"), "C2_hinge": dict(mode="hinge"), "C2_hinge_f02_0p5": dict(mode="hinge", f02=0.5),
            "C2_hinge_f02_1p0": dict(mode="hinge", f02=1.0), "C3_zero_share_1_3": dict(mode="zero", share=1 / 3), "C4_hinge_share_1_3": dict(mode="hinge", share=1 / 3), "C5_hinge_share_1_3_f02_0p5": dict(mode="hinge", share=1 / 3, f02=0.5)}
res = dict(script="run_pose_construction.py", blender=bpy.app.version_string, inputs_sha256=h_before, vertices=NV, triangles=int(len(TRI)), rest_volume_m3=round(V_rest, 6), rest_arm={s: {k: (v.round(5).tolist() if hasattr(v, "round") else round(v, 4)) for k, v in d.items()} for s, d in REST_ARM.items()},
           neutral=NEUT, left_shoulder_region_vertices=int(len(ARM_L)), pose07={}, all_poses={}, ramp={})
for k, v in VARIANTS.items():
    res["pose07"][k] = measure("POSE_07", v, weights4=True); print("P07", k, res["pose07"][k]["joints"]["L"]["clav_to_ua01"], res["pose07"][k]["LBS"]["volume_pct"], round(time.time() - T0, 1), flush=True)
# hand placement and orientation relative to the baseline
b0 = res["pose07"]["B0_baseline"]["joints"]["L"]
for k, r in res["pose07"].items():
    j = r["joints"]["L"]; r["vs_baseline_L"] = dict(wrist_shift_mm=round(float(np.linalg.norm(np.array(j["wrist_head_m"]) - np.array(b0["wrist_head_m"])) * 1e3), 3), hand_orientation_change_deg=round(math.degrees(ST.angle_of(np.array(j["hand_rot"]) @ np.array(b0["hand_rot"]).T)), 2))

# ---------------- swivel scan (POSE_07): the one free parameter of a two-bone reach is the elbow's angle about the shoulder->wrist axis
if "--scan" in sys.argv:
    scan = []; E0 = None
    for phi in range(-180, 180, 10):
        row = dict(swivel_deg=phi)
        for mode in ("hinge", "zero"):
            r = measure("POSE_07", dict(mode=mode, swivel_deg=float(phi)), si=True, dqs=True); c = r["contacts"]["L"]; j = r["joints"]["L"]
            row[mode] = dict(shoulder=j["clav_to_ua01"], elbow=j["elbow"], forearm_arc_axis_vs_arm_plane_normal_deg=c["forearm_arc_axis_vs_arm_plane_normal_deg"], LBS={k: r["LBS"][k] for k in ("volume_pct", "ring_min", "tri_lt_0p5", "tri_lt_0p25", "flips_dom", "self_intersections", "det_blend")},
                             region=r["LBS"]["left_shoulder_region"], DQS={k: r["DQS"][k] for k in ("volume_pct", "tri_lt_0p25", "flips_dom", "self_intersections")}, wrist_to_target_mm=c["wrist_to_target_mm"])
            row["elbow_world_m"] = c["elbow_world_m"]
        if phi == 0: E0 = np.array(row["elbow_world_m"])
        scan.append(row); print("scan", phi, row["hinge"]["shoulder"]["twist_deg"], row["hinge"]["LBS"]["volume_pct"], row["hinge"]["LBS"]["tri_lt_0p25"], row["hinge"]["LBS"]["self_intersections"], "| zero axis err", row["zero"]["forearm_arc_axis_vs_arm_plane_normal_deg"], row["elbow_world_m"], flush=True)
    Sh = None
    for row in scan: row["elbow_shift_from_authored_pole_mm"] = round(float(np.linalg.norm(np.array(row["elbow_world_m"]) - E0) * 1e3), 1)
    json.dump(dict(pose="POSE_07", side="L", note="hinge = rest elbow hinge on the arm-plane normal; zero = one minimal arc from the parent-carried rest. swivel 0 = the authored pole.", inputs_sha256=h_before, scan=scan,
                   inputs_unchanged={f: sha(INP / f) for f in FILES} == h_before), open(OUT / "swivel_scan.json", "w"), indent=1); print("SCAN DONE", round(time.time() - T0, 1)); sys.exit(0)

# ---------------- generalisation: every pose, baseline against the two constructions (LBS; DQS volume only)
if not QUICK:
    for p in sorted(POSES):
        res["all_poses"][p] = {}
        for k in ("B0_baseline", "C1_zero_twist", "C2_hinge"):
            r = measure(p, VARIANTS[k], si=True, dqs=True); res["all_poses"][p][k] = dict(contacts=r["contacts"], shoulder={s: r["joints"][s]["clav_to_ua01"] for s in "LR"}, elbow={s: r["joints"][s]["elbow"] for s in "LR"},
                LBS={kk: r["LBS"][kk] for kk in ("volume_pct", "ring_min", "ring_p1", "tri_lt_0p5", "tri_lt_0p25", "flips_dom", "self_intersections")}, DQS_volume_pct=r["DQS"]["volume_pct"], rig_compat=r["rig_compat"])
        print(p, {k: (v["LBS"]["volume_pct"], v["LBS"]["tri_lt_0p25"], v["LBS"]["self_intersections"]) for k, v in res["all_poses"][p].items()}, round(time.time() - T0, 1), flush=True)

# ---------------- continuity along the cloth ramp t = 0 .. 1 (POSE_07, left arm)
ts = np.linspace(0, 1, 41)
for k in ("B0_baseline", "C1_zero_twist", "C2_hinge"):
    tw, sw, Rs, vol = [], [], [], []
    for t in ts[1:]:                                                                                        # t = 0 has no contact stage in the engine (guarded by t > 0)
        if VARIANTS[k] == "baseline": reset(); RT.apply(POSES["POSE_07"], float(t)); upd()
        else: apply_variant("POSE_07", VARIANTS[k], float(t))
        _, Srot = skin_mats(); d = st_deg(rel(Srot, "clavicle.L", "upperarm01.L")); tw.append(d["twist_deg"]); sw.append(d["swing_deg"]); Rs.append(rot("upperarm01.L")); vol.append(round((volume(wv()) / V_rest - 1) * 100, 3))
    step = [math.degrees(ST.angle_of(Rs[i + 1] @ Rs[i].T)) for i in range(len(Rs) - 1)]; dtw = np.diff(np.unwrap(np.radians(tw)))
    res["ramp"][k] = dict(t=ts[1:].round(4).tolist(), twist_deg=tw, swing_deg=sw, volume_pct_LBS=vol, max_step_rotation_deg=round(max(step), 2), median_step_rotation_deg=round(float(np.median(step)), 2),
                          max_abs_twist_step_deg=round(float(np.degrees(np.abs(dtw)).max()), 2), max_abs_twist_deg=round(float(np.abs(tw).max()), 2), min_volume_pct=min(vol), any_step_over_90_deg=bool(max(step) > 90))
    print("ramp", k, res["ramp"][k]["max_abs_twist_deg"], res["ramp"][k]["max_step_rotation_deg"], res["ramp"][k]["min_volume_pct"], round(time.time() - T0, 1), flush=True)

h_after = {f: sha(INP / f) for f in FILES}; live = {"03_full_rigged.blend": HE / "src_garment_pipeline/human_build/human_engine/mannequin_modes/03_full_rigged/out/03_full_rigged.blend"}
res["inputs_unchanged"] = h_after == h_before; res["live_engine_rig_sha256"] = {k: (sha(v) if v.exists() else None) for k, v in live.items()}; res["runtime_s"] = round(time.time() - T0, 1)
json.dump(res, open(OUT / ("pose_construction_quick.json" if QUICK else "pose_construction.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)); print("DONE", res["runtime_s"], res["inputs_unchanged"])
