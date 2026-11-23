"""posekit.py  Shared code for paper 01 steps 2, 2b and 3: open the canonical rig read-only, pose it by limb-line construction, measure the static gates,
read and write versioned pose files.

Canonical rig: REQ-04's copy of 03_full_rigged.blend (sha256 1be2d4ca..., equal to the live engine file
HUMAN_ENGINE/src_garment_pipeline/human_build/human_engine/mannequin_modes/03_full_rigged/out/03_full_rigged.blend). It is opened, never saved.
The canonical GLB mannequin_full_rigged_122276_morphed.glb is not read or written by anything here.

Frame (P3 report): Blender world, metres, armature and body world matrices identity; the model faces -Y, +X is its left, +Z up. Floor z = 0
(lowest rest body vertex 1.7e-8 m).

Rules from the papers, applied here:
  - the canonical rest pose stays the bind pose; a pose is a set of pose-bone matrix_basis values, never applied as rest (base layer 0 rule 4);
  - every limb rotation is ONE minimal arc from the parent-carried rest that aims the joint-to-joint limb line (base layer 0 rule 3, P3 "zero");
  - linear blend skinning only (the body's Armature modifier is checked to have preserve-volume off; nothing switches it on);
  - metric definitions are the harness's (03_ENGINE/avatar_engine/harness/gates.py), which are P3's and P2's.
"""
import sys, json, math, hashlib, time
from pathlib import Path
import numpy as np
import bpy                                    # import bpy before mathutils (bpy 5.2 module)
from mathutils import Matrix, Quaternion
from mathutils.bvhtree import BVHTree

HERE = Path(__file__).resolve().parent; AE = HERE.parent; WS = AE.parents[1]
sys.path.insert(0, str(WS / "03_ENGINE/prototypes/P3_pose_swing_twist")); sys.path.insert(0, str(WS / "03_ENGINE/prototypes/P2_clearance_field"))
import swing_twist as ST  # noqa: E402

RIG_BLEND = WS / "08_KNOWLEDGE/stage3_engineering/opus_requests/REQ-04_pose_swing_twist_dqs/inputs/03_full_rigged.blend"
RIG_SHA = "1be2d4ca305dca0a197215f6adc8181a2fe9853b03f5d316e77169e3aee82148"
FLOOR_Z = 0.0
POSE_SCHEMA = "betterstyle.avatar_pose/1"
X_, Y_, Z_ = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
DOWN = -Z_; FWD = -Y_


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def unit(v): v = np.asarray(v, float); return v / np.linalg.norm(v)
def ang(u, v): return math.degrees(math.acos(float(np.clip(unit(u) @ unit(v), -1, 1))))
def rot(axis, deg): return ST.axis_angle(unit(axis), math.radians(deg))


def polar(M):
    U, _, Vt = np.linalg.svd(M); R = U @ Vt
    if np.linalg.det(R) < 0: U[:, -1] *= -1; R = U @ Vt
    return R


class Rig:
    """The canonical body and rig, opened read-only. All positions are world = armature space (identity object matrices)."""

    def __init__(self):
        h0 = sha(RIG_BLEND)
        if h0 != RIG_SHA: raise SystemExit(f"canonical rig hash mismatch: {h0}")
        bpy.ops.wm.open_mainfile(filepath=str(RIG_BLEND), load_ui=False)
        self.arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
        self.body = next(o for o in bpy.data.objects if o.type == "MESH" and any(m.type == "ARMATURE" for m in o.modifiers))
        assert np.allclose(np.array(self.arm.matrix_world), np.eye(4)) and np.allclose(np.array(self.body.matrix_world), np.eye(4))
        mods = [m for m in self.body.modifiers if m.type == "ARMATURE"]
        self.lbs = all(not m.use_deform_preserve_volume for m in mods)
        if not self.lbs: raise SystemExit("body armature modifier uses dual quaternion (preserve volume); the papers require LBS")
        self.pbs = self.arm.pose.bones; self.bones = self.arm.data.bones; self.names = [b.name for b in self.bones]
        self.R0 = {b.name: polar(np.array(b.matrix_local)[:3, :3]) for b in self.bones}
        me = self.body.data; me.calc_loop_triangles()
        self.TRI = np.array([lt.vertices[:] for lt in me.loop_triangles], np.int64)
        co = np.array([v.co[:] for v in me.vertices]); _, W = np.unique(np.round(co / 1e-6).astype(np.int64), axis=0, return_inverse=True); self.WELD = W.ravel()
        self.TRI_SETS = [set(t) for t in self.WELD[self.TRI].tolist()]
        self.cz = co[self.TRI].mean(1)[:, 2]                         # bind-pose triangle centroid height, for the harness's region split
        vg = {g.name: g.index for g in self.body.vertex_groups}
        def group_idx(prefixes):
            ids = {vg[n] for n in vg if n.startswith(prefixes)}
            return [v.index for v in me.vertices if any(g.group in ids and g.weight > 0.5 for g in v.groups)]
        self.footv = {}
        for s in "LR":                                                # same rule as capture_walk_blender.py and evidence/measure_bs_walk.py
            cand = group_idx((f"foot.{s}", "toe"))
            self.footv[s] = np.array([i for i in cand if any(self.body.vertex_groups[g.group].name.endswith("." + s) for g in me.vertices[i].groups if g.weight > 0.5)], np.int64)
        best = {}
        for v in me.vertices:
            b = max(v.groups, key=lambda g: g.weight, default=None); best[v.index] = self.body.vertex_groups[b.group].name if b else ""
        self.dominant = np.array([best[i] for i in range(len(me.vertices))])
        self.reset(); self.X0 = self.eval_body(); self.V0 = self.volume(self.X0); self.A0 = self.tri_area(self.X0)
        self.self0 = self.self_pairs(self.X0)

    # ------------------------------------------------------------------ posing
    def reset(self):
        for pb in self.pbs: pb.matrix_basis = Matrix.Identity(4)
        self.update()

    def update(self): bpy.context.view_layer.update()
    def H(self, n): return np.array(self.pbs[n].head)
    def T(self, n): return np.array(self.pbs[n].tail)
    def Rw(self, n): return polar(np.array(self.pbs[n].matrix)[:3, :3])
    def rel(self, n): return self.Rw(n) @ self.R0[n].T                  # rest-relative world rotation of bone n

    def rot_world(self, n, R, pivot=None):
        """Turn bone n (children carried) by the world rotation R about `pivot` (default: its head)."""
        pb = self.pbs[n]; p = self.H(n) if pivot is None else np.asarray(pivot, float)
        M = np.eye(4); M[:3, :3] = R; T = np.eye(4); T[:3, 3] = p; Ti = np.eye(4); Ti[:3, 3] = -p
        pb.matrix = Matrix((T @ M @ Ti @ np.array(pb.matrix)).tolist()); self.update()

    def translate_world(self, n, d):
        pb = self.pbs[n]; M = np.array(pb.matrix); M[:3, 3] += np.asarray(d, float); pb.matrix = Matrix(M.tolist()); self.update()

    def aim(self, n, joint_a, joint_b, target_dir):
        """One minimal arc on bone n that turns the limb line joint_a -> joint_b onto target_dir. Returns the arc angle in degrees."""
        d0 = unit(self.H(joint_b) - self.H(joint_a)); d1 = unit(target_dir); R = ST.min_arc(d0, d1); self.rot_world(n, R, self.H(joint_a))
        return ang(d0, d1)

    def pose_dict(self, tol=1e-9):
        out = {}
        for pb in self.pbs:
            M = np.array(pb.matrix_basis)
            if np.abs(M - np.eye(4)).max() > tol:
                q = Matrix(M.tolist()).to_quaternion().normalized(); t = M[:3, 3]
                out[pb.name] = dict(q_wxyz=[round(float(x), 9) for x in q], t=[round(float(x), 9) for x in t])
        return out

    def apply(self, bones):
        self.reset()
        for n, v in bones.items():
            pb = self.pbs[n]; R = Quaternion(v["q_wxyz"]).to_matrix().to_4x4(); R.translation = v["t"]; pb.matrix_basis = R
        self.update()

    # ------------------------------------------------------------------ evaluation
    def eval_body(self):
        dg = bpy.context.evaluated_depsgraph_get(); ev = self.body.evaluated_get(dg); me = ev.to_mesh(); a = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", a); ev.to_mesh_clear(); return a.reshape(-1, 3)

    def volume(self, X): T = self.TRI; return float(np.einsum("ij,ij->i", X[T[:, 0]], np.cross(X[T[:, 1]], X[T[:, 2]])).sum() / 6)
    def tri_area(self, X): T = self.TRI; return 0.5 * np.linalg.norm(np.cross(X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]]), axis=1)

    def self_pairs(self, X):
        """Non-adjacent intersecting triangle pairs on the position-welded body (P2 / REQ-04 rule), split head (bind z > 1.45 m) /
        armpit band (1.25 to 1.40 m, P2's band) / other, as capture_walk_blender.py does."""
        t = BVHTree.FromPolygons([tuple(p) for p in X.tolist()], self.TRI.tolist(), all_triangles=True, epsilon=0.0); c = np.zeros(3, np.int64); pairs = []
        for i, j in t.overlap(t):
            if i < j and not (self.TRI_SETS[i] & self.TRI_SETS[j]):
                z = (self.cz[i] + self.cz[j]) / 2; r = 0 if z > 1.45 else (1 if 1.25 < z < 1.40 else 2); c[r] += 1
                if r: pairs.append((i, j))
        return dict(head=int(c[0]), armpit_band=int(c[1]), other=int(c[2]), non_head=int(c[1] + c[2]), _pairs=pairs)

    def twist(self, a, b):
        """P3 definition used by the harness: D = Rb^T (Sa^T Sb) Rb, twist about the child bone's own rest Y axis."""
        Rb = self.R0[b]; Sa = self.rel(a); Sb = self.rel(b); D = Rb.T @ (Sa.T @ Sb) @ Rb; tw, sw, _, _, ill = ST.swing_twist(D)
        return dict(twist_deg=round(math.degrees(tw), 3), swing_deg=round(math.degrees(sw), 3), ill_conditioned=bool(ill))

    def foot_axis(self, s):
        """Foot long axis: ankle (foot head) to the second toe's base (toe2-1 head), horizontal. At rest this equals heel centre -> toe2-1 within 0.03 deg.
        Toe-out positive = toes turned away from the midline."""
        d = self.H(f"toe2-1.{s}") - self.H(f"foot.{s}"); a = math.degrees(math.atan2(d[0], -d[1])); return a if s == "L" else -a

    def palm_normal(self, s):
        """Unit normal of the palm, pointing to the palm side. Plane: wrist, index knuckle (finger2-1 head), little-finger knuckle (finger5-1 head).
        Side: the direction the fingers already curl at rest (finger3 tip direction against its base direction)."""
        w, k2, k5 = self.H(f"wrist.{s}"), self.H(f"finger2-1.{s}"), self.H(f"finger5-1.{s}")
        n = unit(np.cross(k2 - w, k5 - w)); d1 = unit(self.T("finger3-1." + s) - self.H("finger3-1." + s)); d3 = unit(self.T("finger3-3." + s) - self.H("finger3-3." + s))
        c = d3 - (d3 @ d1) * d1
        return n if n @ c > 0 else -n

    def metrics(self, X=None, ref=None, label=""):
        """Static measurements on the current pose. ref: dict from an earlier metrics() call (the theta_fit reference) for the 'no more than' gates."""
        X = self.eval_body() if X is None else X; m = dict(label=label)
        fz = {s: float(X[self.footv[s], 2].min() - FLOOR_Z) * 1e3 for s in "LR"}
        m["foot_lowest_mm"] = {s: round(v, 3) for s, v in fz.items()}; m["body_lowest_mm"] = round(float(X[:, 2].min() - FLOOR_Z) * 1e3, 3)
        m["ankle_separation_cm"] = round(float(self.H("foot.L")[0] - self.H("foot.R")[0]) * 100, 3)
        m["hip_separation_cm"] = round(float(self.H("upperleg01.L")[0] - self.H("upperleg01.R")[0]) * 100, 3)
        m["toe_out_deg"] = {s: round(self.foot_axis(s), 3) for s in "LR"}
        m["toe_out_deg_harness_toe1_rule"] = {s: round(abs(math.degrees(math.atan2((self.H(f"toe1-1.{s}") - self.H(f"foot.{s}"))[0], -(self.H(f"toe1-1.{s}") - self.H(f"foot.{s}"))[1]))), 3) for s in "LR"}
        m["foot_tilt_from_rest_deg"] = {s: round(ang(self.rel(f"foot.{s}") @ Z_, Z_), 3) for s in "LR"}
        arm = {}
        for s in "LR":
            S_, E_, W_ = self.H(f"upperarm01.{s}"), self.H(f"lowerarm01.{s}"), self.H(f"wrist.{s}"); a = W_ - S_
            arm[s] = dict(arm_line_from_vertical_deg=round(ang(a, DOWN), 3), elbow_flexion_deg=round(ang(E_ - S_, W_ - E_), 3),
                          upper_arm_from_vertical_deg=round(ang(E_ - S_, DOWN), 3),
                          arm_line_lateral_deg=round(math.degrees(math.atan2(abs(a[0]), -a[2])), 3), arm_line_forward_deg=round(math.degrees(math.atan2(-a[1], -a[2])), 3),
                          palm_normal=[round(float(x), 4) for x in self.palm_normal(s)])
        m["arm"] = arm
        m["twist"] = {f"{a}->{b}": self.twist(a, b) for s in "LR" for a, b in ((f"clavicle.{s}", f"upperarm01.{s}"), (f"pelvis.{s}", f"upperleg01.{s}"))}
        m["volume_pct_vs_bind"] = round((self.volume(X) / self.V0 - 1) * 100, 4)
        r = self.tri_area(X) / self.A0; m["triangles_lt_quarter"] = int((r < 0.25).sum()); m["triangles_lt_half"] = int((r < 0.5).sum())
        sp = self.self_pairs(X); pairs = sp.pop("_pairs"); m["self_intersection"] = sp
        reg = {}
        for i, j in pairs:
            for t in (i, j):
                b = self.dominant[self.TRI[t][0]]; reg[b] = reg.get(b, 0) + 1
        m["self_intersection_non_head_by_dominant_bone"] = dict(sorted(reg.items(), key=lambda kv: -kv[1])[:12])
        # arm-to-torso and thigh-to-thigh distances (build_archetype's collider check; inner-thigh contact is the paper's side effect of a narrow stance)
        from scipy.spatial import cKDTree
        armk = ("clavicle", "shoulder", "upperarm", "lowerarm", "wrist", "finger", "metacarpal")
        isarm = np.array([any(k in b for k in armk) for b in self.dominant]); band = (X[:, 2] > 1.0) & (X[:, 2] < 1.45)
        dd, _ = cKDTree(X[~isarm & band]).query(X[isarm & band]); m["arm_to_torso_min_mm"] = round(float(dd.min()) * 1e3, 2)
        thL = np.array([b.startswith("upperleg") and b.endswith(".L") for b in self.dominant]); thR = np.array([b.startswith("upperleg") and b.endswith(".R") for b in self.dominant])
        dd, _ = cKDTree(X[thR]).query(X[thL]); m["thigh_to_thigh_min_mm"] = round(float(dd.min()) * 1e3, 2)
        if ref is not None:
            m["volume_pct_vs_reference"] = round((self.volume(X) / ref["_volume"] - 1) * 100, 4)
        m["_volume"] = self.volume(X)
        return m


def _np(o): return o.item() if hasattr(o, "item") else (o.tolist() if hasattr(o, "tolist") else str(o))
def canonical_json(obj): return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_np)


def write_pose(path, kind, version, bones, construction, measured, extra=None):
    """Pose file: version, sha256 over the canonical JSON of (kind, version, rig, bones, construction). Measurements are recorded but not hashed."""
    body = dict(schema=POSE_SCHEMA, kind=kind, version=version, rig=dict(file=str(RIG_BLEND.relative_to(WS)).replace("\\", "/"), sha256=RIG_SHA, bones=152,
                bind_pose="canonical rest; this pose is NOT applied as rest"), skinning="linear blend (no dual quaternion)", bones=bones, construction=construction)
    if extra: body.update(extra)
    h = hashlib.sha256(canonical_json(body).encode()).hexdigest()
    out = dict(body); out["sha256"] = h; out["measured"] = {k: v for k, v in measured.items() if not k.startswith("_")} if isinstance(measured, dict) else measured
    Path(path).parent.mkdir(parents=True, exist_ok=True); Path(path).write_text(json.dumps(out, indent=1, default=_np)); return h


def read_pose(path):
    d = json.loads(Path(path).read_text()); body = {k: v for k, v in d.items() if k not in ("sha256", "measured")}
    h = hashlib.sha256(canonical_json(body).encode()).hexdigest()
    if h != d["sha256"]: raise SystemExit(f"pose file {path} hash mismatch")
    return d


def clean(m): return {k: v for k, v in m.items() if not k.startswith("_")}
