"""build_garment_g0.py — Parametric bone-lofted garment reconstruction for BS-TRACK-001 (G0).

Generates dimensionally accurate, multi-panel 3D garment meshes (G0) fitted to the
canonical M2_ALABASTER_CHALK mannequin in theta_fit (arms 12 deg from vertical):
  - Jacket Variant A (bracket chest + horizontal chest piping + BST back shield)
  - Jacket Variant B (orbit star chest + no horizontal chest seam + BST back shield)
  - Track Pants (waistband, curved front/side 3D piping, groin [BETTERSTYLE] bracket logo,
    left-thigh swoop mark, and gathered elastic ankle cuffs)

Run in cloud container or local Blender 5.2.3:
  python3 scripts/build_garment_g0.py --variant both
"""
import argparse, math, re
from pathlib import Path
import numpy as np
import bpy
import bmesh
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

HERE = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
REPO = next((p for p in [HERE, *HERE.parents] if (p / 'BETTER_STYLE_VISUAL_CURATION').is_dir()), HERE)
VIDEO_DIR = REPO / 'BETTER_STYLE_VISUAL_CURATION/14_TRACKSUIT_DUAL_STATE_VIDEO'
GARMENTS_DIR = VIDEO_DIR / 'garments'
TEX_DIR = GARMENTS_DIR / 'textures'

SPEC = {
    'mannequin_blend': str(REPO / 'BETTER_STYLE_VISUAL_CURATION/11_CANONICAL_MANNEQUINS/BLENDER_ASSETS/M2_ALABASTER_CHALK.blend'),
    'rig_name': 'Human.rig',
    'body_name': 'mannequin',
    'fit_pose': {'arm_down_deg': 29.0, 'forearm_straighten_deg': 12.0},
    # Confirmed target measurements (mm -> metres)
    'jacket_length_m': 0.680,
    'half_chest_m': 0.305,       # 610 mm pit-to-pit boxy fit
    'collar_height_m': 0.062,
    'pants_outseam_m': 1.040,
    'piping_radius_m': 0.0028,   # 2.8 mm radius contrast white piping cord
    'piping_proud_m': 0.0025,    # piping sits 2.5 mm proud of the nylon
    'jet_black_rgb': (0.018, 0.018, 0.020, 1.0),
    'piping_white_rgb': (0.95, 0.95, 0.95, 1.0),
    'pushout_floor_m': 0.006,    # any garment vertex closer than this to the body is pushed to loc + n * 0.006
}


def log(*a):
    print('[build_g0]', *a, flush=True)


# ---- REQ-02 instrumentation (logging only). Defaults reproduce the original behaviour exactly.
PROBE = None            # callable(stage_name, payload) or None
PUSH_PASSES = 3         # original: range(3)
PUSH_SMOOTH = True      # original: smooth_vert after passes with it < PUSH_SMOOTH_UNTIL
PUSH_SMOOTH_UNTIL = 2   # original: `if it < 2`
LOCKZ_DISABLED = False  # original: lock_z active


def _probe(name, payload):
    if PROBE is not None: PROBE(name, payload)


def _snap(bm, name, override=None):
    if PROBE is None: return
    P = np.array([tuple(v.co) for v in bm.verts], dtype=np.float64)
    if override:
        for i, co in override.items(): P[i] = co
    PROBE(name, P)


# ============================================================================== 1. DECAL EXTRACTION
def extract_alpha_decal(src_path: Path, dst_path: Path, contrast_boost=2.4, threshold=0.42):
    """Converts a cropped photo of a white-on-black garment logo into a clean RGBA decal."""
    img = bpy.data.images.load(str(src_path))
    w, h = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape((h, w, 4))
    bpy.data.images.remove(img)

    # Luminance from RGB
    lum = 0.299 * px[:, :, 0] + 0.587 * px[:, :, 1] + 0.114 * px[:, :, 2]
    # Remove dark fabric background via percentile floor
    bg_floor = np.percentile(lum, 25)
    norm_lum = np.clip((lum - bg_floor) / max(1e-4, (np.percentile(lum, 99) - bg_floor)), 0.0, 1.0)
    alpha = np.clip((norm_lum - threshold) * contrast_boost, 0.0, 1.0)

    # Feather 4px border to zero so UV projection edges never clamp-bleed
    border = 4
    alpha[:border, :] = 0.0; alpha[-border:, :] = 0.0
    alpha[:, :border] = 0.0; alpha[:, -border:] = 0.0

    out_px = np.ones((h, w, 4), dtype=np.float32)
    out_px[:, :, 0] = 0.95
    out_px[:, :, 1] = 0.95
    out_px[:, :, 2] = 0.94
    out_px[:, :, 3] = alpha

    out_img = bpy.data.images.new('decal_tmp', width=w, height=h, alpha=True)
    out_img.pixels = out_px.ravel().tolist()
    out_img.filepath_raw = str(dst_path)
    out_img.file_format = 'PNG'
    out_img.save()
    bpy.data.images.remove(out_img)
    log(f'extracted RGBA decal: {src_path.name} -> {dst_path.name} ({w}x{h})')


def prepare_all_decals(crop_dir: Path):
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    mapping = {
        'bracket_logo.png': '1_bracket_wordmark_chest.png',
        'bst_shield.png': '2_BST_shield_back.png',
        'orbit_logo.png': '3_orbit_wordmark_chest.png',
    }
    out = {}
    for dst_name, src_name in mapping.items():
        candidates = list(crop_dir.rglob(src_name)) + list(REPO.rglob(src_name))
        dst = TEX_DIR / dst_name
        if candidates:
            extract_alpha_decal(candidates[0], dst, contrast_boost=2.6 if 'BST' not in src_name else 1.8)
            out[dst_name] = dst
        else:
            log(f'WARNING: {src_name} not found; generating procedural fallback decal -> {dst.name}')
            generate_procedural_decal(dst_name, dst)
            out[dst_name] = dst
    return out


def generate_procedural_decal(kind: str, dst_path: Path):
    """Fallback crisp geometric logo generator if raw crop files are in another directory."""
    w, h = 1024, 512
    px = np.zeros((h, w, 4), dtype=np.float32)
    # Draw crisp bracket frame for [BETTERSTYLE]
    if 'bracket' in kind:
        px[80:430, 60:85] = (0.95, 0.95, 0.95, 1.0)
        px[80:105, 60:220] = (0.95, 0.95, 0.95, 1.0)
        px[80:430, 940:965] = (0.95, 0.95, 0.95, 1.0)
        px[395:430, 800:965] = (0.95, 0.95, 0.95, 1.0)
        px[180:330, 130:900] = (0.95, 0.95, 0.95, 0.85)
    img = bpy.data.images.new('proc_decal', width=w, height=h, alpha=True)
    img.pixels = px.ravel().tolist()
    img.filepath_raw = str(dst_path); img.file_format = 'PNG'; img.save()
    bpy.data.images.remove(img)


# ============================================================================== 2. LOAD M2 IN THETA_FIT
def load_theta_fit_mannequin():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    with bpy.data.libraries.load(SPEC['mannequin_blend'], link=False) as (src, dst):
        dst.objects = [SPEC['rig_name'], SPEC['body_name']]
    for o in dst.objects:
        bpy.context.scene.collection.objects.link(o)
    rig, body = dst.objects

    # Apply exact theta_fit (arms 12 deg from vertical)
    M = rig.matrix_world.to_3x3().normalized()
    for side, s in (('L', 1), ('R', -1)):
        pb_u = rig.pose.bones[f'upperarm01.{side}']
        B_u = pb_u.bone.matrix_local.to_3x3()
        R_u = Matrix.Rotation(math.radians(s * SPEC['fit_pose']['arm_down_deg']), 3, 'Y')
        pb_u.rotation_mode = 'QUATERNION'
        pb_u.rotation_quaternion = (B_u.inverted() @ (M.inverted() @ R_u @ M) @ B_u).to_quaternion()

        pb_l = rig.pose.bones[f'lowerarm01.{side}']
        B_l = pb_l.bone.matrix_local.to_3x3()
        R_l = Matrix.Rotation(math.radians(SPEC['fit_pose']['forearm_straighten_deg']), 3, 'X')
        pb_l.rotation_mode = 'QUATERNION'
        pb_l.rotation_quaternion = (B_l.inverted() @ (M.inverted() @ R_l @ M) @ B_l).to_quaternion()

    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    me_eval = bpy.data.meshes.new_from_object(body.evaluated_get(dg))
    me_eval.transform(body.matrix_world)
    baked_body = bpy.data.objects.new('M2_Baked_ThetaFit', me_eval)
    bpy.context.scene.collection.objects.link(baked_body)
    return rig, baked_body



# ============================================================================== 3. BONE-LOFTED PATTERN TUBES
# Nothing here copies or slices the mannequin mesh: garments are analytic quad-grid tubes lofted along the posed bone
# polylines of Human.rig (theta_fit). The body mesh is only used as a BVH push-out floor (+6 mm).
def smoothstep(a, b, x):
    t = min(1.0, max(0.0, (x - a) / (b - a))); return t * t * (3 - 2 * t)


def lerp(a, b, t): return a + (b - a) * t


def bone(rig, name, end='head'):
    pb = rig.pose.bones[name]
    return rig.matrix_world @ (pb.head if end == 'head' else pb.tail)


HEAD_RE = re.compile(r'^(head|jaw|oris|levator|orbicularis|risorius|oculi|temporalis|special|eye|tongue|teeth|nose|brow|ear)')
ARM_RE = re.compile(r'^(upperarm|lowerarm|wrist)')              # sleeve-bearing arm (no hand)
HAND_RE = re.compile(r'^(finger|thumb|metacarpal)')
FOOT_RE = re.compile(r'^(foot|toe)')
LEG_RE = re.compile(r'^(upperleg|lowerleg)')


class BodyRef:
    """Measurement access to the baked theta_fit body: vertex positions plus a dominant-bone category per vertex.
    Garments never reuse these vertices; they only read hull cross-sections and centre lines from them."""
    def __init__(self, baked, orig):
        self.a = np.array([tuple(v.co) for v in baked.data.vertices], dtype=np.float64)
        gname = {g.index: g.name for g in orig.vertex_groups}
        self.name = np.array([gname[max(v.groups, key=lambda e: e.weight).group] if v.groups else 'none' for v in orig.data.vertices])
        assert len(self.name) == len(self.a)
        t = lambda rx: np.array([bool(rx.match(n)) for n in self.name])
        self.head, self.arm, self.hand, self.foot, self.leg = t(HEAD_RE), t(ARM_RE), t(HAND_RE), t(FOOT_RE), t(LEG_RE)
        self.sideL = np.array([n.endswith('.L') for n in self.name]); self.sideR = np.array([n.endswith('.R') for n in self.name])
        self.torso_mask = ~(self.head | self.arm | self.hand | self.foot)                 # spine, pelvis, neck, clavicle, legs above the hem...
        self.torso_arm_mask = ~(self.head | self.hand | self.foot) & ~(self.arm & (self.a[:, 2] < 1.20))   # + shoulder/deltoid verts
        self.lower_mask = ~(self.head | self.arm | self.hand | self.foot)
        self.legL, self.legR = self.leg & self.sideL, self.leg & self.sideR

    def cy(self, z, xmax=0.12):
        m = (np.abs(self.a[:, 2] - z) < 0.012) & (np.abs(self.a[:, 0]) < xmax)
        return float((self.a[m, 1].min() + self.a[m, 1].max()) / 2) if m.any() else -0.03

    def slice_xy(self, mask, z, slab=0.012):
        m = mask & (np.abs(self.a[:, 2] - z) < slab)
        return self.a[m, :2]

    def hull_at(self, mask, z):
        """Convex hull of the body slice at z; widens the slab until the slice has a usable hull."""
        for slab in (0.012, 0.02, 0.03, 0.05):
            H = hull2d(self.slice_xy(mask, z, slab))
            if H is not None: return H
        return None


def superellipse(theta, rx, ry, n):
    c, s = math.cos(theta), math.sin(theta)
    e = 2.0 / n
    return rx * math.copysign(abs(c) ** e, c), ry * math.copysign(abs(s) ** e, s)


class Ring:
    """One cross-section: centre, orthonormal (e1, e2) with e1 x e2 = axis, superellipse radii, gather ripples."""
    def __init__(self, c, e1, e2, rx, ry, n=2.0, amp=0.0, k=0, phase=0.0):
        self.c, self.e1, self.e2, self.rx, self.ry, self.n, self.amp, self.k, self.phase = c, e1, e2, rx, ry, n, amp, k, phase

    def at(self, th):
        px, py = superellipse(th, self.rx, self.ry, self.n)
        if self.amp:
            rip = self.amp * math.sin(self.k * th + self.phase)
            L = math.hypot(px, py) or 1.0; px += rip * px / L; py += rip * py / L
        return self.c + self.e1 * px + self.e2 * py


class Tube:
    def __init__(self, rings, th0=0.0, th1=2 * math.pi, closed=True, nth=64, cap_first=False, cap_last=False):
        self.rings, self.th0, self.th1, self.closed, self.nth = rings, th0, th1, closed, nth
        self.cap_first, self.cap_last = cap_first, cap_last

    def eval(self, th, j):
        j = min(max(j, 0.0), len(self.rings) - 1.0); j0 = int(math.floor(j)); j1 = min(j0 + 1, len(self.rings) - 1); t = j - j0
        return self.rings[j0].at(th).lerp(self.rings[j1].at(th), t)


def axis_frame(a, ref=Vector((0, -1, 0))):
    a = a.normalized(); e1 = (ref - a * ref.dot(a))
    if e1.length < 1e-6: e1 = Vector((1, 0, 0)) - a * a.x
    e1.normalize(); e2 = a.cross(e1); return e1, e2


def add_tube_to_bm(bm, tube, uv_layer, mat_index=0):
    nj, nth = len(tube.rings), tube.nth
    ncol = nth if tube.closed else nth + 1
    vg = []
    for j, ring in enumerate(tube.rings):
        row = []
        for i in range(ncol):
            th = tube.th0 + (tube.th1 - tube.th0) * i / nth
            row.append(bm.verts.new(ring.at(th)))
        vg.append(row)
    faces = []
    for j in range(nj - 1):
        for i in range(nth if not tube.closed else ncol):
            i2 = (i + 1) % ncol if tube.closed else i + 1
            try:
                f = bm.faces.new((vg[j][i], vg[j][i2], vg[j + 1][i2], vg[j + 1][i]))
            except ValueError:
                continue
            f.material_index = mat_index; f.smooth = True
            for loop, (jj, ii) in zip(f.loops, ((j, i), (j, i + 1), (j + 1, i + 1), (j + 1, i))):
                loop[uv_layer].uv = (ii / nth, jj / max(1, nj - 1))
            faces.append(f)
    return vg, faces


# ---------------------------------------------------------------- hull / polar-ring tools
class PointRing:
    """Ring given as N explicit 3D points sampled at theta_i = 2 pi i / N; optional centre and frame for piping."""
    def __init__(self, pts, c=None, e1=None, e2=None):
        self.pts = np.asarray(pts, dtype=np.float64); self.N = len(self.pts); self.c, self.e1, self.e2 = c, e1, e2

    def at(self, th):
        f = (th / (2 * math.pi)) % 1.0 * self.N; i0 = int(math.floor(f)) % self.N; t = f - math.floor(f); i1 = (i0 + 1) % self.N
        p = self.pts[i0] * (1 - t) + self.pts[i1] * t; return Vector(p)


def hull2d(P):
    P = sorted(set((round(float(x), 5), round(float(y), 5)) for x, y in P))
    if len(P) < 3: return None
    def cr(o, a, b): return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in P:
        while len(lo) >= 2 and cr(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(up) >= 2 and cr(up[-2], up[-1], p) <= 0: up.pop()
        up.append(p)
    H = np.array(lo[:-1] + up[:-1]); return H if len(H) >= 3 else None


def poly_centroid(H):
    x, y = H[:, 0], H[:, 1]; x1, y1 = np.roll(x, -1), np.roll(y, -1); cr = x * y1 - x1 * y; A = cr.sum() / 2
    if abs(A) < 1e-12: return H.mean(axis=0)
    return np.array([((x + x1) * cr).sum() / (6 * A), ((y + y1) * cr).sum() / (6 * A)])


def polar_radii(H, c, N):
    """Distance from c to the (convex) hull boundary along N uniformly spaced directions."""
    ths = 2 * math.pi * np.arange(N) / N; D = np.stack([np.cos(ths), np.sin(ths)], 1)
    Pm = H - c; Q = np.roll(H, -1, axis=0) - c; E = Q - Pm; out = np.zeros(N)
    for i, d in enumerate(D):
        det = -d[0] * E[:, 1] + E[:, 0] * d[1]
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (-Pm[:, 0] * E[:, 1] + E[:, 0] * Pm[:, 1]) / det; s = (d[0] * Pm[:, 1] - d[1] * Pm[:, 0]) / det
        ok = (np.abs(det) > 1e-12) & (t > 0) & (s >= -1e-9) & (s <= 1 + 1e-9)
        out[i] = t[ok].max() if ok.any() else 0.0
    return out, D


def clip_x(H, x0, keep_ge):
    """Sutherland-Hodgman: keep the part of convex polygon H with x >= x0 (keep_ge) or x <= x0."""
    out = []; n = len(H)
    inside = (lambda p: p[0] >= x0) if keep_ge else (lambda p: p[0] <= x0)
    for i in range(n):
        p, q = H[i], H[(i + 1) % n]
        if inside(p): out.append(p)
        if inside(p) != inside(q):
            t = (x0 - p[0]) / (q[0] - p[0]); out.append(p + (q - p) * t)
    return np.array(out) if len(out) >= 3 else None


def smooth_rows(P, iters):
    """Axial Laplacian smoothing of an (nj, N, 3) ring stack along j; first/last row fixed."""
    for _ in range(iters): P[1:-1] = (P[:-2] + 2 * P[1:-1] + P[2:]) / 4
    return P


def smooth_circ(P, iters):
    for _ in range(iters): P = (np.roll(P, 1, axis=1) + 2 * P + np.roll(P, -1, axis=1)) / 4
    return P


N_TH = 72
Z_HEM, Z_COLLAR, Z_NECK = 0.855, 1.535, 1.460
Z_WAIST, Z_CUFF = 1.085, 0.045
CHEST_Z = Z_HEM + 0.62 * (Z_COLLAR - Z_HEM)                 # 1.2766 m


# ---------------------------------------------------------------- jacket: hull-of-body-slice torso + raglan sleeves
def torso_ease(z):
    e = lerp(0.026, 0.012, smoothstep(1.33, 1.43, z))          # boxy drape on the trunk, tight flat shoulder line
    return lerp(e, 0.030, smoothstep(1.43, 1.47, z))           # loose stand-up funnel collar


def build_torso_stack(ref, zs, lat=0.0):
    N = N_TH; pts = np.zeros((len(zs), N, 3)); prev = None
    for k, z in enumerate(zs):
        Ht = ref.hull_at(ref.torso_mask, z); Hta = ref.hull_at(ref.torso_arm_mask, z)
        if Ht is None or Hta is None:
            pts[k] = prev if prev is not None else 0; continue
        c = poly_centroid(Ht); r_t, D = polar_radii(Ht, c, N); r_ta, _ = polar_radii(Hta, c, N)
        w = smoothstep(1.24, 1.34, z); r = lerp(r_t, r_ta, w) + torso_ease(z)      # arms join the hull only around the shoulder
        r = r + 0.45 * lat * math.exp(-((z - CHEST_Z) / 0.16) ** 2) * D[:, 0] ** 2     # oversized drop-shoulder: torso shares the lateral ease
        xy = c + r[:, None] * D; pts[k, :, :2] = xy; pts[k, :, 2] = z; prev = pts[k].copy()
        _probe('torso_ring', dict(k=k, z=float(z), c=c.copy(), D=D.copy(), r_hull=lerp(r_t, r_ta, w).copy(), r_tot=r.copy(), w=float(w), lat=float(lat), Ht=Ht.copy(), Hta=Hta.copy()))
    pts = smooth_rows(pts, 8)                                   # 8 axial Laplacian passes: flat woven planes, no horizontal ribbing
    out = smooth_circ(pts, 2)
    _probe('torso_stack', out.copy())
    return out


def sleeve_stack(rig, ref, side, lat):
    sg = 1 if side == 'L' else -1
    S = bone(rig, f'upperarm01.{side}'); E = bone(rig, f'lowerarm01.{side}'); Wr = bone(rig, f'wrist.{side}')
    Wc = Wr - (Wr - E).normalized() * 0.008                     # cuff ends 8 mm proximal to the wrist (hand) joint
    z_start = 1.34; S0 = S.lerp(E, (S.z - z_start) / (S.z - E.z)); L1 = (E - S0).length; L2 = (Wc - E).length; tot = L1 + L2
    nj = 56; pts = []
    for j in range(nj):
        d = j / (nj - 1) * tot; pts.append(S0.lerp(E, d / L1) if d <= L1 else E.lerp(Wc, (d - L1) / L2))
    for _ in range(3): pts = [pts[0]] + [(pts[i - 1] + 2 * pts[i] + pts[i + 1]) / 4 for i in range(1, nj - 1)] + [pts[-1]]
    mask = ref.arm & (ref.sideL if side == 'L' else ref.sideR); A = ref.a[mask]; N = N_TH
    ths = 2 * math.pi * np.arange(N) / N; stack = np.zeros((nj, N, 3)); frames = []; prev = None
    for j, p in enumerate(pts):
        a = (pts[min(j + 1, nj - 1)] - pts[max(j - 1, 0)]).normalized(); e1, e2 = axis_frame(a); u = j / (nj - 1)
        rel = A - np.array(p); ax = rel @ np.array(a); sel = (np.abs(ax) < 0.008) & (np.linalg.norm(rel, axis=1) < 0.2)
        H = hull2d(np.c_[rel[sel] @ np.array(e1), rel[sel] @ np.array(e2)]) if sel.sum() >= 3 else None
        if H is None: stack[j] = prev if prev is not None else 0; frames.append((p, e1, e2)); continue
        c2 = poly_centroid(H); r, _ = polar_radii(H, c2, N); cen = p + e1 * c2[0] + e2 * c2[1]
        up = Vector((0, 0, 1)) - a * a.z; up.normalize()
        top = lerp(0.007, 0.010, smoothstep(0.2, 0.6, u)); sid = lerp(0.022, 0.024, smoothstep(0.2, 0.6, u))
        top = lerp(top, 0.005, smoothstep(0.8, 1.0, u)); sid = lerp(sid, 0.005, smoothstep(0.8, 1.0, u))   # snug elastic cuff
        _eases = []
        for i, th in enumerate(ths):
            d = e1 * math.cos(th) + e2 * math.sin(th); t = 0.5 * (1 + d.dot(up))
            ease = lerp(sid, top, smoothstep(0.4, 1.0, t))                       # +7 mm over the arm top, +22 mm under / beside
            out = max(0.0, d.dot(Vector((sg, 0, 0)))); ease += 0.6 * lat * math.exp(-((p.z - CHEST_Z) / 0.13) ** 2) * out ** 2
            ease += 0.002 * smoothstep(0.85, 1.0, u) * math.sin(14 * th)         # gathered cuff
            stack[j, i] = np.array(cen + d * (r[i] + ease)); _eases.append(ease)
        prev = stack[j].copy(); frames.append((cen, e1, e2))
        _probe('sleeve_ring', dict(side=side, j=j, u=float(u), H=H.copy(), c2=c2.copy(), r_hull=r.copy(), ease=np.array(_eases), cen=np.array(cen), e1=np.array(e1), e2=np.array(e2), axis=np.array(a)))
    stack = smooth_circ(smooth_rows(stack, 3), 1)
    _probe('sleeve_stack', dict(side=side, stack=stack.copy()))
    return stack, frames


def build_lofted_jacket(rig, ref, lat=0.0):
    zs = list(np.linspace(Z_HEM, Z_NECK, 64)) + list(np.linspace(Z_NECK, Z_COLLAR, 14)[1:])
    st = build_torso_stack(ref, zs, lat); rings = [PointRing(p) for p in st]
    tubes = {'torso': Tube(rings[:64], nth=N_TH)}
    # 10 deg V-notch at the front centre. The 70 collar samples sit on the torso ring's 5 deg grid, so collar and torso share
    # vertices exactly: the old 12 deg / 48-sample collar made 3-point partial merges = 3 non-manifold edges (Owner D-TRACK, 2026-11-09)
    half = math.radians(5.0); front = -math.pi / 2
    tubes['collar'] = Tube(rings[63:], th0=front + half, th1=front + 2 * math.pi - half, closed=False, nth=70)
    for side in 'LR':
        stack, fr = sleeve_stack(rig, ref, side, lat); rr = [PointRing(p, *f) for p, f in zip(stack, fr)]
        a = (Vector(stack[-1].mean(axis=0)) - Vector(stack[-2].mean(axis=0))).normalized(); cen = stack[-1].mean(axis=0)
        rr.append(PointRing(cen + (stack[-1] - cen) * 0.93 + np.array(a) * 0.006, *fr[-1]))   # inward-turned cuff edge
        tubes[f'sleeve_{side}'] = Tube(rr, nth=N_TH)
    return tubes


def chest_span(tubes, z0=CHEST_Z, tol=0.006):
    xs = []
    for t in tubes.values():
        if not isinstance(t, Tube): continue
        for r in t.rings:
            m = np.abs(r.pts[:, 2] - z0) < tol; xs += list(r.pts[m, 0])
    return (max(xs) - min(xs)) if xs else 0.0


# ---------------------------------------------------------------- pants: one bifurcating surface, waist to ankle
def ease_leg(z):
    return float(np.interp(z, [0.045, 0.12, 0.30, 0.48, 0.68, 0.80], [0.012, 0.018, 0.040, 0.036, 0.030, 0.026]))


def ease_up(z):
    return float(np.interp(z, [0.80, 1.085], [0.022, 0.018]))


def pants_stack(rig, ref, side, zs):
    sg = 1 if side == 'L' else -1; N = N_TH
    leg = ref.legL if side == 'L' else ref.legR; zmin = float(ref.a[leg, 2].min()); zq_min = zmin + 0.03
    K = bone(rig, f'lowerleg01.{side}'); Aj = bone(rig, f'foot.{side}'); slope = np.array([(Aj.x - K.x) / (Aj.z - K.z), (Aj.y - K.y) / (Aj.z - K.z)])
    stack = np.zeros((len(zs), N, 3)); ths = 2 * math.pi * np.arange(N) / N; D = np.stack([np.cos(ths), np.sin(ths)], 1)
    for k, z in enumerate(zs):
        P_low = P_up = None
        if z < 0.80:
            zq = max(z, zq_min); H = ref.hull_at(leg, zq)
            c = poly_centroid(H); r, _ = polar_radii(H, c, N)
            if z < zq: c = c + slope * (z - zq)
            r = r + ease_leg(z) + 0.0012 * (1 - smoothstep(0.045, 0.10, z)) * np.sin(16 * ths)       # elastic gathered ankle cuff
            P_low = c + r[:, None] * D
        if z > 0.68:
            H = ref.hull_at(ref.lower_mask, z); Hh = clip_x(H, -0.004 * sg, sg > 0) if H is not None else None   # half of the unified hull (8 mm overlap)
            c = poly_centroid(Hh); r, _ = polar_radii(Hh, c, N); r = r + ease_up(z); P_up = c + r[:, None] * D
        w = smoothstep(0.70, 0.80, z)
        P = P_low if P_up is None else (P_up if P_low is None else P_low * (1 - w) + P_up * w)
        stack[k, :, :2] = P; stack[k, :, 2] = z
    return smooth_circ(smooth_rows(stack, 14), 3)


def build_lofted_pants(rig, ref):
    zs = list(np.linspace(Z_CUFF, Z_WAIST, 120)); tubes = {}
    for side in 'LR':
        st = pants_stack(rig, ref, side, zs); tubes[f'leg_{side}'] = Tube([PointRing(p) for p in st], nth=N_TH)
    return tubes


# ---------------------------------------------------------------- push-out + solidify + piping
def pushout(bm_shell, body_bvh, floor, lock_z=lambda v: False):
    """Push vertices to >= floor outside the body. Vertices selected by lock_z keep their z, so hem/collar/cuff heights
    (the confirmed 680 / 1040 mm lengths) are not changed by the push."""
    _snap(bm_shell, 'loft')
    for it in range(PUSH_PASSES):
        moved = 0; _proj = {}
        for _vi, v in enumerate(bm_shell.verts):
            z0 = v.co.z
            loc, nrm, idx, dist = body_bvh.find_nearest(v.co)
            if loc is None: continue
            inside = (v.co - loc).dot(nrm) < 0.0
            if inside or dist < floor:
                v.co = loc + nrm * floor; moved += 1
                if PROBE is not None: _proj[_vi] = tuple(v.co)
                if lock_z(v): v.co.z = z0
        _snap(bm_shell, f'p{it}_proj', _proj)          # positions as projected, before the lock_z restore
        _snap(bm_shell, f'p{it}_lockz')                # actual positions after the lock_z restore
        if PUSH_SMOOTH and it < PUSH_SMOOTH_UNTIL:
            bmesh.ops.smooth_vert(bm_shell, verts=bm_shell.verts, factor=0.25, use_axis_x=True, use_axis_y=True, use_axis_z=False)
            _snap(bm_shell, f'p{it}_smooth')
        log(f'  push-out pass {it}: {moved} verts moved')


def solidify_bm(bm, thickness):
    me = bpy.data.meshes.new('_shell_tmp'); bm.to_mesh(me); o = bpy.data.objects.new('_shell_tmp', me)
    bpy.context.scene.collection.objects.link(o)
    m = o.modifiers.new('Solidify', 'SOLIDIFY'); m.thickness = thickness; m.offset = -1.0; m.use_even_offset = False; m.nonmanifold_thickness_mode = 'CONSTRAINTS'
    dg = bpy.context.evaluated_depsgraph_get(); me2 = bpy.data.meshes.new_from_object(o.evaluated_get(dg))
    bpy.data.objects.remove(o, do_unlink=True); bpy.data.meshes.remove(me)
    bm2 = bmesh.new(); bm2.from_mesh(me2); bpy.data.meshes.remove(me2); return bm2


class Piper:
    def __init__(self, bm, shell_bvh, radius, proud):
        self.bm, self.bvh, self.r, self.proud, self.count = bm, shell_bvh, radius, proud, 0

    def snap_nearest(self, p):
        loc, n, _, _ = self.bvh.find_nearest(p); return loc + n * self.proud if loc is not None else None

    def snap_ray(self, x, z, y_from, direction=1):
        o = Vector((x, y_from, z)); loc, n, _, _ = self.bvh.ray_cast(o, Vector((0, direction, 0)), 2.0)
        if loc is None: return None
        if n.y * direction > 0: n = -n                                       # make the normal face the ray origin (outward)
        return loc + n * self.proud

    def snap_down(self, x, y, z0=2.2):
        loc, n, _, _ = self.bvh.ray_cast(Vector((x, y, z0)), Vector((0, 0, -1)), 3.0)
        if loc is None: return None
        if n.z < 0: n = -n
        return loc + n * self.proud

    def tube(self, pts, closed=False, radius=None, sides=8):
        kept = []
        for p in pts:                                                        # drop samples that jump (ray slipped through the collar notch etc.)
            if p is None: continue
            if kept and (p - kept[-1]).length > 0.04: continue
            kept.append(p)
        pts = kept
        if len(pts) < 3: return
        r = radius or self.r; rings = []; prev_n = None
        for i, p in enumerate(pts):
            a = (pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]).normalized()
            if prev_n is None: e1, e2 = axis_frame(a, Vector((0, 0, 1)) if abs(a.z) < 0.9 else Vector((1, 0, 0)))
            else:
                e1 = (prev_n - a * prev_n.dot(a)); e1 = e1.normalized() if e1.length > 1e-6 else axis_frame(a)[0]; e2 = a.cross(e1)
            prev_n = e1
            rings.append([self.bm.verts.new(p + (e1 * math.cos(2 * math.pi * k / sides) + e2 * math.sin(2 * math.pi * k / sides)) * r) for k in range(sides)])
        for j in range(len(rings) - 1):
            for k in range(sides):
                f = self.bm.faces.new((rings[j][k], rings[j][(k + 1) % sides], rings[j + 1][(k + 1) % sides], rings[j + 1][k]))
                f.material_index = 1; f.smooth = True
        self.count += 1


def jacket_piping(pp, tubes, variant, ref):
    cyf = -0.6
    zs = lambda a, b, n: [lerp(a, b, i / (n - 1)) for i in range(n)]
    # 1. front centre zipper: two tight 1.8 mm cords at x = +-0.004; on the collar they follow the notch edges
    for s in (-1, 1):
        pp.tube([pp.snap_ray(s * lerp(0.004, 0.011, smoothstep(1.43, 1.46, z)), z, cyf, 1) for z in zs(Z_HEM + 0.001, Z_COLLAR - 0.001, 80)], radius=0.0018, sides=6)
    # 2. collar top rim
    col = tubes['collar']; nth = col.nth
    pp.tube([pp.snap_nearest(col.eval(col.th0 + (col.th1 - col.th0) * i / nth, len(col.rings) - 1)) for i in range(nth + 1)])
    # 3. raglan double track (38 mm apart) neck -> shoulder -> wrist cuff
    for side, sg in (('L', 1), ('R', -1)):
        sl = tubes[f'sleeve_{side}']; rings = sl.rings[:-1]
        for off in (-0.019, 0.019):
            top0 = rings[0]; xs0 = float(top0.pts[:, 0].max() if sg > 0 else top0.pts[:, 0].min())
            path = [pp.snap_down(sg * lerp(0.07, abs(xs0) - 0.03, i / 11), ref.cy(1.45) - 0.01 + off) for i in range(12)]
            for r in rings:
                a = r.e1.cross(r.e2); want = Vector((sg, 0.0, 0.35)); want = want - a * want.dot(a)
                th = math.atan2(want.dot(r.e2), want.dot(r.e1)); R = max((r.at(th) - r.c).length, 0.04)
                path.append(pp.snap_nearest(r.at(th + off / R)))
            pp.tube(path)
    # 4. front and back princess seams (+-0.095 at the hem flaring to +-0.145 at the chest, curving into the armhole)
    for s in (-1, 1):
        for y_from, d in ((cyf, 1), (0.6, -1)):
            path = []
            for i in range(44):
                u = i / 43; z = lerp(Z_HEM + 0.002, 1.25, u); x = s * (0.095 + 0.05 * u ** 1.6)
                if u > 0.9: x = s * lerp(0.095 + 0.05 * 0.9 ** 1.6, 0.20, (u - 0.9) / 0.1)
                path.append(pp.snap_ray(x, z, y_from, d))
            pp.tube(path)
    # 5. horizontal chest piping (Variant A only)
    if variant == 'bracket':
        pp.tube([pp.snap_ray(x, 1.245, cyf, 1) for x in zs(-0.21, 0.21, 60)])
    # 6. bungee cord loops with toggles hanging at the hem
    for s in (-1, 1):
        h = pp.snap_ray(s * 0.17, Z_HEM + 0.012, cyf, 1)
        if h is None: continue
        loop = [Vector((h.x + 0.012 * math.cos(t) + s * 0.012, h.y - 0.004, 0.828 + 0.032 * math.sin(t))) for t in [2 * math.pi * i / 20 for i in range(21)]]
        pp.tube(loop, radius=0.0022, sides=6)
        pp.tube([Vector((h.x + s * 0.012, h.y - 0.004, z)) for z in (0.784, 0.794, 0.804)], radius=0.006, sides=8)


def pants_piping(pp, tubes, ref):
    cyf = -0.6
    for s, side in ((1, 'L'), (-1, 'R')):
        leg = tubes[f'leg_{side}']; th = 0.0 if s > 0 else math.pi
        pp.tube([pp.snap_nearest(r.at(th)) for r in leg.rings])                      # continuous outer side seam, cuff to waistband
        kj = min(range(len(leg.rings)), key=lambda j: abs(leg.rings[j].pts[0, 2] - 0.48)); xk = abs(leg.rings[kj].at(th).x)
        # front curve: waistband x=0.085 -> 0.115 at z=0.82 (logo border) -> outer knee x at z=0.48, quadratic through the three anchors
        zA, zB, zC = Z_WAIST - 0.005, 0.82, 0.48; xA, xB, xC = 0.085, 0.115, xk
        def xq(z):
            return (xA * (z - zB) * (z - zC) / ((zA - zB) * (zA - zC)) + xB * (z - zA) * (z - zC) / ((zB - zA) * (zB - zC)) + xC * (z - zA) * (z - zB) / ((zC - zA) * (zC - zB)))
        pp.tube([pp.snap_ray(s * xq(z), z, cyf, 1) for z in [lerp(zA, zC, i / 59) for i in range(60)]])


# ============================================================================== 4. JET BLACK CRINKLE NYLON + DECAL SHADERS
def assign_garment_materials(obj, kind='jacket', variant='bracket', decals=None, y_split=-0.03):
    mat_nylon = bpy.data.materials.new(f'JetBlack_CrinkleNylon_{kind}_{variant}')
    mat_nylon.use_nodes = True
    nt = mat_nylon.node_tree; nodes = nt.nodes; links = nt.links
    nodes.clear()
    out = nodes.new('ShaderNodeOutputMaterial'); bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Roughness'].default_value = 0.42
    bsdf.inputs['Anisotropic'].default_value = 0.62
    bsdf.inputs['Specular IOR Level'].default_value = 0.48
    bsdf.inputs['Coat Weight'].default_value = 0.0
    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    tex_coord = nodes.new('ShaderNodeTexCoord')
    voronoi = nodes.new('ShaderNodeTexVoronoi'); voronoi.inputs['Scale'].default_value = 65.0
    noise = nodes.new('ShaderNodeTexNoise'); noise.inputs['Scale'].default_value = 320.0; noise.inputs['Detail'].default_value = 6.0
    math_add = nodes.new('ShaderNodeMath'); math_add.operation = 'MULTIPLY_ADD'; math_add.inputs[1].default_value = 0.65
    links.new(voronoi.outputs['Distance'], math_add.inputs[0]); links.new(noise.outputs['Fac'], math_add.inputs[2])
    bump = nodes.new('ShaderNodeBump'); bump.inputs['Strength'].default_value = 0.30; bump.inputs['Distance'].default_value = 0.002
    links.new(math_add.outputs['Value'], bump.inputs['Height']); links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])

    base_rgb = nodes.new('ShaderNodeRGB'); base_rgb.outputs[0].default_value = SPEC['jet_black_rgb']
    col = base_rgb.outputs[0]
    if decals:
        if kind == 'jacket':
            front_decal = decals['bracket_logo.png'] if variant == 'bracket' else decals['orbit_logo.png']
            col = add_projected_decal_node(nt, tex_coord, col, front_decal, (-0.145, 0.145), (1.268, 1.353), True, y_split)
            col = add_projected_decal_node(nt, tex_coord, col, decals['bst_shield.png'], (-0.165, 0.165), (1.115, 1.405), False, y_split)
        else:
            col = add_projected_decal_node(nt, tex_coord, col, decals['bracket_logo.png'], (-0.108, 0.108), (0.785, 0.850), True, y_split)
    nt.links.new(col, bsdf.inputs['Base Color'])

    mat_piping = bpy.data.materials.new('Contrast_White_Piping'); mat_piping.use_nodes = True
    p = mat_piping.node_tree.nodes['Principled BSDF']
    p.inputs['Base Color'].default_value = SPEC['piping_white_rgb']; p.inputs['Roughness'].default_value = 0.20
    p.inputs['Emission Color'].default_value = (1, 1, 1, 1); p.inputs['Emission Strength'].default_value = 0.15
    obj.data.materials.clear(); obj.data.materials.append(mat_nylon); obj.data.materials.append(mat_piping)
    # materials.clear() resets every polygon's slot to 0, so assign the piping slot (faces appended after the nylon shell) now
    mi = np.zeros(len(obj.data.polygons), dtype=np.int32); mi[obj['n_nylon_faces']:] = 1
    obj.data.polygons.foreach_set('material_index', mi); obj.data.update()
    assert int(mi.sum()) > 0, 'no piping faces assigned'


def add_projected_decal_node(nt, tex_coord, prev_color_socket, img_path, x_bounds, z_bounds, front_only, y_split):
    """Object-space planar decal. Front/back gate is relative to the torso centre line y_split (front = -y)."""
    nodes = nt.nodes; links = nt.links; x0, x1 = x_bounds; z0, z1 = z_bounds
    sep = nodes.new('ShaderNodeSeparateXYZ'); links.new(tex_coord.outputs['Object'], sep.inputs['Vector'])
    map_u = nodes.new('ShaderNodeMapRange'); map_u.clamp = True
    map_u.inputs['From Min'].default_value = x0 if front_only else x1; map_u.inputs['From Max'].default_value = x1 if front_only else x0
    map_v = nodes.new('ShaderNodeMapRange'); map_v.clamp = True
    map_v.inputs['From Min'].default_value = z0; map_v.inputs['From Max'].default_value = z1
    links.new(sep.outputs['X'], map_u.inputs['Value']); links.new(sep.outputs['Z'], map_v.inputs['Value'])
    comb = nodes.new('ShaderNodeCombineXYZ'); links.new(map_u.outputs['Result'], comb.inputs['X']); links.new(map_v.outputs['Result'], comb.inputs['Y'])
    tex = nodes.new('ShaderNodeTexImage'); tex.image = bpy.data.images.load(str(img_path)); tex.extension = 'CLIP'
    links.new(comb.outputs['Vector'], tex.inputs['Vector'])
    cmp_y = nodes.new('ShaderNodeMath'); cmp_y.operation = 'LESS_THAN' if front_only else 'GREATER_THAN'; cmp_y.inputs[1].default_value = y_split
    links.new(sep.outputs['Y'], cmp_y.inputs[0])
    mask = nodes.new('ShaderNodeMath'); mask.operation = 'MULTIPLY'
    links.new(tex.outputs['Alpha'], mask.inputs[0]); links.new(cmp_y.outputs['Value'], mask.inputs[1])
    mix = nodes.new('ShaderNodeMix'); mix.data_type = 'RGBA'
    links.new(mask.outputs['Value'], mix.inputs['Factor']); links.new(prev_color_socket, mix.inputs[6]); links.new(tex.outputs['Color'], mix.inputs[7])
    return mix.outputs[2]


# ============================================================================== 5. ASSEMBLE, EXPORT
def assemble(kind, variant, rig, ref, body_bvh, decals):
    bm = bmesh.new(); uv = bm.loops.layers.uv.new('UVMap')
    if kind == 'jacket':
        lat = 0.0; hist = []
        for it in range(10):                                                  # secant solve: lateral ease -> 610 mm chest span
            tubes = build_lofted_jacket(rig, ref, lat); span = chest_span(tubes); hist.append((lat, span))
            log(f'  chest calibration {it}: span={span * 1000:.1f} mm lat={lat * 1000:.1f} mm')
            if abs(span - 0.610) < 0.0015: break
            if len(hist) == 1: lat += (0.610 - span) / 1.1
            else:
                (l0, s0), (l1, s1) = hist[-2], hist[-1]; slope = (s1 - s0) / (l1 - l0) if abs(l1 - l0) > 1e-9 else 1.0
                lat += (0.610 - span) / max(slope, 0.2)
    else:
        tubes = build_lofted_pants(rig, ref)
    shell_tubes = {k: t for k, t in tubes.items() if isinstance(t, Tube)}
    for k, t in shell_tubes.items(): add_tube_to_bm(bm, t, uv, 0)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    bm.verts.ensure_lookup_table()
    log(f'{kind}/{variant}: {len(bm.verts)} verts, {len(bm.faces)} faces before push-out')
    _probe('chest_calibration', list(hist) if kind == 'jacket' else None)
    _lz = (lambda v: abs(v.co.x) < 0.15) if kind == 'jacket' else (lambda v: True)
    if LOCKZ_DISABLED: _lz = lambda v: False
    pushout(bm, body_bvh, SPEC['pushout_floor_m'], lock_z=_lz)
    bm.normal_update()
    zs_n = [v.co.z for v in bm.verts if abs(v.co.x) < 0.15]; xz = np.array([(v.co.x, v.co.z) for v in bm.verts])
    props = {}
    if kind == 'jacket':
        props['z_hem'], props['z_collar'] = min(zs_n), max(zs_n); props['jacket_length_mm'] = round((max(zs_n) - min(zs_n)) * 1000, 1)
        ch = xz[np.abs(xz[:, 1] - CHEST_Z) < 0.006]; props['chest_width_mm'] = round(float(ch[:, 0].max() - ch[:, 0].min()) * 1000, 1)
    else:
        zs_all = xz[:, 1]; props['z_cuff'], props['z_waist'] = float(zs_all.min()), float(zs_all.max())
        props['pants_length_mm'] = round(float(zs_all.max() - zs_all.min()) * 1000, 1)
    _probe('props', dict(props))
    shell = solidify_bm(bm, 0.0035); bm.free()
    _snap(shell, 'solidify')
    shell.faces.ensure_lookup_table(); n_shell_faces = len(shell.faces); shell_bvh = BVHTree.FromBMesh(shell)
    pp = Piper(shell, shell_bvh, SPEC['piping_radius_m'], SPEC['piping_proud_m'])
    if kind == 'jacket': jacket_piping(pp, tubes, variant, ref)
    else: pants_piping(pp, tubes, ref)
    log(f'{kind}/{variant}: {pp.count} piping tubes')
    me = bpy.data.meshes.new(f'G0_{kind}_{variant}' if kind == 'jacket' else 'G0_pants'); shell.to_mesh(me); shell.free()
    for p in me.polygons: p.use_smooth = True
    obj = bpy.data.objects.new(me.name, me); bpy.context.scene.collection.objects.link(obj)
    obj['n_nylon_faces'] = int(n_shell_faces)
    for k, v in props.items(): obj[k] = v
    log(f'  measured: {props}')
    return obj


def export_glb(obj, path):
    """Export only `obj` (real selection state; use_selection ignores temp_override)."""
    for o in bpy.context.scene.objects: o.select_set(False)
    obj.select_set(True); bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.gltf(filepath=str(path), use_selection=True, export_format='GLB', export_materials='EXPORT')
    assert Path(path).stat().st_size > 1000, f'empty GLB: {path}'


def build_all(variant_choice='both', crop_dir=VIDEO_DIR):
    GARMENTS_DIR.mkdir(parents=True, exist_ok=True)
    decals = prepare_all_decals(Path(crop_dir))
    rig, baked_body = load_theta_fit_mannequin()
    ref = BodyRef(baked_body, bpy.data.objects[SPEC['body_name']])
    bm = bmesh.new(); bm.from_mesh(baked_body.data); bm.verts.ensure_lookup_table(); body_bvh = BVHTree.FromBMesh(bm); bm.free()
    variants = ['bracket', 'orbit'] if variant_choice == 'both' else [variant_choice]
    for var in variants:
        jacket = assemble('jacket', var, rig, ref, body_bvh, decals)
        assign_garment_materials(jacket, 'jacket', var, decals, y_split=ref.cy(1.3))
        out_j = GARMENTS_DIR / f'bs_track_001_jacket_{var}.glb'; export_glb(jacket, out_j)
        log(f'exported Jacket ({var}) -> {out_j}  length={jacket["jacket_length_mm"]} mm chest={jacket["chest_width_mm"]} mm')
    pants = assemble('pants', 'bracket', rig, ref, body_bvh, decals)
    assign_garment_materials(pants, 'pants', 'bracket', decals, y_split=ref.cy(0.8))
    out_p = GARMENTS_DIR / 'bs_track_001_pants.glb'; export_glb(pants, out_p)
    log(f'exported Pants -> {out_p}  length={pants["pants_length_mm"]} mm')
    master_blend = GARMENTS_DIR / 'bs_track_001_g0_master.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(master_blend), compress=True)
    bpy.ops.file.make_paths_relative()
    bpy.ops.wm.save_as_mainfile(filepath=str(master_blend), compress=True)
    log(f'saved master G0 library -> {master_blend}')


if __name__ == '__main__':
    import sys
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['bracket', 'orbit', 'both'], default='both')
    ap.add_argument('--crop-dir', default=str(VIDEO_DIR))
    args = ap.parse_args(argv)
    build_all(args.variant, args.crop_dir)
