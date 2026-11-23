"""build_shirt.py - rebuild the camp shirt's neck: V opening + one-piece camp collar with revers (lapels).
The live master is opened read-only; the result is saved to the workspace.
usage: python -I -B build_shirt.py <camp_shirt_master.blend> <out.blend> <out_collar_grids.npz>
Frame: Blender world, Z up, wearer's front = -Y, metres."""
import sys, math
import numpy as np
import bpy, bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree

SRC, DST, NPZ = sys.argv[1:4]
bpy.ops.wm.open_mainfile(filepath=SRC, load_ui=False)
tee = bpy.data.objects['Tee']; me = tee.data
gcol = bpy.data.collections['Garment']

# ------------------------------------------------------------------ design constants (metres)
Z_V = 1.335                    # bottom of the V opening
NS = np.array([0.082, -0.028, 1.472])     # roll line at the side of the neck
NB = np.array([0.0, 0.050, 1.503])        # roll line at centre back (stand height above the neck seam)
G_FRAC = 0.74                  # gorge point as a fraction of the roll line from the V bottom to NS
LP_XZ = (0.136, 1.404)         # lapel point (front view x, z)
CP_XZ = (0.158, 1.425)         # collar point
EB_Z = 1.418                   # collar outer edge at centre back
SH_X = 0.172                   # collar outer edge over the shoulder
H_CLEAR = 0.0024               # collar underside clearance above the shell
THICK = 0.0018

# ------------------------------------------------------------------ outer-shell BVH of the Tee
nv = len(me.vertices); V = np.zeros(nv * 3); me.vertices.foreach_get('co', V); V = V.reshape(-1, 3)
nf = len(me.polygons); FN = np.zeros(nf * 3); me.polygons.foreach_get('normal', FN); FN = FN.reshape(-1, 3)
FC = np.zeros(nf * 3); me.polygons.foreach_get('center', FC); FC = FC.reshape(-1, 3)
lt = np.zeros(nf, np.int64); me.polygons.foreach_get('loop_total', lt); ls = np.zeros(nf, np.int64); me.polygons.foreach_get('loop_start', ls)
lv = np.zeros(len(me.loops), np.int64); me.loops.foreach_get('vertex_index', lv)
C0 = np.array([0.0, -0.03, 1.28])
outer = np.einsum('ij,ij->i', FN, FC - C0) > 0
upper = FC[:, 2] > 1.20
ids = np.nonzero(outer & upper & (lt == 4))[0]
quads = lv[ls[ids][:, None] + np.arange(4)]
bvh = BVHTree.FromPolygons(V.tolist(), quads.tolist())
print('outer upper faces', len(ids))


def surf(q):
    """nearest outer-shell point and outward unit normal"""
    loc, n, idx, d = bvh.find_nearest(Vector(q))
    n = np.array(n); c = np.array(loc)
    if n @ (c - C0) < 0: n = -n
    return c, n


def front_surface(x, z):
    loc, n, idx, d = bvh.ray_cast(Vector((x, -1.0, z)), Vector((0, 1, 0)), 2.0)
    return None if (loc is None or loc[1] > -0.012) else np.array(loc)         # a hit behind the shoulder line came through the neck opening


def top_surface(x, y):
    loc, n, idx, d = bvh.ray_cast(Vector((x, y, 2.0)), Vector((0, 0, -1)), 2.0)
    return None if loc is None else np.array(loc)


def back_surface(x, z):
    loc, n, idx, d = bvh.ray_cast(Vector((x, 1.0, z)), Vector((0, -1, 0)), 2.0)
    return None if loc is None else np.array(loc)


# ------------------------------------------------------------------ 1. V opening: delete front faces inside the V, trim placket + top button
VB = front_surface(0.0, Z_V - 0.004); VB[2] = Z_V
print('V bottom', VB.round(4))
half_w = lambda z: (z - Z_V) / (NS[2] - Z_V) * NS[0]
bm = bmesh.new(); bm.from_mesh(me); kill = []
for f in bm.faces:
    c = f.calc_center_median()
    if c.y < -0.02 and Z_V < c.z < 1.46 and abs(c.x) < half_w(c.z) - 0.0005: kill.append(f)
print('V faces removed', len(kill)); bmesh.ops.delete(bm, geom=kill, context='FACES'); bm.to_mesh(me); bm.free(); me.update()
for nm in ('placket_overlap', 'placket_buttons'):
    o = bpy.data.objects[nm]; bm = bmesh.new(); bm.from_mesh(o.data)
    if nm == 'placket_overlap':
        kill = [f for f in bm.faces if f.calc_center_median().z > Z_V - 0.006]
    else:                                   # whole button islands whose centre is above the V bottom
        seen = set(); kill = []
        for f0 in bm.faces:
            if f0 in seen: continue
            isl = [f0]; seen.add(f0); st = [f0]
            while st:
                f = st.pop()
                for e in f.edges:
                    for g in e.link_faces:
                        if g not in seen: seen.add(g); isl.append(g); st.append(g)
            if np.mean([f.calc_center_median().z for f in isl]) > Z_V - 0.02: kill += isl
    print(nm, 'faces removed', len(kill)); bmesh.ops.delete(bm, geom=kill, context='FACES'); bm.to_mesh(o.data); bm.free()
old = bpy.data.objects['collar']; rib_mat = old.data.materials[0]; bpy.data.objects.remove(old, do_unlink=True)


# ------------------------------------------------------------------ 2. curves
def roll_front(u, sx):
    """roll (break) line of the lapel from the V bottom (u=0) to the neck side (u=1): on the chest low down, lifting to NS."""
    x = sx * NS[0] * u; z = Z_V + (NS[2] - Z_V) * u
    y = VB[1] + (NS[1] - VB[1]) * u ** 1.6
    s = front_surface(x, z)
    if s is not None: y = min(y, s[1] - 0.0012)          # never behind the shell
    return np.array([x, y, z])


def roll_back(phi, sx):
    """phi 0..pi/2 from the neck side to centre back"""
    return np.array([sx * NS[0] * math.cos(phi), NS[1] + (NB[1] - NS[1]) * math.sin(phi), NS[2] + (NB[2] - NS[2]) * math.sin(phi)])


def smooth(x): x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)


C1 = np.array([0.0, 0.005, 1.36])                 # radial pole inside the upper chest: the shell is single-valued along rays from here


def radial(Q):
    d = Q - C1; r = np.linalg.norm(d); d = d / r
    loc, n, idx, dist = bvh.ray_cast(Vector(C1 + d * 0.7), Vector(-d), 0.7)
    return (None if loc is None else np.array(loc)), d, r


def conform(Q, lift):
    X, d, r = radial(Q)
    return Q if X is None else X + d * lift


def section(R, E, nvs, stand_blend):
    """leaf section from the roll point R to the outer point E: straight line blended into the shell-conformed surface, a soft roll at R."""
    out = []
    for j in range(nvs):
        v = j / (nvs - 1); Q = (1 - v) * R + v * E
        w = smooth(v / stand_blend)
        P = (1 - w) * Q + w * conform(Q, H_CLEAR + THICK)
        X, d, r = radial(P)
        if X is not None:
            rs = np.linalg.norm(X - C1) + H_CLEAR + THICK
            if r < rs: P = C1 + d * rs
        bulge = 0.003 * math.exp(-(v / 0.12) ** 2) * (1 - math.exp(-(v / 0.03) ** 2))
        out.append(P + d * bulge)
    return out


grids = {}
for sx, tag in ((1.0, 'L'), (-1.0, 'R')):                                    # +x is the wearer's left
    lp = front_surface(sx * LP_XZ[0], LP_XZ[1]); cp = front_surface(sx * CP_XZ[0], CP_XZ[1])
    G = roll_front(G_FRAC, sx)
    # ---- lapel: u along the roll line V bottom -> G, v from roll to outer edge (V bottom -> LP)
    NU, NVv = 60, 28; rows = []
    for i in range(NU):
        u = i / (NU - 1); R = roll_front(u * G_FRAC, sx)
        ex = sx * LP_XZ[0] * (u ** 0.85); ez = Z_V + (LP_XZ[1] - Z_V) * (u ** 1.15)         # slightly convex outer edge
        E = front_surface(ex, ez)
        if E is None: E = conform(np.array([ex, -0.05, ez]), 0.0)
        if i == 0: E = R.copy()
        rows.append(section(R, E, NVv, 0.25))
    grids['lapel_' + tag] = np.array(rows)
    # ---- collar leaf half: t from the gorge (front) round to centre back
    NT = 110; rows = []
    from scipy.interpolate import splprep, splev
    def resample(P, n):
        P = np.asarray(P); d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        return np.stack([np.interp(np.linspace(0, d[-1], n), d, P[:, k]) for k in range(3)], 1)
    Rp = resample([roll_front(G_FRAC + (1 - G_FRAC) * a, sx) for a in np.linspace(0, 1, 40)] + [roll_back(ph, sx) for ph in np.linspace(0, math.pi / 2, 160)[1:]], NT)
    ctrl = np.array([[CP_XZ[0], cp[1], CP_XZ[1]], [0.176, -0.045, 1.445], [0.181, -0.005, 1.455], [0.160, 0.036, 1.458], [0.120, 0.066, 1.462], [0.062, 0.082, 1.462], [0.0, 0.086, 1.462], [-0.062, 0.082, 1.462]])
    ctrl[:, 0] *= sx
    tck, uu = splprep(ctrl.T, s=0, k=3); Ep = resample(np.array(splev(np.linspace(0, uu[-2], 400), tck)).T, NT)
    for i in range(NT):
        E = conform(Ep[i], 0.0) if i else cp
        rows.append(section(Rp[i], E, 30, 0.45))
    grids['leaf_' + tag] = np.array(rows)
    # ---- inner stand (visible through the opening): roll line down to the neck seam
    rows = []
    for i in range(NT):
        R = Rp[i]; drop = 0.004 + 0.028 * smooth(i / 30)
        inw = np.array([-R[0], NS[1] + 0.02 - R[1], 0.0]); inw /= max(np.linalg.norm(inw), 1e-9)
        rows.append([R + k / 5 * (np.array([0, 0, -drop]) + inw * 0.0025) for k in range(6)])
    grids['stand_' + tag] = np.array(rows)

# make the two leaf halves meet exactly at centre back, the lapels meet at the V bottom
for k in ('leaf', 'stand'):
    m = 0.5 * (grids[k + '_L'][-1] + grids[k + '_R'][-1]); m[:, 0] = 0.0; grids[k + '_L'][-1] = m; grids[k + '_R'][-1] = m


# ------------------------------------------------------------------ 3. mesh
def add_grid(name, Gd, flip, uv_scale=1 / 0.30):
    nu, nvv = Gd.shape[:2]; verts = Gd.reshape(-1, 3).tolist(); faces = []
    for i in range(nu - 1):
        for j in range(nvv - 1):
            a, b, c, d = i * nvv + j, (i + 1) * nvv + j, (i + 1) * nvv + j + 1, i * nvv + j + 1
            faces.append((a, d, c, b) if flip else (a, b, c, d))
    m = bpy.data.meshes.new(name); m.from_pydata(verts, [], faces); m.update()
    # metric UV: u = arc length along the roll edge, v = arc length across
    su = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Gd[:, nvv // 2], axis=0), axis=1))])
    uvl = m.uv_layers.new(name='UVMap_Fabric')
    sv = np.concatenate([np.zeros((nu, 1)), np.cumsum(np.linalg.norm(np.diff(Gd, axis=1), axis=2), axis=1)], 1)
    for p in m.polygons:
        for li in p.loop_indices:
            vi = m.loops[li].vertex_index; i, j = divmod(vi, nvv); uvl.data[li].uv = (su[i] * uv_scale, sv[i, j] * uv_scale)
    o = bpy.data.objects.new(name, m); gcol.objects.link(o); m.materials.append(rib_mat)
    for p in m.polygons: p.use_smooth = True
    return o


objs = []
for k, Gd in grids.items():
    kind, tag = k.split('_')
    # outward-facing winding differs per side; decide from the first face normal against the outward direction
    o = add_grid('collar_' + k, Gd, False)
    n = np.array(o.data.polygons[len(o.data.polygons) // 2].normal); c = np.array(o.data.polygons[len(o.data.polygons) // 2].center)
    want = (c - C0) if kind != 'stand' else (np.array([0, NS[1] + 0.02, c[2]]) - c)
    if n @ want < 0:
        bpy.data.objects.remove(o, do_unlink=True); o = add_grid('collar_' + k, Gd, True)
    if kind != 'stand':
        sm = o.modifiers.new('solid', 'SOLIDIFY'); sm.thickness = THICK; sm.offset = -1.0; sm.use_even_offset = True
    ss = o.modifiers.new('subd', 'SUBSURF'); ss.levels = 1; ss.render_levels = 1
    objs.append(o)
# apply modifiers and join into one 'collar' object
dg = bpy.context.evaluated_depsgraph_get()
bm = bmesh.new()
for o in objs:
    ev = o.evaluated_get(dg); tmp = bpy.data.meshes.new_from_object(ev); bm.from_mesh(tmp); bpy.data.meshes.remove(tmp)
cm = bpy.data.meshes.new('collar'); bm.to_mesh(cm); bm.free(); cm.materials.append(rib_mat)
for p in cm.polygons: p.use_smooth = True
for o in objs: bpy.data.objects.remove(o, do_unlink=True)
co = bpy.data.objects.new('collar', cm); gcol.objects.link(co)
print('collar', len(cm.vertices), 'v', len(cm.polygons), 'f')
np.savez(NPZ, VB=VB, **grids)
bpy.ops.wm.save_as_mainfile(filepath=DST, check_existing=False)
print('SAVED', DST)
