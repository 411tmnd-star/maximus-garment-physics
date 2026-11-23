"""build_trousers.py - tailored straight-leg trousers for the BST two-piece (new geometry: no trouser pattern or mesh existed)
and their stone layout (outseam stud rows + crowned BST shield on the wearer's left thigh).
usage: python -B build_trousers.py <out.blend> <out_prefix>
Construction (metres, Z up, front = -Y): each leg is a lofted tube of elliptical sections clipped at the centre plane above the
crotch, pressed crease front and back, separate waistband with belt loops, fly and button. Proportions follow the reference image;
they are NOT a production pattern."""
import sys, math, json
sys.dont_write_bytecode = True
import numpy as np
import bpy, bmesh
from mathutils import Vector
import stones_core as C, art

DST, PREFIX = sys.argv[1:3]; MM = C.MM
HE = "C:/Users/USER/Desktop/4041_MechMix/02_engines/human_engine/"
bpy.ops.wm.read_factory_settings(use_empty=True)
gcol = bpy.data.collections.new('Garment'); bpy.context.scene.collection.children.link(gcol)
Z_W, Z_C, Z_H = 1.045, 0.775, 0.075                    # waist seam (under the band), crotch, hem
NZ, NT = 340, 200
rng = np.random.default_rng(7)


def prof(z):
    """leg section at height z -> (x centre, semi-axis a across, semi-axis b front-back, y centre)"""
    if z >= Z_C:
        t = (z - Z_C) / (Z_W - Z_C); s = t * t * (3 - 2 * t)
        a = 0.108 + (0.110 - 0.108) * s; xc = 0.097 + (0.066 - 0.097) * s; b = 0.100 + (0.108 - 0.100) * s; yc = 0.005
        hip = math.exp(-((z - 0.92) / 0.08) ** 2); xc += 0.010 * hip; b += 0.004 * hip
    else:
        t = (Z_C - z) / (Z_C - Z_H); s = t ** 0.8
        a = 0.108 + (0.082 - 0.108) * s; b = 0.100 + (0.078 - 0.100) * s; xc = 0.097 + (0.112 - 0.097) * t; yc = 0.005 + 0.012 * t
    return xc, a, b, yc


def leg(sx):
    Vg = np.zeros((NZ, NT, 3)); zs = np.linspace(Z_H, Z_W, NZ)
    for i, z in enumerate(zs):
        xc, a, b, yc = prof(z)
        for j in range(NT):
            th = 2 * math.pi * j / NT                      # 0 = outseam, pi/2 = back, pi = inseam, 3pi/2 = front
            ce, se = math.cos(th), math.sin(th)
            # pressed crease: lens-shaped section (sharper front and back)
            k = 1.0 + 0.10 * abs(se) ** 6
            x = xc + a * ce; y = yc + b * se * k
            fade = min(1.0, max(0.0, (0.98 - z) / 0.12))
            y += -0.004 * fade * math.exp(-((th - 1.5 * math.pi) / 0.05) ** 2) + 0.003 * fade * math.exp(-((th - 0.5 * math.pi) / 0.05) ** 2)
            # soft drape: long vertical folds and a break above the hem
            wr = 0.0022 * math.sin(3 * th + 5.0 * z + (0.7 if sx > 0 else 2.1)) * min(1.0, (Z_W - z) / 0.25) + 0.0035 * math.exp(-((z - 0.16) / 0.035) ** 2) * math.cos(2 * th + 1.0) * (1 if se < 0.3 else 0.4)
            x += wr * ce; y += wr * se
            x = max(x, 0.0)                                # centre seam plane above the crotch
            Vg[i, j] = (sx * x, y, z)
    return Vg, zs


def add_tube(name, Vg, mat, flip):
    nz, nt = Vg.shape[:2]; verts = Vg.reshape(-1, 3).tolist(); faces = []
    for i in range(nz - 1):
        for j in range(nt):
            j2 = (j + 1) % nt; q = (i * nt + j, i * nt + j2, (i + 1) * nt + j2, (i + 1) * nt + j)
            faces.append(q[::-1] if flip else q)
    m = bpy.data.meshes.new(name); m.from_pydata(verts, [], faces); m.update()
    su = np.concatenate([np.zeros((nz, 1)), np.cumsum(np.linalg.norm(np.diff(Vg, axis=1), axis=2), axis=1)], 1)
    uvl = m.uv_layers.new(name='UVMap_Fabric')
    for p in m.polygons:
        vs = [m.loops[li].vertex_index for li in p.loop_indices]
        js = [v % nt for v in vs]; wrap = max(js) - min(js) > nt // 2
        for li, v in zip(p.loop_indices, vs):
            i, j = divmod(v, nt); u = su[i, j] if not (wrap and j == 0) else su[i, -1] + np.linalg.norm(Vg[i, 0] - Vg[i, -1])
            uvl.data[li].uv = (u / 0.30, Vg[i, j, 2] / 0.30)
        p.use_smooth = True
    o = bpy.data.objects.new(name, m); gcol.objects.link(o); m.materials.append(mat)
    sm = o.modifiers.new('solid', 'SOLIDIFY'); sm.thickness = 0.0012; sm.offset = -1.0
    return o


body = bpy.data.materials.new('Trouser_Body'); trim = bpy.data.materials.new('Trouser_Trim'); btn = bpy.data.materials.new('Button')
grids = {}
for sx, tag in ((1.0, 'L'), (-1.0, 'R')):
    Vg, zs = leg(sx); grids[tag] = Vg
    o = add_tube('Leg_' + tag, Vg, body, flip=False)
    n = np.array(o.data.polygons[NT * 10].normal); c = np.array(o.data.polygons[NT * 10].center)
    if n @ (c - np.array([sx * 0.12, 0.01, c[2]])) < 0:
        bpy.data.objects.remove(o, do_unlink=True); add_tube('Leg_' + tag, Vg, body, flip=True)
# waistband: ring following the waist section, 38 mm tall, 1.5 mm proud
nb = 2 * NT; WB = np.zeros((12, nb, 3)); top = np.concatenate([grids['L'][-1], grids['R'][-1]])
# outer hull of the two half sections ordered by angle about the centre
ang = np.arctan2(top[:, 1] - 0.005, top[:, 0]); keep = np.abs(top[:, 0]) > 1e-4; ring = top[keep][np.argsort(ang[keep])]
d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(np.vstack([ring, ring[:1]]), axis=0), axis=1))])
ring = np.stack([np.interp(np.linspace(0, d[-1], nb, endpoint=False), d, np.append(ring[:, k], ring[0, k])) for k in range(3)], 1)
cen = ring.mean(0); out_dir = ring - cen; out_dir[:, 2] = 0; out_dir /= np.linalg.norm(out_dir, axis=1, keepdims=True)
for i in range(12):
    WB[i] = ring + out_dir * 0.0018; WB[i, :, 2] = Z_W - 0.002 + 0.040 * i / 11
wb = add_tube('Waistband', WB, trim, flip=False)
n = np.array(wb.data.polygons[5].normal); c = np.array(wb.data.polygons[5].center)
if n @ (c - cen) < 0: bpy.data.objects.remove(wb, do_unlink=True); wb = add_tube('Waistband', WB, trim, flip=True)
wb.modifiers['solid'].thickness = 0.0028


def box(name, c, sx_, sy_, sz_, mat, rot=None):
    bm = bmesh.new(); bmesh.ops.create_cube(bm, size=1.0); me = bpy.data.meshes.new(name)
    for v in bm.verts: v.co = Vector((v.co.x * sx_, v.co.y * sy_, v.co.z * sz_))
    bm.to_mesh(me); bm.free(); o = bpy.data.objects.new(name, me); o.location = c; gcol.objects.link(o); me.materials.append(mat)
    if rot is not None: o.rotation_euler = rot
    bv = o.modifiers.new('bev', 'BEVEL'); bv.width = 0.0008; bv.segments = 2
    return o


def ring_at(frac):                               # point + outward dir on the waistband at a fraction of its perimeter (0 = +x side)
    k = int(frac * nb) % nb; return ring[k] + out_dir[k] * 0.0035, out_dir[k]


for fr in (0.045, 0.20, 0.30, 0.455, 0.545, 0.70, 0.80, 0.955):          # belt loops
    p, od = ring_at(fr); box('BeltLoop', (p[0], p[1], Z_W + 0.018), 0.010, 0.0022, 0.050, trim, rot=(0, 0, math.atan2(od[1], od[0]) + math.pi / 2))
front = min(range(nb), key=lambda k: (abs(ring[k, 0]), ring[k, 1]))        # centre front of the band
pf = ring[front] + out_dir[front] * 0.004
bm = bmesh.new(); bmesh.ops.create_cone(bm, cap_ends=True, segments=24, radius1=0.0075, radius2=0.0068, depth=0.0025); me = bpy.data.meshes.new('WaistButton'); bm.to_mesh(me); bm.free()
ob = bpy.data.objects.new('WaistButton', me); ob.location = (pf[0] + 0.012, pf[1] - 0.0012, Z_W + 0.018); ob.rotation_euler = (math.pi / 2, 0, 0); gcol.objects.link(ob); me.materials.append(btn)
# fly: a narrow raised facing from the band down the centre front
fy = [grids['L'][i, int(0.75 * NT)] for i in range(NZ) if zs[i] > 0.86]
for q in fy[::6]:
    box('Fly', (0.016, min(q[1], grids['L'][-1, int(0.75 * NT), 1]) - 0.0006, q[2]), 0.0012, 0.0012, 0.012, trim)

# apply modifiers into one evaluated set, save
dg = bpy.context.evaluated_depsgraph_get()
for o in list(gcol.objects):
    ev = o.evaluated_get(dg); me = bpy.data.meshes.new_from_object(ev); me.transform(o.matrix_world); o.modifiers.clear(); o.data = me; o.matrix_world = np.eye(4).T.tolist() if False else o.matrix_world
    o.location = (0, 0, 0); o.rotation_euler = (0, 0, 0)
    for p in me.polygons: p.use_smooth = o.name.startswith(('Leg', 'Waist'))
bpy.ops.wm.save_as_mainfile(filepath=DST, check_existing=False); print('SAVED', DST)

# ================================================================== stones
legs = {t: C.Mesh('Leg_' + t) for t in 'LR'}; wbm = C.Mesh('Waistband')
seater = C.Seater([legs['L'].bvh, legs['R'].bvh, wbm.bvh]); stones = []; report = dict(zones={}, rejects={})


def rej(zone, why): report['rejects'].setdefault(zone, {}); report['rejects'][zone][why] = report['rejects'][zone].get(why, 0) + 1


# outseam stud rows: both rows are offsets (+-11 mm of arc) of the same outseam line on the same sections
for tag, sx in (('L', 1.0), ('R', -1.0)):
    Vg = grids[tag]; zone = 'G_outseam_' + tag; n_ = 0
    arc = np.concatenate([np.zeros((NZ, 1)), np.cumsum(np.linalg.norm(np.diff(np.concatenate([Vg, Vg[:, :1]], 1), axis=1), axis=2), axis=1)], 1)
    for off, phase in ((-11 * MM, 0.0), (11 * MM, 0.5)):
        line = []
        for i in range(NZ):
            if not (Z_H + 0.012 <= zs[i] <= Z_W - 0.012): continue
            s = off % arc[i, -1]; j = int(np.searchsorted(arc[i], s)) - 1; w = (s - arc[i, j]) / max(arc[i, j + 1] - arc[i, j], 1e-12)
            line.append((1 - w) * Vg[i, j % NT] + w * Vg[i, (j + 1) % NT])
        Q, L = C.resample(np.array(line), 0.05 * MM); R = C.SS['SS30']
        for s in C.even_positions(L, 13.0 * MM, False, phase):
            q = min(int(round(s / (0.05 * MM))), len(Q) - 1); p = Q[q]; tg = Q[min(q + 5, len(Q) - 1)] - Q[max(q - 5, 0)]
            xc, a, b, yc = prof(p[2]); n0 = C.unit(np.array([(p[0] - sx * xc) / a ** 2, (p[1] - yc) / b ** 2, 0.0]))
            st, why = seater.seat(p, n0, tg, R)
            if st is None: rej(zone, why); continue
            stones.append(dict(P=st['P'], N=st['N'], T=st['T'], R=R, zone=zone, cls=0, prio=0)); n_ += 1
    report['zones'][zone] = dict(rows=2, row_offset_mm=11.0, pitch_mm=13.0, size='SS30 (6.4 mm)', stones=n_)

# shield on the wearer's left thigh (planar projection)
shield, _, _ = art.tile_shield(HE + 'src_catalog/better_style_catalog/assets/artwork_camp/lattice_tile.png', 181.4, 1.0)
c0 = np.array([0.105, -0.2, 0.86]); loc, nn, idx, dd = legs['L'].bvh.ray_cast(Vector(c0), Vector((0, 1, 0)), 0.5); c = np.array(loc)
n = C.unit(np.array(nn)); ex = C.unit(np.cross([0, 0, 1.0], n)); ex = ex if ex[0] > 0 else -ex; ey = np.cross(n, ex); ey = ey if ey[2] > 0 else -ey
k = 0
for x, y, r, cl in shield:
    q = c + x * MM * ex - y * MM * ey; loc, nn, idx, dd = legs['L'].bvh.ray_cast(Vector(q + 0.03 * n), Vector(-n), 0.06)
    if loc is None: rej('H_thigh_shield', 'off_panel'); continue
    st, why = seater.seat(np.array(loc), np.array(nn), ex, r * MM)
    if st is None: rej('H_thigh_shield', why); continue
    stones.append(dict(P=st['P'], N=st['N'], T=st['T'], R=r * MM, zone='H_thigh_shield', cls=0, prio=1)); k += 1
report['zones']['H_thigh_shield'] = dict(planned=len(shield), placed=k, drawing_scale=1.0)
P = np.array([s['P'] for s in stones]); N = np.array([s['N'] for s in stones]); R = np.array([s['R'] for s in stones]); prio = np.array([s['prio'] for s in stones])
keep, vrep = C.verify_exact(P, N, R, prio); report['exact_clearance'] = vrep; stones = [s for s, kk in zip(stones, keep) if kk]
zz = np.array([s['zone'] for s in stones]); report['counts'] = {z: int((zz == z).sum()) for z in sorted(set(zz))}; report['total'] = len(stones)
report['geometry'] = dict(waist_z=Z_W, crotch_z=Z_C, hem_z=Z_H, note='proportions read from the reference image; not a production pattern')
C.write_outputs(PREFIX, stones, report); print(json.dumps(report['counts']), vrep)
