"""solve_shirt.py - stone layout for the BS-BSTSET shirt on the rebuilt camp shirt.
usage: python -B solve_shirt.py <camp_rebuilt.blend> <collar_grids.npz> <out_prefix>
Zones: A collar leaf (parallel rows), B sleeve hems (parallel rows), C back crown + script, D all-over crowned BST shields,
E elephant crest (front, wearer's right hem)."""
import sys, os, math, time, json
sys.dont_write_bytecode = True
import numpy as np
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
import stones_core as C, art

BLEND, GRIDS, PREFIX = sys.argv[1:4]
HE = "C:/Users/USER/Desktop/4041_MechMix/02_engines/human_engine/"
ONLY = set(next((a.split('=')[1] for a in sys.argv if a.startswith('--zones=')), 'A,B,C,D,E').split(','))
MM = C.MM; T0 = time.time()
bpy.ops.wm.open_mainfile(filepath=BLEND, load_ui=False)
def log(*a): print('[%6.1fs]' % (time.time() - T0), *a, flush=True)

tee = C.Mesh('Tee'); col = C.Mesh('collar'); man = C.Mesh('Mannequin')
others = [C.Mesh(n) for n in ('placket_overlap', 'placket_buttons')]
cache = PREFIX + '_outer.npy'
if os.path.exists(cache) and len(np.load(cache)) == len(tee.T): tee.outer = np.load(cache)
else: tee.outer_mask(man.bvh); np.save(cache, tee.outer)
log('meshes', len(tee.T), 'tee tris, outer', int(tee.outer.sum()))
seater = C.Seater([tee.bvh, col.bvh] + [o.bvh for o in others])
stones = []; report = dict(zones={}, rejects={})
CANVAS = (638.0, 700.0)


def add(st, zone, R, cls=0, prio=1):
    stones.append(dict(P=st['P'], N=st['N'], T=st['T'], R=R, zone=zone, cls=cls, prio=prio))


def rej(zone, why): report['rejects'].setdefault(zone, {}); report['rejects'][zone][why] = report['rejects'][zone].get(why, 0) + 1


def place_rows(zone, sections, s_edge, rows, closed, prio=0):
    """sections: (ns, nj, 3) dense surface sections ordered along the band, each running from the band's source edge inward;
    s_edge: distance field phi of every section sample from the band's source edge (ns, nj). rows: [(d, R, phase)].
    Every row is the iso-line phi = d_k of the SAME field -> rows are parallel offsets of the edge and their separation is d_{k+1} - d_k by construction."""
    ns = len(sections); out = dict(rows=[])
    for k, (d, R, phase) in enumerate(rows):
        pts = np.full((ns, 3), np.nan); nrm = np.zeros((ns, 3))
        for i in range(ns):
            s = s_edge[i]; ok = np.isfinite(s)
            if ok.sum() < 3 or np.nanmax(s) < d + R + 0.3 * MM: continue
            idx = np.nonzero(ok)[0]; cr = np.nonzero((s[idx][:-1] < d) & (s[idx][1:] >= d))[0]      # first crossing of the level along the section
            if not len(cr): continue
            a, b = idx[cr[0]], idx[cr[0] + 1]; w = (d - s[a]) / max(s[b] - s[a], 1e-12); pts[i] = (1 - w) * sections[i, a] + w * sections[i, b]
        valid = np.isfinite(pts[:, 0])
        # runs of consecutive valid sections
        runs = []; i = 0
        while i < ns:
            if valid[i]:
                j = i
                while j + 1 < ns and valid[j + 1]: j += 1
                runs.append((i, j)); i = j + 1
            else: i += 1
        is_closed = closed and len(runs) == 1 and runs[0] == (0, ns - 1)
        placed = 0; Ltot = 0.0
        for (i0, i1) in runs:
            Pl = pts[i0:i1 + 1]
            if len(Pl) < 4: continue
            Q, L = C.resample(Pl, 0.05 * MM, closed=is_closed); Ltot += L
            pitch = 2 * R + 0.20 * MM
            for s in C.even_positions(L, pitch, is_closed, phase):
                q = int(round(s / (0.05 * MM))); q = min(q, len(Q) - 1); p = Q[q]
                tg = Q[min(q + 4, len(Q) - 1)] - Q[max(q - 4, 0)]
                # section direction at this point gives the outward normal guess: n = unit(tangent x across)
                ii = i0 + int(round(q / max(len(Q) - 1, 1) * (i1 - i0))); sec = sections[ii]; okj = np.isfinite(sec[:, 0]); sj = sec[okj]
                jj = int(np.argmin(np.linalg.norm(sj - p, axis=1))); ac = sj[min(jj + 1, len(sj) - 1)] - sj[max(jj - 1, 0)]
                n0 = C.unit(np.cross(tg, ac))
                if n0 @ (p - CORE(p)) < 0: n0 = -n0
                st, why = seater.seat(p, n0, tg, R)
                if st is None: rej(zone, why); continue
                add(st, zone, R, 0, prio); placed += 1
        out['rows'].append(dict(d_mm=round(d / MM, 3), R_mm=R / MM, runs=len(runs), closed=bool(is_closed), length_mm=round(Ltot / MM, 1), stones=placed))
    return out


# ================================================================== Zone A: collar leaf
if 'A' in ONLY:
    CORE = lambda p: np.array([0.0, 0.005, 1.36])
    g = np.load(GRIDS); L_, R_ = g['leaf_L'], g['leaf_R']
    G = np.concatenate([L_, R_[-2::-1]], 0)                                   # wearer's-left front -> centre back -> right front
    ti = np.arange(len(G)); tf = np.linspace(0, len(G) - 1, (len(G) - 1) * 10 + 1)
    G = CubicSpline(ti, G, axis=0)(tf)                                        # dense along the band
    vj = np.arange(G.shape[1]); vf = np.linspace(0, G.shape[1] - 1, (G.shape[1] - 1) * 8 + 1)
    G = CubicSpline(vj, G, axis=1)(vf)[:, ::-1]                               # across, re-ordered: outer edge first
    arc = np.concatenate([np.zeros((len(G), 1)), np.cumsum(np.linalg.norm(np.diff(G, axis=1), axis=2), axis=1)], 1)
    width = arc[:, -1].copy(); keep_j = arc <= (width[:, None] - 9.0 * MM)                 # stay clear of the rolled fold
    src = np.vstack([C.resample(G[0, ::-1], 0.1 * MM)[0], C.resample(G[:, 0], 0.1 * MM)[0], C.resample(G[-1], 0.1 * MM)[0]])   # front edge, outer edge, front edge
    s_edge = cKDTree(src).query(G.reshape(-1, 3))[0].reshape(G.shape[:2]); s_edge[~keep_j] = np.nan
    rows = [(2.3 * MM, C.SS['SS10'], 0.0)]; d = 2.3 * MM + C.SS['SS10'] + 0.35 * MM + C.SS['SS16']; k = 1
    while d < np.nanmax(s_edge):
        rows.append((d, C.SS['SS16'], 0.5 * (k % 2))); d += 2 * C.SS['SS16'] + 0.35 * MM; k += 1
    rep = place_rows('A_collar', G, s_edge, rows, closed=False, prio=0)
    rep['leaf_width_mm'] = [round(float(width.min() / MM), 1), round(float(width.max() / MM), 1)]
    report['zones']['A_collar'] = rep; log('zone A', sum(r['stones'] for r in rep['rows']), 'stones in', len(rep['rows']), 'rows')

# ================================================================== Zone B: sleeve hems
if 'B' in ONLY:
    ob = bpy.data.objects['Tee']; me = ob.data
    for side in ('L', 'R'):
        gi = ob.vertex_groups[f'set_in_sleeve_{side}'].index; w = np.zeros(len(tee.V))
        for v in me.vertices:
            for gg in v.groups:
                if gg.group == gi: w[v.index] = gg.weight
        sm = w > 0.5; mo = bpy.data.objects['Mannequin']; am = np.zeros(len(man.V), bool)
        for nm in (f'upperarm01.{side}', f'upperarm02.{side}'):
            gj = mo.vertex_groups[nm].index
            for v in mo.data.vertices:
                for gg in v.groups:
                    if gg.group == gj and gg.weight > 0.5: am[v.index] = True
        Pa = man.V[am]; c0 = Pa.mean(0); a = np.linalg.svd(Pa - c0)[2][0]
        if (tee.V[sm].mean(0) - c0) @ a < 0: a = -a
        if a[2] > 0 and abs(a[0]) < 0.3: a = -a
        tri = np.nonzero(np.all(sm[tee.T], 1) & tee.outer)[0]
        sbvh = BVHTree.FromPolygons(tee.V.tolist(), tee.T[tri].tolist(), all_triangles=True)
        NTH = 1080
        for pas in (0, 1):
            e1 = C.unit(np.cross(a, [0, 1, 0])); e2 = np.cross(a, e1)
            tt = (tee.V[sm] - c0) @ a; t_hi = tt.max() + 2 * MM; t_lo = tt.max() - (90 if pas == 0 else 45) * MM
            ts = np.arange(t_hi, t_lo, -(1.0 if pas == 0 else 0.25) * MM); nth = 180 if pas == 0 else NTH; sec = np.full((nth, len(ts), 3), np.nan)
            for i in range(nth):
                th = 2 * math.pi * i / nth; dr = math.cos(th) * e1 + math.sin(th) * e2
                for j, t in enumerate(ts):
                    loc, nn, idx, dd = sbvh.ray_cast(Vector(c0 + t * a + 0.30 * dr), Vector(-dr), 0.30)
                    if loc is not None: sec[i, j] = loc
            if pas == 0:                                                           # hem ring -> its plane normal is the sleeve's own axis
                ring = np.array([sec[i, np.argmax(np.isfinite(sec[i, :, 0]))] for i in range(nth) if np.isfinite(sec[i, :, 0]).any()])
                rc = ring.mean(0); nrm = np.linalg.svd(ring - rc)[2][2]
                if nrm @ a < 0: nrm = -nrm
                tilt = math.degrees(math.acos(min(1.0, abs(nrm @ a)))); a = nrm; c0 = rc - 0.03 * a
        # hem edge per section = first hit from the distal side; smooth the edge position round the loop (the SOURCE, once)
        first = np.array([np.argmax(np.isfinite(sec[i, :, 0])) if np.isfinite(sec[i, :, 0]).any() else -1 for i in range(NTH)])
        good = first >= 0; tf_ = np.where(good, ts[np.clip(first, 0, None)], np.nan)
        tf_ = np.interp(np.arange(NTH), np.nonzero(good)[0], tf_[good], period=NTH); tf_s = gaussian_filter1d(tf_, 6, mode='wrap')
        s_edge = np.full(sec.shape[:2], np.nan); ring = np.full((NTH, 3), np.nan)
        for i in range(NTH):
            okj = np.isfinite(sec[i, :, 0]) & (ts <= tf_s[i] + 0.25 * MM)
            if okj.sum() < 4: continue
            idx = np.nonzero(okj)[0]; P = sec[i, idx]; stp = np.linalg.norm(np.diff(P, axis=0), axis=1)
            brk = np.nonzero(stp > 1.5 * MM)[0]                                   # a jump = the ray moved to another fold: stop the section there
            if len(brk): idx = idx[:brk[0] + 1]; P = P[:brk[0] + 1]
            s_edge[i, idx] = 0.0; ring[i] = P[0]
        okr = np.isfinite(ring[:, 0]); ring = np.stack([np.interp(np.arange(NTH), np.nonzero(okr)[0], ring[okr, k], period=NTH) for k in range(3)], 1)
        ring = gaussian_filter1d(ring, 5, axis=0, mode='wrap'); src = C.resample(ring, 0.1 * MM, closed=True)[0]
        m_ = np.isfinite(s_edge); s_edge[m_] = cKDTree(src).query(sec[m_])[0]
        CORE = (lambda c0_, a_: (lambda p: c0_ + ((p - c0_) @ a_) * a_))(c0, a)
        d1 = 2.3 * MM; d2 = d1 + C.SS['SS10'] + 0.35 * MM + C.SS['SS16']; d3 = d2 + 2 * C.SS['SS16'] + 0.35 * MM; d4 = d3 + 2 * C.SS['SS16'] + 0.35 * MM
        d5 = d4 + C.SS['SS16'] + 0.35 * MM + C.SS['SS10']
        rows = [(d1, C.SS['SS10'], 0.0), (d2, C.SS['SS16'], 0.5), (d3, C.SS['SS16'], 0.0), (d4, C.SS['SS16'], 0.5), (d5, C.SS['SS10'], 0.0)]
        rep = place_rows(f'B_cuff_{side}', sec, s_edge, rows, closed=True, prio=0); rep['axis'] = a.round(4).tolist(); rep['hem_plane_vs_arm_axis_deg'] = round(tilt, 1); rep['sections_with_edge'] = int(good.sum())
        report['zones'][f'B_cuff_{side}'] = rep; log('zone B', side, sum(r['stones'] for r in rep['rows']), [(r['runs'], r['stones']) for r in rep['rows']])

# ================================================================== artwork zones on the body (UV-mapped)
CORE = lambda p: np.array([0.0, -0.03, p[2]])
if ONLY & {'C', 'D', 'E'}:
    for key, layer in (('front', 'UVMap_Art_Front'), ('back', 'UVMap_Art_Back')):
        uvt = tee.loop_uv(layer)[tee.TL]; inside = np.all((uvt >= 0) & (uvt <= 1), axis=(1, 2))
        log('uv', key, tee.build_uv(key, uvt, tee.outer & inside))


def canvas_point(key, x_mm, y_mm):
    h = tee.uv_to_surface(key, (x_mm / CANVAS[0], 1.0 - y_mm / CANVAS[1])); return None if h is None else h[0]


def fit_frame(centre, ex0, ey0, half_w, half_h):
    """least-squares plane through the outer surface under a design; x axis kept as close to ex0 as the plane allows."""
    n0 = C.unit(np.cross(ex0, ey0)); pts = []
    for ax in np.linspace(-1, 1, 9):
        for ay in np.linspace(-1, 1, 9):
            q = centre + ax * half_w * ex0 + ay * half_h * ey0
            loc, nn, idx, dd = tee.bvh.ray_cast(Vector(q + 0.06 * n0), Vector(-n0), 0.12)
            if loc is not None: pts.append(np.array(loc))
    pts = np.array(pts); c = pts.mean(0); n = np.linalg.svd(pts - c)[2][2]
    if n @ n0 < 0: n = -n
    ex = C.unit(ex0 - (ex0 @ n) * n); ey = np.cross(n, ex)
    if ey @ ey0 < 0: ey = -ey
    return ex, ey, n


def place_planar(zone, pts, centre, ex, ey, n, prio, depth=0.03, guard=None):
    """pts (x_mm, y_mm down, R_mm, cls) about `centre`; projected along -n onto the outer surface. A parallel projection onto a
    surface can only lengthen distances, so gaps designed in the plane are never reduced on the garment."""
    k = 0
    for x, y, r, c in pts:
        q = centre + x * MM * ex - y * MM * ey
        loc, nn, idx, dd = tee.bvh.ray_cast(Vector(q + depth * n), Vector(-n), 2 * depth)
        if loc is None: rej(zone, 'off_panel'); continue
        P0 = np.array(loc); nn = np.array(nn)
        if nn @ n < 0.55: rej(zone, 'steep'); continue
        if guard is not None and guard(P0): continue
        st, why = seater.seat(P0, nn, ex, r * MM)
        if st is None: rej(zone, why); continue
        add(st, zone, r * MM, c, prio); k += 1
    return k


def canvas_frame(key, cx, cy, hw, hh):
    c = canvas_point(key, cx, cy); a_ = canvas_point(key, cx + 60, cy); b_ = canvas_point(key, cx - 60, cy); u_ = canvas_point(key, cx, cy - 60); d_ = canvas_point(key, cx, cy + 60)
    ex0 = C.unit(a_ - b_); ey0 = C.unit(u_ - d_); ey0 = C.unit(ey0 - (ey0 @ ex0) * ex0)
    return (c,) + fit_frame(c, ex0, ey0, hw * MM, hh * MM)


if 'E' in ONLY:
    m, cimg, mp = art.elephant_masks(HE + 'engineering/PHASE3_2026-11-09/elephant_evidence/elephant_final_548x852.png', height_mm=235.0)
    S = art.contour_fill(m, mp, 1.0, gap=0.25, ensure_min=False, cls_img=cimg); ew, eh = m.shape[1] * mp, m.shape[0] * mp
    loc, nn_, idx_, dd_ = tee.bvh.ray_cast(Vector((-0.138, -1.0, 1.015)), Vector((0, 1, 0)), 2.0); c = np.array(loc)     # wearer's right, low on the front
    ex, ey, n = fit_frame(c, np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), ew / 2 * MM, eh / 2 * MM)
    k = place_planar('E_elephant', [(x - ew / 2, y - eh / 2, 1.0, cc) for x, y, _, cc in S], c, ex, ey, n, 1)
    report['zones']['E_elephant'] = dict(planned=len(S), placed=k, size_mm=[round(ew, 1), round(eh, 1)], plane_normal=n.round(3).tolist()); log('zone E', k, '/', len(S))
if 'C' in ONLY:
    m, mp = art.script_mask(width_mm=300.0); S = art.contour_fill(m, mp, 1.0, gap=0.25); sw, sh = m.shape[1] * mp, m.shape[0] * mp
    c, ex, ey, n = canvas_frame('back', 319.0, 300.0, sw / 2, sh / 2 + 40)
    n1 = place_planar('C_script', [(x - sw / 2, y - sh / 2, 1.0, 0) for x, y, _, _ in S], c, ex, ey, n, 1)
    mc, mp2 = art.crown_mask(74.0); Sc = art.contour_fill(mc, mp2, 1.0, gap=0.25); cw, ch = mc.shape[1] * mp2, mc.shape[0] * mp2
    n2 = place_planar('C_crown', [(x - cw / 2 + 14, y - sh / 2 - ch + 4, 1.0, 0) for x, y, _, _ in Sc], c, ex, ey, n, 1)
    report['zones']['C_script'] = dict(planned=len(S), placed=n1, size_mm=[round(sw, 1), round(sh, 1)], font='Brush Script MT (stand-in)')
    report['zones']['C_crown'] = dict(planned=len(Sc), placed=n2, size_mm=[round(cw, 1), round(ch, 1)]); log('zone C', n1, n2)

if 'D' in ONLY:
    SC = 0.90; shield, (fx, fy), (Wm, Hm) = art.tile_shield(HE + 'src_catalog/better_style_catalog/assets/artwork_camp/lattice_tile.png', 181.4, SC)
    uvp, mir = C.print_uv_tri(tee); tee.build_uv('print', uvp, tee.outer)
    art_pts = np.array([s['P'] for s in stones if s['zone'][0] in 'CE'])
    art_kd = cKDTree(art_pts) if len(art_pts) else None
    lab = [canvas_point('front', (1321 + ax * 307) / 3.21, (407 + ay * 85) / 3.21) for ax in np.linspace(0, 1, 12) for ay in np.linspace(0, 1, 4)]
    lab_kd = cKDTree(np.array([h for h in lab if h is not None]))
    def guard(P0):
        if art_kd is not None and art_kd.query(P0)[0] < 6.0 * MM: rej('D_monogram', 'under_art'); return True
        if lab_kd.query(P0)[0] < 7.0 * MM: rej('D_monogram', 'under_label'); return True
        return False
    um, vm = uvp[tee.outer][..., 0], uvp[tee.outer][..., 1]; tw_uv, th_uv = (181.4 * MM) / 0.30, (181.4 * Hm / Wm * MM) / 0.30
    occ = set(zip(np.floor(um.ravel() / tw_uv).astype(int).tolist(), np.floor(vm.ravel() / th_uv).astype(int).tolist()))
    def P_uv(u, v):
        h = tee.uv_to_surface('print', (u, v)); return None if h is None else h[0]
    centres = []; k = 0; inst = 0; dl = 0.02 / 0.30
    for (i, j) in sorted(occ):
        for (ax, ay) in ((fx, fy), ((fx + 0.5) % 1.0, (fy + 0.5) % 1.0)):
            u0, v0 = (i + ax) * tw_uv, (j + 1 - ay) * th_uv; c = P_uv(u0, v0)
            if c is None: continue
            nb = [P_uv(u0 + dl, v0), P_uv(u0 - dl, v0), P_uv(u0, v0 + dl), P_uv(u0, v0 - dl)]
            if any(q is None for q in nb): rej('D_monogram', 'instance_at_panel_edge'); continue
            n0 = C.unit(np.cross(nb[0] - nb[1], nb[2] - nb[3])); hd = tee.uv_to_surface('print', (u0, v0))
            if n0 @ tee.FN[hd[4]] < 0: n0 = -n0
            up = np.array([0.0, 0.0, 1.0]); ey0 = up - (up @ n0) * n0
            if np.linalg.norm(ey0) < 0.35: rej('D_monogram', 'instance_on_horizontal_surface'); continue      # shoulder top: no upright direction
            ey0 = C.unit(ey0); ex0 = np.cross(ey0, n0)                                   # upright and readable from outside on every panel
            if abs(c[0]) < 0.060 and c[1] < 0: rej('D_monogram', 'instance_on_placket'); continue
            if centres and np.min(np.linalg.norm(np.array(centres) - c, axis=1)) < 70 * MM: rej('D_monogram', 'instance_overlaps_other_panel'); continue
            ex, ey, n = fit_frame(c, ex0, ey0, 0.03, 0.03); start = len(stones)
            kk = place_planar('D_monogram', shield, c, ex, ey, n, 2, depth=0.02, guard=guard)
            if kk < 0.72 * len(shield):                                                 # a shield is whole or absent: no fragments at seams, collar or art
                del stones[start:]; rej('D_monogram', 'instance_incomplete'); continue
            centres.append(c); inst += 1; k += kk
    report['zones']['D_monogram'] = dict(stones_per_shield=len(shield), drawing_scale=SC, instances=inst, placed=k); log('zone D', k, 'stones in', inst, 'shields of', len(shield))

# ================================================================== exact clearance certificate over the whole garment
P = np.array([s['P'] for s in stones]); N = np.array([s['N'] for s in stones]); R = np.array([s['R'] for s in stones]); prio = np.array([s['prio'] for s in stones])
# band stones outrank everything; a monogram stone inside a band's footprint is dropped before the exact test (planned exclusion)
band = prio == 0
if band.any() and (~band).any():
    kd = cKDTree(P[band]); dmin, _ = kd.query(P[~band]); drop = np.nonzero(~band)[0][dmin < 4.6 * MM]
    keep0 = np.ones(len(P), bool); keep0[drop] = False; report['band_exclusion_removed'] = int(len(drop))
    stones = [s for s, k in zip(stones, keep0) if k]; P, N, R, prio = P[keep0], N[keep0], R[keep0], prio[keep0]
keep, vrep = C.verify_exact(P, N, R, prio); report['exact_clearance'] = vrep; log('exact', vrep)
zones = np.array([s['zone'] for s in stones]); report['removed_by_zone'] = {z: int((~keep[zones == z]).sum()) for z in sorted(set(zones))}
stones = [s for s, k in zip(stones, keep) if k]
keep2, vrep2 = C.verify_exact(P[keep], N[keep], R[keep], prio[keep]); report['exact_clearance_final'] = vrep2
zones = np.array([s['zone'] for s in stones]); Rf = np.array([s['R'] for s in stones])
report['counts'] = {z: int((zones == z).sum()) for z in sorted(set(zones))}; report['total'] = len(stones)
report['by_size'] = {k: int(np.isclose(Rf, v).sum()) for k, v in C.SS.items()}
C.write_outputs(PREFIX, stones, report); log('TOTAL', len(stones), report['counts'], report['by_size']); log('rejects', report['rejects'])
