"""solve_royal_shirt.py - BST Royal Camp Set shirt (BS-BSTSET-001, recipe of 2026-11-12): stones and raised beds on the rebuilt camp shirt.
usage: python -B solve_royal_shirt.py <camp_rebuilt.blend> <collar_grids.npz> <out_prefix> [--zones=A,B,C,D,E,F] [--outer=<cached outer mask .npy>]

Zones (recipe numbering in brackets):
  A [3] collar diadem      double-row base rail on the leaf and lapel edges, centre-back keystone, pave crowns, bezel solitaires between
  B [1] sleeve cuffs       triple rail / X-lattice / rosette / V-gap accent bracelet, laid out in the cuff's own (arc, distance-from-hem) chart
  C [4] back               satin patch + double-row script + three-row underline; separate crown patch above
  D [F] jacquard pinpoints one crystal at the crown apex of every woven shield (the shields themselves are thread, not stones)
  E [2] elephant           relief bed + structural stone design
  F [4] chest signet       satin badge with embroidered lettering and pocket welt (no stones)
The garment .blend is only read. Outputs: <prefix>_stones.npz/.f32/.report.json, <prefix>_beds.npz, <prefix>_jacquard.png."""
import sys, os, math, time, json, shutil
sys.dont_write_bytecode = True
import numpy as np
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
sys.path.insert(0,os.path.normpath(os.path.expanduser('~/Desktop/better_style_execution_workspace/solvers/rhinestone_engine/garments/tools')))
import stones_core as C
import royal_core as K, royal_art as A, royal_elephant as EL

BLEND, GRIDS, PREFIX = sys.argv[1:4]
ONLY = set(next((a.split('=')[1] for a in sys.argv if a.startswith('--zones=')), 'A,B,C,D,E,F').split(','))
OUTER = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--outer=')), None)
MM = C.MM; T0 = time.time()
bpy.ops.wm.open_mainfile(filepath=BLEND, load_ui=False)
def log(*a): print('[%6.1fs]' % (time.time() - T0), *a, flush=True)

tee = C.Mesh('Tee'); col = C.Mesh('collar'); man = C.Mesh('Mannequin'); others = [C.Mesh(n) for n in ('placket_overlap', 'placket_buttons')]
cache = PREFIX + '_outer.npy'
if not os.path.exists(cache) and OUTER and os.path.exists(OUTER): shutil.copyfile(OUTER, cache)
if os.path.exists(cache) and len(np.load(cache)) == len(tee.T): tee.outer = np.load(cache)
else: tee.outer_mask(man.bvh); np.save(cache, tee.outer)
log('meshes', len(tee.T), 'tee tris, outer', int(tee.outer.sum()))
seater = C.Seater([tee.bvh, col.bvh] + [o.bvh for o in others])
stones = []; beds = []; bed_bvhs = []; report = dict(zones={}, rejects={}, recipe='BST Royal Camp Set, Owner recipe 2026-11-12')


def rej(zone, why): report['rejects'].setdefault(zone, {}); report['rejects'][zone][why] = report['rejects'][zone].get(why, 0) + 1


def add(st, zone, s, prio):
    stones.append(dict(P=st['P'], N=st['N'], T=st['T'], R=s['R'] * MM, shape=s['shape'], L=s['L'] * MM, W=s['W'] * MM, bezel=s['bezel'], cls=s['cls'], zone=zone, tag=s['tag'], prio=prio))


def rnd(size, tag='', bezel=None):
    R = K.rad(size); return dict(R=R, shape='round', L=2 * R, W=2 * R, bezel=(R >= 2.3) if bezel is None else bezel, cls=0, tag=tag, rot=0.0)


def cast_top(q, n, depth, targets):
    """first surface met coming down along -n onto `targets` (highest layer wins) -> (point, face normal) or None"""
    best = None
    for b in targets:
        loc, nn, idx, d = b.ray_cast(Vector(q + depth * n), Vector(-n), 2 * depth)
        if loc is not None and (best is None or d < best[2]): best = (np.array(loc), np.array(nn), d)
    if best is None: return None
    nn = best[1] if best[1] @ n > 0 else -best[1]; return best[0], nn


def fpoint(fr, x, y): return fr[0] + x * MM * fr[1] - y * MM * fr[2]


def add_bed(zone, b, fr, targets, depth=0.03, cell=0.4):
    """mesh a design bed (mask + height field in the design plane) onto the garment: every grid corner is projected along -n onto
    `targets` and raised by the height; the rim is sunk 0.3 mm so the patch closes onto the cloth."""
    c, ex, ey, n = fr; f = max(1, int(round(cell / b['mmpp']))); m = b['mask']; H, W = m.shape; Hc, Wc = H // f, W // f
    mm_ = m[:Hc * f, :Wc * f].reshape(Hc, f, Wc, f); cnt = mm_.sum((1, 3)); mc = cnt >= 0.5 * f * f
    hc = (b['h'][:Hc * f, :Wc * f] * m[:Hc * f, :Wc * f]).reshape(Hc, f, Wc, f).sum((1, 3)) / np.maximum(cnt, 1)
    if isinstance(b['mat'], str): idc = np.zeros((Hc, Wc), np.uint8); mats = [b['mat']]
    else: idc = b['mat'][0][f // 2:Hc * f:f, f // 2:Wc * f:f][:Hc, :Wc]; mats = list(b['mat'][1])
    n4 = np.zeros((Hc + 1, Wc + 1)); hs = np.zeros((Hc + 1, Wc + 1))
    for di in (0, 1):
        for dj in (0, 1): n4[di:di + Hc, dj:dj + Wc] += mc; hs[di:di + Hc, dj:dj + Wc] += np.where(mc, hc, 0)
    used = n4 > 0; hv = np.where(used, hs / np.maximum(n4, 1), 0); hv[used & (n4 < 4)] = -0.3
    cs = b['mmpp'] * f; vid = -np.ones((Hc + 1, Wc + 1), np.int64); V = []
    for i, j in zip(*np.nonzero(used)):
        hit = cast_top(fpoint(fr, b['org'][0] + j * cs, b['org'][1] + i * cs), n, depth, targets)
        if hit is None: continue
        vid[i, j] = len(V); V.append(hit[0] + hv[i, j] * MM * n)
    F = []; M = []
    for i, j in zip(*np.nonzero(mc)):
        q = (vid[i, j], vid[i + 1, j], vid[i + 1, j + 1], vid[i, j + 1])
        if min(q) >= 0: F.append(q); M.append(int(idc[i, j]))
    if not F: rej(zone, 'bed_off_panel'); return None
    V = np.array(V); F = np.array(F); bv = BVHTree.FromPolygons(V.tolist(), F.tolist())
    beds.append(dict(name=zone + ':' + b['name'], V=V, F=F, M=np.array(M, np.uint8), mats=mats)); bed_bvhs.append(bv); seater.bvhs.append(bv)
    report['zones'].setdefault(zone, {}).setdefault('beds', []).append(dict(name=b['name'], faces=len(F), cells_lost=int(mc.sum() - len(F)), materials=mats, max_height_mm=round(float(hv.max()), 2)))
    return bv


def place_design(zone, D, fr, prio, targets, depth=0.03, guard=None, steep=0.5, dx=0.0, dy=0.0):
    c, ex, ey, n = fr; k = 0
    for s in D.S:
        hit = cast_top(fpoint(fr, s['x'] + dx, s['y'] + dy), n, depth, targets)
        if hit is None: rej(zone, 'off_panel'); continue
        P0, nn = hit
        if nn @ n < steep: rej(zone, 'steep'); continue
        if guard is not None and guard(P0): continue
        t_raw = math.cos(s['rot']) * ex - math.sin(s['rot']) * ey
        st, why = seater.seat(P0, nn, t_raw, (s['R'] if s['shape'] == 'round' else 0.5 * math.sqrt(s['L'] * s['W'])) * MM)
        if st is None: rej(zone, why); continue
        add(st, zone, s, prio); k += 1
    return k


def place_rows(zone, sections, s_edge, rows, closed, prio=0, skip=None, tag='rail'):
    """solve_shirt.place_rows: every row is the iso-line phi = d_k of ONE distance field from the band's source edge, so the rows
    are parallel offsets of that edge by construction. rows: [(d, size, phase)]."""
    ns = len(sections); out = dict(rows=[])
    for k, (d, size, phase) in enumerate(rows):
        R = K.rad(size) * MM; pts = np.full((ns, 3), np.nan)
        for i in range(ns):
            s = s_edge[i]; ok = np.isfinite(s)
            if ok.sum() < 3 or np.nanmax(s) < d + R + 0.3 * MM: continue
            idx = np.nonzero(ok)[0]; cr = np.nonzero((s[idx][:-1] < d) & (s[idx][1:] >= d))[0]
            if not len(cr): continue
            a, b = idx[cr[0]], idx[cr[0] + 1]; w = (d - s[a]) / max(s[b] - s[a], 1e-12); pts[i] = (1 - w) * sections[i, a] + w * sections[i, b]
        valid = np.isfinite(pts[:, 0]); runs = []; i = 0
        while i < ns:
            if valid[i]:
                j = i
                while j + 1 < ns and valid[j + 1]: j += 1
                runs.append((i, j)); i = j + 1
            else: i += 1
        is_closed = closed and len(runs) == 1 and runs[0] == (0, ns - 1); placed = 0; Ltot = 0.0
        for (i0, i1) in runs:
            Pl = pts[i0:i1 + 1]
            if len(Pl) < 4: continue
            Q, L = C.resample(Pl, 0.05 * MM, closed=is_closed); Ltot += L
            for s in C.even_positions(L, 2 * R + 0.25 * MM, is_closed, phase):
                q = min(int(round(s / (0.05 * MM))), len(Q) - 1); p = Q[q]
                if skip is not None and skip(p): continue
                tg = Q[min(q + 4, len(Q) - 1)] - Q[max(q - 4, 0)]
                ii = i0 + int(round(q / max(len(Q) - 1, 1) * (i1 - i0))); sec = sections[ii]; okj = np.isfinite(sec[:, 0]); sj = sec[okj]
                jj = int(np.argmin(np.linalg.norm(sj - p, axis=1))); ac = sj[min(jj + 1, len(sj) - 1)] - sj[max(jj - 1, 0)]
                n0 = C.unit(np.cross(tg, ac))
                if n0 @ (p - CORE(p)) < 0: n0 = -n0
                st, why = seater.seat(p, n0, tg, R)
                if st is None: rej(zone, why); continue
                add(st, zone, rnd(size, tag), prio); placed += 1
        out['rows'].append(dict(d_mm=round(d / MM, 3), size=size, runs=len(runs), closed=bool(is_closed), length_mm=round(Ltot / MM, 1), stones=placed))
    return out


def dense_grid(G, ku, kv):
    ti = np.arange(len(G)); G = CubicSpline(ti, G, axis=0)(np.linspace(0, len(G) - 1, (len(G) - 1) * ku + 1))
    vj = np.arange(G.shape[1]); return CubicSpline(vj, G, axis=1)(np.linspace(0, G.shape[1] - 1, (G.shape[1] - 1) * kv + 1))


def surf_frame(p, along, up, core):
    """frame on a surface point: design x = viewer's right with `up` up, n outward from `core`"""
    n = C.unit(np.cross(along, up))
    if n @ (p - core) < 0: n = -n
    up = C.unit(up - (up @ n) * n); ex = np.cross(up, n); return p, ex, np.cross(n, ex), n


def single(size, tag): D = K.Design(); D.put(0, 0, size, tag=tag); return D


# ================================================================== Zone A: collar diadem
if 'A' in ONLY:
    CORE = lambda p: np.array([0.0, 0.005, 1.36]); core = CORE(None); z = 'A_collar'
    g = np.load(GRIDS); L_, R_ = g['leaf_L'], g['leaf_R']
    G = dense_grid(np.concatenate([L_, R_[-2::-1]], 0), 10, 8)[:, ::-1]            # along: wearer's-left front -> centre back -> right front; across: outer edge first
    arc = np.concatenate([np.zeros((len(G), 1)), np.cumsum(np.linalg.norm(np.diff(G, axis=1), axis=2), axis=1)], 1)
    width = arc[:, -1].copy(); keep_j = arc <= (width[:, None] - 9.0 * MM)          # stay clear of the rolled fold
    src = np.vstack([C.resample(G[0, ::-1], 0.1 * MM)[0], C.resample(G[:, 0], 0.1 * MM)[0], C.resample(G[-1], 0.1 * MM)[0]])
    s_edge = cKDTree(src).query(G.reshape(-1, 3))[0].reshape(G.shape[:2]); s_edge[~keep_j] = np.nan
    along = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(G[:, 0], axis=0), axis=1))]); i_cb = (len(L_) - 1) * 10; a_cb = along[i_cb]

    def chart(a_mm, d_mm):
        """(arc from centre back along the outer edge, distance in from the outer edge) -> frame with 'up' toward the roll line"""
        i = int(np.clip(np.searchsorted(along, a_cb + a_mm * MM), 6, len(G) - 7)); j = int(np.clip(np.searchsorted(arc[i], d_mm * MM), 4, G.shape[1] - 5))
        return surf_frame(G[i, j], G[i + 6, j] - G[i - 6, j], G[i, j + 4] - G[i, j - 4], core)
    r1, r2 = 2.3, 2.3 + 2 * 1.4 + 0.25; d_key = 1.0 + 4.3 + K.BEZEL
    _fr = chart(0.0, d_key); _c, _ex, _ey, _n = _fr; _D = single('SS40', 'keystone'); _tr = {'d_key_mm': d_key, 'chart_point': _c.tolist(), 'n': _n.tolist()}
    _s0 = _D.S[0]; _q = fpoint(_fr, _s0['x'], _s0['y']); _tr['design_point'] = _q.tolist()
    _hit = cast_top(_q, _n, 0.03, [col.bvh]); _tr['cast_top_hit'] = None if _hit is None else dict(P=_hit[0].tolist(), nn=_hit[1].tolist())
    if _hit is not None:
        _tr['dot_n_nn'] = float(_hit[1] @ _n); _tr['dist_design_to_hit_mm'] = float(np.linalg.norm(_hit[0] - _q) / MM)
        _st, _why = seater.seat(_hit[0], _hit[1], math.cos(0) * _ex, K.rad('SS40') * MM); _tr['seat_reason'] = _why
        if _st is not None: _tr['seated_P'] = _st['P'].tolist()
        else:                                                                  # which seat() test fired, with the numbers
            _R = K.rad('SS40') * MM; _N0 = C.unit(_hit[1]); _c2 = seater.cast(_hit[0], _N0); _tr['seat_center_cast'] = None if _c2 is None else dict(lift_mm=float((_c2 - _hit[0]) @ _N0 / MM))
            if _c2 is not None:
                T_ = C.unit(_ex - (_ex @ _N0) * _N0); B_ = np.cross(_N0, T_); el = []
                for ang in 2 * math.pi * np.arange(8) / 8:
                    x = seater.cast(_c2 + _R * (math.cos(ang) * T_ + math.sin(ang) * B_), _N0); el.append(None if x is None else float((x - _c2) @ _N0 / MM))
                _tr['seat_ring_elevations_mm'] = el
                top = 0.0; Nf = _N0
                for r_ in (0.0, 0.4, 0.75, 1.0):
                    for k_ in range(1 if r_ == 0 else 12):
                        th = 2 * math.pi * k_ / 12; x = seater.cast(_c2 + r_ * _R * (math.cos(th) * T_ + math.sin(th) * B_), Nf)
                        if x is not None: top = max(top, float((x - _c2) @ Nf))
                _tr['seat_footprint_top_mm_untilted'] = top / MM; _tr['seat_max_lift_mm'] = 0.6
    n_key = place_design(z, _D, _fr, 0, [col.bvh]); _tr['place_design_result'] = n_key; _tr['rejects_after'] = json.loads(json.dumps(report['rejects']))
    json.dump(_tr, open(os.path.normpath(os.path.expanduser('~/Desktop/better_style_execution_workspace/solvers/rhinestone_engine/garments/out/BS-BSTSET-001/royal_zoneA_diagnostic/keystone_trace.json')), 'w'), indent=1)
    pk = np.array([s['P'] for s in stones if s['tag'] == 'keystone'])
    skip = (lambda p: bool(len(pk)) and np.linalg.norm(pk - p, axis=1).min() < (4.3 + K.BEZEL + 1.4 + 0.3) * MM)
    rep = place_rows(z, G, s_edge, [(r1 * MM, 'SS10', 0.0), (r2 * MM, 'SS10', 0.5)], closed=False, prio=0, skip=skip, tag='base_rail')
    base = r2 + 1.4 + 2.2; crowns = []; sol = []
    for a_mm, w, h, b in ((0.0, 46.0, 32.0, d_key + 4.3 + K.BEZEL + 2.0), (-88.0, 38.0, 26.0, base), (88.0, 38.0, 26.0, base), (-236.0, 34.0, 23.0, base), (236.0, 34.0, 23.0, base)):
        Dc = A.crown_pave(w, h, tips=('SS10', 'SS8')); crowns.append(dict(arc_mm=a_mm, w=w, h=h, stones=place_design(z, Dc, chart(a_mm, b), 0, [col.bvh])))
    for a_mm in (-162.0, -44.0, 44.0, 162.0): sol.append(place_design(z, single('SS30', 'solitaire'), chart(a_mm, base + 12.0), 0, [col.bvh]))
    rep.update(leaf_width_mm=[round(float(width.min() / MM), 1), round(float(width.max() / MM), 1)], keystone=n_key, crowns=crowns, solitaires=sol)
    # lapels: rail along the outer and gorge edges, crowns climbing the lapel upright
    lap = {}
    for tag in ('L', 'R'):
        Gl = dense_grid(g['lapel_' + tag], 6, 6)[:, ::-1]                       # sections from the outer edge inward to the roll line
        arcl = np.concatenate([np.zeros((len(Gl), 1)), np.cumsum(np.linalg.norm(np.diff(Gl, axis=1), axis=2), axis=1)], 1); wl = arcl[:, -1]
        srcl = np.vstack([C.resample(Gl[:, 0], 0.1 * MM)[0], C.resample(Gl[-1], 0.1 * MM)[0]])
        se = cKDTree(srcl).query(Gl.reshape(-1, 3))[0].reshape(Gl.shape[:2]); se[arcl > (wl[:, None] - 9.0 * MM)] = np.nan; se[wl < 20 * MM] = np.nan
        lr = place_rows(z, Gl, se, [(r1 * MM, 'SS10', 0.0), (r2 * MM, 'SS10', 0.5)], closed=False, prio=0, tag='lapel_rail'); cnt = []
        for u, v, kind in ((0.80, 0.50, (26.0, 18.0)), (0.67, 0.50, 'SS20'), (0.55, 0.50, (17.0, 12.0)), (0.44, 0.50, 'SS16')):
            i = int(u * (len(Gl) - 1)); j = int(v * (Gl.shape[1] - 1)); p = Gl[i, j]
            fr = surf_frame(p, Gl[i, min(j + 4, Gl.shape[1] - 1)] - Gl[i, j - 4], np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.03, p[2]]))
            if isinstance(kind, tuple): Dc = A.crown_pave(kind[0], kind[1], tips=('SS8', 'SS6')); cnt.append(place_design(z, Dc, fr, 0, [col.bvh], dy=kind[1] / 2))
            else: cnt.append(place_design(z, single(kind, 'solitaire'), fr, 0, [col.bvh]))
        lap[tag] = dict(rail=[r['stones'] for r in lr['rows']], motifs=cnt)
    rep['lapels'] = lap; report['zones'][z] = rep
    log('zone A', sum(1 for s in stones if s['zone'] == z), 'stones; crowns', [c['stones'] for c in crowns], 'lapels', lap)

# ================================================================== Zone B: sleeve cuff bracelets
if 'B' in ONLY:
    ob = bpy.data.objects['Tee']; me = ob.data
    for side in ('L', 'R'):
        z = f'B_cuff_{side}'; gi = ob.vertex_groups[f'set_in_sleeve_{side}'].index; w = np.zeros(len(tee.V))
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
            tt = (tee.V[sm] - c0) @ a; t_hi = tt.max() + 2 * MM; t_lo = tt.max() - (90 if pas == 0 else 56) * MM
            ts = np.arange(t_hi, t_lo, -(1.0 if pas == 0 else 0.25) * MM); nth = 180 if pas == 0 else NTH; sec = np.full((nth, len(ts), 3), np.nan)
            for i in range(nth):
                th = 2 * math.pi * i / nth; dr = math.cos(th) * e1 + math.sin(th) * e2
                for j, t in enumerate(ts):
                    loc, nn, idx, dd = sbvh.ray_cast(Vector(c0 + t * a + 0.30 * dr), Vector(-dr), 0.30)
                    if loc is not None: sec[i, j] = loc
            if pas == 0:
                ring = np.array([sec[i, np.argmax(np.isfinite(sec[i, :, 0]))] for i in range(nth) if np.isfinite(sec[i, :, 0]).any()])
                rc = ring.mean(0); nrm = np.linalg.svd(ring - rc)[2][2]
                if nrm @ a < 0: nrm = -nrm
                a = nrm; c0 = rc - 0.03 * a
        first = np.array([np.argmax(np.isfinite(sec[i, :, 0])) if np.isfinite(sec[i, :, 0]).any() else -1 for i in range(NTH)])
        good = first >= 0; tf_ = np.where(good, ts[np.clip(first, 0, None)], np.nan)
        tf_ = np.interp(np.arange(NTH), np.nonzero(good)[0], tf_[good], period=NTH); tf_s = gaussian_filter1d(tf_, 6, mode='wrap')
        s_edge = np.full(sec.shape[:2], np.nan); ring = np.full((NTH, 3), np.nan)
        for i in range(NTH):
            okj = np.isfinite(sec[i, :, 0]) & (ts <= tf_s[i] + 0.25 * MM)
            if okj.sum() < 4: continue
            idx = np.nonzero(okj)[0]; P = sec[i, idx]; stp = np.linalg.norm(np.diff(P, axis=0), axis=1)
            brk = np.nonzero(stp > 1.5 * MM)[0]
            if len(brk): idx = idx[:brk[0] + 1]; P = P[:brk[0] + 1]
            s_edge[i, idx] = 0.0; ring[i] = P[0]
        okr = np.isfinite(ring[:, 0]); ring = np.stack([np.interp(np.arange(NTH), np.nonzero(okr)[0], ring[okr, k], period=NTH) for k in range(3)], 1)
        ring = gaussian_filter1d(ring, 5, axis=0, mode='wrap'); src = C.resample(ring, 0.1 * MM, closed=True)[0]
        m_ = np.isfinite(s_edge); s_edge[m_] = cKDTree(src).query(sec[m_])[0]
        CORE = (lambda c0_, a_: (lambda p: c0_ + ((p - c0_) @ a_) * a_))(c0, a)
        # chart: point at (section i, level d) for a regular level table
        lv = np.arange(0.0, 40.01, 0.25) * MM; PT = np.full((NTH, len(lv), 3), np.nan)
        for i in range(NTH):
            ok = np.isfinite(s_edge[i])
            if ok.sum() < 4: continue
            s = s_edge[i][ok]; P = sec[i][ok]; o = np.argsort(s); s = s[o]; P = P[o]; inr = lv <= s[-1]
            for k in range(3): PT[i, inr, k] = np.interp(lv[inr], s, P[:, k])
        kmid = int(round(20.0 / 0.25)); mid = PT[:, kmid]; okm = np.isfinite(mid[:, 0])
        midf = np.stack([np.interp(np.arange(NTH), np.nonzero(okm)[0], mid[okm, k], period=NTH) for k in range(3)], 1)
        seg = np.linalg.norm(np.roll(midf, -1, 0) - midf, axis=1); cum = np.concatenate([[0], np.cumsum(seg)]); Lc = cum[-1] / MM
        D, info = A.cuff_band(Lc); placed = 0
        for s in D.S:
            fi = np.interp(s['x'] * MM, cum, np.arange(NTH + 1)); i0 = int(fi) % NTH; i1 = (i0 + 1) % NTH; wi = fi - int(fi); fk = s['y'] / 0.25; k0 = int(fk); wk = fk - k0
            if k0 + 1 >= len(lv): rej(z, 'above_band'); continue
            q = np.array([PT[i0, k0], PT[i1, k0], PT[i0, k0 + 1], PT[i1, k0 + 1]])
            if not np.isfinite(q).all(): rej(z, 'fold_gap'); continue
            p = (1 - wk) * ((1 - wi) * q[0] + wi * q[1]) + wk * ((1 - wi) * q[2] + wi * q[3]); tg = (q[1] - q[0]); ac = (q[2] - q[0])
            n0 = C.unit(np.cross(tg, ac))
            if n0 @ (p - CORE(p)) < 0: n0 = -n0
            st, why = seater.seat(p, n0, tg, s['R'] * MM)
            if st is None: rej(z, why); continue
            add(st, z, s, 0); placed += 1
        info.update(circumference_mm=round(Lc, 1), planned=len(D.S), placed=placed, sections_valid_at_mid=int(okm.sum())); report['zones'][z] = info
        log('zone B', side, 'L=%.1f mm' % Lc, placed, '/', len(D.S), report['rejects'].get(z))

# ================================================================== body: planar frames
CORE = lambda p: np.array([0.0, -0.03, p[2]]); CANVAS = (638.0, 700.0)
for key, layer in (('front', 'UVMap_Art_Front'), ('back', 'UVMap_Art_Back')):
    uvt = tee.loop_uv(layer)[tee.TL]; inside = np.all((uvt >= 0) & (uvt <= 1), axis=(1, 2)); tee.build_uv(key, uvt, tee.outer & inside)


def canvas_point(key, x_mm, y_mm):
    h = tee.uv_to_surface(key, (x_mm / CANVAS[0], 1.0 - y_mm / CANVAS[1])); return None if h is None else h[0]


def fit_frame(centre, ex0, ey0, half_w, half_h):
    n0 = C.unit(np.cross(ex0, ey0)); pts = []
    for ax in np.linspace(-1, 1, 11):
        for ay in np.linspace(-1, 1, 11):
            q = centre + ax * half_w * ex0 + ay * half_h * ey0
            loc, nn, idx, dd = tee.bvh.ray_cast(Vector(q + 0.08 * n0), Vector(-n0), 0.16)
            if loc is not None: pts.append(np.array(loc))
    pts = np.array(pts); c = pts.mean(0); n = np.linalg.svd(pts - c)[2][2]
    if n @ n0 < 0: n = -n
    ex = C.unit(ex0 - (ex0 @ n) * n); ey = np.cross(n, ex)
    return centre, ex, ey, n


def rotated(fr, deg):
    c, ex, ey, n = fr; t = math.radians(deg); return c, math.cos(t) * ex + math.sin(t) * ey, -math.sin(t) * ex + math.cos(t) * ey, n


if 'C' in ONLY:
    z = 'C_back'; D, info = A.script_design(width=300.0); A.patch_bed(D, name='script_patch')
    x0, y0, x1, y1 = D.bbox(); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    c = canvas_point('back', 319.0, 318.0); a_ = canvas_point('back', 379.0, 318.0); b_ = canvas_point('back', 259.0, 318.0); u_ = canvas_point('back', 319.0, 258.0); d_ = canvas_point('back', 319.0, 378.0)
    ex0 = C.unit(a_ - b_); ey0 = C.unit(u_ - d_); ey0 = C.unit(ey0 - (ey0 @ ex0) * ex0)
    fr = rotated(fit_frame(c, ex0, ey0, 0.16, 0.07), 6.0)
    bv = add_bed(z, D.beds[0], (fr[0] - cx * MM * fr[1] + cy * MM * fr[2],) + fr[1:], [tee.bvh])
    n1 = place_design(z, D, fr, 1, [bv, tee.bvh], dx=-cx, dy=-cy)
    Dc = A.crown_pave(62.0, 40.0, cross=True, tips=('SS12', 'SS10')); A.patch_bed(Dc, name='crown_patch')
    ys = [s['y'] for s in D.S if abs(s['x'] - (x0 + 0.52 * (x1 - x0))) < 34]; top = min(ys) if ys else y0
    ox, oy = (x0 + 0.52 * (x1 - x0)) - cx, (top - 5.5) - cy                                       # crown base 5.5 mm above the lettering below it
    bv2 = add_bed(z, Dc.beds[0], (fr[0] + ox * MM * fr[1] - oy * MM * fr[2],) + fr[1:], [tee.bvh])
    n2 = place_design(z, Dc, fr, 1, [bv2, tee.bvh], dx=ox, dy=oy)
    for s in stones[-n2:]: s['tag'] = 'back_crown_' + s['tag']
    info.update(planned=len(D.S), placed=n1, crown_planned=len(Dc.S), crown_placed=n2, font='Harlow Solid Italic (stand-in)', slope_deg=6.0, plane_normal=fr[3].round(3).tolist())
    report['zones'][z].update(info); log('zone C script', n1, '/', len(D.S), 'crown', n2, '/', len(Dc.S))

if 'F' in ONLY:
    z = 'F_signet'; D = A.signet()
    loc, nn_, idx_, dd_ = tee.bvh.ray_cast(Vector((0.105, -1.0, 1.300)), Vector((0, 1, 0)), 2.0); c = np.array(loc)
    fr = fit_frame(c, np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), 0.06, 0.025)
    b0 = add_bed(z, D.beds[0], fr, [tee.bvh]); add_bed(z, D.beds[1], fr, [b0, tee.bvh]); add_bed(z, D.beds[2], fr, [tee.bvh])
    report['zones'][z].update(centre=c.round(4).tolist(), size_mm=[74.0, 19.0], note='satin badge, embroidered lettering, pocket welt; no stones'); log('zone F signet beds', len(report['zones'][z]['beds']))

if 'E' in ONLY:
    z = 'E_elephant'; D, info = EL.elephant()
    X_TRUNK, Y_END = 1660 * EL.S_PX, 3545 * EL.S_PX; xr = -0.024                                    # trunk's outer edge 8 mm from the placket; ray ends at the hem
    hz = [zz for zz in np.arange(0.86, 1.0, 0.001) if (lambda h: h[0] is not None and h[0][1] < -0.02)(tee.bvh.ray_cast(Vector((-0.10, -1.0, zz)), Vector((0, 1, 0)), 2.0))]
    z_hem = min(hz); dcx, dcy = 34.0, 125.0
    wx = xr + (dcx - X_TRUNK) * MM; wz = z_hem + 0.004 + (Y_END - dcy) * MM
    loc, nn_, idx_, dd_ = tee.bvh.ray_cast(Vector((wx, -1.0, wz)), Vector((0, 1, 0)), 2.0); c = np.array(loc)
    fr = fit_frame(c, np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), 0.085, 0.125)
    bv = add_bed(z, D.beds[0], (fr[0] - dcx * MM * fr[1] + dcy * MM * fr[2],) + fr[1:], [tee.bvh], depth=0.05)
    k = place_design(z, D, fr, 1, [bv, tee.bvh], depth=0.05, dx=-dcx, dy=-dcy, steep=0.45)
    by = {}
    for s in stones[-k:]: by[s['tag']] = by.get(s['tag'], 0) + 1
    info.update(placed=k, placed_by_structure=dict(sorted(by.items())), hem_z=round(z_hem, 4), centre=c.round(4).tolist(), plane_normal=fr[3].round(3).tolist())
    report['zones'][z].update(info); log('zone E', k, '/', len(D.S), report['rejects'].get(z))

if 'D' in ONLY:
    z = 'D_pinpoint'; tile = A.jacquard_tile(PREFIX + '_jacquard.png'); tw, th = tile['tile_mm']
    uvp, mir = C.print_uv_tri(tee); tee.build_uv('print', uvp, tee.outer); tw_uv, th_uv = (tw * MM) / 0.30, (th * MM) / 0.30
    kd = cKDTree(np.array([s['P'] for s in stones])) if stones else None
    um, vm = uvp[tee.outer][..., 0], uvp[tee.outer][..., 1]
    occ = set(zip(np.floor(um.ravel() / tw_uv).astype(int).tolist(), np.floor(vm.ravel() / th_uv).astype(int).tolist())); k = 0; cand = 0
    for (i, j) in sorted(occ):
        for (fx, fy) in tile['apex']:
            h = tee.uv_to_surface('print', ((i + fx) * tw_uv, (j + 1 - fy) * th_uv))
            if h is None: continue
            cand += 1; P0, nn = h[0], h[1]
            if kd is not None and kd.query(P0)[0] < 9.0 * MM: rej(z, 'inside_another_zone'); continue
            if abs(P0[0]) < 0.024 and P0[1] < -0.05: rej(z, 'on_placket'); continue
            if any(b.ray_cast(Vector(P0 + 0.004 * nn), Vector(-nn), 0.008)[0] is not None for b in bed_bvhs): rej(z, 'under_patch'); continue
            if col.bvh.ray_cast(Vector(P0 + 0.0005 * nn), Vector(nn), 0.09)[0] is not None: rej(z, 'under_collar'); continue
            up = h[3] - (h[3] @ nn) * nn
            st, why = seater.seat(P0, nn, up if np.linalg.norm(up) > 1e-9 else np.array([0, 0, 1.0]), K.rad('SS8') * MM)
            if st is None: rej(z, why); continue
            add(st, z, rnd('SS8', 'crown_apex_pinpoint'), 3); k += 1
    report['zones'][z] = dict(shield_apexes_on_outer_surface=cand, placed=k, size='SS8', tile_mm=[tw, th], shield_width_mm=60.0)
    report['print_tile'] = dict(path=os.path.basename(PREFIX + '_jacquard.png'), tile_mm=[tw, th], scale_uv=[1.0 / tw_uv, 1.0 / th_uv], uv='UVMap_Print', px=list(tile['px']))
    log('zone D pinpoints', k, '/', cand, report['rejects'].get(z))

# ================================================================== exact clearance certificate (round model for rounds; covering cylinders for bezels and fancy cuts)
sys.path.insert(0, r'C:/Users/USER/Desktop/better_style_execution_workspace/solvers/rhinestone_engine/garments/../../prototypes/P1_bstset_exact_hull')
import convex_gap as CG


def verify(stones, g_min=C.G_MIN):
    Pv, Nv, Rv, Hv, own, plain = [], [], [], [], [], []
    for i, s in enumerate(stones):
        B = np.cross(s['T'], s['N']); simple = s['shape'] == 'round' and not s['bezel']
        for a, b, r in K.cover_discs(s['shape'], s['L'] / MM, s['W'] / MM, K.BEZEL if s['bezel'] else 0.0):
            Pv.append(s['P'] + a * MM * s['T'] + b * MM * B); Nv.append(s['N']); Rv.append(r * MM); Hv.append(0.30 * s['W']); own.append(i); plain.append(simple)
    Pv, Nv, Rv, Hv, own, plain = np.array(Pv), np.array(Nv), np.array(Rv), np.array(Hv), np.array(own), np.array(plain)
    prio = np.array([s['prio'] for s in stones]); size = np.array([s['R'] for s in stones]); keep = np.ones(len(stones), bool)
    pairs = cKDTree(Pv).query_pairs(2 * Rv.max() + 0.4 * MM, output_type='ndarray'); pairs = pairs[own[pairs[:, 0]] != own[pairs[:, 1]]]
    i, j = pairs[:, 0], pairs[:, 1]; dist = np.linalg.norm(Pv[i] - Pv[j], axis=1); d = dist - Rv[i] - Rv[j]; cosn = np.einsum('ij,ij->i', Nv[i], Nv[j])
    bound = np.hypot(Rv, Hv)                                                    # every solid lies inside the sphere of this radius about its base centre
    sure = ((d > 0.12 * MM) & (cosn > 0.94)) | (dist - bound[i] - bound[j] > g_min)
    cand = pairs[~sure]; exact = np.full(len(cand), np.inf); sup = {}

    def S(k):
        if k not in sup:
            sup[k] = CG.round_support(C.CHATON, Pv[k], Nv[k], 2 * Rv[k]) if plain[k] else CG.CirclesSupport(np.array([Pv[k], Pv[k] + Hv[k] * Nv[k]]), Nv[k], np.array([Rv[k], Rv[k]]))
        return sup[k]
    for k, (a, b) in enumerate(cand):
        up, lo, it, hit = CG.gjk_distance(S(int(a)), S(int(b))); exact[k] = 0.0 if hit else lo
    bad = exact < g_min; removed = 0
    for k in np.argsort(exact):
        if not bad[k]: break
        a, b = own[cand[k, 0]], own[cand[k, 1]]
        if keep[a] and keep[b]:
            lose = a if (prio[a], -size[a]) > (prio[b], -size[b]) else b; keep[lose] = False; removed += 1
    okk = exact[~bad]; alive = keep[own[pairs[:, 0]]] & keep[own[pairs[:, 1]]]
    rep = dict(stones=len(stones), clearance_solids=int(len(Pv)), neighbour_pairs=int(len(pairs)), pairs_decided_by_bound=int(sure.sum()), pairs_exact=int(len(cand)),
               pairs_below_min=int(bad.sum()), pairs_overlapping=int((exact <= 0).sum()), removed=int(removed),
               min_exact_gap_kept_mm=float(okk.min() / MM) if len(okk) else None,
               min_disc_gap_among_bound_decided_mm=float(d[sure & alive].min() / MM) if (sure & alive).any() else None,
               model='rounds: CirclesSupport chaton; bezel-set and fancy cuts: covering cylinders (outline + %.2f mm bezel wall, full stone height); g_min %.3f mm' % (K.BEZEL, g_min / MM))
    return keep, rep


np.savez(os.path.normpath(os.path.expanduser('~/Desktop/better_style_execution_workspace/solvers/rhinestone_engine/garments/out/BS-BSTSET-001/royal_zoneA_diagnostic/pre_clearance.npz')), P=np.array([x['P'] for x in stones]), N=np.array([x['N'] for x in stones]), T=np.array([x['T'] for x in stones]), R=np.array([x['R'] for x in stones]), tag=np.array([x['tag'] for x in stones]), shape=np.array([x['shape'] for x in stones]), L=np.array([x['L'] for x in stones]), W=np.array([x['W'] for x in stones]), bezel=np.array([x['bezel'] for x in stones]))
keep, vrep = verify(stones); report['exact_clearance'] = vrep
np.save(os.path.normpath(os.path.expanduser('~/Desktop/better_style_execution_workspace/solvers/rhinestone_engine/garments/out/BS-BSTSET-001/royal_zoneA_diagnostic/keep_mask.npy')), keep); log('exact', vrep)
zones = np.array([s['zone'] for s in stones]); report['removed_by_zone'] = {zz: int((~keep[zones == zz]).sum()) for zz in sorted(set(zones))}
report['removed_stones'] = [dict(zone=s['zone'], tag=s['tag'], shape=s['shape'], d_mm=round(2 * s['R'] / MM, 2)) for s, kk in zip(stones, keep) if not kk][:200]
stones = [s for s, kk in zip(stones, keep) if kk]
keep2, vrep2 = verify(stones); report['exact_clearance_final'] = vrep2; log('exact final', vrep2)
zones = np.array([s['zone'] for s in stones]); report['counts'] = {zz: int((zones == zz).sum()) for zz in sorted(set(zones))}; report['total'] = len(stones)
bysz = {}
for s in stones:
    k = s['shape'] if s['shape'] != 'round' else next((n for n, v in K.SS.items() if abs(v * MM - s['R']) < 1e-9), 'other'); bysz[k] = bysz.get(k, 0) + 1
report['by_cut'] = dict(sorted(bysz.items())); report['bezel_set'] = int(sum(s['bezel'] for s in stones))
tags = {}
for s in stones: tags.setdefault(s['zone'], {}); tags[s['zone']][s['tag']] = tags[s['zone']].get(s['tag'], 0) + 1
report['structures'] = {zz: dict(sorted(v.items())) for zz, v in sorted(tags.items())}

P = np.array([s['P'] for s in stones]); N = np.array([s['N'] for s in stones]); T = np.array([s['T'] for s in stones]); R = np.array([s['R'] for s in stones])
np.savez(PREFIX + '_stones.npz', P=P, N=N, T=T, R=R, zone=zones, cls=np.array([s['cls'] for s in stones]), shape=np.array([s['shape'] for s in stones]),
         L=np.array([s['L'] for s in stones]), W=np.array([s['W'] for s in stones]), bezel=np.array([s['bezel'] for s in stones]), tag=np.array([s['tag'] for s in stones]))
gl = lambda Aa: np.stack([Aa[:, 0], Aa[:, 2], -Aa[:, 1]], 1)
np.concatenate([gl(P), gl(N), np.sqrt(np.array([s['L'] * s['W'] for s in stones]))[:, None]], 1).astype('<f4').tofile(PREFIX + '_stones.f32')
bd = dict(names=np.array([b['name'] for b in beds]))
for k, b in enumerate(beds): bd.update({f'V{k}': b['V'], f'F{k}': b['F'], f'M{k}': b['M'], f'mats{k}': np.array(b['mats'])})
np.savez_compressed(PREFIX + '_beds.npz', **bd)
json.dump(report, open(PREFIX + '_stones.report.json', 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
log('TOTAL', len(stones), report['counts'], report['by_cut']); log('rejects', report['rejects'])
