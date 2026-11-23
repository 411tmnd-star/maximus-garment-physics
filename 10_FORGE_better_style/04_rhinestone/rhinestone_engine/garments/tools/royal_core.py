"""royal_core.py - 2D design layer for the BST Royal Camp Set (BS-BSTSET-001, recipe of 2026-11-12).

A motif is authored in a millimetre plane (x right, y DOWN) as a `Design`: an ordered list of stones plus optional `beds`
(raised textile / metal patches under the stones). Rule enforced here, before anything touches the garment:
every stone is put through one shared `Packer`, so no stone is ever accepted closer than `gap` to another one. Fancy cuts
(pear, marquise, oval, baguette, octagon, trapezoid) are represented for clearance by a chain of discs that COVERS their
outline (plus the bezel wall when bezel-set), i.e. the clearance model is never smaller than the stone.
No bpy here: everything in this file can be previewed and checked without Blender."""
import math
import numpy as np, cv2
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

SS = dict(SS3=0.70, SS6=1.00, SS8=1.20, SS10=1.40, SS12=1.55, SS16=2.00, SS20=2.35, SS24=2.65, SS30=3.20, SS34=3.60, SS40=4.30, SS48=5.50)   # girdle radius, mm
BEZEL = 0.28                 # bezel wall thickness, mm (large rounds and every fancy cut are bezel-set)
SHAPES = ('round', 'oval', 'marquise', 'pear', 'baguette', 'octagon', 'trapezoid')


def rad(size): return SS[size] if isinstance(size, str) else float(size)


# ------------------------------------------------------------------ stone outlines (local frame: a = long axis, b across)
def outline(shape, L, W, n=48):
    """closed polygon (n,2), counter-clockwise in (a, b)."""
    if shape in ('round', 'oval'):
        t = 2 * math.pi * np.arange(n) / n; return np.stack([L / 2 * np.cos(t), W / 2 * np.sin(t)], 1)
    if shape == 'marquise':
        r = ((L / 2) ** 2 + (W / 2) ** 2) / W; a = np.linspace(-L / 2, L / 2, n // 2 + 1); w = np.sqrt(np.maximum(r * r - a * a, 0)) - (r - W / 2)
    elif shape == 'pear':                                   # round end at -a, point at +a
        c = -L / 2 + W / 2; a = np.concatenate([-L / 2 + W / 2 * (1 - np.cos(np.linspace(0, math.pi / 2, n // 4 + 1))), np.linspace(c, L / 2, n // 4 + 1)[1:]])
        u = np.clip((a - c) / (L / 2 - c), 0, 1); w = np.where(a <= c, np.sqrt(np.maximum((W / 2) ** 2 - (a - c) ** 2, 0)), W / 2 * np.cos(u * math.pi / 2) ** 0.85)
    elif shape in ('baguette', 'octagon'):
        c = (0.20 if shape == 'baguette' else 0.29) * W
        return np.array([(L / 2 - c, -W / 2), (L / 2, -W / 2 + c), (L / 2, W / 2 - c), (L / 2 - c, W / 2), (-L / 2 + c, W / 2), (-L / 2, W / 2 - c), (-L / 2, -W / 2 + c), (-L / 2 + c, -W / 2)])
    elif shape == 'trapezoid':                              # wide end at -a
        k = 0.55; return np.array([(L / 2, -k * W / 2), (L / 2, k * W / 2), (-L / 2, W / 2), (-L / 2, -W / 2)])
    else: raise ValueError(shape)
    up = np.stack([a, w], 1); return np.vstack([up[::-1], np.stack([a, -w], 1)[1:-1]])[::-1]


def _dense(poly, step=0.04):
    out = []
    for i in range(len(poly)):
        p, q = poly[i], poly[(i + 1) % len(poly)]; m = max(1, int(np.linalg.norm(q - p) / step))
        out.append(p + (q - p) * (np.arange(m) / m)[:, None])
    return np.vstack(out)


_COVER = {}


def cover_discs(shape, L, W, margin=0.0, step=0.35):
    """[(a, b, r)] discs (local frame) whose union contains the outline grown by `margin`.
    Curved cuts: inscribed discs along the long axis. Flat-ended cuts: inscribed discs along the axis plus a ring of small discs
    just inside the outline, so the cover follows the corners instead of bulging past them."""
    key = (shape, round(L, 4), round(W, 4), round(margin, 4))
    if key in _COVER: return _COVER[key]
    if shape == 'round': out = [(0.0, 0.0, W / 2 + margin)]
    else:
        poly = outline(shape, L, W, 96); B = _dense(poly); flat = shape in ('baguette', 'octagon', 'trapezoid'); ends = []
        for sgn in (-1, 1):                                  # inset of the first centre = radius of the inscribed circle that sits in this end
            e = 0.0
            for t in np.arange(0.05, L / 2, 0.05):
                if np.hypot(B[:, 0] - sgn * (L / 2 - t), B[:, 1]).min() < t - 0.02: break
                e = t
            ends.append(max(0.15, e))
        lo, hi = -L / 2 + ends[0], L / 2 - ends[1]
        if hi <= lo: lo = hi = 0.0
        ca = np.linspace(lo, hi, max(2, int(math.ceil((hi - lo) / step)) + 1)) if hi > lo else np.array([0.0])
        if flat:
            out = [(float(a), 0.0, float(np.hypot(B[:, 0] - a, B[:, 1]).min() + margin)) for a in ca]; rho = 0.45
            cen = poly.mean(0); ins = []
            for i in range(len(poly)):                       # ring: centres on the outline moved inward by rho
                p, q = poly[i], poly[(i + 1) % len(poly)]; t = (q - p) / np.linalg.norm(q - p); n = np.array([-t[1], t[0]])
                if n @ (cen - p) < 0: n = -n
                m = max(1, int(math.ceil(np.linalg.norm(q - p) / 0.4)))
                for k in range(m + 1): ins.append(p + (q - p) * k / m + n * rho)
            out += [(float(x), float(y), rho * 1.42 + margin) for x, y in ins]                 # 1.42: a centre at a corner still reaches the corner
        else:
            k = np.argmin(np.abs(B[:, 0][:, None] - ca[None, :]), 1); d = np.hypot(B[:, 0] - ca[k], B[:, 1])
            out = [(float(ca[i]), 0.0, float(d[k == i].max() + margin)) for i in range(len(ca)) if (k == i).any()]
    _COVER[key] = out; return out


# ------------------------------------------------------------------ packer
class Packer:
    """minimum-gap acceptance between discs with different owners (mm plane)."""

    def __init__(self, cell=3.0): self.c = cell; self.g = {}; self.n = 0

    def _cells(self, x, y, r):
        c = self.c
        for i in range(int((x - r) // c), int((x + r) // c) + 1):
            for j in range(int((y - r) // c), int((y + r) // c) + 1): yield (i, j)

    def free(self, discs, gap):
        for x, y, r in discs:
            for key in self._cells(x, y, r + gap):
                for qx, qy, qr in self.g.get(key, ()):
                    if (x - qx) ** 2 + (y - qy) ** 2 < (r + qr + gap - 1e-6) ** 2: return False
        return True

    def add(self, discs):
        for x, y, r in discs:
            for key in self._cells(x, y, r): self.g.setdefault(key, []).append((x, y, r))
        self.n += 1


class Design:
    def __init__(self, name='', gap=0.25):
        self.name = name; self.gap = gap; self.S = []; self.pk = Packer(); self.beds = []; self.rej = 0

    def put(self, x, y, size, shape='round', L=None, W=None, rot=0.0, cls=0, bezel=None, tag='', gap=None, force=False):
        """place one stone; False (and nothing placed) if it would come closer than `gap` to any stone already there."""
        if shape == 'round': R = rad(size); L = W = 2 * R
        else: R = W / 2
        bz = (shape != 'round' or R >= 2.3) if bezel is None else bezel
        loc = cover_discs(shape, L, W, BEZEL if bz else 0.0); ca, sa = math.cos(rot), math.sin(rot)
        discs = [(x + a * ca - b * sa, y + a * sa + b * ca, r) for a, b, r in loc]
        if not force and not self.pk.free(discs, self.gap if gap is None else gap): self.rej += 1; return False
        self.pk.add(discs)
        self.S.append(dict(x=float(x), y=float(y), R=float(R), shape=shape, L=float(L), W=float(W), rot=float(rot), cls=int(cls), bezel=bool(bz), tag=tag)); return True

    def block(self, x, y, r): self.pk.add([(x, y, r)])        # keep-out disc (a void that stays stone-free)

    def bed(self, mask, org, mmpp, height, mat, name=''):
        """raised patch: mask (bool, y down) with pixel (0,0) at org (mm); height (mm, same shape); mat = material tag or uint8 map + list."""
        self.beds.append(dict(mask=mask.astype(bool), org=(float(org[0]), float(org[1])), mmpp=float(mmpp), h=height.astype(np.float32), mat=mat, name=name))

    def shift(self, dx, dy):
        for s in self.S: s['x'] += dx; s['y'] += dy
        for b in self.beds: b['org'] = (b['org'][0] + dx, b['org'][1] + dy)
        return self

    def bbox(self):
        xs = [s['x'] for s in self.S]; ys = [s['y'] for s in self.S]; x0, x1, y0, y1 = min(xs) - 6, max(xs) + 6, min(ys) - 6, max(ys) + 6
        for b in self.beds:
            x0 = min(x0, b['org'][0]); y0 = min(y0, b['org'][1]); x1 = max(x1, b['org'][0] + b['mask'].shape[1] * b['mmpp']); y1 = max(y1, b['org'][1] + b['mask'].shape[0] * b['mmpp'])
        return x0, y0, x1, y1

    def discs(self):
        """all clearance discs: (x, y, r, owner)"""
        out = []
        for i, s in enumerate(self.S):
            ca, sa = math.cos(s['rot']), math.sin(s['rot'])
            out += [(s['x'] + a * ca - b * sa, s['y'] + a * sa + b * ca, r, i) for a, b, r in cover_discs(s['shape'], s['L'], s['W'], BEZEL if s['bezel'] else 0.0)]
        return np.array(out)

    def min_gap(self, period=None):
        """smallest gap between clearance discs of different stones (mm); period = wrap length in x for closed bands."""
        A = self.discs()
        if not len(A): return None
        if period: A = np.vstack([A, A + [period, 0, 0, 0]])
        kd = cKDTree(A[:, :2]); pr = kd.query_pairs(2 * A[:, 2].max() + 1.0, output_type='ndarray')
        pr = pr[A[pr[:, 0], 3] != A[pr[:, 1], 3]]
        if not len(pr): return None
        return float((np.linalg.norm(A[pr[:, 0], :2] - A[pr[:, 1], :2], axis=1) - A[pr[:, 0], 2] - A[pr[:, 1], 2]).min())

    def counts(self):
        c = {}
        for s in self.S:
            k = s['shape'] if s['shape'] != 'round' else next((n for n, v in SS.items() if abs(v - s['R']) < 1e-6), 'R%.2f' % s['R']); c[k] = c.get(k, 0) + 1
        return dict(sorted(c.items()))

    # -------------------------------------------------------------- preview
    def preview(self, path, ppmm=8.0, bg=(22, 52, 128), under=None):
        """under = (image RGB array, mm-per-pixel, (x0_mm, y0_mm) of its pixel (0,0)) drawn dimmed behind the design."""
        x0, y0, x1, y1 = self.bbox(); W = int((x1 - x0) * ppmm) + 1; H = int((y1 - y0) * ppmm) + 1
        im = np.zeros((H, W, 3), np.uint8); im[:] = bg[::-1]
        if under is not None:
            src, mp, (ux, uy) = under; M = np.array([[mp * ppmm, 0, (ux - x0) * ppmm], [0, mp * ppmm, (uy - y0) * ppmm]], np.float64)
            u = cv2.warpAffine(src[:, :, ::-1], M, (W, H), flags=cv2.INTER_AREA); m = u.sum(2) > 0; im[m] = (0.55 * u[m]).astype(np.uint8)
        MAT = dict(backing=(34, 22, 14), eye=(0, 0, 0), silver=(96, 100, 108), satin=(70, 28, 12), cord=(120, 70, 40), thread=(235, 235, 240), welt=(150, 70, 30), gold=(60, 170, 210))
        for b in self.beds:
            m = b['mask']; sc = b['mmpp'] * ppmm; M = np.array([[sc, 0, (b['org'][0] - x0) * ppmm], [0, sc, (b['org'][1] - y0) * ppmm]], np.float64)
            mm = cv2.warpAffine(m.astype(np.uint8) * 255, M, (W, H), flags=cv2.INTER_LINEAR) > 127
            if isinstance(b['mat'], str): im[mm] = MAT.get(b['mat'], (90, 90, 90))
            else:
                ids, names = b['mat']; idw = cv2.warpAffine(ids, M, (W, H), flags=cv2.INTER_NEAREST)
                for k, nm in enumerate(names): im[mm & (idw == k)] = MAT.get(nm, (90, 90, 90))
        sh = 4; f = 1 << sh
        for s in self.S:
            ca, sa = math.cos(s['rot']), math.sin(s['rot']); P = outline(s['shape'], s['L'], s['W'], 28)
            for k, col in ((1.0 + (2 * BEZEL / s['W'] if s['bezel'] else 0), (60, 62, 70)), (1.0, (236, 240, 246) if s['cls'] == 0 else (120, 200, 255)), (0.55, (255, 255, 255))):
                if k > 1.0 or k <= 1.0:
                    Q = np.stack([s['x'] + k * (P[:, 0] * ca - P[:, 1] * sa), s['y'] + k * (P[:, 0] * sa + P[:, 1] * ca)], 1)
                    cv2.fillPoly(im, [np.round((Q - [x0, y0]) * ppmm * f).astype(np.int32)], col, lineType=cv2.LINE_AA, shift=sh)
            if s['R'] * ppmm > 5 or s['shape'] != 'round':
                Q = np.stack([s['x'] + (P[:, 0] * ca - P[:, 1] * sa), s['y'] + (P[:, 0] * sa + P[:, 1] * ca)], 1)
                cv2.polylines(im, [np.round((Q - [x0, y0]) * ppmm * f).astype(np.int32)], True, (150, 156, 170), 1, cv2.LINE_AA, shift=sh)
        cv2.imwrite(path, im); return im


# ------------------------------------------------------------------ curves
def spline(pts, step=0.1, closed=False):
    """dense polyline (spacing ~step mm) through control points."""
    P = np.asarray(pts, float)
    if closed: P = np.vstack([P, P[:1]])
    t = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1) ** 0.5)])           # centripetal
    cs = CubicSpline(t, P, axis=0, bc_type='periodic' if closed else 'not-a-knot') if len(P) > 2 else None
    if cs is None: Q = P
    else: Q = cs(np.linspace(0, t[-1], max(8, int(20 * len(P)))))
    return resample(Q, step)


def arclen(P): return np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])


def resample(P, step):
    P = np.asarray(P, float); d = arclen(P)
    if d[-1] < step: return P[[0, -1]]
    s = np.linspace(0, d[-1], int(round(d[-1] / step)) + 1); return np.stack([np.interp(s, d, P[:, k]) for k in range(P.shape[1])], 1)


def tangents(P):
    T = np.gradient(P, axis=0); return T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)


def offset(P, d):
    """parallel curve; d > 0 is to the LEFT of the direction of travel as seen on screen (y down)."""
    T = tangents(P); d = np.broadcast_to(np.asarray(d, float), (len(P),)); return P + d[:, None] * np.stack([T[:, 1], -T[:, 0]], 1)


def stations(L, pitch, closed=False, phase=0.0, fit=True):
    """arc positions. closed: integer count, pitch stretched so the loop closes. open + fit: end stones exactly on both ends
    (pitch stretched, never compressed); open + not fit: centred run, phase 0.5 = brick offset."""
    if closed:
        n = int(math.floor(L / pitch))
        if n < 1: return np.array([])
        p = L / n; return (phase * p + p * np.arange(n)) % L
    if fit:
        n = int(math.floor(L / pitch + 1e-9))
        if n < 1: return np.array([L / 2])
        return np.linspace(0, L, n + 1) if not phase else (np.arange(n) + 0.5) * (L / n)
    n = int(math.floor(L / pitch)); off = 0.5 * (L - n * pitch)
    return off + pitch * (np.arange(n) + 0.5) if phase else off + pitch * np.arange(n + 1)


def rail(D, P, size, rows=1, closed=False, gap=None, row_gap=None, brick=False, fit=True, tag='', cls=0, shape='round', LW=None, bezel=None, side=0.0, s0=0.0, s1=None, rows_at=None):
    """`rows` parallel single-file rows of one size centred on polyline P (side shifts the pack sideways, mm).
    rows_at(s_frac) -> number of rows wanted at that fraction of the length (taper), default constant."""
    R = rad(size) if shape == 'round' else LW[1] / 2; g = D.gap if gap is None else gap; rg = 2 * R + (g if row_gap is None else row_gap)
    pitch = (2 * R if shape == 'round' else LW[0]) + g + (2 * BEZEL if (bezel or (bezel is None and (shape != 'round' or R >= 2.3))) else 0); n0 = len(D.S)
    P = resample(P, 0.05); Ltot = arclen(P)[-1]
    for k in range(rows):
        Q = offset(P, side + (k - (rows - 1) / 2) * rg); Q = resample(Q, 0.05); d = arclen(Q); T = tangents(Q)
        a, b = s0 * d[-1], (d[-1] if s1 is None else s1 * d[-1])
        for s in a + stations(b - a, pitch, closed, 0.5 * (k % 2) if brick else 0.0, fit):
            i = min(int(round(s / d[-1] * (len(Q) - 1))), len(Q) - 1)
            if rows_at is not None and rows_at(s / d[-1]) <= k: continue
            D.put(float(np.interp(s, d, Q[:, 0])), float(np.interp(s, d, Q[:, 1])), size, shape, *(LW or (None, None)), rot=math.atan2(T[i, 1], T[i, 0]), cls=cls, bezel=bezel, tag=tag, gap=g)
    return len(D.S) - n0


# ------------------------------------------------------------------ raster canvas (masks)
class Canvas:
    def __init__(self, x0, y0, x1, y1, mmpp=0.1):
        self.org = (x0, y0); self.mmpp = mmpp; self.shape = (int(math.ceil((y1 - y0) / mmpp)) + 1, int(math.ceil((x1 - x0) / mmpp)) + 1)

    def px(self, P): return np.round((np.asarray(P, float) - self.org) / self.mmpp * 16).astype(np.int32)

    def poly(self, *polys):
        m = np.zeros(self.shape, np.uint8)
        for P in polys: cv2.fillPoly(m, [self.px(P)], 1, shift=4)
        return m.astype(bool)

    def stroke(self, P, width, closed=False):
        m = np.zeros(self.shape, np.uint8); cv2.polylines(m, [self.px(P)], closed, 1, max(1, int(round(width / self.mmpp))), cv2.LINE_8, shift=4); return m.astype(bool)

    def disc(self, x, y, r):
        m = np.zeros(self.shape, np.uint8); cv2.circle(m, tuple(int(v) for v in self.px((x, y))), int(round(r / self.mmpp * 16)), 1, -1, shift=4); return m.astype(bool)

    def stones(self, S, grow=0.0):
        """footprint of a stone list grown by `grow` mm"""
        m = np.zeros(self.shape, np.uint8)
        for s in S:
            ca, sa = math.cos(s['rot']), math.sin(s['rot'])
            for a, b, r in cover_discs(s['shape'], s['L'], s['W'], BEZEL if s['bezel'] else 0.0):
                cv2.circle(m, tuple(int(v) for v in self.px((s['x'] + a * ca - b * sa, s['y'] + a * sa + b * ca))), int(round((r + grow) / self.mmpp * 16)), 1, -1, shift=4)
        return m.astype(bool)


def edt(mask, mmpp): return cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1] * mmpp


def smooth_mask(mask, mmpp, r):
    k = int(r / mmpp) * 2 + 1; return cv2.GaussianBlur(mask.astype(np.float32), (k, k), r / mmpp / 2) > 0.5


def dome(mask, mmpp, hmax, w, base=0.0):
    """relief height: `base` at the edge rising smoothly to hmax over a distance w from the edge"""
    t = np.clip(edt(mask, mmpp) / w, 0, 1); return np.where(mask, base + (hmax - base) * t * t * (3 - 2 * t), 0.0).astype(np.float32)


def fill(D, mask, org, mmpp, sizes, gap=None, edge=0.15, leftover=('SS6',), tag='', cls=0, max_rows=None, start=0.0):
    """contour rows inside a mask: row k is the iso-line of ONE distance field at depth level_k and carries sizes[min(k, last)]
    (so the size can be graduated from the rim inward). `leftover` sizes then close the remaining gaps (pave)."""
    g = D.gap if gap is None else gap; m = np.pad(mask.astype(np.uint8), 2); E = cv2.distanceTransform(m, cv2.DIST_L2, 5) * mmpp; n0 = len(D.S)
    ox, oy = org[0] - 2 * mmpp, org[1] - 2 * mmpp; k = 0; lvl = start + rad(sizes[0]) + edge
    while lvl <= E.max() and (max_rows is None or k < max_rows):
        sz = sizes[min(k, len(sizes) - 1)]; R = rad(sz); pitch = 2 * R + g
        cs, _ = cv2.findContours((E >= lvl).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for c in sorted(cs, key=lambda c: -len(c)):
            Pl = c[:, 0, :].astype(float) * mmpp + [ox, oy]
            if len(Pl) < 8: D.put(*Pl.mean(0), sz, cls=cls, tag=tag, gap=g); continue
            w = max(3, int(0.4 / mmpp)) | 1; Pp = np.vstack([Pl[-w:], Pl, Pl[:w]]); ker = np.ones(w) / w
            Pl = np.stack([np.convolve(Pp[:, a], ker, 'same') for a in (0, 1)], 1)[w:-w]
            Q = resample(np.vstack([Pl, Pl[:1]]), 0.05); L = arclen(Q)[-1]
            if L < pitch: D.put(*Pl.mean(0), sz, cls=cls, tag=tag, gap=g); continue
            for s in stations(L, pitch, True):
                if not D.put(*Q[min(int(round(s / 0.05)), len(Q) - 1)], sz, cls=cls, tag=tag, gap=g):
                    for ds in np.arange(0.15, pitch * 0.95, 0.15):
                        if D.put(*Q[int(round((s + ds) / 0.05)) % len(Q)], sz, cls=cls, tag=tag, gap=g): break
        nxt = sizes[min(k + 1, len(sizes) - 1)]; lvl += R + g + rad(nxt); k += 1
    for sz in leftover:
        R = rad(sz); ys, xs = np.nonzero(E >= start + R + edge); step = max(1, int(0.3 / mmpp)); sel = (ys % step == 0) & (xs % step == 0)
        for y, x in zip(ys[sel], xs[sel]): D.put(x * mmpp + ox, y * mmpp + oy, sz, cls=cls, tag=tag, gap=g)
    return len(D.S) - n0


def zhang_suen(img):
    I = (img > 0).astype(np.uint8).copy(); I = np.pad(I, 1)
    while True:
        changed = False
        for it in (0, 1):
            nb = [np.roll(np.roll(I, dy, 0), dx, 1) for dy, dx in ((1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1))]
            P2, P3, P4, P5, P6, P7, P8, P9 = nb; B = sum(n.astype(np.int32) for n in nb)
            seq = nb + [P2]; A = sum(((seq[k] == 0) & (seq[k + 1] == 1)).astype(np.int32) for k in range(8))
            c = ((P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0)) if it == 0 else ((P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0))
            m = (I == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
            if m.any(): I[m] = 0; changed = True
        if not changed: break
    return I[1:-1, 1:-1]
