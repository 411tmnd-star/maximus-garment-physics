"""royal_elephant.py - Zone 2: the Royal Elephant & Crown insignia as a structural design (mm, y down).
Landmarks are read from the Owner's macro reference (1696 x 2528 px); one reference pixel = S_PX mm on the garment, which makes
the whole motif (crown apex to hem) about 245 mm tall on the engine's camp shirt. The reference shows the motif down to the cape;
the rays and cape tiers are continued on their own curves to the hem and side seam.
Every stone belongs to a named structure: circlet rail, peak, rim, eye ring, cranium ring, fan rib, trunk row, tusk track, ray track,
cape tier, or is a named solitaire. Nothing is scattered."""
import math
import numpy as np, cv2
from royal_core import *

S_PX = 0.0707


def mm(pts): return np.asarray(pts, float) * S_PX


def _tube(cl, widths):
    """polygon of a tube of varying full width along a centre line (control points in mm)"""
    P = spline(cl, 0.2); f = np.linspace(0, 1, len(widths)); t = arclen(P) / arclen(P)[-1]; w = np.interp(t, f, widths) / 2
    return np.vstack([offset(P, w), offset(P, -w)[::-1]])


def _x_at(P, y): return float(np.interp(y, P[:, 1], P[:, 0]))


def elephant():
    D = Design('elephant', 0.22); C = Canvas(-52, 0, 126, 254, 0.1); mp = C.mmpp; parts = []; info = {}

    def part(mask, hmax, w, base=0.45): parts.append((mask, dome(mask, mp, hmax, w, base)))

    def ring(curve, size, clip=None, tag='', closed=True):
        Q = resample(np.vstack([curve, curve[:1]]) if closed else curve, 0.05); L = arclen(Q)[-1]; R = rad(size); n = 0
        for s in stations(L, 2 * R + D.gap + (2 * BEZEL if R >= 2.3 else 0), closed):
            q = Q[min(int(round(s / L * (len(Q) - 1))), len(Q) - 1)]
            if clip is not None:
                iy, ix = int((q[1] - C.org[1]) / mp), int((q[0] - C.org[0]) / mp)
                if not (0 <= iy < clip.shape[0] and 0 <= ix < clip.shape[1]) or clip[iy, ix] < R + 0.25: continue
            n += D.put(q[0], q[1], size, tag=tag)
        return n

    # ================================================================ sovereign crown
    lower = spline(mm([(592, 640), (893, 620), (1194, 644)])); upper = spline(mm([(556, 480), (893, 456), (1230, 484)]))
    for x, y in ((650, 560), (768, 548), (893, 540), (1018, 548), (1136, 562)): D.put(*mm((x, y)), 'SS30', tag='crown_solitaire')
    rail(D, lower, 'SS6', tag='crown_rail'); rail(D, upper, 'SS6', tag='crown_rail')
    band = C.poly(np.vstack([offset(upper, 1.5), offset(lower, -1.5)[::-1]])); part(band, 1.3, 3.0)
    D.put(*mm((880, 168)), None, 'pear', L=9.2, W=5.3, rot=math.pi / 2, tag='crown_pear')                # central cross peak, crowned with a pear
    D.put(*mm((880, 60)), None, 'oval', L=4.8, W=3.3, rot=math.pi / 2, tag='crown_finial')
    D.put(*mm((752, 262)), 'SS12', tag='crown_cross', bezel=True); D.put(*mm((1008, 258)), 'SS12', tag='crown_cross', bezel=True)
    cross = C.poly(mm([(843, 452), (943, 452), (932, 330), (852, 330)]), mm([(735, 300), (1030, 296), (1030, 222), (735, 226)]), mm([(805, 332), (955, 332), (962, 100), (798, 100)]))
    D.put(*mm((650, 312)), 'SS24', tag='crown_tip'); D.put(*mm((1118, 312)), 'SS24', tag='crown_tip')    # round-tipped inner peaks
    inner = C.poly(mm([(655, 474), (735, 470), (692, 346), (626, 352)]), mm([(1048, 470), (1128, 474), (1144, 352), (1078, 346)]))
    D.put(*mm((480, 262)), None, 'oval', L=5.6, W=3.5, rot=math.atan2(-1, -0.28), tag='crown_tip')       # oval-tipped outer spires
    D.put(*mm((1302, 268)), None, 'oval', L=5.6, W=3.5, rot=math.atan2(-1, 0.28), tag='crown_tip')
    outer = C.poly(mm([(528, 476), (648, 472), (600, 425), (512, 302), (460, 306)]), mm([(1140, 472), (1262, 478), (1328, 306), (1278, 302), (1190, 425)]))
    peaks = cross | inner | outer
    fill(D, peaks, C.org, mp, ['SS6'], edge=0.12, leftover=('SS6',), tag='crown_peak')
    part(cv2.dilate((peaks | C.stones([s for s in D.S if s['tag'].startswith('crown_')], 0.7)).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool), 1.1, 2.5)

    # ================================================================ features that everything else parts around
    D.put(*mm((1268, 726)), None, 'marquise', L=10.5, W=5.6, rot=math.atan2(110, 120), tag='temple_marquise')
    D.put(*mm((1358, 872)), None, 'pear', L=9.6, W=6.8, rot=math.atan2(145, 100), tag='temple_pear')
    root = mm((640, 1000))
    for (x, y), shp, L_, W_ in (((245, 1015), 'baguette', 10.4, 5.6), ((415, 1005), 'baguette', 8.4, 3.0), ((290, 1150), 'octagon', 7.0, 7.0), ((455, 1245), 'trapezoid', 8.6, 6.2),
                                ((300, 850), 'trapezoid', 8.0, 5.6)):
        c = mm((x, y)); D.put(c[0], c[1], None, shp, L=L_, W=W_, rot=math.atan2(c[1] - root[1], c[0] - root[0]) + math.pi, tag='ear_' + shp)
    D.put(*mm((835, 1535)), None, 'pear', L=15.0, W=9.0, rot=math.pi / 2, tag='chin_pear')                # inverted pear solitaire
    D.put(*mm((722, 1465)), None, 'marquise', L=9.0, W=4.0, rot=1.40, tag='chin_navette')
    # orbital eye: stone-free almond void, then concentric rings
    ec = mm((925, 1035)); ea, eb, et = 55 * S_PX, 33 * S_PX, math.radians(-8)

    def ell(da):
        t = 2 * math.pi * np.arange(240) / 240; x = (ea + da) * np.cos(t); y = (eb + da) * np.sin(t)
        return np.stack([ec[0] + x * math.cos(et) - y * math.sin(et), ec[1] + x * math.sin(et) + y * math.cos(et)], 1)
    for q in ell(-eb * 0.5)[::8]: D.block(q[0], q[1], eb * 0.5 - 0.08)
    D.block(ec[0], ec[1], eb - 0.08)
    eye = C.poly(ell(0.0))

    # ================================================================ regions
    head_pts = mm([(600, 668), (750, 652), (900, 648), (1050, 655), (1200, 672), (1310, 690), (1400, 790), (1442, 930), (1392, 992), (1250, 1012), (1120, 1060), (1040, 1150),
                   (1000, 1260), (975, 1380), (900, 1420), (760, 1400), (650, 1330), (615, 1200), (625, 1050), (650, 900), (630, 780)])
    head = C.poly(spline(head_pts, 0.3, closed=True))
    ear_pts = mm([(596, 672), (520, 640), (400, 628), (280, 660), (170, 750), (105, 880), (95, 1000), (120, 1110), (190, 1215), (300, 1275), (420, 1300), (520, 1350), (585, 1400),
                  (638, 1338), (612, 1200), (622, 1050), (646, 900), (626, 780)])
    k3 = np.ones((7, 7), np.uint8)
    ear = C.poly(spline(ear_pts, 0.3, closed=True)) & ~cv2.dilate(head.astype(np.uint8), k3).astype(bool)
    trunk = C.poly(_tube(mm([(1232, 930), (1250, 1300), (1275, 1600), (1300, 1800), (1345, 1960), (1440, 2040), (1540, 2010), (1575, 1920), (1555, 1840)]),
                         np.array([300, 280, 215, 195, 170, 150, 135, 125, 112]) * S_PX)) & ~cv2.dilate(head.astype(np.uint8), k3).astype(bool)
    tusk_cl = spline(mm([(1400, 1005), (1500, 1090), (1585, 1230), (1625, 1420), (1620, 1600), (1588, 1715)]))
    tusk_l = C.poly(mm([(1028, 1425), (1114, 1438), (1058, 1606)]))
    chin = C.poly(spline(mm([(672, 1398), (840, 1405), (1008, 1402), (1002, 1560), (905, 1682), (790, 1692), (688, 1600)]), 0.3, closed=True)) & ~cv2.dilate((head | trunk).astype(np.uint8), k3).astype(bool)

    # ---- eye rings (micro then small), head rim, cranium rings graduated outward, pave remainder
    Eh = edt(head, mp)
    ring(ell(1.05), 'SS3', Eh, 'eye_ring'); ring(ell(3.0), 'SS6', Eh, 'eye_ring')
    fill(D, head, C.org, mp, ['SS6'], edge=0.15, leftover=(), max_rows=1, tag='head_rim')
    off = 3.0 + 1.0
    for size in ('SS10', 'SS12', 'SS16', 'SS20', 'SS20', 'SS16'):
        R = rad(size); bz = BEZEL if R >= 2.3 else 0.0; off += 0.3 + R + bz; ring(ell(off), size, Eh, 'cranium_ring'); off += R + bz
    fill(D, head, C.org, mp, ['SS6'], edge=0.15, leftover=('SS6',), max_rows=0, tag='head_pave')
    part(head, 2.4, 9.0, 0.5)

    # ---- ear: single-stone rim, radiating ribs cradling the fancy stones
    fill(D, ear, C.org, mp, ['SS8'], edge=0.15, leftover=(), max_rows=1, tag='ear_rim')
    Ee = edt(ear, mp)
    for a in np.linspace(104, 256, 10):
        d = np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])
        for r in np.arange(6.5, 60, 2.22):
            q = root + r * d; iy, ix = int((q[1] - C.org[1]) / mp), int((q[0] - C.org[0]) / mp)
            if 0 <= iy < ear.shape[0] and 0 <= ix < ear.shape[1] and Ee[iy, ix] >= 3.6: D.put(q[0], q[1], 'SS6', tag='ear_rib')
    part(ear, 1.1, 5.0, 0.4)

    # ---- trunk: concentric rows following the S-sweep; tusks
    fill(D, trunk, C.org, mp, ['SS8', 'SS10', 'SS10', 'SS12'], edge=0.15, leftover=('SS6',), tag='trunk_row')
    part(trunk, 2.2, 5.0, 0.5)
    rail(D, tusk_cl, 'SS6', rows=2, tag='tusk_right', rows_at=lambda f: 2 if f < 0.82 else 1)
    part(C.stroke(tusk_cl, 5.4), 0.9, 2.0)
    fill(D, tusk_l, C.org, mp, ['SS6'], edge=0.1, leftover=('SS6',), tag='tusk_left'); part(cv2.dilate(tusk_l.astype(np.uint8), k3).astype(bool), 0.8, 1.5)

    # ---- chin medallion halo
    pr = next(s for s in D.S if s['tag'] == 'chin_pear'); O = outline('pear', pr['L'], pr['W'], 96); O = np.stack([pr['x'] - O[:, 1], pr['y'] + O[:, 0]], 1)     # rot = +90 deg
    dh = BEZEL + D.gap + 1.0 + 0.1; H = offset(np.vstack([O, O[:1]]), dh)[:-1]
    if np.linalg.norm(H - [pr['x'], pr['y']], axis=1).mean() < np.linalg.norm(O - [pr['x'], pr['y']], axis=1).mean(): H = offset(np.vstack([O, O[:1]]), -dh)[:-1]
    ring(H, 'SS6', edt(chin, mp), 'chin_halo')
    part(chin, 0.9, 3.0)

    # ================================================================ regalia rays (single- and double-row tracks to the hem and side seam)
    RAYS = dict(
        L1=([(500, 1385), (300, 1410), (0, 1470), (-380, 1580), (-700, 1700)], 'SS6', 2),
        L2=([(560, 1458), (300, 1565), (0, 1705), (-350, 1905), (-700, 2150)], 'SS8', 2),
        L3=([(588, 1568), (380, 1755), (150, 1970), (-150, 2310), (-450, 2710), (-700, 3080)], 'SS10', 1),
        L4=([(612, 1672), (455, 1900), (300, 2150), (110, 2550), (-80, 3000), (-270, 3545)], 'SS6', 2),
        L4a=([(630, 1700), (548, 1950), (470, 2200), (360, 2600), (250, 3050), (150, 3545)], 'SS8', 1),
        L5=([(660, 1712), (652, 1950), (644, 2200), (630, 2600), (614, 3050), (600, 3545)], 'SS6', 2),         # cape borders
        R1=([(972, 1708), (978, 1950), (984, 2200), (998, 2600), (1014, 3050), (1030, 3545)], 'SS6', 2),
        R1a=([(998, 1692), (1052, 1950), (1108, 2200), (1188, 2600), (1258, 3050), (1312, 3545)], 'SS8', 1),
        R3=([(1014, 1660), (1112, 1980), (1238, 2320), (1388, 2740), (1492, 3150), (1556, 3545)], 'SS6', 2),
        R2=([(1024, 1612), (1107, 1850), (1217, 2085), (1372, 2290), (1512, 2520), (1612, 2850), (1667, 3200), (1692, 3545)], 'SS8', 1))
    curve = {k: spline(mm(v[0]), 0.1) for k, v in RAYS.items()}; halfw = {k: (v[2] * (2 * rad(v[1]) + D.gap)) / 2 for k, v in RAYS.items()}

    def between(a, b, plan, tag):
        """solitaires centred between two rays; the size is whatever the local free width allows (so the stones ascend as the rays part)"""
        A, B = curve[a], curve[b]
        for t, kind in plan:
            pa = A[int(t * (len(A) - 1))]; pb = B[int(t * (len(B) - 1))]; c = (pa + pb) / 2; free = np.linalg.norm(pa - pb) - halfw[a] - halfw[b] - 2 * (BEZEL + 0.45)
            tg = A[min(int(t * (len(A) - 1)) + 20, len(A) - 1)] - A[max(int(t * (len(A) - 1)) - 20, 0)]; ang = math.atan2(tg[1], tg[0])
            if kind == 'pear':
                W_ = min(free, 10.5)
                if W_ >= 3.0: D.put(c[0], c[1], None, 'pear', L=1.42 * W_, W=W_, rot=ang + math.pi, tag=tag)       # point toward the chin, weight toward the hem
            else:
                ok = [n for n, v in SS.items() if 2 * v <= min(free, 8.7) and v >= 1.4]
                if ok: D.put(c[0], c[1], ok[-1], tag=tag, bezel=True)
    between('L4a', 'L5', ((0.24, 'round'), (0.36, 'pear'), (0.49, 'pear'), (0.61, 'round'), (0.73, 'pear'), (0.87, 'pear')), 'cape_solitaire')
    between('R1', 'R1a', ((0.30, 'round'), (0.44, 'pear'), (0.57, 'round'), (0.70, 'pear'), (0.85, 'pear')), 'cape_solitaire')
    between('L4', 'L4a', ((0.45, 'round'), (0.64, 'round'), (0.82, 'round'), (0.95, 'round')), 'ray_solitaire')
    between('R1a', 'R3', ((0.55, 'round'), (0.74, 'round'), (0.92, 'round')), 'ray_solitaire')
    between('L1', 'L2', ((0.42, 'round'), (0.66, 'round'), (0.88, 'round')), 'ray_solitaire')
    between('L2', 'L3', ((0.52, 'round'), (0.80, 'round')), 'ray_solitaire')
    chan = np.zeros(C.shape, bool)
    for k, (pts, size, rows) in RAYS.items():
        rail(D, curve[k], size, rows=rows, tag='ray_' + k, fit=False); chan |= C.stroke(curve[k], 2 * halfw[k] + 1.0)
    chan &= ~cv2.dilate((head | ear | trunk | chin).astype(np.uint8), k3).astype(bool); part(chan, 0.55, 1.2, 0.35)
    solo = C.stones([s for s in D.S if s['tag'] in ('cape_solitaire', 'ray_solitaire')], 0.8); part(solo, 0.7, 1.5, 0.4)

    # ================================================================ cape: chevron tiers of graduated rounds between the inner rays
    Lc, Rc = offset(curve['L5'], -(halfw['L5'] + 0.9)), offset(curve['R1'], halfw['R1'] + 0.9)                     # inner edges (L5 runs downward: its right-hand side is the centre)
    ys = np.array([1728, 2085, 2445, 2805, 3165, 3545]) * S_PX; drop = np.array([45, 170, 170, 170, 170, 170]) * S_PX; tiers = 0
    for k in range(len(ys) - 1):
        ya, yb = ys[k], ys[k + 1] - 1.0; yy = np.linspace(ya, yb, 40)
        left = np.stack([[_x_at(Lc, y) for y in yy], yy], 1); right = np.stack([[_x_at(Rc, y) for y in yy], yy], 1); xm_a = (left[0, 0] + right[0, 0]) / 2; xm_b = (left[-1, 0] + right[-1, 0]) / 2
        poly = np.vstack([[(xm_a, ya + drop[k])], right, [(xm_b, yb + drop[k + 1])], left[::-1]])
        m = C.poly(poly) & ~cv2.dilate(chin.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
        if m.sum() < 400: continue
        m = smooth_mask(m, mp, 0.6); fill(D, m, C.org, mp, ['SS6', 'SS8', 'SS10', 'SS12', 'SS16'], edge=0.15, leftover=('SS6',), tag='cape_tier'); part(m, 1.0, 3.5, 0.4); tiers += 1

    # ================================================================ bed: antiqued silver relief on a dark backing; black enamel eye
    Hs = np.zeros(C.shape, np.float32); silver = np.zeros(C.shape, bool)
    for m, h in parts: Hs = np.maximum(Hs, h); silver |= m
    kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)); body = (head | ear | trunk | chin | band | peaks)
    back = cv2.dilate(cv2.morphologyEx(body.astype(np.uint8), cv2.MORPH_CLOSE, kk), np.ones((9, 9), np.uint8)).astype(bool) | cv2.dilate(silver.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    Hh = np.where(back, np.maximum(Hs, 0.25), 0.0); ids = np.where(silver, 1, 0).astype(np.uint8); ids[eye & head] = 2
    D.bed(back, C.org, mp, Hh, (ids, ['backing', 'silver', 'eye']), 'elephant_relief')
    tags = {}
    for s in D.S: tags[s['tag']] = tags.get(s['tag'], 0) + 1
    info.update(stones=len(D.S), by_structure=dict(sorted(tags.items())), counts=D.counts(), tiers=tiers, min_gap_mm=round(D.min_gap(), 4), rejected_candidates=D.rej,
                extent_mm=[round(v, 1) for v in D.bbox()], mm_per_ref_px=S_PX)
    return D, info
