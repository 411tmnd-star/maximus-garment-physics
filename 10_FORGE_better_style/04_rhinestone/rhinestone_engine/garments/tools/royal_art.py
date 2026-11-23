"""royal_art.py - the motifs of the BST Royal Camp Set as 2D designs (mm, y down). Recipe zones:
  cuff_band      Zone 1  triple rail / X-lattice / rosette / V-gap accent bracelet
  crown_pave     Zone 3+4  five-point pave crown (collar diadem, back patch crown)
  script_design  Zone 4  double-row calligraphy channel + tapered three-row underline, on a satin bed with a corded border
  signet         Zone 4  front chest badge (satin bed + embroidered lettering) and pocket welt
  jacquard_tile  Foundation  tonal woven crowned double-lined BST shield, half-drop repeat (+ crown-apex positions for the pinpoints)
  thigh_crest    Zone 5  crowned BST shield in embroidered linework
The elephant (Zone 2) is in royal_elephant.py."""
import math
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from royal_core import *

FONTS = 'C:/Windows/Fonts/'
SCRIPT_FONT = 'HARLOWSI.TTF'            # stand-in: the brand lettering has not been supplied as vectors
BLOCK_FONT = 'impact.ttf'


# ------------------------------------------------------------------ Zone 1: sleeve cuff bracelet
def cuff_band(L, lip=5.0, rail='SS6', hl_target=15.9, gap=0.25):
    """Chart design for a closed band of circumference L (x along the hem, y = distance from the hem edge).
    Bottom to top: bare hem lip, triple rail, X-lattice of open diamonds with a 5-stone rosette in each and a single accent stone
    in every upper and lower V-notch, triple rail. Returns (Design, info). Periodic in x by construction."""
    D = Design('cuff', gap); R = rad(rail); p = 2 * R + gap; info = {}
    n_cell = max(1, int(round(L / hl_target))); w = L / n_cell
    rows_lo = [lip + R + k * p for k in range(3)]; y_v0 = rows_lo[-1] + p; y_v1 = y_v0 + hl_target; rows_hi = [y_v1 + p + k * p for k in range(3)]; mid = 0.5 * (y_v0 + y_v1)
    for y in rows_lo + rows_hi:
        n = int(math.floor(L / p)); pp = L / n
        for i in range(n): D.put(i * pp, y, rail, tag='rail', force=True)
    seg = math.hypot(w / 2, hl_target / 2); m = max(1, int(math.floor(seg / (2 * R + 0.20))))
    al = math.atan2(w / 2, hl_target / 2); t = (hl_target / 2 + p - R + R) / (1 + math.sin(al))          # accent: equal clearance to both lattice legs and the rail
    for c in range(n_cell):
        xc = (c + 0.5) * w
        D.put(xc, y_v0, rail, tag='lattice', force=True); D.put(xc, y_v1, rail, tag='lattice', force=True); D.put(c * w, mid, rail, tag='lattice', force=True)
        for (xa, ya), (xb, yb) in (((xc - w / 2, mid), (xc, y_v1)), ((xc, y_v1), (xc + w / 2, mid)), ((xc + w / 2, mid), (xc, y_v0)), ((xc, y_v0), (xc - w / 2, mid))):
            for k in range(1, m): D.put(xa + (xb - xa) * k / m, ya + (yb - ya) * k / m, rail, tag='lattice', force=True)
        D.put(xc, mid, 'SS16', tag='rosette', bezel=False, force=True)
        d = rad('SS16') + rad('SS10') + gap
        for a in range(4): D.put(xc + d * math.cos(a * math.pi / 2), mid + d * math.sin(a * math.pi / 2), 'SS10', tag='rosette', force=True)
        for sgn in (-1, 1): D.put(c * w, mid + sgn * t, 'SS16', tag='accent', bezel=False, force=True)
    info.update(cells=n_cell, cell_w=round(w, 3), lattice_h=hl_target, seg_intervals=m, seg_pitch=round(seg / m, 3), band=[round(lip, 2), round(rows_hi[-1] + R, 2)],
                min_gap_mm=round(D.min_gap(period=L), 4), counts=D.counts())
    return D, info


# ------------------------------------------------------------------ five-point pave crown
def crown_pave(w, h, size='SS6', band_rows=2, cross=False, tips=('SS10', 'SS8'), gap=0.25):
    """Local frame: base centre at (0, 0), peaks toward -y. Base band of `band_rows` rows; five pave peaks built as stacked rows that
    narrow from the band to a single file (centre tallest, inner pair round-tipped, outer pair flaring outward); a larger stone
    crowns every peak; optional cross apex on the centre peak. Every stone position is constructed, none is scattered."""
    D = Design('crown', gap); R = rad(size); p = 2 * R + gap; r0, r1 = rad(tips[0]), rad(tips[1])
    n0 = max(2, int(round(0.92 * w / (5 * p)))); nb = 5 * n0; pw = n0 * p
    for k in range(band_rows):
        for q in range(nb): D.put((q - (nb - 1) / 2) * p, -(R + k * p), size, tag='band')
    y0 = -(band_rows * p + R); hp = h - band_rows * p
    for i, hf, lean, rt in ((-2, 0.70, -0.20, r1), (-1, 0.82, -0.04, r0), (0, 1.0, 0.0, r0), (1, 0.82, 0.04, r0), (2, 0.70, 0.20, r1)):
        extra = (2 * R + gap) if (cross and i == 0) else 0.0
        nr = max(2, int((hf * hp - 2 * rt - gap - extra) // p)); xc = i * pw
        for j in range(nr):
            n = max(1, int(round(n0 - (n0 - 1) * j / max(nr - 1, 1) * 1.0))) if j < nr - 1 else 1
            x = xc + lean * j * p
            for q in range(n): D.put(x + (q - (n - 1) / 2) * p, y0 - j * p, size, tag='peak')
        xt = xc + lean * nr * p; yt = y0 - (nr - 1) * p - R - gap - rt
        D.put(xt, yt, tips[0] if rt == r0 else tips[1], tag='tip', bezel=False)
        if cross and i == 0:
            D.put(xt, yt - rt - gap - R, size, tag='cross')
            for sx in (-1, 1): D.put(xt + sx * (rt + gap + R), yt, size, tag='cross')
    return D


def patch_bed(D, grow=2.4, h=0.55, cord=1.1, cord_h=0.45, mats=('satin', 'cord'), smooth=1.2, name='patch', mmpp=0.2):
    """Applique bed under every stone of D: footprint grown by `grow`, smoothed; a corded border `cord` mm wide stands proud."""
    x0, y0, x1, y1 = D.bbox(); C = Canvas(x0 - grow - 2, y0 - grow - 2, x1 + grow + 2, y1 + grow + 2, mmpp)
    m = C.stones(D.S, grow); k = int(2 * grow / mmpp) | 1
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool); m = smooth_mask(m, mmpp, smooth)
    e = edt(m, mmpp); ring = m & (e < cord); t = np.clip(e / cord, 0, 1)
    H = np.where(m, h + np.where(ring, cord_h * np.clip(np.sin(math.pi * t), 0, 1) ** 0.7, 0.0), 0.0)
    D.bed(m, C.org, mmpp, H, (ring.astype(np.uint8), list(mats)), name); return D


# ------------------------------------------------------------------ script
def text_mask(text, font, width_mm, mmpp=0.1, px=700, stroke=0):
    f = ImageFont.truetype(FONTS + font, px); l, t, r, b = f.getbbox(text, stroke_width=stroke)
    im = Image.new('L', (r - l + 80, b - t + 80), 0); ImageDraw.Draw(im).text((40 - l, 40 - t), text, font=f, fill=255, stroke_width=stroke, stroke_fill=255)
    a = np.asarray(im) > 127; ys, xs = np.nonzero(a); a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    sc = (width_mm / mmpp) / a.shape[1]
    a = cv2.resize(a.astype(np.uint8) * 255, (int(round(a.shape[1] * sc)), int(round(a.shape[0] * sc))), interpolation=cv2.INTER_AREA) > 127
    base = (40 - t + f.getbbox('x')[3] - ys.min()) * sc * mmpp                # baseline of the lower case, mm below the mask top
    return a, base


def script_design(text='BetterStyle', width=300.0, size='SS6', gap=0.25, thr=1.15, mmpp=0.1):
    """Calligraphy channel: the lettering is reduced to its centre line and re-inked at exactly two stones wide (one stone wide where
    the stroke is a hairline), so every stroke carries a strict double row. Underline: three rows narrowing to two at the left tail."""
    D = Design('script', gap); R = rad(size); p = 2 * R + gap
    m, base = text_mask(text, SCRIPT_FONT, width, mmpp); m = np.pad(m, 40); org = (-40 * mmpp, -40 * mmpp)
    sk = zhang_suen(m).astype(bool); e = edt(m, mmpp)
    d2 = cv2.distanceTransform((~(sk & (e >= thr))).astype(np.uint8), cv2.DIST_L2, 5) * mmpp; d1 = cv2.distanceTransform((~sk).astype(np.uint8), cv2.DIST_L2, 5) * mmpp
    ink = (d2 <= p - 0.02 + 0.13) | (d1 <= R + 0.17)                            # half-width: one pitch (two rows) or one radius (single row)
    k = int(0.5 / mmpp) | 1; ink = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
    fill(D, ink, org, mmpp, [size], edge=0.12, leftover=(), max_rows=1, tag='script')
    n_script = len(D.S)
    # underline: from under the y's descender back to beyond the B; rows 0..2 are parallel straight lines
    xs = [s['x'] for s in D.S]; ys_ = [s['y'] for s in D.S]; yl = max(ys_) + 2 * p; xr = max(xs) - 0.20 * width; xl = min(xs) - 0.02 * width
    rail(D, [(xr, yl), (xl, yl)], size, rows=3, tag='underline', fit=False, rows_at=lambda f: 3 if f < 0.70 else 2)
    return D, dict(script=n_script, underline=len(D.S) - n_script, baseline_mm=round(base, 2), size_mm=[round(max(xs) - min(xs), 1), round(max(ys_) - min(ys_), 1)])


def signet(width=74.0, height=19.0):
    """Front chest badge: satin bed with corded border, embroidered script lettering on top; pocket welt strip below. No stones."""
    D = Design('signet'); mmpp = 0.1; C = Canvas(-width / 2 - 2, -height / 2 - 2, width / 2 + 2, height / 2 + 2, mmpp)
    r = 2.2; m = C.poly([(-width / 2 + r, -height / 2), (width / 2 - r, -height / 2), (width / 2, -height / 2 + r), (width / 2, height / 2 - r), (width / 2 - r, height / 2), (-width / 2 + r, height / 2), (-width / 2, height / 2 - r), (-width / 2, -height / 2 + r)])
    m = smooth_mask(m, mmpp, 0.8); e = edt(m, mmpp); ring = m & (e < 1.0)
    D.bed(m, C.org, mmpp, np.where(m, 0.55 + np.where(ring, 0.4 * np.sin(math.pi * np.clip(e / 1.0, 0, 1)), 0), 0), (ring.astype(np.uint8), ['satin', 'cord']), 'signet_patch')
    t, base = text_mask('BetterStyle', SCRIPT_FONT, width * 0.80, mmpp); th, tw = t.shape
    D.bed(t, (-tw * mmpp / 2, -th * mmpp / 2), mmpp, np.where(t, 0.35, 0), 'thread', 'signet_text')
    wl, wh = 118.0, 6.5; Cw = Canvas(-wl / 2 - 1, 0, wl / 2 + 1, wh + 2, mmpp); wm = smooth_mask(Cw.poly([(-wl / 2, 1), (wl / 2, 1), (wl / 2, 1 + wh), (-wl / 2, 1 + wh)]), mmpp, 0.5)
    D.bed(wm, (Cw.org[0], height / 2 + 7.0), mmpp, dome(wm, mmpp, 0.9, 1.6, 0.3), 'welt', 'pocket_welt')
    return D


# ------------------------------------------------------------------ crowned BST shield line art (jacquard + thigh crest)
def shield_lines(sw=60.0, line=1.0, mmpp=0.05, crown=True):
    """line-art mask of one double-lined BST shield with a bar over the letters and a five-point crown on top.
    -> (mask, (cx, cy) of the shield centre in px, apex (x, y) px of the crown's centre peak, mmpp)"""
    sh = sw * 1.12; ch = sw * 0.27; cw = sw * 0.46; cg = sw * 0.05; pad = 4.0
    W = int((sw + 2 * pad) / mmpp); H = int((sh + ch + cg + 2 * pad) / mmpp); im = Image.new('L', (W, H), 0); d = ImageDraw.Draw(im)
    X = lambda x: (x + sw / 2 + pad) / mmpp; Y = lambda y: (y + ch + cg + pad) / mmpp       # shield top edge at y = 0, centre line x = 0
    lw = max(1, int(round(line / mmpp)))

    def plate(inset, wline):
        a = sw / 2 - inset; top = inset; knee = sh * 0.60 - inset * 0.3; tip = sh - inset * 1.5
        pts = [(-a, top), (a, top), (a, knee), (0, tip), (-a, knee), (-a, top)]
        d.line([(X(x), Y(y)) for x, y in pts], fill=255, width=max(1, int(round(wline / mmpp))), joint='curve')
    plate(0.0, line * 1.25); plate(sw * 0.075, line * 0.8)
    d.line([(X(-sw * 0.33), Y(sw * 0.20)), (X(sw * 0.33), Y(sw * 0.20))], fill=255, width=lw)                       # bar over the letters
    # letters: outlined condensed capitals
    f = ImageFont.truetype(FONTS + BLOCK_FONT, int(sw * 0.50 / mmpp)); l, t, r, b = f.getbbox('BST'); tw, thh = r - l, b - t
    sc = min((sw * 0.62 / mmpp) / tw, 1.0); tim = Image.new('L', (tw + 40, thh + 40), 0); ImageDraw.Draw(tim).text((20 - l, 20 - t), 'BST', font=f, fill=255)
    ta = np.asarray(tim) > 127; kk = max(3, int(line * 0.8 / mmpp)) | 1
    ta = (cv2.morphologyEx(ta.astype(np.uint8), cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))) > 0)
    ta = cv2.resize(ta.astype(np.uint8) * 255, (int(ta.shape[1] * sc), ta.shape[0]), interpolation=cv2.INTER_AREA) > 100
    a = np.asarray(im).copy(); y0 = int(Y(sw * 0.255)); x0 = int(X(0) - ta.shape[1] / 2); a[y0:y0 + ta.shape[0], x0:x0 + ta.shape[1]] |= (ta * 255).astype(np.uint8)
    im = Image.fromarray(a); d = ImageDraw.Draw(im)
    apex = (X(0), Y(-cg - ch))
    if crown:                                                                    # five-point crown: band + zig-zag + orb on each point
        yb = -cg; y1 = yb - ch * 0.30; tipsx = [-0.5, -0.25, 0, 0.25, 0.5]; tipsy = [0.80, 0.90, 1.0, 0.90, 0.80]
        d.rectangle((X(-cw * 0.44), Y(y1), X(cw * 0.44), Y(yb)), outline=255, width=lw)
        pts = [(-cw * 0.44, y1)]
        for i in range(5):
            pts.append((tipsx[i] * cw, yb - ch * tipsy[i]))
            if i < 4: pts.append(((tipsx[i] + tipsx[i + 1]) / 2 * cw, y1 - ch * 0.16))
        pts.append((cw * 0.44, y1)); d.line([(X(x), Y(y)) for x, y in pts], fill=255, width=lw, joint='curve')
    return np.asarray(im) > 127, (X(0), Y(sh / 2)), apex, mmpp


def jacquard_tile(path, tile_w=190.0, tile_h=126.0, px_w=2048, sw=60.0):
    """seamless half-drop tile (white = woven line). -> dict(apex=[(fx, fy from the top), ...], tile_mm=(w, h))"""
    m, (cx, cy), apex, mp = shield_lines(sw, line=1.4); W = int(tile_w / mp); H = int(tile_h / mp); big = np.zeros((3 * H, 3 * W), np.uint8); ap = []
    midy = m.shape[0] / 2
    for (fx, fy) in ((0.25, 0.25), (0.75, 0.75)):
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                x0 = int(W + (fx + i) * W - cx); y0 = int(H + (fy + j) * H - midy)
                xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + m.shape[1], 3 * W), min(y0 + m.shape[0], 3 * H)
                if xb > xa and yb > ya: big[ya:yb, xa:xb] |= m[ya - y0:yb - y0, xa - x0:xb - x0].astype(np.uint8)
        ap.append((fx + (apex[0] - cx) / W, (fy + (apex[1] - midy) / H) % 1.0))
    t = big[H:2 * H, W:2 * W] * 255; px_h = int(round(px_w * tile_h / tile_w))
    t = cv2.GaussianBlur(cv2.resize(t, (px_w, px_h), interpolation=cv2.INTER_AREA), (0, 0), 0.8)
    Image.fromarray(t).save(path); return dict(apex=ap, tile_mm=(tile_w, tile_h), px=(px_w, px_h))


def thigh_crest(sw=46.0):
    """Crowned BST shield in embroidered linework (bed, no stone outline) with crystal points on the five crown tips."""
    m, (cx, cy), apex, mp = shield_lines(sw, line=1.25, mmpp=0.1); D = Design('crest')
    k = 3; m = cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    org = (-cx * mp, -cy * mp); D.bed(m, org, mp, dome(m, mp, 0.5, 0.5, 0.25), 'thread', 'crest_lines')
    ch = sw * 0.27; cw = sw * 0.46; ax, ay = org[0] + apex[0] * mp, org[1] + apex[1] * mp
    for tx, ty in zip([-0.5, -0.25, 0, 0.25, 0.5], [0.80, 0.90, 1.0, 0.90, 0.80]):
        D.put(ax + tx * cw, ay + (1.0 - ty) * ch - 1.6, 'SS8' if tx else 'SS10', tag='crest_tip', bezel=False)
    return D
