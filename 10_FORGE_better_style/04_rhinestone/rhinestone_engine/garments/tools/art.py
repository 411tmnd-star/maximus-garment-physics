"""art.py - design masks and planar stone packing (millimetre plane) for the artwork zones.
contour_fill: rows are iso-contours of the distance transform of ONE mask (one field for every row), so rows are parallel by
construction; strokes narrower than a stone get a single row on their medial line."""
import math
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from stones_core import Grid2, resample, even_positions

FONTS = 'C:/Windows/Fonts/'


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


def contour_fill(mask, mmpp, R, gap=0.2, edge=0.10, ensure_min=True, max_rows=None, grid=None, fill_pass=True, cls_img=None):
    """mask (bool, y down), mmpp mm per pixel, R stone radius mm. -> list of (x_mm, y_mm, row_index, cls). `grid` = shared Grid2 so
    several calls (different stone sizes / components) respect each other."""
    mask = mask.astype(np.uint8); grid = grid or Grid2(4.0)
    if ensure_min:
        sk = zhang_suen(mask)
        d_sk = cv2.distanceTransform((1 - sk).astype(np.uint8), cv2.DIST_L2, 5)
        mask = (mask.astype(bool) | (d_sk <= (R + edge + 0.06) / mmpp)).astype(np.uint8)
    mask = np.pad(mask, 2)                                                  # the distance transform needs a zero border
    edt = cv2.distanceTransform(mask, cv2.DIST_L2, 5) * mmpp
    out = []; pitch = 2 * R + gap; j = 0; o = 2 * mmpp

    def cls_at(x, y):
        if cls_img is None: return 0
        return int(cls_img[min(max(int(round(y / mmpp)), 0), cls_img.shape[0] - 1), min(max(int(round(x / mmpp)), 0), cls_img.shape[1] - 1)])

    def put(p, row):
        if grid.ok(p, R, gap): grid.add(p, R); out.append((p[0] - o, p[1] - o, row, cls_at(p[0] - o, p[1] - o))); return True
        return False
    while True:
        lvl = R + edge + j * pitch
        if lvl > edt.max() or (max_rows is not None and j >= max_rows): break
        cs, _ = cv2.findContours((edt >= lvl).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for c in sorted(cs, key=lambda c: -len(c)):
            Pl = c[:, 0, :].astype(float) * mmpp
            if len(Pl) < 8: put(Pl.mean(0), j); continue
            k = max(3, int(0.4 / mmpp)) | 1                               # light smoothing of the pixel staircase (same for every row)
            Pp = np.vstack([Pl[-k:], Pl, Pl[:k]]); ker = np.ones(k) / k
            Pl = np.stack([np.convolve(Pp[:, a], ker, 'same') for a in (0, 1)], 1)[k:-k]
            Q, L = resample(Pl, 0.05, closed=True)
            if L < pitch: put(Pl.mean(0), j); continue
            for s in even_positions(L, pitch, True):
                if not put(Q[min(int(round(s / 0.05)), len(Q) - 1)], j):  # slide forward to the first free spot before the next slot
                    for ds in np.arange(0.15, pitch * 0.95, 0.15):
                        if put(Q[int(round((s + ds) / 0.05)) % len(Q)], j): break
        j += 1
    if fill_pass:                                                         # leftovers on ridges / centres
        ys, xs = np.nonzero(edt >= R + edge); step = max(1, int(0.35 / mmpp))
        sel = (ys % step == 0) & (xs % step == 0)
        for y, x in zip(ys[sel], xs[sel]): put((x * mmpp, y * mmpp), 99)
    return out


# ------------------------------------------------------------------ monogram tile
def tile_shield(tile_png, tile_w_mm, scale=1.0):
    """One crowned BST shield of the half-drop repeat -> (stones [(x_mm, y_mm, R_mm, cls)] about the shield centre, y down;
    shield centre in tile fractions (fx, fy from the top); tile size mm). `scale` shrinks the drawing before packing (stones keep
    their size). Outer frame: one row SS16 on its medial line. Inner outline, crown, letters: SS6."""
    im = np.asarray(Image.open(tile_png).convert('L')) > 127; H, W = im.shape; mmpp = tile_w_mm / W
    big = np.tile(im, (2, 2)).astype(np.uint8)
    n, lab, stats, cen = cv2.connectedComponentsWithStats(big, connectivity=8)
    x0, y0, x1, y1 = int(0.03 * W), int(0.90 * H), int(0.47 * W), int(1.60 * H)
    ids = [i for i in range(1, n) if stats[i, 0] >= x0 and stats[i, 1] >= y0 and stats[i, 0] + stats[i, 2] <= x1 and stats[i, 1] + stats[i, 3] <= y1 and stats[i, 4] > 30]
    frame = max(ids, key=lambda i: stats[i, 2] * stats[i, 3]); pad = 40
    bx0 = stats[frame, 0] - pad; by0 = stats[frame, 1] - pad; bx1 = stats[frame, 0] + stats[frame, 2] + pad; by1 = stats[frame, 1] + stats[frame, 3] + pad
    sub = lab[by0:by1, bx0:bx1]; fm = (sub == frame).astype(np.uint8); rest = np.isin(sub, [i for i in ids if i != frame]).astype(np.uint8)
    if scale != 1.0:
        sz = (int(sub.shape[1] * scale), int(sub.shape[0] * scale))
        fm = (cv2.resize(fm * 255, sz, interpolation=cv2.INTER_AREA) > 127).astype(np.uint8); rest = (cv2.resize(rest * 255, sz, interpolation=cv2.INTER_AREA) > 127).astype(np.uint8)
    grid = Grid2(4.0); stones = []
    sk = zhang_suen(fm); d_sk = cv2.distanceTransform((1 - sk).astype(np.uint8), cv2.DIST_L2, 5)
    st = contour_fill(d_sk <= (2.0 + 0.16) / mmpp, mmpp, 2.0, gap=0.3, ensure_min=False, max_rows=1, grid=grid, fill_pass=False)
    stones += [(x, y, 2.0, 0) for x, y, _, _ in st]
    st = contour_fill(rest, mmpp, 1.0, gap=0.25, ensure_min=True, grid=grid)
    stones += [(x, y, 1.0, 0) for x, y, _, _ in st]
    cx, cy = fm.shape[1] * mmpp / 2, fm.shape[0] * mmpp / 2
    fx = ((bx0 + bx1) / 2 % W) / W; fy = ((by0 + by1) / 2 % H) / H
    return [(x - cx, y - cy, r, c) for x, y, r, c in stones], (fx, fy), (W * mmpp, H * mmpp)


# ------------------------------------------------------------------ back script + crown
def script_mask(text='BetterStyle', width_mm=315.0, mmpp=0.1, slope_deg=5.0):
    """Baseball-style script with an underline swoosh. Font is a local stand-in (Brush Script MT): the brand lettering has not
    been supplied as vectors."""
    f = ImageFont.truetype(FONTS + 'BRUSHSCI.TTF', 600)
    l, t, r, b = f.getbbox(text); im = Image.new('L', (r - l + 400, b - t + 700), 0); d = ImageDraw.Draw(im)
    d.text((200 - l, 150 - t), text, font=f, fill=255, stroke_width=5)
    W0 = r - l; base = 150 + (f.getbbox('x')[3] - t)                      # baseline of the lower-case letters
    pts_top, pts_bot = [], []
    for k in range(61):                                                   # swoosh: tapered arc from under the B to past the final e
        u = k / 60; x = 200 + W0 * (0.04 + 1.02 * u); y = base + 105 + 30 * math.sin(math.pi * u) - 40 * u
        th = 62 * math.sin(math.pi * min(u * 1.15, 1.0)) ** 0.6 * (1 - 0.55 * u) + 6
        pts_top.append((x, y - th / 2)); pts_bot.append((x, y + th / 2))
    d.polygon(pts_top + pts_bot[::-1], fill=255)
    im = im.rotate(slope_deg, resample=Image.BICUBIC, expand=True)
    a = np.asarray(im) > 127; ys, xs = np.nonzero(a); a = a[ys.min() - 5:ys.max() + 6, xs.min() - 5:xs.max() + 6]
    sc = (width_mm / mmpp) / a.shape[1]
    a = cv2.resize(a.astype(np.uint8) * 255, (int(a.shape[1] * sc), int(a.shape[0] * sc)), interpolation=cv2.INTER_AREA) > 127
    return a, mmpp


def crown_mask(width_mm=78.0, mmpp=0.1):
    """Five-point crown with a base band and orb tips (procedural stand-in for the brand crown)."""
    W = int(width_mm / mmpp); H = int(W * 0.62); im = Image.new('L', (W, H), 0); d = ImageDraw.Draw(im)
    bx0, bx1 = 0.10 * W, 0.90 * W; by0, by1 = 0.74 * H, 0.97 * H
    d.rounded_rectangle((bx0, by0, bx1, by1), radius=0.05 * H, fill=255)
    tips = [(0.06, 0.30), (0.28, 0.20), (0.50, 0.10), (0.72, 0.20), (0.94, 0.30)]; valleys = [(0.19, 0.60), (0.39, 0.56), (0.61, 0.56), (0.81, 0.60)]
    poly = [(bx0, by0 + 2)]
    for i, tp in enumerate(tips):
        poly.append((tp[0] * W, tp[1] * H))
        if i < 4: poly.append((valleys[i][0] * W, valleys[i][1] * H))
    poly.append((bx1, by0 + 2)); d.polygon(poly, fill=255)
    for tp in tips:
        r = 0.05 * W; d.ellipse((tp[0] * W - r, tp[1] * H - r - 0.02 * H, tp[0] * W + r, tp[1] * H + r - 0.02 * H), fill=255)
    return np.asarray(im) > 127, mmpp


def elephant_masks(png, height_mm=252.0, mmpp=0.1):
    """-> (fill mask, class image: 0 body / 1 regalia, mmpp). A thin gap is left along the body/regalia boundary so the packed rows
    follow the drawing's internal shapes."""
    im = Image.open(png).convert('RGBA'); sc = (height_mm / mmpp) / im.size[1]
    im = im.resize((int(im.size[0] * sc), int(im.size[1] * sc)), Image.LANCZOS); a = np.asarray(im).astype(np.float32)
    alpha = a[..., 3] > 128
    hsv = cv2.cvtColor(a[..., :3].astype(np.uint8), cv2.COLOR_RGB2HSV)
    reg = (hsv[..., 1] > 95) & (hsv[..., 2] > 95) & (hsv[..., 0] < 32)       # saturated orange / gold = crown, cloak, chains
    k = int(0.9 / mmpp) | 1
    reg = cv2.morphologyEx(reg.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 2 + 1, k * 2 + 1)))
    reg = cv2.morphologyEx(reg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool) & alpha
    alpha = cv2.morphologyEx(alpha.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
    edge = cv2.morphologyEx(reg.astype(np.uint8), cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(1.3 / mmpp) | 1,) * 2)).astype(bool)
    # drawn lines inside the body (ear, trunk, legs): strong luminance edges of the smoothed picture become stone-free channels
    g = cv2.GaussianBlur(cv2.cvtColor(a[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY), (0, 0), 1.2 / mmpp)
    lines = cv2.Canny(g, 14, 34) > 0
    lines = cv2.dilate(lines.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(1.1 / mmpp) | 1,) * 2)).astype(bool) & ~reg
    return alpha & ~edge, reg.astype(np.uint8), mmpp
