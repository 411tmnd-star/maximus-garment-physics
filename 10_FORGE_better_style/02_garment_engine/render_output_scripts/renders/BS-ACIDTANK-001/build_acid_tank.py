"""build_acid_tank.py - acid-wash sleeveless, catalogue flat-front visual in the BS-TANK-001 pattern
(1200 x 1600 transparent PNG, garment centred, plus the copy on the catalogue ground #F3EEE7).
usage: python build_acid_tank.py <reference.jpg> <out_dir>
Method: the Owner's reference is the design source. The garment silhouette, wash distribution and construction details are taken
from it; the two printed artworks (BST shield, flame outline) are separated from the fabric, re-drawn as clean ink layers at 4x
and laid back on a cleaned fabric layer, so print edges are sharp and the fabric carries no compression blocks.
Nothing is invented: no back view, no measurements."""
import sys, json, hashlib
from pathlib import Path
import numpy as np, cv2
from PIL import Image

REF, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
W, H = 1200, 1600; GROUND = '#F3EEE7'
bgr = cv2.imread(str(REF)); L = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

# ---- 1. silhouette
m = (L < 205).astype(np.uint8)
n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8); k = 1 + int(np.argmax(st[1:, 4])); m = (lab == k).astype(np.uint8)
hn, hl, hs, _ = cv2.connectedComponentsWithStats(1 - m, 4)                                           # fill pin-holes only: the neck opening stays open
bg_lab = hl[0, 0]                                                                                    # outside background
inner = [i for i in range(hn) if i != bg_lab and (1 - m)[hl == i].any()]
neck = max(inner, key=lambda i: hs[i, 4] if hs[i, 1] < 0.25 * L.shape[0] else -1)                  # the largest hole in the top quarter
for i in inner:
    if i != neck: m[hl == i] = 1
m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))                                # drop the light compression fringe at the cut edge
ys, xs = np.nonzero(m); x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
s = 853.0 / (x1 - x0 + 1)                                                                            # same garment width as the tank tile
ox = (W - (x1 - x0 + 1) * s) / 2 - x0 * s; oy = 200 - y0 * s
A = np.float32([[s, 0, ox], [0, s, oy]])
warp = lambda img, interp=cv2.INTER_LANCZOS4: cv2.warpAffine(img, A, (W, H), flags=interp, borderMode=cv2.BORDER_REPLICATE)
mask = cv2.warpAffine(m.astype(np.float32), A, (W, H), flags=cv2.INTER_CUBIC, borderValue=0)
mask = cv2.GaussianBlur(mask, (0, 0), 1.6); alpha = np.clip((mask - 0.5) / 0.12 + 0.5, 0, 1)        # smooth, anti-aliased edge

# ---- 2. ink layers (logo: black + white; flames: black line), detected against the local fabric tone
fab_local = cv2.medianBlur(L, 41).astype(np.int16); d = L.astype(np.int16) - fab_local
logo_box = (455, 300, 715, 520)                                   # reference pixels: x0, y0, x1, y1
flame_box = (190, 670, 1000, 1000)
ink_k = np.zeros(L.shape, np.float32); ink_w = np.zeros(L.shape, np.float32)
bx0, by0, bx1, by1 = logo_box
ink_k[by0:by1, bx0:bx1] = np.clip((-d[by0:by1, bx0:bx1] - 30) / 30.0, 0, 1)
ink_w[by0:by1, bx0:bx1] = np.clip((d[by0:by1, bx0:bx1] - 35) / 35.0, 0, 1)
fx0, fy0, fx1, fy1 = flame_box
ink_k[fy0:fy1, fx0:fx1] = np.maximum(ink_k[fy0:fy1, fx0:fx1], np.clip((-d[fy0:fy1, fx0:fx1] - 22) / 25.0, 0, 1))
m_in = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))); ink_k *= m_in; ink_w *= m_in     # edge shading is not ink


def crisp(layer, thr=0.5):
    """4x supersampled re-draw of a soft ink mask: smooth, threshold, area-average back -> sharp anti-aliased edges"""
    A4 = np.float32([[4 * s, 0, 4 * ox], [0, 4 * s, 4 * oy]])
    big = cv2.warpAffine(layer, A4, (4 * W, 4 * H), flags=cv2.INTER_CUBIC)
    big = cv2.GaussianBlur(big, (0, 0), 2.2); big = (big > thr).astype(np.float32)
    return cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)


K = crisp(ink_k, 0.45); Wh = crisp(ink_w, 0.5)

# ---- 3. fabric: reference with the inks removed, de-blocked, then fine wash grain added at output resolution
ink_any = ((ink_k > 0.15) | (ink_w > 0.15)).astype(np.uint8)
ink_any = cv2.dilate(ink_any, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
fab = cv2.inpaint(bgr, ink_any * 255, 6, cv2.INPAINT_TELEA)
fab = cv2.bilateralFilter(fab, 7, 18, 5)
fab = warp(fab).astype(np.float32) / 255.0
rng = np.random.default_rng(31)
def noise(sig, amp):
    n_ = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), sig); return amp * n_ / (n_.std() + 1e-9)
grain = noise(0.7, 0.022) + noise(2.2, 0.018) + noise(7.0, 0.014)                     # speckle, mottle, cloud: the wash at three scales
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
knit = 0.006 * np.sin(xx * 2 * np.pi / 3.1) * (0.6 + 0.4 * np.sin(yy * 2 * np.pi / 2.3))
fab = np.clip(fab * (1.0 + grain + knit)[..., None], 0, 1)

# ---- 4. composite: black ink absorbs (keeps a trace of the fabric relief), white ink covers with slight wash wear
relief = (fab.mean(2, keepdims=True) / max(fab.mean(), 1e-6))
black = np.float32([0.060, 0.060, 0.062])[None, None] * (0.75 + 0.25 * relief)
white = np.float32([0.925, 0.925, 0.915])[None, None] * (0.93 + 0.07 * relief)
out = fab * (1 - K[..., None]) + black * K[..., None]
out = out * (1 - Wh[..., None]) + white * Wh[..., None]
rgba = np.dstack([out[..., ::-1], alpha]); rgba[alpha <= 0] = 0
png = OUT / 'acidtank_grey_front_v1.png'; Image.fromarray(np.round(rgba * 255).astype(np.uint8), 'RGBA').save(png)
g = np.array([int(GROUND[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255.0
og = rgba[..., :3] * alpha[..., None] + g * (1 - alpha[..., None])
ogp = OUT / 'acidtank_grey_front_v1_on_ground.png'; Image.fromarray(np.round(og * 255).astype(np.uint8), 'RGB').save(ogp)
ys, xs = np.nonzero(alpha > 0.03)
for nm, (cx0, cy0, cx1, cy1) in (('crop_logo', logo_box), ('crop_flames', flame_box)):
    q = [int(cx0 * s + ox), int(cy0 * s + oy), int(cx1 * s + ox), int(cy1 * s + oy)]
    Image.fromarray(np.round(og * 255).astype(np.uint8), 'RGB').crop(q).save(OUT / f'acidtank_grey_front_v1_{nm}.png')
rec = dict(file=png.name, size=[W, H], alpha_bbox=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], scale_from_reference=round(float(s), 4),
           reference=dict(file=REF.name, sha256=hashlib.sha256(REF.read_bytes()).hexdigest(), size=[bgr.shape[1], bgr.shape[0]]),
           sha256=hashlib.sha256(png.read_bytes()).hexdigest(), on_ground=dict(file=ogp.name, hex=GROUND, sha256=hashlib.sha256(ogp.read_bytes()).hexdigest()),
           ink_coverage_px=dict(black=int((K > 0.5).sum()), white=int((Wh > 0.5).sum())),
           method='owner reference cut-out; inks re-drawn at 4x; fabric inpainted under the inks, de-blocked, wash grain added',
           not_claimed=['back view', 'measurements', 'fabric weight', '3D geometry'])
(OUT / 'acidtank_grey_front_v1.json').write_text(json.dumps(rec, indent=1)); print(json.dumps(rec, indent=1))
