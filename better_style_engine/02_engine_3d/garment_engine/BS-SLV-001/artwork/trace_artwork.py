"""trace_artwork.py  BS-SLV-001 mockup-grade artwork traced from the specification PDF's rendered views (paper 03 section 3: "The flame and shield
vectors ... can be traced from the PDF's rendered views, which gives artwork good enough for a mockup. Production artwork must come from the designer's files.")

usage:  python -I -B trace_artwork.py

Sources (embedded images extracted from ~/Desktop/Sleevless..pdf into ../source/images/):
  p05_0.jpeg  front technical flat     -> front flame field (deboss fill + contour lines)
  p06_0.jpeg  back technical flat      -> side wrap flames (left, right) and the centre-back ember
  p08_0.jpeg  shield macro render      -> BST shield (perspective-rectified to the PDF's 92 x 78 mm pentagon)
Scale: every traced shape gets a metric scale from a measured feature of the same view (garment hem width = POM-12, 58.0 cm fold to fold), then,
where the specification table gives the shape's own size, a correction to that size. Every factor is recorded in trace_report.json.
Outputs (metric): <name>.svg (mm units) and <name>_*.png masks at 5 px/mm. Nothing here is placed on the garment (paper 03 stage 4)."""
import json, math, hashlib
from pathlib import Path
import numpy as np
import cv2

HERE = Path(__file__).resolve().parent; SRC = HERE.parent / "source/images"; SPEC = json.loads((HERE.parent / "spec/bs_slv_001_spec.json").read_text())
PXMM = 5.0                                                   # output raster: 5 px per mm
REP = dict(method=__doc__.split("\n\n")[0], px_per_mm=PXMM, artworks={}, flags=[])
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def skeleton(mask):
    """Zhang-Suen thinning (numpy), mask uint8 0/1"""
    img = mask.copy().astype(np.uint8)
    while True:
        changed = False
        for step in (0, 1):
            P = np.pad(img, 1); p2, p3, p4, p5, p6, p7, p8, p9 = (P[:-2, 1:-1], P[:-2, 2:], P[1:-1, 2:], P[2:, 2:], P[2:, 1:-1], P[2:, :-2], P[1:-1, :-2], P[:-2, :-2])
            nb = [p2, p3, p4, p5, p6, p7, p8, p9]; B = sum(x.astype(int) for x in nb)
            A = sum(((nb[i] == 0) & (nb[(i + 1) % 8] == 1)).astype(int) for i in range(8))
            c = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0) if step == 0 else (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            rm = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
            if rm.any(): img[rm] = 0; changed = True
        if not changed: return img


def garment_mask(gray):
    lines = (gray < 150).astype(np.uint8); closed = cv2.dilate(lines, np.ones((3, 3), np.uint8))
    ff = (1 - closed).astype(np.uint8).copy(); h, w = ff.shape; m = np.zeros((h + 2, w + 2), np.uint8)
    for seed in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if ff[seed[1], seed[0]] == 1: cv2.floodFill(ff, m, seed, 2)
    return (ff != 2).astype(np.uint8), lines


def to_svg(path, layers, w_mm, h_mm):
    """layers: list of (name, mask at PXMM, fill colour). Contours of each mask written as mm paths (even-odd fill)."""
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w_mm:.1f}mm" height="{h_mm:.1f}mm" viewBox="0 0 {w_mm:.2f} {h_mm:.2f}">']
    for name, mask, col in layers:
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE); d = []
        for c in cs:
            c = cv2.approxPolyDP(c, 0.6, True)[:, 0, :] / PXMM
            if len(c) >= 3: d.append("M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in c) + " Z")
        out.append(f'<path id="{name}" fill="{col}" fill-rule="evenodd" d="{" ".join(d)}"/>')
    Path(path).write_text("\n".join(out + ["</svg>"]))


def save_png(path, mask): cv2.imwrite(str(path), (mask > 0).astype(np.uint8) * 255)


def flame_field(gray, box, cm_per_px, anchor_row, target_h_cm=None, name="", fill_mode="from_top", exclude=None):
    """trace one flame field. box = (x0, y0, x1, y1) in the source image; anchor_row = source row of the flame base line.
    Returns masks at PXMM: deboss fill, contour ridge lines, plus the measured size."""
    x0, y0, x1, y1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    if y1 <= y0 or x1 <= x0:
        raise ValueError(f"flame_field({name!r}): degenerate box {(x0, y0, x1, y1)}")
    g = gray[y0:y1, x0:x1]; lines = np.ascontiguousarray((g < 165).astype(np.uint8))
    if exclude is not None: lines[exclude[y0:y1, x0:x1] > 0] = 0
    if anchor_row - y0 < lines.shape[0]:
        lines[max(0, anchor_row - y0):, :] = 0                                       # nothing below the base line
    lines = np.ascontiguousarray(lines, dtype=np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(lines, 8); keep = np.zeros_like(lines)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= 12: keep[lab == i] = 1
    # deboss fill: everything in the band not reachable from the open side over non-line pixels, closed by the base line
    closed = cv2.dilate(keep, np.ones((3, 3), np.uint8)); band = closed.copy(); band[anchor_row - y0:, :] = 1
    ff = (1 - band).astype(np.uint8); h, w = ff.shape; m = np.zeros((h + 2, w + 2), np.uint8)
    seeds = [(x, 0) for x in range(0, w, 3)] if fill_mode == "from_top" else [(x, 0) for x in range(0, w, 3)] + [(w - 1, y) for y in range(0, h, 3)]
    for sx, sy in seeds:
        if ff[sy, sx] == 1: cv2.floodFill(ff, m, (sx, sy), 2)
    fill = ((ff != 2) & (np.arange(h)[:, None] < anchor_row - y0)).astype(np.uint8)
    ys = np.nonzero(keep.any(1))[0]; top = int(ys.min()); height_cm = (anchor_row - y0 - top) * cm_per_px
    sy = (target_h_cm / height_cm) if target_h_cm else 1.0
    W_mm = w * cm_per_px * 10; H_mm = (anchor_row - y0 - top) * cm_per_px * 10 * sy
    def up(mask):                                                                     # crop to [top, anchor), scale to PXMM (x and y factors may differ)
        mm = mask[top:anchor_row - y0].astype(np.float32); return (cv2.resize(mm, (int(round(W_mm * PXMM)), int(round(H_mm * PXMM))), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
    skel = skeleton(keep[top:anchor_row - y0]); sk = np.zeros_like(keep); sk[top:anchor_row - y0] = skel
    ridge_core = up(cv2.dilate(sk, np.ones((2, 2), np.uint8)))
    ridge = cv2.dilate(skeleton(ridge_core), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(1.2 * PXMM), int(1.2 * PXMM))))   # 1.2 mm stroke (PDF p8)
    return dict(fill=up(fill), ridge=ridge, width_cm=W_mm / 10, height_as_drawn_cm=height_cm, height_cm=H_mm / 10, y_scale=sy)


# ======================================================================================= front flames (p05)
def front():
    src = SRC / "p05_0.jpeg"; g = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE); gm, lines = garment_mask(g); H, W = g.shape
    cols = np.nonzero(gm.any(0))[0]; rows = np.nonzero(gm.any(1))[0]; bottom = int(rows.max())
    hem_w_px = np.ptp(np.nonzero(gm[bottom - 25])[0]); cmpp = SPEC["poms"]["POM-12_half_hem_sweep"]["value"] * 100 / hem_w_px     # POM-12 measured just above the vents
    # top of the hem turn-up = the upper of the dashed twin-needle stitch lines near the bottom (rows with many short dark runs)
    xs0, xs1 = int(cols.min() + 0.15 * np.ptp(cols)), int(cols.max() - 0.15 * np.ptp(cols)); frac = lines[:, xs0:xs1].mean(1)
    cand = [r for r in range(bottom - 60, bottom - 3) if frac[r] > 0.25]; anchor = int(min(cand)) if cand else bottom - 10
    # exclude the garment outline and the side seam lines: pixels within 7 px of the silhouette edge, and tall thin line components
    edge = cv2.dilate(cv2.morphologyEx(gm, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)), np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(lines, 8); tall = np.zeros_like(lines)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_HEIGHT] > 120 and st[i, cv2.CC_STAT_WIDTH] < 25: tall[lab == i] = 1
    excl = np.maximum(edge, cv2.dilate(tall, np.ones((5, 5), np.uint8)))
    box = (int(cols.min()) + 2, int(0.52 * H), int(cols.max()) - 1, anchor)
    r = flame_field(g, box, cmpp, anchor, target_h_cm=SPEC["poms"]["POM-16_flame_max_height"]["value"] * 100, name="front", exclude=excl)
    out = HERE / "flames_front"; save_png(f"{out}_deboss_fill.png", r["fill"]); save_png(f"{out}_hd_ridge.png", r["ridge"])
    to_svg(f"{out}.svg", [("deboss_fill", r["fill"], "#2e2f31"), ("hd_ridge", r["ridge"], "#0b0b0c")], r["width_cm"] * 10, r["height_cm"] * 10)
    REP["artworks"]["flames_front"] = dict(source=src.name, source_sha256=sha(src), cm_per_source_px=round(cmpp, 4), scale_from="hem width = POM-12 58.0 cm",
        anchor="upper twin-needle stitch line of the hem turn-up, source row %d (garment bottom row %d)" % (anchor, bottom),
        width_cm=round(r["width_cm"], 2), peak_height_as_drawn_cm=round(r["height_as_drawn_cm"], 2), peak_height_cm=round(r["height_cm"], 2),
        vertical_correction=round(r["y_scale"], 4), spec_peak_cm=21.5, files=[f"{out.name}.svg", f"{out.name}_deboss_fill.png", f"{out.name}_hd_ridge.png"],
        note="deboss fill = band under the flame silhouette, closed at the hem anchor; ridges = traced contour centrelines re-stroked at 1.2 mm (PDF p8 echo-line width)")


# ======================================================================================= back wraps + ember (p06)
def back():
    src = SRC / "p06_0.jpeg"; g = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE); gm, lines = garment_mask(g); H, W = g.shape
    rows = np.nonzero(gm.any(1))[0]; bottom = int(rows.max()); hem_cols = np.nonzero(gm[bottom - 25])[0]; xl, xr = int(hem_cols.min()), int(hem_cols.max())
    cmpp = SPEC["poms"]["POM-12_half_hem_sweep"]["value"] * 100 / (xr - xl)
    xs0, xs1 = xl + int(0.1 * (xr - xl)), xr - int(0.1 * (xr - xl)); frac = lines[:, xs0:xs1].mean(1)
    cand = [r for r in range(bottom - 60, bottom - 3) if frac[r] > 0.25]; anchor = int(min(cand)) if cand else bottom - 10
    edge = cv2.dilate(cv2.morphologyEx(gm, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)), np.ones((7, 7), np.uint8))
    # rib gusset hatching (dense vertical lines) and the centre-back seam are excluded
    hatch = cv2.morphologyEx((g < 165).astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 1), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats((g < 165).astype(np.uint8), 8); tall = np.zeros_like(gm)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_HEIGHT] > 120 and st[i, cv2.CC_STAT_WIDTH] < 25: tall[lab == i] = 1
    excl = np.maximum(np.maximum(edge, cv2.dilate(tall, np.ones((5, 5), np.uint8))), cv2.dilate(hatch, np.ones((3, 3), np.uint8)))
    wrap_target_w = 12.0                                                                            # PDF p13 / p20: 12.0 cm from the side seam
    for side, (bx0, bx1) in (("left_side", (xl + 1, xl + int(0.30 * (xr - xl)))), ("right_side", (xr - int(0.30 * (xr - xl)), xr - 1))):
        # gusset base: lowest row of the upper-torso hatched rib area (strictly above the hem anchor)
        hs = np.nonzero(hatch[:int(0.65 * H), bx0:bx1].any(1))[0]
        ytop = int(hs.max()) + 3 if len(hs) else int(0.4 * H)
        if ytop >= anchor - 10: ytop = int(0.4 * H)
        r = flame_field(g, (bx0, ytop, bx1, anchor), cmpp, anchor, target_h_cm=None, exclude=excl, fill_mode="from_side")
        ink = np.nonzero(r["ridge"].any(0))[0]; drawn_w = (ink.max() - ink.min()) / PXMM / 10 if len(ink) else 0
        out = HERE / f"flames_back_wrap_{side}"; save_png(f"{out}_deboss_fill.png", r["fill"]); save_png(f"{out}_hd_ridge.png", r["ridge"])
        to_svg(f"{out}.svg", [("deboss_fill", r["fill"], "#2e2f31"), ("hd_ridge", r["ridge"], "#0b0b0c")], r["width_cm"] * 10, r["height_cm"] * 10)
        REP["artworks"][f"flames_back_wrap_{side}"] = dict(source=src.name, source_sha256=sha(src), cm_per_source_px=round(cmpp, 4), scale_from="hem width = POM-12 58.0 cm",
            box_source_px=[bx0, ytop, bx1, anchor], drawn_ink_width_from_fold_cm=round(drawn_w, 2), spec_wrap_cm="12.0 from the side seam = 10.0 past the fold",
            height_as_drawn_cm=round(r["height_as_drawn_cm"], 2), files=[f"{out.name}.svg", f"{out.name}_deboss_fill.png", f"{out.name}_hd_ridge.png"],
            note="kept at the drawn proportions; not rescaled (see flags)")
    REP["flags"].append("Back flat side wraps are drawn about %.0f cm tall; the table's side crests are 14.5 cm and the side-seam crossing window is 4.5-16.5 cm. "
                        "Traced as drawn; the stage 4 placement must fit them to the window." % REP["artworks"]["flames_back_wrap_left_side"]["height_as_drawn_cm"])
    # ember: the small flame at the base of the centre-back seam, scaled to the table's 5.0 x 8.5 cm
    cx = (xl + xr) // 2; ex0, ex1 = cx - int(8 / cmpp), cx + int(8 / cmpp)
    r = flame_field(g, (ex0, int(0.6 * H), ex1, anchor), cmpp, anchor, target_h_cm=8.5, exclude=excl)
    ink = np.nonzero(r["fill"].any(0))[0]; fx = (5.0 * 10 * PXMM) / max(1, ink.max() - ink.min())
    crop_f = r["fill"][:, ink.min():ink.max() + 1]; crop_r = r["ridge"][:, ink.min():ink.max() + 1]
    sz = (int(round(crop_f.shape[1] * fx)), crop_f.shape[0]); ef = cv2.resize(crop_f, sz, interpolation=cv2.INTER_NEAREST); er = cv2.resize(crop_r, sz, interpolation=cv2.INTER_NEAREST)
    out = HERE / "flames_back_ember"; save_png(f"{out}_deboss_fill.png", ef); save_png(f"{out}_hd_ridge.png", er)
    to_svg(f"{out}.svg", [("deboss_fill", ef, "#2e2f31"), ("hd_ridge", er, "#0b0b0c")], sz[0] / PXMM, sz[1] / PXMM)
    REP["artworks"]["flames_back_ember"] = dict(source=src.name, drawn_width_cm=round((ink.max() - ink.min()) / PXMM / 10, 2), drawn_height_cm=round(r["height_as_drawn_cm"], 2),
        spec_cm=[5.0, 8.5], scaled_to_cm=[round(sz[0] / PXMM / 10, 2), round(sz[1] / PXMM / 10, 2)], x_scale=round(fx, 4), y_scale=round(r["y_scale"], 4),
        note="the centre-back seam line runs through the ember in the flat and is traced as part of it", files=[f"{out.name}.svg", f"{out.name}_deboss_fill.png", f"{out.name}_hd_ridge.png"])


# ======================================================================================= shield (p08)
def shield():
    src = SRC / "p08_0.jpeg"; g = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE); H, W = g.shape
    light = (g > 150).astype(np.uint8); light = cv2.morphologyEx(light, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cs, _ = cv2.findContours(light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); c = max(cs, key=cv2.contourArea); hull = cv2.convexHull(c)
    eps = 0.01 * cv2.arcLength(hull, True); poly = cv2.approxPolyDP(hull, eps, True)
    while len(poly) > 5: eps *= 1.15; poly = cv2.approxPolyDP(hull, eps, True)
    P = poly[:, 0, :].astype(np.float32)
    # PDF: 92 (vertical) x 78 (horizontal) mm, flat top, vertical sides, 45 deg lower bevels meeting at the bottom point (p13)
    Wm, Hm = 78.0, 92.0; side = Hm - Wm / 2; T = np.array([[0, 0], [Wm, 0], [Wm, side], [Wm / 2, Hm], [0, side]], np.float32)
    # order the detected corners: the longest edge is the top edge (78 mm against 53 and 55 mm)
    L = [np.linalg.norm(P[(i + 1) % 5] - P[i]) for i in range(5)]; i0 = int(np.argmax(L))
    best = None
    for direction in (1, -1):
        order = [P[(i0 + direction * k) % 5] for k in range(5)] if direction == 1 else [P[(i0 + 1 - k) % 5] for k in range(5)]
        Hm_, _ = cv2.findHomography(np.array(order), T); res = np.linalg.norm(cv2.perspectiveTransform(np.array(order)[None], Hm_)[0] - T, axis=1)
        if best is None or res.max() < best[1].max(): best = (Hm_, res, order)
    Hmat, res, order = best
    S = np.diag([PXMM, PXMM, 1.0]); warp = cv2.warpPerspective(g, S @ Hmat, (int(Wm * PXMM), int(Hm * PXMM)), flags=cv2.INTER_CUBIC, borderValue=0)
    inside = np.zeros_like(warp); cv2.fillPoly(inside, [np.round(T * PXMM).astype(np.int32)], 1)
    raised = ((warp > 165) & (inside > 0)).astype(np.uint8); raised = cv2.morphologyEx(raised, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # the letter B: the left-most large raised component that is not the frame; its outline encloses the louvred field
    n, lab, st, cen = cv2.connectedComponentsWithStats(raised, 8); comps = sorted([i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 300], key=lambda i: -st[i, cv2.CC_STAT_AREA])
    frame = comps[0]; letters = [i for i in comps[1:] if st[i, cv2.CC_STAT_AREA] > 800]; letters.sort(key=lambda i: cen[i][0])
    fins = np.zeros_like(raised); b_field = np.zeros_like(raised)
    if letters:
        bi = letters[0]; bmask = (lab == bi).astype(np.uint8); filled = bmask.copy(); cs2, hier = cv2.findContours(bmask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        outer = [cs2[k] for k in range(len(cs2)) if hier[0][k][3] < 0]; cv2.drawContours(filled, outer, -1, 1, -1)
        holes = [cs2[k] for k in range(len(cs2)) if hier[0][k][3] >= 0]; areas = sorted([cv2.contourArea(h) for h in holes], reverse=True)
        # the B outline is a stroke: its largest interior holes are the two counters (left dark) and the stroke-enclosed field; fins go in the field
        b_field = (filled & (1 - bmask)).astype(np.uint8)
        counters = np.zeros_like(raised)
        for h in holes:
            if cv2.contourArea(h) < 0.25 * (areas[0] if areas else 1): cv2.drawContours(counters, [h], -1, 1, -1)
        b_field = (b_field & (1 - counters)).astype(np.uint8)
        yy, xx = np.mgrid[0:raised.shape[0], 0:raised.shape[1]] / PXMM
        louver = ((((xx - yy) / math.sqrt(2)) % 4.0) < 2.2).astype(np.uint8)                       # 45 deg, 4.0 mm pitch, 2.2 mm raised (PDF p13)
        fins = (louver & b_field).astype(np.uint8)
    frame_m = (lab == frame).astype(np.uint8); letter_m = np.isin(lab, letters).astype(np.uint8)
    out = HERE / "shield_bst"
    for nm, m in (("frame", frame_m), ("letters", letter_m), ("b_louver_fins", fins), ("base_twill", inside)): save_png(f"{out}_{nm}.png", m)
    to_svg(f"{out}.svg", [("base_twill", inside, "#1a1a1b"), ("frame", frame_m, "#e8e2d4"), ("letters", letter_m, "#e8e2d4"), ("b_louver_fins", fins, "#e8e2d4")], Wm, Hm)
    REP["artworks"]["shield_bst"] = dict(source=src.name, source_sha256=sha(src), corners_source_px=np.round(np.array(order), 1).tolist(),
        target_mm="pentagon 78 wide x 92 tall, vertical sides 53 mm, 45 deg bevels (from PDF p13: 92 x 78 mm, 1:1.18, 45 deg lower bevel)",
        rectification_residual_mm=np.round(res, 3).tolist(), letters_found=len(letters), louver=dict(angle_deg=45, pitch_mm=4.0, raised_mm=2.2, channel_mm=1.8),
        files=[f"{out.name}.svg"] + [f"{out.name}_{n}.png" for n in ("frame", "letters", "b_louver_fins", "base_twill")],
        note="source is a perspective macro render; a homography maps its 5 outer corners to the specified pentagon. Letterforms are as rendered, not the brand vectors")
    if len(letters) != 3: REP["flags"].append(f"shield: {len(letters)} letter components found (expected 3: B, S, T)")


def yoke_bar():
    Wm, Hm = 48.0, 6.0; m = np.ones((int(Hm * PXMM), int(Wm * PXMM)), np.uint8); out = HERE / "yoke_bar"; save_png(f"{out}.png", m)
    to_svg(f"{out}.svg", [("silicone_bar", m, "#0b0b0c")], Wm, Hm)
    REP["artworks"]["yoke_bar"] = dict(source="PDF p13 / p20 numbers (48 x 6 mm, 4.5 cm below the CB collar seam); geometric, not traced", files=[f"{out.name}.svg", f"{out.name}.png"])


for f in (front, back, shield, yoke_bar): f()
REP["flags"].append("All artwork is mockup grade, traced from generated views in the PDF. Production artwork must come from the designer's files (paper 03 section 3).")
(HERE / "trace_report.json").write_text(json.dumps(REP, indent=1, default=float)); print(json.dumps(REP, indent=1, default=float)[:4000])
