import sys, time, numpy as np, cv2, os
sys.dont_write_bytecode = True
import bpy
import art
HE = "C:/Users/USER/Desktop/4041_MechMix/02_engines/human_engine/"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'work') + os.sep


def draw(stones, W, H, name, s=8):
    im = np.zeros((int(H * s), int(W * s), 3), np.uint8)
    for x, y, r, c in stones: cv2.circle(im, (int(x * s), int(y * s)), int(r * s), (255, 255, 255) if c == 0 else (60, 160, 255), -1, cv2.LINE_AA)
    cv2.imwrite(OUT + name, im)


t = time.time(); st, (Wm, Hm) = art.tile_shields(HE + 'src_catalog/better_style_catalog/assets/artwork_camp/lattice_tile.png', 181.4)
print('tile stones', len(st), round(time.time() - t, 1), 's', Wm, Hm); draw(st, Wm, Hm, 'dbg_tile.png', 6)
t = time.time(); m, mp = art.script_mask(); S = art.contour_fill(m, mp, 1.0); print('script', m.shape, len(S), round(time.time() - t, 1))
draw([(x, y, 1.0, 0) for x, y, _, _ in S], m.shape[1] * mp, m.shape[0] * mp, 'dbg_script.png', 5)
m, mp = art.crown_mask(); S = art.contour_fill(m, mp, 1.0); print('crown', len(S)); draw([(x, y, 1.0, 0) for x, y, _, _ in S], m.shape[1] * mp, m.shape[0] * mp, 'dbg_crown.png', 8)
t = time.time(); m, c, mp = art.elephant_masks(HE + 'engineering/PHASE3_2026-11-09/elephant_evidence/elephant_final_548x852.png')
S = art.contour_fill(m, mp, 1.0, ensure_min=False, cls_img=c); print('elephant', len(S), round(time.time() - t, 1))
draw([(x, y, 1.0, k) for x, y, _, k in S], m.shape[1] * mp, m.shape[0] * mp, 'dbg_elephant.png', 4)
