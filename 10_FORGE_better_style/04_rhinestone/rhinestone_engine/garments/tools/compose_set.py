"""compose_set.py - two-piece set sheets in the reference layout (shirt over trousers; front column, back column) from the
catalogue alpha renders. usage: python compose_set.py <renders_dir> <colour>"""
import sys, json, hashlib
from pathlib import Path
import numpy as np
from PIL import Image
D = Path(sys.argv[1]); col = sys.argv[2]; GROUND = (0xF3, 0xEE, 0xE7)


def crop(name):
    im = Image.open(D / name).convert('RGBA'); return im.crop(im.getbbox())


def column(view):
    s = crop(f'BS-BSTSET-001_shirt_{col}_floating_{view}_alpha.png'); t = crop(f'BS-BSTSET-001_trousers_{col}_floating_{view}_alpha.png')
    # both renders are framed to the same occupancy rule, not the same scale: bring them to one scale from their real sizes
    # shirt body width about 0.673 m, trousers height 1.012 m (waistband top to hem) in the source geometry
    ppm_s = s.size[0] / 0.673; ppm_t = t.size[1] / 1.012; k = ppm_s / ppm_t
    t = t.resize((int(t.size[0] * k), int(t.size[1] * k)), Image.LANCZOS)
    W = max(s.size[0], t.size[0]) + 120; gap = 40; H = s.size[1] + t.size[1] + gap + 160
    c = Image.new('RGBA', (W, H), GROUND + (255,)); c.alpha_composite(s, ((W - s.size[0]) // 2, 80)); c.alpha_composite(t, ((W - t.size[0]) // 2, 80 + s.size[1] + gap)); return c


f, b = column('front'), column('back'); H = max(f.size[1], b.size[1]); sheet = Image.new('RGB', (f.size[0] + b.size[0] + 24, H), GROUND)
sheet.paste(f.convert('RGB'), (0, 0)); sheet.paste(b.convert('RGB'), (f.size[0] + 24, 0))
for x in range(f.size[0] + 8, f.size[0] + 16):
    for y in range(H): sheet.putpixel((x, y), (255, 255, 255))
out = D / f'BS-BSTSET-001_set_{col}_front_back.png'; sheet.save(out)
f.convert('RGB').save(D / f'BS-BSTSET-001_set_{col}_front.png'); b.convert('RGB').save(D / f'BS-BSTSET-001_set_{col}_back.png')
print(out.name, sheet.size, hashlib.sha256(out.read_bytes()).hexdigest()[:16])
