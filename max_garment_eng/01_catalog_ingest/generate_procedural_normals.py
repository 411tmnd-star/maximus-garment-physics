"""generate_procedural_normals.py — Synthesize seamless 512x512 tangent-space fabric normal maps (CPU NumPy + PIL)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

OUT_TEX_DIR = Path(__file__).resolve().parent / "repaired_masters" / "textures"


def _height_to_normal_rgba(height: np.ndarray, strength: float = 2.0) -> np.ndarray:
    """Convert a tileable scalar height field in [0, 1] into tangent-space RGB normal map (OpenGL Y+)."""
    h = np.asarray(height, dtype=np.float64)
    gx = 0.5 * (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) * strength
    gy = 0.5 * (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) * strength
    nx = -gx
    ny = -gy
    nz = np.ones_like(h)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx /= norm
    ny /= norm
    nz /= norm
    rgb = np.stack([(nx * 0.5 + 0.5), (ny * 0.5 + 0.5), (nz * 0.5 + 0.5)], axis=-1)
    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)


def build_textures(size: int = 512) -> dict[str, str]:
    OUT_TEX_DIR.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64) / float(size)
    rng = np.random.default_rng(411)

    # 1. 280 GSM Combed Cotton Jersey (fine V-shaped single-jersey interlock wales, tight smooth knit)
    wales_x = 64.0
    courses_y = 84.0
    v_shape = np.sin(2.0 * np.pi * xx * wales_x)
    course = np.cos(2.0 * np.pi * (yy * courses_y + 0.14 * np.abs(v_shape)))
    micro_grain = rng.normal(0.0, 0.03, size=(size, size))
    h_cotton = 0.5 + 0.28 * v_shape * (0.65 + 0.35 * course) + micro_grain
    cotton_path = OUT_TEX_DIR / "cotton_jersey_280gsm_normal.png"
    Image.fromarray(_height_to_normal_rgba(h_cotton, strength=1.35), mode="RGB").save(cotton_path)

    # 2. Parametric 2x2 Rib Knit (strong alternating vertical raised ribs for BS-TANK-001)
    rib_freq = 32.0  # 32 complete 2x2 rib cycles across tile width
    rib_wave = np.sin(2.0 * np.pi * xx * rib_freq)
    rib_groove = np.tanh(2.1 * rib_wave)
    knit_loop = 0.18 * np.sin(2.0 * np.pi * yy * 72.0 + 0.25 * rib_wave)
    h_rib = 0.5 + 0.38 * rib_groove + knit_loop + rng.normal(0.0, 0.025, size=(size, size))
    rib_path = OUT_TEX_DIR / "tank_2x2_rib_normal.png"
    Image.fromarray(_height_to_normal_rgba(h_rib, strength=2.6), mode="RGB").save(rib_path)

    # 3. 12 oz Heavyweight 3x1 Right-Hand Denim Twill (diagonal whale + crisp warp/weft grain for BS-DENIMSHIRT-001)
    twill_diag = np.sin(2.0 * np.pi * (xx * 48.0 - yy * 48.0))
    twill_sharp = np.where(twill_diag > -0.25, 0.45 * twill_diag + 0.2, -0.35)
    warp_slub = 0.16 * np.sin(2.0 * np.pi * xx * 96.0) + 0.08 * np.sin(2.0 * np.pi * xx * 12.0)
    weft = 0.12 * np.cos(2.0 * np.pi * yy * 64.0)
    h_denim = 0.5 + 0.45 * twill_sharp + warp_slub + weft + rng.normal(0.0, 0.03, size=(size, size))
    denim_path = OUT_TEX_DIR / "denim_12oz_twill_normal.png"
    Image.fromarray(_height_to_normal_rgba(h_denim, strength=2.2), mode="RGB").save(denim_path)

    return {
        "cotton_jersey_280gsm": str(cotton_path),
        "tank_2x2_rib": str(rib_path),
        "denim_12oz_twill": str(denim_path),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(build_textures(), indent=2))
