"""06_splat_binding_and_relight/splat_pack_cli.py — Deterministic SOG v3 & SPZ v4 Packing Wrapper.

Licence: the upstream dependency `playcanvas/splat-transform` is MIT; this wrapper script is
covered by this repository's LICENSE.

Preserves explicit Gaussian slot ordering (i <-> i) so index-matched pose morphs and per-garment ID masks
stay aligned without nearest-neighbor jumps. Encodes garment_id:u16 in (R, G) with A = 255 (-exact lossless WebP)
to prevent WebKit WebGPU alpha premultiplication corruption.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

SOG_MAX_BYTES = 7 * 1024 * 1024  # 7.0 MB SOG lite tier cap


def _cli_base() -> list[str]:
    exe = shutil.which("splat-transform")
    return [exe] if exe else ["npx", "--yes", "@playcanvas/splat-transform@3.10.3"]


def encode_garment_id_rgba(garment_ids: np.ndarray, width: int | None = None) -> np.ndarray:
    """Encode garment_id:u16 into (R, G) with B=0 and A=255 so WebGPU premultiplied-alpha never zeroes RGB."""
    gids = np.asarray(garment_ids, dtype=np.uint16).reshape(-1)
    n = len(gids)
    w = int(width) if width is not None else int(np.ceil(np.sqrt(max(n, 1))))
    h = int(np.ceil(max(n, 1) / w))
    rgba = np.zeros((h * w, 4), dtype=np.uint8)
    rgba[:n, 0] = (gids & 0xFF).astype(np.uint8)
    rgba[:n, 1] = ((gids >> 8) & 0xFF).astype(np.uint8)
    rgba[:, 3] = 255
    return rgba.reshape((h, w, 4))


def write_blind_garment_id_webp(garment_ids: np.ndarray, out_webp: Path, width: int | None = None) -> Path:
    """Write lossless WebP with A=255 (use cwebp -lossless -exact if available, else Pillow lossless)."""
    from PIL import Image  # type: ignore

    rgba = encode_garment_id_rgba(garment_ids, width=width)
    out_webp.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(out_webp, format="WEBP", lossless=True, quality=100)
    return out_webp


def build_splat_transform_sog_cmd(
    src_ply: Path,
    dst_sog: Path,
    sh_bands: int = 3,
    sh_iterations: int = 10,
    sog_version: int = 3,
    sh_palette_size: int = 65536,
    blind_attr: str = "garment_id:u16",
    preserve_order: bool = True,
    filter_floaters: bool = False,
) -> list[str]:
    """Build SOG v3 command omitting `-m` Morton reorder across paired pose files."""
    cmd = [
        *_cli_base(),
        str(src_ply),
        str(dst_sog),
        "-w",
        "--sog-version",
        str(sog_version),
        "--sh-palette-size",
        str(sh_palette_size),
        "-i",
        str(sh_iterations),
    ]
    if blind_attr:
        cmd += ["--sog-blind", blind_attr]
    if 0 <= sh_bands <= 4 and sh_bands < 3:
        cmd += ["-H", str(sh_bands)]
    if filter_floaters:
        cmd += ["-F"]
    if not preserve_order:
        # Never inject `-m` (--morton-order) on index-matched pose pairs.
        cmd += ["-m"]
    return cmd


def build_splat_transform_spz_cmd(src_ply: Path, dst_spz: Path, spz_version: int = 4, sh_bands: int = 3) -> list[str]:
    cmd = [*_cli_base(), str(src_ply), str(dst_spz), "-w", "--spz-version", str(spz_version)]
    if 0 <= sh_bands <= 4 and sh_bands < 3:
        cmd += ["-H", str(sh_bands)]
    return cmd


def build_splat_transform_html_cmd(src_sog_or_ply: Path, dst_html: Path, unbundled: bool = True) -> list[str]:
    cmd = [*_cli_base(), str(src_sog_or_ply), str(dst_html), "-w"]
    if unbundled:
        cmd.append("--unbundled")
    return cmd


if __name__ == "__main__":
    sample_rgba = encode_garment_id_rgba(np.array([0, 1, 258, 65535], dtype=np.uint16), width=2)
    print(
        json.dumps(
            {
                "sog_cmd_index_locked": build_splat_transform_sog_cmd(Path("pose0.ply"), Path("pose0.sog"), preserve_order=True),
                "spz_v4_cmd": build_splat_transform_spz_cmd(Path("pose0.ply"), Path("pose0.spz"), spz_version=4),
                "blind_rgba_sample": sample_rgba.tolist(),
                "sog_max_bytes": SOG_MAX_BYTES,
            },
            indent=2,
        )
    )
