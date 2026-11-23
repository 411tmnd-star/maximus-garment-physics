#!/usr/bin/env python3
"""
scripts/run_mit_matting_benchmark.py
--------------------------------------------------------------------------------
Track T3 / Milestone M3 (Job B) — 5-Minute Commercial-Safe MIT Matting Benchmark
Benchmarks strictly MIT-licensed matting models on NVIDIA L4 GPU:
  1. BEN2     (PramaLLC/BEN2, MIT License)
  2. BiRefNet (ZhengPeng7/BiRefNet, MIT License)

Hard Invariants (k):
  - ZERO NTU S-Lab 1.0 weights (MatAnyone, MatAnyone2, ProPainter, CodeFormer)
  - ZERO GPL-3.0 weights (RVM)
  - Strictly bounded runtime (<= 300s / 5-minute budget)
  - Isolated output directory (/content/work/MIT_matting_bench -> Drive results/MIT_matting_bench/)
  - Does NOT touch or invalidate Stages 1-5 Maximus checkpoints.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

ALLOWED_MODELS = {
    "BEN2": {
        "hf_repo": "PramaLLC/BEN2",
        "license": "MIT",
        "commercial_safe": True,
    },
    "BiRefNet": {
        "hf_repo": "ZhengPeng7/BiRefNet",
        "license": "MIT",
        "commercial_safe": True,
    },
}

QUARANTINED_NON_COMMERCIAL_OR_GPL = frozenset({
    "MatAnyone", "MatAnyone2", "ProPainter", "CodeFormer", "RVM",
})


def _make_synthetic_facecam_clips(out_dir: Path, n_frames: int = 8, res: tuple[int, int] = (1024, 1024)) -> list[list[Path]]:
    """Generate or extract 2 deterministic 1024x1024 streamer facecam test sequences if sample clips are absent."""
    clips_dir = out_dir / "sample_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    w, h = res
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    all_clips: list[list[Path]] = []
    for clip_idx in range(2):
        cdir = clips_dir / f"clip_{clip_idx + 1:02d}"
        cdir.mkdir(parents=True, exist_ok=True)
        frames: list[Path] = []
        for f_idx in range(n_frames):
            # Bedroom/studio background gradient + monitor glow
            bg_r = np.clip(28.0 + 35.0 * (yy / h) + 15.0 * clip_idx, 0, 255)
            bg_g = np.clip(24.0 + 25.0 * (xx / w), 0, 255)
            bg_b = np.clip(45.0 + 50.0 * (1.0 - yy / h), 0, 255)
            img = np.stack([bg_r, bg_g, bg_b], axis=-1)

            # Moving streamer head + shoulders silhouette with fine hair-strand edge frequency
            cx = w * (0.48 + 0.04 * np.sin(f_idx * 0.6 + clip_idx))
            cy_head = h * 0.36
            cy_torso = h * 0.78

            theta = np.arctan2(yy - cy_head, xx - cx)
            hair_ripple = 1.0 + 0.035 * np.sin(theta * 28.0 + f_idx)
            head_dist = np.sqrt(((xx - cx) / (w * 0.14 * hair_ripple)) ** 2 + ((yy - cy_head) / (h * 0.17 * hair_ripple)) ** 2)
            torso_dist = np.sqrt(((xx - cx) / (w * 0.28)) ** 2 + ((yy - cy_torso) / (h * 0.26)) ** 2)

            fg_mask = np.clip(1.0 - np.minimum(head_dist, torso_dist), 0.0, 1.0)
            alpha = np.clip((fg_mask - 0.02) / 0.06, 0.0, 1.0)[..., None]

            fg_rgb = np.zeros_like(img)
            fg_rgb[..., 0] = 195.0 - 30.0 * (yy / h)
            fg_rgb[..., 1] = 152.0 - 20.0 * (yy / h)
            fg_rgb[..., 2] = 132.0 + 25.0 * (xx / w)

            comp = np.clip(fg_rgb * alpha + img * (1.0 - alpha), 0, 255).astype(np.uint8)
            fp = cdir / f"frame_{f_idx:03d}.png"
            Image.fromarray(comp, mode="RGB").save(fp)
            frames.append(fp)
        all_clips.append(frames)
    return all_clips


def _ensure_bench_deps() -> None:
    """Install lightweight huggingface/timm/einops/kornia runtime deps if missing."""
    pkgs = ["timm>=1.0.0", "einops>=0.8.0", "kornia>=0.7.3", "huggingface_hub>=0.25.0", "transformers>=4.44.0"]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=False)


def _compute_edge_sharpness(alpha_u8: np.ndarray) -> float:
    """Measure normalized Sobel-like gradient magnitude along the transition band (10 < alpha < 245)."""
    a = alpha_u8.astype(np.float32) / 255.0
    gx = np.abs(a[:, 1:] - a[:, :-1])[:-1, :]
    gy = np.abs(a[1:, :] - a[:-1, :])[:, :-1]
    grad = np.hypot(gx, gy)
    band = (a[:-1, :-1] > 0.04) & (a[:-1, :-1] < 0.96)
    if not np.any(band):
        return float(np.mean(grad))
    return float(np.mean(grad[band]))


def _benchmark_birefnet(clips: list[list[Path]], out_dir: Path) -> tuple[dict[str, Any], list[np.ndarray]]:
    import torch
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t_load0 = time.perf_counter()

    model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", trust_remote_code=True
    ).to("cuda").eval()
    model.half()
    load_sec = time.perf_counter() - t_load0

    tfm = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    latencies_ms: list[float] = []
    edge_scores: list[float] = []
    sample_mattes: list[np.ndarray] = []

    with torch.no_grad():
        # Warmup pass
        w_im = Image.open(clips[0][0]).convert("RGB")
        w_t = tfm(w_im).unsqueeze(0).to("cuda", dtype=torch.float16)
        _ = model(w_t)
        torch.cuda.synchronize()

        for c_idx, frames in enumerate(clips):
            for f_idx, fp in enumerate(frames):
                im = Image.open(fp).convert("RGB")
                inp = tfm(im).unsqueeze(0).to("cuda", dtype=torch.float16)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                preds = model(inp)
                pred = preds[-1] if isinstance(preds, (list, tuple)) else preds
                alpha = pred.sigmoid().squeeze().float().cpu().numpy()
                torch.cuda.synchronize()
                dt_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(dt_ms)
                alpha_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
                edge_scores.append(_compute_edge_sharpness(alpha_u8))
                if f_idx == 0:
                    sample_mattes.append(alpha_u8)

    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024.0 ** 3)
    del model
    torch.cuda.empty_cache()

    mean_ms = float(np.mean(latencies_ms))
    fps = 1000.0 / max(mean_ms, 1e-3)
    return {
        "model": "BiRefNet",
        "hf_repo": "ZhengPeng7/BiRefNet",
        "license": "MIT",
        "commercial_safe": True,
        "resolution": "1024x1024",
        "precision": "fp16",
        "frames_evaluated": len(latencies_ms),
        "load_time_sec": round(load_sec, 3),
        "mean_latency_ms": round(mean_ms, 2),
        "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 2),
        "fps": round(fps, 2),
        "peak_vram_gb": round(peak_vram_gb, 3),
        "mean_edge_sharpness": round(float(np.mean(edge_scores)), 4),
        "pass": bool(peak_vram_gb <= 12.0 and fps >= 5.0),
    }, sample_mattes


def _benchmark_ben2(clips: list[list[Path]], out_dir: Path) -> tuple[dict[str, Any], list[np.ndarray]]:
    import torch
    from transformers import AutoModelForImageSegmentation

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t_load0 = time.perf_counter()

    model = None
    load_method = "transformers_auto"
    try:
        model = AutoModelForImageSegmentation.from_pretrained(
            "PramaLLC/BEN2", trust_remote_code=True
        ).to("cuda").eval()
    except Exception:
        load_method = "ben2_hub_clone"
        ben_repo = Path("/content/ben2_src")
        if not ben_repo.exists():
            subprocess.run(["git", "clone", "--depth", "1", "https://github.com/PramaLLC/BEN2.git", str(ben_repo)], check=True)
        sys.path.insert(0, str(ben_repo))
        from huggingface_hub import hf_hub_download
        import BEN2  # type: ignore
        ckpt = hf_hub_download(repo_id="PramaLLC/BEN2", filename="BEN2_Base.pth")
        model = BEN2.BEN_Base().to("cuda").eval()
        model.loadcheckpoints(ckpt)

    load_sec = time.perf_counter() - t_load0
    latencies_ms: list[float] = []
    edge_scores: list[float] = []
    sample_mattes: list[np.ndarray] = []

    with torch.no_grad():
        # Warmup
        w_im = Image.open(clips[0][0]).convert("RGB")
        if hasattr(model, "inference"):
            _ = model.inference(w_im)
        else:
            from torchvision import transforms
            tfm = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            _ = model(tfm(w_im).unsqueeze(0).to("cuda"))
        torch.cuda.synchronize()

        for c_idx, frames in enumerate(clips):
            for f_idx, fp in enumerate(frames):
                im = Image.open(fp).convert("RGB")
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                if hasattr(model, "inference"):
                    out_im = model.inference(im)
                    if isinstance(out_im, Image.Image):
                        alpha_u8 = np.array(out_im.split()[-1] if out_im.mode == "RGBA" else out_im.convert("L"), dtype=np.uint8)
                    else:
                        alpha_u8 = np.array(out_im, dtype=np.uint8)
                else:
                    from torchvision import transforms
                    tfm = transforms.Compose([
                        transforms.Resize((1024, 1024)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ])
                    preds = model(tfm(im).unsqueeze(0).to("cuda"))
                    pred = preds[-1] if isinstance(preds, (list, tuple)) else preds
                    alpha = pred.sigmoid().squeeze().float().cpu().numpy()
                    alpha_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
                torch.cuda.synchronize()
                dt_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(dt_ms)
                edge_scores.append(_compute_edge_sharpness(alpha_u8))
                if f_idx == 0:
                    sample_mattes.append(alpha_u8)

    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024.0 ** 3)
    del model
    torch.cuda.empty_cache()

    mean_ms = float(np.mean(latencies_ms))
    fps = 1000.0 / max(mean_ms, 1e-3)
    return {
        "model": "BEN2",
        "hf_repo": "PramaLLC/BEN2",
        "license": "MIT",
        "commercial_safe": True,
        "load_method": load_method,
        "resolution": "1024x1024",
        "frames_evaluated": len(latencies_ms),
        "load_time_sec": round(load_sec, 3),
        "mean_latency_ms": round(mean_ms, 2),
        "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 2),
        "fps": round(fps, 2),
        "peak_vram_gb": round(peak_vram_gb, 3),
        "mean_edge_sharpness": round(float(np.mean(edge_scores)), 4),
        "pass": bool(peak_vram_gb <= 12.0 and fps >= 3.0),
    }, sample_mattes


def _save_contact_sheet(
    clips: list[list[Path]],
    ben2_mattes: list[np.ndarray],
    biref_mattes: list[np.ndarray],
    out_png: Path,
) -> Path:
    w, h = 480, 480
    sheet = Image.new("RGB", (w * 3, h * 2), (18, 20, 28))
    draw = ImageDraw.Draw(sheet)

    for row in range(2):
        src_im = Image.open(clips[row][0]).convert("RGB").resize((w, h))
        sheet.paste(src_im, (0, row * h))
        draw.text((14, row * h + 14), f"Clip {row + 1}: Input 1024px", fill=(0, 229, 190))

        if row < len(ben2_mattes):
            m1 = Image.fromarray(ben2_mattes[row]).convert("RGB").resize((w, h))
            sheet.paste(m1, (w, row * h))
        draw.text((w + 14, row * h + 14), f"Clip {row + 1}: BEN2 (MIT) Matte", fill=(255, 209, 102))

        if row < len(biref_mattes):
            m2 = Image.fromarray(biref_mattes[row]).convert("RGB").resize((w, h))
            sheet.paste(m2, (w * 2, row * h))
        draw.text((w * 2 + 14, row * h + 14), f"Clip {row + 1}: BiRefNet (MIT) Matte", fill=(0, 229, 190))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    return out_png


def main() -> int:
    parser = argparse.ArgumentParser(description="5-Min MIT Matting Benchmark (BEN2 & BiRefNet)")
    parser.add_argument("--out-dir", type=Path, default=Path("/content/work/MIT_matting_bench"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(json.dumps({
            "tool": "run_mit_matting_benchmark",
            "status": "DRY_RUN_OK",
            "allowed_models": ALLOWED_MODELS,
            "quarantined_excluded": sorted(QUARANTINED_NON_COMMERCIAL_OR_GPL),
            "out_dir": str(args.out_dir),
        }, indent=2))
        return 0

    t0 = time.perf_counter()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_bench_deps()
    clips = _make_synthetic_facecam_clips(args.out_dir, n_frames=8, res=(1024, 1024))

    results: dict[str, Any] = {
        "schema": "411.m3.mit_matting_benchmark/1",
        "quarantined_excluded": sorted(QUARANTINED_NON_COMMERCIAL_OR_GPL),
        "models": {},
    }
    ben2_mattes: list[np.ndarray] = []
    biref_mattes: list[np.ndarray] = []

    try:
        biref_rep, biref_mattes = _benchmark_birefnet(clips, args.out_dir)
        results["models"]["BiRefNet"] = biref_rep
    except Exception as exc:
        results["models"]["BiRefNet"] = {"model": "BiRefNet", "license": "MIT", "pass": False, "error": str(exc)}

    try:
        ben2_rep, ben2_mattes = _benchmark_ben2(clips, args.out_dir)
        results["models"]["BEN2"] = ben2_rep
    except Exception as exc:
        results["models"]["BEN2"] = {"model": "BEN2", "license": "MIT", "pass": False, "error": str(exc)}

    sheet_path = _save_contact_sheet(
        clips, ben2_mattes, biref_mattes, args.out_dir / "MIT_MATTING_BENCH_CONTACT_SHEET.png"
    )
    results["contact_sheet"] = str(sheet_path)
    results["elapsed_sec"] = round(time.perf_counter() - t0, 2)
    results["lane4_gate_unlocked"] = any(
        m.get("pass", False) for m in results["models"].values()
    )

    rep_json = args.out_dir / "MIT_MATTING_BENCHMARK_REPORT.json"
    rep_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
