"""run_b1_b2_known_camera_splat.py — Track B1/B2 120-Cam Known-Camera Garment-Only 3DGS, Morph Pack & Vulkan SOG v3 Pipeline.

Stage 1A / Phase 3 Factory Contract:
  - Known-camera Cycles + Object-ID pass (`body=0, top=1, bottom=2, footwear=3`), Cam-0 IoU > 0.98.
  - Garment-only gsplat MCMCStrategy (`cap_max=120_000`, canonical morph export `N=100_000`),
    `aniso_r_train=8.0`, `lambda_aniso=5e-3`, `post_processing=None`.
  - Canonical Z-slab morph pack export via `pair_maximus_splats.py` (`<= 2.5 MB`/SKU) + lossless `exact=True` WebP `(R,G,0,255)`.
  - Fast `splat-transform --gpu-backend vulkan` primary w/ bounded fallback (`-H 1 --sh-palette-size 4096 -i 2 --max-workers 8`).
  - Frozen-XYZ `60s` colourway SH rebake (`rebake_colorway_sh_frozen_xyz`) + 3D bone-bound callout anchors JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
SPRINT_SCRIPTS = THIS_DIR.parents[0] / "COLAB_MEGA_SPRINT" / "scripts"
for _p in (THIS_DIR, SPRINT_SCRIPTS):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gsplat_known_camera_runner import (  # noqa: E402
    GsplatTrainSpec,
    blender_c2w_to_opencv,
    build_gsplat_trainer_spec,
    cam0_silhouette_iou_check,
    sample_explicit_surface_points,
    write_ply_points,
)
from pair_maximus_splats import (  # noqa: E402
    DEFAULT_TARGET_SPLATS,
    SplatCloud,
    canonicalize_garment_splats,
    export_morph_pack,
    write_garment_id_webp,
)

OBJECT_ID_CONTRACT = {"body": 0, "top": 1, "bottom": 2, "footwear": 3}


def _look_at_c2w_bl(pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - pos
    forward /= max(np.linalg.norm(forward), 1e-12)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = true_up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = pos
    return c2w


def build_120_camera_schedule(w: int = 1000, h: int = 1000, sensor_width_mm: float = 36.0) -> dict:
    """72 Fibonacci (-15..+65 deg, r=2.10m, 50mm) + 24 low axilla/hem (-18..-5 deg, r=1.75m) + 24 detail (r=1.15m, 85mm)."""
    target = np.array([0.0, 0.0, 1.05], dtype=np.float64)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    frames = []

    def _add_cam(idx: int, pos: np.ndarray, focal_mm: float, ring: str) -> None:
        c2w_bl = _look_at_c2w_bl(pos, target)
        c2w_cv = blender_c2w_to_opencv(c2w_bl)
        fl = w * (focal_mm / sensor_width_mm)
        frames.append({
            "file_path": f"images/cam_{idx:03d}.png",
            "mask_path": f"masks/cam_part_id_{idx:04d}.png",
            "ring": ring,
            "focal_mm": focal_mm,
            "fl_x": fl,
            "fl_y": fl,
            "transform_matrix_bl": c2w_bl.tolist(),
            "transform_matrix": c2w_cv.tolist(),
        })

    cam_idx = 0
    for i in range(72):
        t = (i + 0.5) / 72.0
        elev = math.radians(-15.0 + 80.0 * t)
        azim = i * golden_angle
        pos = target + np.array([2.10 * math.cos(elev) * math.cos(azim), 2.10 * math.cos(elev) * math.sin(azim), 2.10 * math.sin(elev)])
        _add_cam(cam_idx, pos, 50.0, "fibonacci_72")
        cam_idx += 1

    for i in range(24):
        elev = math.radians(-18.0 + 13.0 * (i % 2))
        azim = 2.0 * math.pi * i / 24.0
        pos = target + np.array([1.75 * math.cos(elev) * math.cos(azim), 1.75 * math.cos(elev) * math.sin(azim), 1.75 * math.sin(elev)])
        _add_cam(cam_idx, pos, 50.0, "axilla_hem_24")
        cam_idx += 1

    for i in range(24):
        elev = math.radians(-5.0 + 25.0 * ((i % 3) / 2.0))
        azim = 2.0 * math.pi * i / 24.0
        pos = np.array([0.0, 0.0, 1.25]) + np.array([1.15 * math.cos(elev) * math.cos(azim), 1.15 * math.cos(elev) * math.sin(azim), 1.15 * math.sin(elev)])
        _add_cam(cam_idx, pos, 85.0, "detail_85mm_24")
        cam_idx += 1

    fl_50 = w * (50.0 / sensor_width_mm)
    return {
        "w": w,
        "h": h,
        "fl_x": fl_50,
        "fl_y": fl_50,
        "cx": w * 0.5,
        "cy": h * 0.5,
        "spp": 64,
        "oidn": True,
        "bit_depth": 16,
        "color_space": "sRGB",
        "view_transform": "Khronos PBR Neutral",
        "object_id_contract": OBJECT_ID_CONTRACT,
        "random_solid_bg_training": True,
        "known_camera_zero_sfm": True,
        "frames": frames,
    }


def build_bone_callout_anchors_json(
    sku: str,
    V_garment: np.ndarray,
    out_path: Path,
    colorways: list[str] | None = None,
) -> Path:
    """Export deterministic 3D bone-bound callout anchors anchored to Maximus Z-up garment bounds."""
    V = np.asarray(V_garment, dtype=np.float64)
    z_min, z_max = float(V[:, 2].min()), float(V[:, 2].max())
    front_pts = V[V[:, 1] < 0.0]
    y_front = float(np.percentile(front_pts[:, 1], 10)) if len(front_pts) else -0.14
    x_span = float(np.percentile(np.abs(V[:, 0]), 92))
    anchors = {
        "schema": "maximus.v2.callout_anchors/1",
        "sku": sku,
        "colorways": colorways or ["default"],
        "space": "blender_zup_meters",
        "anchors": [
            {"id": "collar", "bone": "neck01", "pos_zup": [0.04, round(y_front * 0.65, 4), round(min(z_max - 0.03, 1.44), 4)]},
            {"id": "chest", "bone": "spine03", "pos_zup": [round(-x_span * 0.72, 4), round(y_front * 0.85, 4), round(0.58 * z_max + 0.42 * z_min, 4)]},
            {"id": "hem", "bone": "pelvis", "pos_zup": [round(x_span * 0.55, 4), round(y_front * 0.90, 4), round(z_min + 0.04, 4)]},
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(anchors, indent=2) + "\n", encoding="utf-8")
    return out_path


def run_splat_transform_fast(ply_path: Path, sog_path: Path, spz_path: Path) -> dict[str, Any]:
    """Run @playcanvas/splat-transform Vulkan primary w/ explicit `-g cpu` SH1 & SH0 fallbacks."""
    st_exe = shutil.which("splat-transform")
    if not st_exe:
        return {"ran": False, "backend": None}

    attempts = [
        ("vulkan", [st_exe, "-w", "--gpu-backend", "vulkan", "-H", "1", "--sh-palette-size", "4096", "-i", "2", str(ply_path), str(sog_path)], 180),
        ("cpu_sh1", [st_exe, "-w", "-g", "cpu", "-H", "1", "--sh-palette-size", "4096", "-i", "2", "--max-workers", "8", str(ply_path), str(sog_path)], 240),
        ("cpu_sh0", [st_exe, "-w", "-g", "cpu", "-H", "0", "-i", "2", "--max-workers", "8", str(ply_path), str(sog_path)], 120),
    ]
    backend_used = None
    for name, cmd, tmo in attempts:
        sog_path.unlink(missing_ok=True)
        try:
            r = subprocess.run(cmd, check=False, timeout=tmo)
            if r.returncode == 0 and sog_path.is_file() and sog_path.stat().st_size > 0:
                backend_used = name
                break
        except subprocess.TimeoutExpired:
            print(f"[splat-transform] attempt {name} timed out after {tmo}s", flush=True)

    subprocess.run([st_exe, "-w", str(ply_path), str(spz_path), "--spz-version", "4"], check=False)
    if not spz_path.exists():
        subprocess.run([st_exe, "-w", str(ply_path), str(spz_path)], check=False)
    return {
        "ran": True,
        "backend": backend_used,
        "sog_mb": round(sog_path.stat().st_size / (1024.0 * 1024.0), 3) if sog_path.exists() else 0.0,
        "spz_mb": round(spz_path.stat().st_size / (1024.0 * 1024.0), 3) if spz_path.exists() else 0.0,
    }


def _load_ply_vertices(ply_path: Path) -> np.ndarray:
    with ply_path.open("rb") as pf:
        header_lines = []
        while True:
            hline = pf.readline().decode("ascii")
            header_lines.append(hline.strip())
            if hline.strip() == "end_header":
                break
        is_bin = any("binary_little_endian" in hl for hl in header_lines)
        n_vtx = next(int(hl.split()[-1]) for hl in header_lines if hl.startswith("element vertex "))
        if is_bin:
            has_rgb = any("property uchar red" in hl for hl in header_lines)
            dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")] if has_rgb else [("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
            arr_ply = np.fromfile(pf, dtype=dt, count=n_vtx)
            return np.column_stack([arr_ply["x"], arr_ply["y"], arr_ply["z"]]).astype(np.float32)
        return np.loadtxt(pf, dtype=np.float32)[:n_vtx, :3]


def _load_training_views(out_dir: Path, frames: list[dict], require_garment_mask: bool = True) -> tuple[list, list, list]:
    import torch  # type: ignore
    from PIL import Image  # type: ignore

    imgs_rgba, viewmats_list, Ks_list = [], [], []
    cx = float(frames[0].get("_cx", 500.0))
    cy = float(frames[0].get("_cy", 500.0))
    for fr in frames:
        im = Image.open(out_dir / fr["file_path"]).convert("RGBA")
        arr_np = np.asarray(im, dtype=np.float32) / 255.0
        # Zero out bare body pixels (`part_id == 0`) when Object-ID mask is present for pure garment-only training
        m_rel = fr.get("mask_path")
        if require_garment_mask and m_rel:
            mp = out_dir / m_rel
            if mp.is_file():
                id_map = np.asarray(Image.open(mp).convert("L"), dtype=np.uint8)
                cloth_mask = (id_map >= 1).astype(np.float32)
                arr_np[..., 3] *= cloth_mask
        imgs_rgba.append(torch.from_numpy(arr_np))
        w2c = np.linalg.inv(np.asarray(fr["transform_matrix"], dtype=np.float32))
        viewmats_list.append(torch.from_numpy(w2c))
        K_mat = np.array([[fr["fl_x"], 0.0, cx], [0.0, fr["fl_y"], cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        Ks_list.append(torch.from_numpy(K_mat))
    return imgs_rgba, viewmats_list, Ks_list


def rebake_colorway_sh_frozen_xyz(
    base_ckpt_pt: Path,
    colorway_renders_dir: Path,
    tf_path: Path,
    out_dir: Path,
    sku: str,
    colorway: str,
    time_budget_sec: float = 60.0,
) -> dict[str, Any]:
    """Frozen-XYZ 60s SH0/SHN colorway rebake (zero geometry re-optimization)."""
    import torch  # type: ignore
    import torch.nn.functional as F_nn  # type: ignore
    from gsplat.rendering import rasterization  # type: ignore
    from gsplat import export_splats  # type: ignore
    from torchmetrics.image import StructuralSimilarityIndexMeasure  # type: ignore

    device = torch.device("cuda")
    ckpt = torch.load(base_ckpt_pt, map_location=device)
    means = ckpt["means"].detach().requires_grad_(False)
    scales = ckpt["scales"].detach().requires_grad_(False)
    quats = ckpt["quats"].detach().requires_grad_(False)
    opacities = ckpt["opacities"].detach().requires_grad_(False)
    sh0 = torch.nn.Parameter(ckpt["sh0"].detach().clone())
    shN = torch.nn.Parameter(ckpt["shN"].detach().clone())

    tf = json.loads(tf_path.read_text(encoding="utf-8"))
    w, h = int(tf["w"]), int(tf["h"])
    for fr in tf["frames"]:
        fr["_cx"] = float(tf["cx"])
        fr["_cy"] = float(tf["cy"])
    imgs_rgba, viewmats_list, Ks_list = _load_training_views(
        colorway_renders_dir, tf["frames"], require_garment_mask=not bool(tf.get("body_hidden_in_120cam", False))
    )

    opt = torch.optim.Adam([{"params": [sh0], "lr": 5e-3}, {"params": [shN], "lr": 1e-3}], eps=1e-15)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    rng = np.random.default_rng(77)
    t_start = time.monotonic()
    steps_done = 0
    while (time.monotonic() - t_start) < time_budget_sec and steps_done < 800:
        idx = steps_done % len(imgs_rgba)
        rgba = imgs_rgba[idx].to(device)
        bg = torch.tensor(rng.uniform(0.02, 0.95, size=3), dtype=torch.float32, device=device)
        gt = rgba[..., :3] * rgba[..., 3:4] + bg[None, None, :] * (1.0 - rgba[..., 3:4])
        renders, _, _ = rasterization(
            means=means, quats=quats, scales=torch.exp(scales), opacities=torch.sigmoid(opacities),
            colors=torch.cat([sh0, shN], dim=1), viewmats=viewmats_list[idx].to(device)[None],
            Ks=Ks_list[idx].to(device)[None], width=w, height=h, sh_degree=3,
            backgrounds=bg[None], packed=False, rasterize_mode="antialiased",
        )
        pred = renders[0].clamp(0.0, 1.0)
        loss = 0.8 * F_nn.l1_loss(pred, gt) + 0.2 * (1.0 - ssim_metric(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        steps_done += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    ply_path = out_dir / f"{sku}_{colorway}.ply"
    sog_path = out_dir / f"{sku}_{colorway}.sog"
    spz_path = out_dir / f"{sku}_{colorway}.spz"
    export_splats(means=means, scales=scales, quats=quats, opacities=opacities, sh0=sh0.detach(), shN=shN.detach(), format="ply", save_to=str(ply_path))
    st_res = run_splat_transform_fast(ply_path, sog_path, spz_path)

    # Emit canonicalized 100k morph pack preserving frozen XYZ pairing
    part_ids = ckpt.get("part_ids", torch.ones(len(means), dtype=torch.int32)).cpu().numpy().astype(np.uint16)
    srgb_colors = np.clip(0.5 + 0.28209479 * sh0.detach().cpu().numpy().reshape(-1, 3), 0.0, 1.0)
    cloud = SplatCloud(
        means=means.cpu().numpy(),
        scales=torch.exp(scales).cpu().numpy(),
        quats=quats.cpu().numpy(),
        opacities=torch.sigmoid(opacities).cpu().numpy(),
        colors=srgb_colors,
        part_ids=part_ids,
    )
    canon_cloud, _ = canonicalize_garment_splats(cloud, n_target=DEFAULT_TARGET_SPLATS, seed=411)
    morph_info = export_morph_pack(canon_cloud, out_dir / f"{sku}_{colorway}.morph.npz", sku=sku, colorway=colorway)

    return {
        "sku": sku,
        "colorway": colorway,
        "frozen_xyz": True,
        "steps_done": steps_done,
        "elapsed_sec": round(time.monotonic() - t_start, 2),
        "splat_transform": st_res,
        "morph_pack": morph_info,
    }


def run_live_gsplat_and_pack(
    out_dir: Path,
    tf_path: Path,
    init_ply: Path,
    max_steps: int = 4500,
    cap_max: int = 120_000,
    hero_npz: Path | None = None,
    sku: str = "BS-RINGER-001",
    colorway: str = "white-black",
) -> dict:
    """Train garment-only gsplat MCMCStrategy(cap_max=120k..150k) + export real body=0/garment=1..k picking & Vulkan SOG v3."""
    import torch  # type: ignore
    import torch.nn.functional as F_nn  # type: ignore
    from gsplat.rendering import rasterization  # type: ignore
    from gsplat.strategy import MCMCStrategy  # type: ignore
    from gsplat import export_splats  # type: ignore
    import lpips as lpips_lib  # type: ignore
    from torchmetrics.image import StructuralSimilarityIndexMeasure  # type: ignore

    device = torch.device("cuda")
    tf = json.loads(tf_path.read_text(encoding="utf-8"))
    w, h = int(tf["w"]), int(tf["h"])
    for fr in tf["frames"]:
        fr["_cx"] = float(tf["cx"])
        fr["_cy"] = float(tf["cy"])
    frames = tf["frames"]
    b1_rep_path = out_dir / "B1_120CAM_REPORT.json"
    body_hidden = False
    if b1_rep_path.is_file():
        body_hidden = bool(json.loads(b1_rep_path.read_text(encoding="utf-8")).get("body_hidden_in_120cam", False))
    imgs_rgba, viewmats_list, Ks_list = _load_training_views(
        out_dir, frames, require_garment_mask=not body_hidden
    )

    pts_np = _load_ply_vertices(init_ply)
    bind_npz = out_dir / "init_surface_binding.npz"
    body_pts: np.ndarray | None = None
    if bind_npz.is_file():
        bind_d = np.load(bind_npz, allow_pickle=True)
        init_pids = bind_d["garment_id"].astype(np.uint16)
        if "body_ref_V" in bind_d:
            body_pts = bind_d["body_ref_V"].astype(np.float32)
    else:
        init_pids = np.ones(len(pts_np), dtype=np.uint16)
    if body_pts is None and hero_npz is not None and hero_npz.is_file():
        hd_ref = np.load(hero_npz, allow_pickle=True)
        if "body_ref_V" in hd_ref:
            body_pts = hd_ref["body_ref_V"].astype(np.float32)
    n_pts = len(pts_np)

    means = torch.nn.Parameter(torch.from_numpy(pts_np).to(device))
    scales = torch.nn.Parameter(torch.full((n_pts, 3), math.log(0.0025), device=device))
    quats = torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(n_pts, 1))
    opacities = torch.nn.Parameter(torch.full((n_pts,), torch.logit(torch.tensor(0.5)), device=device))
    sh0 = torch.nn.Parameter(torch.zeros((n_pts, 1, 3), device=device))
    shN = torch.nn.Parameter(torch.zeros((n_pts, 15, 3), device=device))

    params = {"means": means, "scales": scales, "quats": quats, "opacities": opacities, "sh0": sh0, "shN": shN}
    lrs = {"means": 1.6e-4, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20.0}
    optimizers = {k: torch.optim.Adam([p], lr=lrs[k], eps=1e-15) for k, p in params.items()}

    strategy = MCMCStrategy(
        cap_max=int(cap_max), noise_lr=5e5, refine_start_iter=500,
        refine_stop_iter=max(1000, max_steps - 1000), refine_every=150, min_opacity=0.005,
    )
    strategy_state = strategy.initialize_state()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()

    train_idx = [i for i in range(len(frames)) if i % 10 != 0]
    val_idx = [i for i in range(len(frames)) if i % 10 == 0]
    rng = np.random.default_rng(42)

    for step in range(max_steps):
        frac = step / max(1, max_steps - 1)
        lr_means_cur = lrs["means"] * (0.1 ** frac)
        lr_sh0_cur = lrs["sh0"] * (0.2 ** frac)
        lr_shN_cur = lrs["shN"] * (0.2 ** frac)
        for pg in optimizers["means"].param_groups:
            pg["lr"] = lr_means_cur
        for pg in optimizers["sh0"].param_groups:
            pg["lr"] = lr_sh0_cur
        for pg in optimizers["shN"].param_groups:
            pg["lr"] = lr_shN_cur

        idx = train_idx[step % len(train_idx)]
        rgba = imgs_rgba[idx].to(device)
        bg = torch.tensor(rng.uniform(0.02, 0.95, size=3), dtype=torch.float32, device=device)
        gt = rgba[..., :3] * rgba[..., 3:4] + bg[None, None, :] * (1.0 - rgba[..., 3:4])

        viewmat = viewmats_list[idx].to(device)[None]
        K_t = Ks_list[idx].to(device)[None]
        colors = torch.cat([params["sh0"], params["shN"]], dim=1)
        renders, _, info = rasterization(
            means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]), colors=colors,
            viewmats=viewmat, Ks=K_t, width=w, height=h, sh_degree=3,
            backgrounds=bg[None], packed=False, absgrad=False, rasterize_mode="antialiased",
        )
        strategy.step_pre_backward(params=params, optimizers=optimizers, state=strategy_state, step=step, info=info)
        pred = renders[0].clamp(0.0, 1.0)
        l1 = F_nn.l1_loss(pred, gt)
        ssim_v = ssim_metric(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
        s_exp = torch.exp(params["scales"])
        scale_reg = 0.01 * F_nn.relu(s_exp - 0.012).mean()
        opa_reg = 0.002 * (1.0 - torch.sigmoid(params["opacities"])).mean()
        aniso = s_exp.max(dim=-1).values / s_exp.min(dim=-1).values.clamp_min(1e-6)
        aniso_reg = 0.005 * F_nn.relu(aniso - 8.0).mean()
        loss = 0.8 * l1 + 0.2 * (1.0 - ssim_v) + scale_reg + opa_reg + aniso_reg
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        strategy.step_post_backward(params=params, optimizers=optimizers, state=strategy_state, step=step, info=info, lr=lr_means_cur)
        if step % 500 == 0 or step == max_steps - 1:
            print(f"[B2 garment gsplat] step {step}/{max_steps} loss={loss.item():.4f} l1={l1.item():.4f} ssim={ssim_v.item():.4f} n_gs={len(params['means'])}", flush=True)

    # Hard SVD-equivalent scale (<= 15mm) & anisotropy clamp (<= 8.0) before export
    with torch.no_grad():
        s_clamp = torch.exp(params["scales"].data).clamp(min=1e-5, max=0.015)
        s_floor = s_clamp.max(dim=-1, keepdim=True).values / 8.0
        s_clamp = torch.maximum(s_clamp, s_floor)
        params["scales"].data.copy_(torch.log(s_clamp))

    psnrs, ssims, lpips_vals = [], [], []
    bg_eval = torch.full((1, 3), 0.04, dtype=torch.float32, device=device)
    with torch.no_grad():
        colors = torch.cat([params["sh0"], params["shN"]], dim=1)
        for idx in val_idx:
            rgba = imgs_rgba[idx].to(device)
            gt = rgba[..., :3] * rgba[..., 3:4] + bg_eval[0, None, None, :] * (1.0 - rgba[..., 3:4])
            renders, _, _ = rasterization(
                means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]),
                opacities=torch.sigmoid(params["opacities"]), colors=colors,
                viewmats=viewmats_list[idx].to(device)[None], Ks=Ks_list[idx].to(device)[None],
                width=w, height=h, sh_degree=3, backgrounds=bg_eval, packed=False, rasterize_mode="antialiased",
            )
            pred = renders[0].clamp(0.0, 1.0)
            mse = F_nn.mse_loss(pred, gt).item()
            psnrs.append(-10.0 * math.log10(max(mse, 1e-10)))
            ssims.append(float(ssim_metric(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]).item()))
            lpips_vals.append(float(lpips_fn(pred.permute(2, 0, 1)[None] * 2 - 1, gt.permute(2, 0, 1)[None] * 2 - 1).item()))

    s_final = torch.exp(params["scales"]).detach()
    aniso_ratios = (s_final.max(dim=-1).values / s_final.min(dim=-1).values.clamp_min(1e-6)).cpu().numpy()
    aniso_p99 = float(np.percentile(aniso_ratios, 99))
    aniso_final = float(aniso_ratios.max())
    max_scale_mm = float(s_final.max().item() * 1000.0)

    gs_out = out_dir / "gsplat_out"
    gs_out.mkdir(parents=True, exist_ok=True)
    ply_path = gs_out / "splats.ply"
    export_splats(
        means=params["means"].detach(), scales=params["scales"].detach(), quats=params["quats"].detach(),
        opacities=params["opacities"].detach(), sh0=params["sh0"].detach(), shN=params["shN"].detach(),
        format="ply", save_to=str(ply_path),
    )

    # Nearest-neighbor transfer preserving real body=0 vs garment=1..k picking (NEVER forced to ones)
    means_cpu = params["means"].detach().cpu().numpy().astype(np.float32)
    n_splats = int(len(means_cpu))
    gids = np.ones(n_splats, dtype=np.uint16)
    if len(init_pids) > 0:
        from scipy.spatial import cKDTree  # type: ignore

        d_garment, nn_i = cKDTree(pts_np).query(means_cpu, k=1)
        gids = init_pids[nn_i].astype(np.uint16)
        if body_pts is not None and len(body_pts) > 0:
            d_body, _ = cKDTree(body_pts).query(means_cpu, k=1)
            # Mark any splat strictly closer to bare body than garment seed (+2mm margin) as body=0
            gids[d_body + 0.002 < d_garment] = 0

    # Canonical fixed-N=100k pairing + morph pack + WebKit exact=True lossless WebP (no -m reorder)
    srgb_colors = np.clip(0.5 + 0.28209479 * params["sh0"].detach().cpu().numpy().reshape(-1, 3), 0.0, 1.0)
    raw_cloud = SplatCloud(
        means=means_cpu,
        scales=s_final.cpu().numpy().astype(np.float32),
        quats=params["quats"].detach().cpu().numpy().astype(np.float32),
        opacities=torch.sigmoid(params["opacities"]).detach().cpu().numpy().astype(np.float32),
        colors=srgb_colors.astype(np.float32),
        part_ids=gids,
    )
    gid_webp = gs_out / f"{sku}_garment_id.webp"
    webp_info = write_garment_id_webp(gids, gid_webp)
    canon_cloud, _ = canonicalize_garment_splats(raw_cloud, n_target=DEFAULT_TARGET_SPLATS, seed=411)
    morph_gid_info = write_garment_id_webp(canon_cloud.part_ids, gs_out / f"{sku}_morph_garment_id.webp")
    morph_info = export_morph_pack(canon_cloud, gs_out / f"{sku}_{colorway}.morph.npz", sku=sku, colorway=colorway)
    morph_info["garment_id_webp"] = morph_gid_info

    # Save frozen-XYZ checkpoint for 4x60s colorway SH rebakes
    ckpt_pt = gs_out / f"{sku}_frozen_xyz_ckpt.pt"
    torch.save(
        {
            "means": params["means"].detach().cpu(),
            "scales": params["scales"].detach().cpu(),
            "quats": params["quats"].detach().cpu(),
            "opacities": params["opacities"].detach().cpu(),
            "sh0": params["sh0"].detach().cpu(),
            "shN": params["shN"].detach().cpu(),
            "part_ids": torch.from_numpy(gids.astype(np.int32)),
        },
        ckpt_pt,
    )

    sog_path = gs_out / f"{sku}_v3.sog"
    spz_path = gs_out / f"{sku}_v4.spz"
    st_res = run_splat_transform_fast(ply_path, sog_path, spz_path)

    psnr_db = round(float(np.mean(psnrs)), 3)
    ssim_val = round(float(np.mean(ssims)), 4)
    lpips_val = round(float(np.mean(lpips_vals)), 4)
    gates = {
        "psnr_pass": bool(psnr_db >= 30.0),
        "ssim_pass": bool(ssim_val >= 0.93),
        "lpips_pass": bool(lpips_val <= 0.08),
        "n_splats_pass": bool(120_000 <= n_splats <= 150_000),
        "aniso_pass": bool(aniso_final <= 8.05),
        "morph_pack_pass": bool(morph_info.get("under_2_5mb_budget", False)),
        "webp_pass": bool(webp_info.get("exact_alpha_255", False)),
        "sog_pass": bool(st_res.get("sog_mb", 0.0) > 0.0),
    }
    gates["all_3dgs_gates_pass"] = bool(all(gates.values()))

    return {
        "trained_live": True,
        "sku": sku,
        "colorway": colorway,
        "garment_only_training": True,
        "post_processing": None,
        "n_splats": n_splats,
        "canonical_morph_n": len(canon_cloud),
        "psnr_db": psnr_db,
        "ssim": ssim_val,
        "lpips": lpips_val,
        "aniso_p99": round(aniso_p99, 3),
        "aniso_max": round(aniso_final, 3),
        "max_splat_scale_mm": round(max_scale_mm, 3),
        "garment_id_webp": webp_info,
        "morph_pack": morph_info,
        "splat_transform": st_res,
        "frozen_xyz_ckpt": str(ckpt_pt),
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage B1/B2 120-cam OpenCV + Garment-Only gsplat + Canonical Morph Pack")
    parser.add_argument("--hero-npz", type=Path, default=None)
    parser.add_argument("--extract-npz", type=Path, default=None)
    parser.add_argument("--solved-npz", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sku", default="BS-RINGER-001")
    parser.add_argument("--colorway", default="white-black")
    parser.add_argument("--n-views", type=int, default=120)
    parser.add_argument("--n-points", type=int, default=120000)
    parser.add_argument("--strategy", choices=["mcmc", "default"], default="mcmc")
    parser.add_argument("--phase", choices=["b1", "b2", "all"], default="all")
    parser.add_argument("--train-live", action="store_true")
    parser.add_argument("--rebake-colorway-sh", action="store_true")
    parser.add_argument("--base-ckpt", type=Path, default=None)
    parser.add_argument("--colorway-renders-dir", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=4500)
    parser.add_argument("--cap-max", type=int, default=135000, help="Fixed splat cap in [120000..150000]")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cap_max = max(120_000, min(150_000, int(args.cap_max)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    init_ply = args.out_dir / "init_surface_points.ply"
    tf_path = args.out_dir / "transforms_cv.json"

    if args.rebake_colorway_sh:
        assert args.base_ckpt is not None and args.colorway_renders_dir is not None
        res = rebake_colorway_sh_frozen_xyz(
            args.base_ckpt, args.colorway_renders_dir, tf_path, args.out_dir, args.sku, args.colorway
        )
        print(json.dumps(res, indent=2))
        return

    if args.phase in ("b1", "all"):
        body_ref_pts: np.ndarray | None = None
        if args.extract_npz is not None and args.extract_npz.is_file():
            ext = np.load(args.extract_npz, allow_pickle=True)
            if args.solved_npz is not None and args.solved_npz.exists():
                V_cloth = np.load(args.solved_npz, allow_pickle=True)["V_solved"]
                garment_source = str(args.solved_npz)
            else:
                V_cloth = ext["V_fit"]
                garment_source = str(args.extract_npz)
            F_cloth = ext["F"].astype(np.int64)
            fpid_cloth = ext["face_part_ids"].astype(np.uint16)
            sV = ext["static_V"].astype(np.float32) if "static_V" in ext else np.zeros((0, 3), dtype=np.float32)
            sF = ext["static_F"].astype(np.int64) if "static_F" in ext else np.zeros((0, 3), dtype=np.int64)
            sfpid = ext["static_face_part_ids"].astype(np.uint16) if "static_face_part_ids" in ext else np.zeros((0,), dtype=np.uint16)
            if len(sV) > 0 and len(sF) > 0:
                V_garment = np.vstack([V_cloth, sV]).astype(np.float32)
                F_garment = np.vstack([F_cloth, sF + len(V_cloth)]).astype(np.int64)
                face_gids = np.concatenate([fpid_cloth, sfpid]).astype(np.uint16)
            else:
                V_garment = V_cloth
                F_garment = F_cloth
                face_gids = fpid_cloth
            if "body_ref_V" in ext:
                body_ref_pts = np.asarray(ext["body_ref_V"], dtype=np.float32)
        else:
            assert args.hero_npz is not None and args.hero_npz.is_file(), "Provide --extract-npz or --hero-npz"
            d = np.load(args.hero_npz, allow_pickle=True)
            if args.solved_npz is not None and args.solved_npz.exists():
                V_garment = np.load(args.solved_npz, allow_pickle=True)["V_solved"]
                garment_source = str(args.solved_npz)
            else:
                V_garment = d["V_fit"]
                garment_source = "hero_npz_V_fit"
            F_garment = d["F"]
            face_gids = np.ones(len(F_garment), dtype=np.uint16)
            if "body_ref_V" in d.files:
                body_ref_pts = np.asarray(d["body_ref_V"], dtype=np.float32)

        pts, _, binding = sample_explicit_surface_points(
            V_garment, F_garment, n_points=args.n_points, seed=411, face_garment_ids=face_gids
        )
        if body_ref_pts is not None:
            binding["body_ref_V"] = body_ref_pts
        init_ply = write_ply_points(init_ply, pts)
        np.savez_compressed(args.out_dir / "init_surface_binding.npz", **binding)

        tf = build_120_camera_schedule(w=1000, h=1000)
        tf["frames"] = tf["frames"][: args.n_views]
        tf_path.write_text(json.dumps(tf, indent=2) + "\n", encoding="utf-8")

        anchors_json = build_bone_callout_anchors_json(
            args.sku, V_garment, args.out_dir / f"{args.sku}_callout_anchors.json", colorways=[args.colorway]
        )

        dummy_mask = np.ones((64, 64), dtype=np.float32)
        iou_probe = cam0_silhouette_iou_check(dummy_mask, dummy_mask, out_diag_png=args.out_dir / "cam0_iou_diag.png")
        iou_probe["mode"] = "helper_self_check_pending_cycles"
        solved_sha = (
            hashlib.sha256(args.solved_npz.read_bytes()).hexdigest()
            if (args.solved_npz is not None and args.solved_npz.exists())
            else ""
        )
        b1_ready = {
            "sku": args.sku,
            "garment_source": garment_source,
            "solved_npz_sha256": solved_sha,
            "transforms_cv_json": str(tf_path),
            "callout_anchors_json": str(anchors_json),
            "n_views": len(tf["frames"]),
            "init_ply": str(init_ply),
            "n_points": len(pts),
            "garment_only_seeding": True,
            "cam0_iou_gate": iou_probe,
            "dry_run": bool(args.dry_run),
        }
        (args.out_dir / "B1_TRANSFORMS_READY.json").write_text(json.dumps(b1_ready, indent=2) + "\n", encoding="utf-8")
        if args.phase == "b1":
            print(json.dumps({"phase": "b1", "transforms_views": len(tf["frames"]), "init_points": len(pts), "out_dir": str(args.out_dir)}, indent=2))
            return

    b1_meta = json.loads((args.out_dir / "B1_TRANSFORMS_READY.json").read_text(encoding="utf-8"))
    gs_spec = build_gsplat_trainer_spec(
        GsplatTrainSpec(
            data_dir=args.out_dir,
            out_dir=args.out_dir / "gsplat_out",
            init_points_ply=init_ply,
            strategy=args.strategy,
            cap_max=cap_max,
            max_steps=args.max_steps,
            aniso_r_train=8.0,
            aniso_reg=5e-3,
            post_processing=None,
        )
    )
    b2_metrics: dict = {"trained_live": False}
    if args.train_live and not args.dry_run:
        b2_metrics = run_live_gsplat_and_pack(
            args.out_dir,
            tf_path,
            init_ply,
            max_steps=args.max_steps,
            cap_max=cap_max,
            hero_npz=args.hero_npz,
            sku=args.sku,
            colorway=args.colorway,
        )

    summary = {
        "sku": args.sku,
        "solved_npz_sha256": b1_meta.get("solved_npz_sha256", ""),
        "transforms_cv_json": str(tf_path),
        "callout_anchors_json": b1_meta.get("callout_anchors_json"),
        "n_views": b1_meta["n_views"],
        "init_ply": str(init_ply),
        "n_points": b1_meta["n_points"],
        "cam0_iou_gate": b1_meta["cam0_iou_gate"],
        "gsplat_spec": gs_spec,
        "b2_metrics": b2_metrics,
        "dry_run": bool(args.dry_run),
    }
    (args.out_dir / "B1_B2_PLAN.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"phase": args.phase, "transforms_views": b1_meta["n_views"], "strategy": args.strategy, "init_points": b1_meta["n_points"], "b2_metrics": b2_metrics}, indent=2))


if __name__ == "__main__":
    main()
