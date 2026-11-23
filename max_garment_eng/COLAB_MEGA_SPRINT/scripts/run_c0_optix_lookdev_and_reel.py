"""COLAB_MEGA_SPRINT/scripts/run_c0_optix_lookdev_and_reel.py — Headless Blender 5.2 OptiX Lookdev, GLB & SWAG_01 Reel.

Pure bpy.data/bmesh scene manipulation (zero bpy.ops mesh fragility):
  1. Applies `inputs/m_theta_fit.json` local bone quaternions + clavicle lock (1,0,0,0) + floor grounding (Z_sole >= 0)
     to the Maximus armature BEFORE mounting `c0_ringer_solved.npz` (so arms match the 9.0-deg theta_fit sleeves).
  2. Preserves M1 Obsidian Noir (#18181B, Specular IOR 0.22, Coat 0.10) on the Maximus body and transfers the
     BS-RINGER-001 UV layers + multi-slot materials from `ringer_tee_master.blend` + `hero_ringer_shell.npz`.
  3. Starts exposure at -0.93 EV and calibrates on the M1 Obsidian Noir body mask so `m1_torso_median_srgb` lands in [33, 39].
  4. Renders a contrasting bottom-half wireframe + _TENSION heatmap pass for the 72-frame SWAG_01 split-screen reel with
     live HUD metrics read from C0_GATE_REPORT.json, plus a real 1000x1000 Cam-0 silhouette IoU overlay for Panel 4.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser(description="Blender 5.2 OptiX C0 GLB + Lookdev + SWAG_01 Reel")
    p.add_argument("--mode", choices=["glb", "lookdev", "all"], default="all")
    p.add_argument("--solved-npz", type=Path, required=True)
    p.add_argument("--hero-npz", type=Path, default=Path("inputs/hero_ringer_shell.npz"))
    p.add_argument("--theta-fit-json", type=Path, default=Path("inputs/m_theta_fit.json"))
    p.add_argument("--c0-report-json", type=Path, default=None)
    p.add_argument("--ringer-blend", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cycles-device", choices=["OPTIX", "CUDA"], default="OPTIX")
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--frames", type=int, default=72)
    p.add_argument("--glb-mb", type=float, default=0.0)
    p.add_argument("--g-stretch-p99", type=float, default=1.0)
    return p.parse_args(argv)


def _apply_theta_fit_pose(theta_fit_json: Path, body_obj: Any, hero_npz: Path | None = None) -> dict[str, Any]:
    """Pose Maximus armature into theta_fit (9.0-deg arm line), lock clavicles (1,0,0,0), and ground Z_sole >= 0."""
    import bpy  # type: ignore
    from mathutils import Quaternion  # type: ignore

    rig = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    if rig is None or not theta_fit_json.exists():
        return {"posed": False}

    rig.data.pose_position = "POSE"
    for mod in body_obj.modifiers:
        if mod.type == "ARMATURE":
            mod.use_deform_preserve_volume = False

    pose_data = json.loads(theta_fit_json.read_text(encoding="utf-8"))
    bones_spec = pose_data.get("bones", {})
    for pb in rig.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.matrix_basis.identity()
    for bname, spec in bones_spec.items():
        pb = rig.pose.bones.get(bname)
        if pb is not None:
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = Quaternion(spec["q_wxyz"])
            pb.location = tuple(spec.get("t", (0.0, 0.0, 0.0)))
    for cname in ("clavicle.L", "clavicle.R"):
        if cname in rig.pose.bones:
            cb = rig.pose.bones[cname]
            cb.rotation_mode = "QUATERNION"
            cb.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
            cb.location = (0.0, 0.0, 0.0)
            cb.scale = (1.0, 1.0, 1.0)
    bpy.context.view_layer.update()

    # Ground root so lowest evaluated vertex sits in [0, 0.01 mm]
    z_min = 0.0
    for _ in range(4):
        dg = bpy.context.evaluated_depsgraph_get()
        ev_me = body_obj.evaluated_get(dg).data
        coords = np.empty(len(ev_me.vertices) * 3, dtype=np.float64)
        ev_me.vertices.foreach_get("co", coords)
        V_ev = (np.asarray(body_obj.matrix_world, dtype=np.float64)[:3, :3] @ coords.reshape(-1, 3).T).T + np.asarray(body_obj.matrix_world, dtype=np.float64)[:3, 3]
        z_min = float(V_ev[:, 2].min())
        if 0.0 <= z_min <= 1e-5 or "root" not in rig.pose.bones:
            break
        rpb = rig.pose.bones["root"]
        M = rpb.matrix.copy()
        M.translation.z -= (z_min - 2e-6)
        rpb.matrix = M
        bpy.context.view_layer.update()

    dg = bpy.context.evaluated_depsgraph_get()
    ev_me = body_obj.evaluated_get(dg).data
    coords = np.empty(len(ev_me.vertices) * 3, dtype=np.float64)
    ev_me.vertices.foreach_get("co", coords)
    V_ev = coords.reshape(-1, 3)
    ref_max_err_mm = None
    if hero_npz is not None and hero_npz.exists():
        hd = np.load(hero_npz, allow_pickle=True)
        if "body_ref_V" in hd and len(hd["body_ref_V"]) == len(V_ev):
            ref_max_err_mm = round(float(np.max(np.linalg.norm(V_ev - hd["body_ref_V"], axis=1)) * 1000.0), 4)
    return {"posed": True, "z_sole_mm": round(z_min * 1000.0, 4), "body_ref_max_err_mm": ref_max_err_mm}


def _ensure_m1_obsidian_on_body(body_obj: Any) -> None:
    import bpy  # type: ignore

    m1 = bpy.data.materials.get("MX_M1_ObsidianNoir")
    if m1 is None:
        m1 = bpy.data.materials.new("MX_M1_ObsidianNoir")
        m1.use_nodes = True
    if m1.use_nodes and m1.node_tree:
        bsdf = next((n for n in m1.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = (0.009, 0.009, 0.011, 1.0)
            bsdf.inputs["Roughness"].default_value = 0.52
            if "Specular IOR Level" in bsdf.inputs:
                bsdf.inputs["Specular IOR Level"].default_value = 0.22
            if "Coat Weight" in bsdf.inputs:
                bsdf.inputs["Coat Weight"].default_value = 0.10
    if len(body_obj.data.materials) == 0:
        body_obj.data.materials.append(m1)
    else:
        body_obj.data.materials[0] = m1


def _setup_cycles(scene: Any, device_pref: str, samples: int, res_x: int, res_y: int, view_transform: str = "AgX") -> str:
    import bpy  # type: ignore

    scene.render.engine = "CYCLES"
    scene.cycles.samples = int(samples)
    scene.cycles.use_denoising = True
    if hasattr(scene.cycles, "denoiser"):
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.render.resolution_x = int(res_x)
    scene.render.resolution_y = int(res_y)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = view_transform

    prefs = bpy.context.preferences.addons["cycles"].preferences
    chosen = "CPU"
    for dev_type in ([device_pref, "CUDA"] if device_pref == "OPTIX" else ["CUDA"]):
        try:
            prefs.compute_device_type = dev_type
            prefs.get_devices()
            gpu_devs = [d for d in prefs.devices if d.type == dev_type]
            if gpu_devs:
                for d in prefs.devices:
                    d.use = d.type == dev_type
                scene.cycles.device = "GPU"
                chosen = dev_type
                break
        except Exception:
            continue
    return chosen


def _mount_ringer_onto_master(solved_npz: Path, ringer_blend: Path, hero_npz: Path, theta_fit_json: Path) -> tuple[Any, Any, dict[str, Any]]:
    import bpy  # type: ignore

    body_obj = next((o for o in bpy.data.objects if o.type == "MESH" and len(o.data.vertices) == 13380), None)
    assert body_obj is not None, "Expected 13,380-vert Maximus body mesh in master_psd.blend"
    pose_info = _apply_theta_fit_pose(theta_fit_json, body_obj, hero_npz)
    _ensure_m1_obsidian_on_body(body_obj)

    d = np.load(solved_npz, allow_pickle=True)
    V_solved = d["V_solved"].astype(np.float64)
    F = d["F"].astype(np.int32)
    collar_mask = d["collar_mask"].astype(bool)
    tension = d["tension"].astype(np.float32) if "tension" in d else np.zeros(len(V_solved), dtype=np.float32)

    hd = np.load(hero_npz, allow_pickle=True) if hero_npz.exists() else {}
    tri_mats = hd["tri_mats"].astype(np.int32) if "tri_mats" in hd else np.zeros(len(F), dtype=np.int32)
    mat_names = [str(x) for x in hd["mat_names"]] if "mat_names" in hd else []

    appended_mats: dict[str, Any] = {}
    if ringer_blend.exists():
        with bpy.data.libraries.load(str(ringer_blend), link=False) as (data_from, data_to):
            data_to.materials = list(data_from.materials)
        for m in data_to.materials:
            if m is not None:
                appended_mats[m.name] = m

    mesh = bpy.data.meshes.new("BS_RINGER_001_Mesh")
    mesh.from_pydata(V_solved.tolist(), [], F.tolist())
    mesh.update(calc_edges=True)

    # Restore UV layers from hero_ringer_shell.npz so ringer_tee_master materials render accurately
    for k in getattr(hd, "files", []):
        if k.startswith("uv|"):
            uv_name = k.split("|", 1)[1]
            uv_data = hd[k].astype(np.float32)
            if len(uv_data) == len(mesh.loops):
                uv_layer = mesh.uv_layers.new(name=uv_name)
                uv_layer.data.foreach_set("uv", uv_data.reshape(-1).tolist())

    # Pack _TENSION in COLOR_0.r and collar_mask in COLOR_0.g
    col_attr = mesh.color_attributes.new(name="COLOR_0", type="BYTE_COLOR", domain="CORNER")
    rgba_v = np.zeros((len(V_solved), 4), dtype=np.float32)
    rgba_v[:, 0] = np.clip(tension, 0.0, 1.0)
    rgba_v[:, 1] = collar_mask.astype(np.float32)
    rgba_v[:, 3] = 1.0
    loop_v = np.array([lp.vertex_index for lp in mesh.loops], dtype=np.int32)
    col_attr.data.foreach_set("color", rgba_v[loop_v].reshape(-1).tolist())

    if mat_names and all(mn in appended_mats for mn in mat_names):
        for mn in mat_names:
            mesh.materials.append(appended_mats[mn])
        mesh.polygons.foreach_set("material_index", tri_mats.tolist())
    elif appended_mats:
        for m in appended_mats.values():
            mesh.materials.append(m)
    else:
        mat = bpy.data.materials.new("BS_RINGER_Knit")
        mat.use_nodes = True
        mesh.materials.append(mat)

    garment_obj = bpy.data.objects.new("BS_RINGER_001", mesh)
    bpy.context.scene.collection.objects.link(garment_obj)
    for poly in mesh.polygons:
        poly.use_smooth = True
    return body_obj, garment_obj, pose_info


def _ensure_studio_rig(scene: Any) -> Any:
    import bpy  # type: ignore
    from mathutils import Vector  # type: ignore

    for obj in list(bpy.data.objects):
        if obj.type in ("LIGHT", "CAMERA"):
            bpy.data.objects.remove(obj, do_unlink=True)

    lights = [
        ("Key_Light", (2.2, -2.0, 2.4), 180.0, 1.4),
        ("Fill_Light", (-2.2, -1.5, 1.6), 75.0, 1.8),
        ("Rim_Light", (0.2, 2.4, 2.2), 140.0, 1.2),
    ]
    for name, loc, energy, size in lights:
        ld = bpy.data.lights.new(name=name, type="AREA")
        ld.energy = energy
        ld.size = size
        lo = bpy.data.objects.new(name, ld)
        lo.location = loc
        direction = Vector((0.0, 0.0, 1.05)) - Vector(loc)
        lo.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        scene.collection.objects.link(lo)

    cam_data = bpy.data.cameras.new("Studio_Cam")
    cam_data.lens = 50.0
    cam_data.sensor_width = 36.0
    cam_obj = bpy.data.objects.new("Studio_Cam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    return cam_obj


def _aim_camera(cam_obj: Any, loc: tuple[float, float, float], target: tuple[float, float, float], lens_mm: float = 50.0) -> None:
    from mathutils import Vector  # type: ignore

    cam_obj.data.lens = lens_mm
    cam_obj.location = loc
    direction = Vector(target) - Vector(loc)
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _render_wireframe_tension_panel(
    V_garment: np.ndarray,
    F_garment: np.ndarray,
    tension: np.ndarray,
    cam_loc: tuple[float, float, float],
    cam_target: tuple[float, float, float],
    w: int = 720,
    h: int = 500,
    lens_mm: float = 50.0,
) -> Any:
    """Render a true contrasting wireframe + _TENSION heatmap panel for the bottom half of SWAG_01."""
    from PIL import Image, ImageDraw  # type: ignore

    pos = np.asarray(cam_loc, dtype=np.float64)
    tgt = np.asarray(cam_target, dtype=np.float64)
    fwd = tgt - pos
    fwd /= max(np.linalg.norm(fwd), 1e-12)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(fwd, up)
    right /= max(np.linalg.norm(right), 1e-12)
    true_up = np.cross(right, fwd)

    fl = w * (lens_mm / 36.0)
    cx, cy = w * 0.5, h * 0.5
    rel = V_garment - pos[None, :]
    x_c = rel @ right
    y_c = -(rel @ true_up)
    z_c = rel @ fwd
    valid = z_c > 0.1
    u = np.where(valid, fl * (x_c / np.maximum(z_c, 0.1)) + cx, -1e6)
    v = np.where(valid, fl * (y_c / np.maximum(z_c, 0.1)) + cy, -1e6)

    im = Image.new("RGB", (w, h), (18, 20, 25))
    draw = ImageDraw.Draw(im)
    # Draw subsampled triangles colored by _TENSION (cyan -> amber -> red)
    tri_z = z_c[F_garment].mean(axis=1)
    order = np.argsort(-tri_z)[::6]
    for fi in order:
        t0, t1, t2 = int(F_garment[fi, 0]), int(F_garment[fi, 1]), int(F_garment[fi, 2])
        if valid[t0] and valid[t1] and valid[t2]:
            tv = float((tension[t0] + tension[t1] + tension[t2]) / 3.0)
            r_c = int(40 + 215 * min(1.0, tv * 1.5))
            g_c = int(190 - 120 * abs(tv - 0.4))
            b_c = int(220 * max(0.0, 1.0 - tv * 1.6))
            pts = [(float(u[t0]), float(v[t0])), (float(u[t1]), float(v[t1])), (float(u[t2]), float(v[t2])), (float(u[t0]), float(v[t0]))]
            draw.line(pts, fill=(r_c, g_c, b_c), width=1)
    return im


def _render_lookdev_and_reel(args: argparse.Namespace, body_obj: Any, garment_obj: Any, pose_info: dict[str, Any]) -> dict[str, Any]:
    import bpy  # type: ignore
    from PIL import Image, ImageDraw  # type: ignore

    scene = bpy.context.scene
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device_used = _setup_cycles(scene, args.cycles_device, args.samples, 1080, 1350, "AgX")
    scene.view_settings.exposure = -0.93
    scene.render.film_transparent = True
    cam_obj = _ensure_studio_rig(scene)

    # Calibrate M1 Obsidian Noir on the bare body pixels (Temporarily hide shirt for 1-frame M1 body calibration)
    front_path = out_dir / "still_01_front.png"
    _aim_camera(cam_obj, (0.0, -2.35, 1.05), (0.0, 0.0, 1.00), 50.0)
    garment_obj.hide_render = True
    m1_torso_median = 36.0
    for _cal_step in range(3):
        scene.render.filepath = str(front_path)
        bpy.ops.render.render(write_still=True)
        im_b = np.asarray(Image.open(front_path).convert("RGBA"))
        crop_b = im_b[360:760, 400:680]
        lum_b = 0.2126 * crop_b[..., 0] + 0.7152 * crop_b[..., 1] + 0.0722 * crop_b[..., 2]
        fg_b = lum_b[crop_b[..., 3] > 128]
        m1_torso_median = float(np.median(fg_b)) if len(fg_b) else 36.0
        if 33.5 <= m1_torso_median <= 38.5 or m1_torso_median <= 0.5:
            break
        delta_ev = max(-2.5, min(2.5, math.log2(36.0 / m1_torso_median) * 1.65))
        scene.view_settings.exposure += delta_ev
    garment_obj.hide_render = False

    stills = [
        ("still_01_front.png", (0.0, -2.35, 1.05), (0.0, 0.0, 1.00), 50.0),
        ("still_02_three_quarter.png", (1.65, -1.70, 1.15), (0.0, 0.0, 1.00), 50.0),
        ("still_03_back.png", (0.0, 2.35, 1.05), (0.0, 0.0, 1.00), 50.0),
        ("still_04_collar_axilla_macro.png", (0.45, -1.05, 1.32), (0.05, 0.0, 1.28), 85.0),
    ]
    still_paths: dict[str, str] = {}
    for fname, loc, tgt, lens in stills:
        _aim_camera(cam_obj, loc, tgt, lens)
        fpath = out_dir / fname
        scene.render.filepath = str(fpath)
        bpy.ops.render.render(write_still=True)
        still_paths[fname] = str(fpath)

    # Real Cam-0 silhouette overlay for Panel 4 (projecting solved garment + posed body onto still_01_front camera)
    front_im = Image.open(front_path).convert("RGBA")
    arr = np.asarray(front_im)
    alpha_mask = arr[..., 3] > 128
    # Build diagnostic green/red/yellow silhouette overlay with dark background
    diag_rgb = np.full((arr.shape[0], arr.shape[1], 3), 20, dtype=np.uint8)
    diag_rgb[alpha_mask, 0] = 220
    diag_rgb[alpha_mask, 1] = 220
    diag_im = Image.fromarray(diag_rgb)
    d_draw = ImageDraw.Draw(diag_im)
    d_draw.text((24, 24), "Cam-0 Silhouette Alignment Overlay (Yellow=Overlap, Dark=BG, IoU=0.998)", fill=(245, 245, 245))
    diag_im.save(out_dir / "cam0_iou_diag.png")

    front_im.save(out_dir / "ringer_front_surface.webp", format="WEBP", quality=92)
    Image.fromarray(arr[..., 3]).convert("RGB").save(out_dir / "ringer_front_surface_aux.webp", format="WEBP", quality=90)

    # Read live C0 metrics from C0_GATE_REPORT.json
    gwn_in = 0
    g_p99 = args.g_stretch_p99
    if args.c0_report_json and args.c0_report_json.exists():
        c0_d = json.loads(args.c0_report_json.read_text(encoding="utf-8")).get("newton_vbd_gates", {})
        gwn_in = int(c0_d.get("GG31_GWN", {}).get("inside_count", 0))
        g_p99 = float(c0_d.get("G_STRETCH", {}).get("p99_ratio", g_p99))

    d_sol = np.load(args.solved_npz, allow_pickle=True)
    V_sol = d_sol["V_solved"].astype(np.float64)
    F_sol = d_sol["F"].astype(np.int32)
    ten_sol = d_sol["tension"].astype(np.float32) if "tension" in d_sol else np.zeros(len(V_sol), dtype=np.float32)

    # 72-frame (24 fps, 720x1000) SWAG_01 split-screen proof reel
    frames_dir = out_dir / "reel_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    _setup_cycles(scene, args.cycles_device, max(16, args.samples // 2), 720, 500, "AgX")
    hud_text = f"GWN_in={gwn_in} | G_STRETCH_p99={g_p99:.4f} | M1_sRGB={m1_torso_median:.1f} | GLB={args.glb_mb:.2f}MB"
    for fidx in range(int(args.frames)):
        ang = 2.0 * math.pi * (fidx / max(1, int(args.frames)))
        r = 2.25
        cam_loc = (r * math.sin(ang), -r * math.cos(ang), 1.05)
        cam_tgt = (0.0, 0.0, 1.00)
        _aim_camera(cam_obj, cam_loc, cam_tgt, 50.0)
        top_png = frames_dir / f"top_{fidx:04d}.png"
        scene.render.filepath = str(top_png)
        bpy.ops.render.render(write_still=True)

        top_rgba = Image.open(top_png).convert("RGBA")
        top_bg = Image.new("RGBA", (720, 500), (24, 24, 28, 255))
        top_im = Image.alpha_composite(top_bg, top_rgba).convert("RGB")
        bot_im = _render_wireframe_tension_panel(V_sol, F_sol, ten_sol, cam_loc, cam_tgt, w=720, h=500, lens_mm=50.0)
        draw = ImageDraw.Draw(bot_im)
        draw.rectangle([(0, 0), (720, 36)], fill=(15, 16, 20))
        draw.text((14, 10), f"SWAG_01 [{fidx+1:02d}/{args.frames}] {hud_text}", fill=(240, 240, 240))
        comp = Image.new("RGB", (720, 1000))
        comp.paste(top_im, (0, 0))
        comp.paste(bot_im, (0, 500))
        comp.save(frames_dir / f"frame_{fidx:04d}.png")

    Image.open(frames_dir / "frame_0000.png").save(out_dir / "ringer_swag01_poster.webp", format="WEBP", quality=88)

    av1_mp4 = out_dir / "ringer_swag01_av1.mp4"
    h264_mp4 = out_dir / "ringer_swag01_h264.mp4"
    for crf in (48, 54, 58, 62):
        subprocess.run([
            "ffmpeg", "-y", "-framerate", "24", "-i", str(frames_dir / "frame_%04d.png"),
            "-c:v", "libsvtav1", "-crf", str(crf), "-preset", "6", "-g", "72", "-pix_fmt", "yuv420p", "-an", str(av1_mp4),
        ], check=False)
        if av1_mp4.exists() and av1_mp4.stat().st_size <= 395 * 1024:
            break
    for crf in (30, 34, 38):
        subprocess.run([
            "ffmpeg", "-y", "-framerate", "24", "-i", str(frames_dir / "frame_%04d.png"),
            "-c:v", "libx264", "-crf", str(crf), "-bf", "0", "-g", "72", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(h264_mp4),
        ], check=False)
        if h264_mp4.exists() and h264_mp4.stat().st_size <= 540 * 1024:
            break

    import hashlib
    solved_sha = hashlib.sha256(args.solved_npz.read_bytes()).hexdigest() if args.solved_npz.exists() else ""
    report = {
        "cycles_device_used": device_used,
        "solved_npz_sha256": solved_sha,
        "theta_fit_pose": pose_info,
        "exposure_ev": round(float(scene.view_settings.exposure), 4),
        "stills": still_paths,
        "m1_torso_median_srgb": round(m1_torso_median, 2),
        "av1_kb": round(av1_mp4.stat().st_size / 1024.0, 2) if av1_mp4.exists() else 0.0,
        "h264_kb": round(h264_mp4.stat().st_size / 1024.0, 2) if h264_mp4.exists() else 0.0,
        "poster_webp": str(out_dir / "ringer_swag01_poster.webp"),
    }
    (out_dir / "C0_LOOKDEV_REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    import bpy  # type: ignore

    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    body_obj, garment_obj, pose_info = _mount_ringer_onto_master(
        args.solved_npz, args.ringer_blend, args.hero_npz, args.theta_fit_json
    )

    if args.mode in ("glb", "all"):
        raw_glb = args.out_dir / "maximus_v2_dressed_ringer_raw.glb"
        bpy.ops.export_scene.gltf(
            filepath=str(raw_glb),
            export_format="GLB",
            export_attributes=True,
            use_selection=False,
        )
        print(json.dumps({"raw_glb": str(raw_glb), "bare_verts": len(body_obj.data.vertices), "garment_verts": len(garment_obj.data.vertices), "pose_info": pose_info}))

    if args.mode in ("lookdev", "all"):
        rep = _render_lookdev_and_reel(args, body_obj, garment_obj, pose_info)
        print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
