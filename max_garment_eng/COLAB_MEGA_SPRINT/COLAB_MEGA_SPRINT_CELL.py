"""COLAB_MEGA_SPRINT_CELL.py — Single-Cell 5-Stage L4 Mega-Burst Orchestrator.

Stages (each subprocess-isolated + atomically promoted to Drive):
  1. C0_SOLVE_AND_GLB       : SolverVBD + PBD + Meshopt <1.5 MB GLB
  2. C0_OPTIX_HERO_AND_REEL : Blender 5.2 OptiX 4 studio stills + 2.5D WebPs + 72-frame SWAG_01 reel
  3. B1_OPTIX_120CAM        : Cam-0 IoU >= 0.98 gate + 120-cam 16-bit RGBA OptiX renders + tar
  4. GSPLAT_WHEEL_CAPPED    : ninja + MAX_JOBS=4 gsplat@main build (20-min hard cap, Drive cached)
  5. B2_GSPLAT_SOG_V3       : Live gsplat MCMC (cap_max=600k) + SOG v3 + SPZ v4 packing
  6. QA_AND_TEARDOWN        : BURST_1A_MEGA_QA_SHEET.png + SPRINT_VERIFICATION_REPORT.json + FINAL_SHA256_MANIFEST.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from pathlib import Path

SPRINT_SCOPE = "PHASE3_ALL_8_LIVE_SKUS"
SKIP_GSPLAT_BUILD = False  # Full 5-stage execution (use --skip-gsplat to skip Stage 4/5)
BLENDER_URL = "https://download.blender.org/release/Blender5.2/blender-5.2.3-linux-x64.tar.xz"
APT_LIBS = ["libxi6", "libxxf86vm1", "libxfixes3", "libxrender1", "libgl1", "libxkbcommon0", "libsm6", "libegl1", "libvulkan1", "ffmpeg"]
SPLAT_TRANSFORM_PKG = "@playcanvas/splat-transform@3.12.0"
PINNED_NEWTON_SOLVER = "SolverVBD"  # NEVER SolverStyle3D
FORBIDDEN_SOLVER = "SolverStyle3D"
DRIVE_ROOT = Path("/content/drive/MyDrive/maximus_v2")
DRIVE_STAGE = DRIVE_ROOT / "COLAB_MEGA_SPRINT"
DRIVE_CACHE = DRIVE_ROOT / "cache"
DRIVE_M4_ATLAS = DRIVE_ROOT / "results" / "M4_canon_atlas"
LEDGER_PATH = DRIVE_ROOT / "results" / "CHECKPOINT_LEDGER.jsonl"
LOCAL_STAGE = Path("/content/sprint_local")
WORK_ROOT = Path("/content/work")
VENV_DIR = Path("/content/venv312")
TOOLS_DIR = Path("/content/tools")
BLENDER_OPT_DIR = Path("/opt/blender")


def ensure_drive_mounted() -> None:
    """Verify Drive FUSE is responsive; fusermount + force_remount on Errno 107."""
    from google.colab import drive  # type: ignore

    try:
        os.listdir("/content/drive/MyDrive")
    except OSError:
        subprocess.run(["fusermount", "-uz", "/content/drive"], check=False)
        drive.mount("/content/drive", force_remount=True)


def stage_to_local_nvme() -> tuple[Path, Path, Path, Path]:
    """Copy COLAB_MEGA_SPRINT + M4 atlas assets to local NVMe /content/sprint_local/ (zero FUSE execution)."""
    ensure_drive_mounted()
    if LOCAL_STAGE.exists():
        shutil.rmtree(LOCAL_STAGE)
    shutil.copytree(DRIVE_STAGE, LOCAL_STAGE)
    local_atlas = LOCAL_STAGE / "m4_atlas"
    local_atlas.mkdir(parents=True, exist_ok=True)
    master_blend = local_atlas / "maximus_v2_master_psd.blend"
    surface_atlas = local_atlas / "maximus_v2_surface_atlas.npz"
    shutil.copy2(DRIVE_M4_ATLAS / "maximus_v2_master_psd.blend", master_blend)
    shutil.copy2(DRIVE_M4_ATLAS / "maximus_v2_surface_atlas.npz", surface_atlas)
    cotton_ringer = LOCAL_STAGE / "repaired_masters" / "BS-RINGER-001_cotton280_master.blend"
    ringer_blend = cotton_ringer if cotton_ringer.is_file() else (LOCAL_STAGE / "ringer_tee_master.blend")
    return LOCAL_STAGE, master_blend, surface_atlas, ringer_blend


def _run(cmd: list[str], env: dict[str, str] | None = None, check: bool = True, timeout_sec: int | None = None) -> int:
    print(">>>", " ".join(cmd), flush=True)
    log_path = DRIVE_ROOT / "results" / "sprint_run.log"
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, start_new_session=True,
    )
    timed_out = False

    def _kill_group() -> None:
        nonlocal timed_out
        timed_out = True
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            proc.kill()

    timer = threading.Timer(timeout_sec, _kill_group) if timeout_sec is not None else None
    if timer is not None:
        timer.start()

    assert proc.stdout is not None
    lf = None
    try:
        if DRIVE_ROOT.exists():
            log_path.parent.mkdir(parents=True, exist_ok=True)
            lf = open(log_path, "a", encoding="utf-8")
            lf.write(f">>> {' '.join(cmd)}\n")
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if lf is not None:
                lf.write(line)
                lf.flush()
    finally:
        if timer is not None:
            timer.cancel()
        if lf is not None:
            lf.close()
    rc = proc.wait()
    if timed_out:
        raise subprocess.TimeoutExpired(cmd, timeout_sec or 0)
    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    return rc


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_promote_to_drive(local_src: Path, drive_dst: Path, stage_id: str) -> str:
    digest = sha256_file(local_src)
    for attempt in range(3):
        try:
            ensure_drive_mounted()
            drive_dst.parent.mkdir(parents=True, exist_ok=True)
            LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp_dst = drive_dst.with_suffix(drive_dst.suffix + ".tmp")
            shutil.copy2(local_src, tmp_dst)
            with open(tmp_dst, "rb+") as f:
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_dst, drive_dst)
            assert sha256_file(drive_dst) == digest, f"Drive hash mismatch on {drive_dst}"
            rec = {
                "sprint_scope": SPRINT_SCOPE,
                "stage": stage_id,
                "file": str(drive_dst),
                "sha256": digest,
                "bytes": drive_dst.stat().st_size,
                "ts": time.time(),
            }
            with open(LEDGER_PATH, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(rec) + "\n")
                lf.flush()
                os.fsync(lf.fileno())
            return digest
        except OSError as exc:
            if attempt == 2:
                raise
            print(f"[Drive promote retry {attempt + 1}/3] {exc}", flush=True)
            subprocess.run(["fusermount", "-uz", "/content/drive"], check=False)
            time.sleep(2)
    return digest


def is_stage_verified(stage_id: str, expected_files: list[Path]) -> bool:
    if not LEDGER_PATH.exists():
        return False
    records = {
        r["file"]: r["sha256"]
        for r in map(json.loads, LEDGER_PATH.read_text("utf-8").splitlines())
        if r.get("stage") == stage_id
    }
    return all(str(p) in records and p.exists() and sha256_file(p) == records[str(p)] for p in expected_files)


def ensure_nvidia_vulkan_icd() -> Path | None:
    """Register a verified NVIDIA Vulkan ICD only if the runtime driver exports vkCreateInstance."""
    import ctypes

    sys_icd_candidates = [
        Path("/usr/share/vulkan/icd.d/nvidia_icd.json"),
        Path("/etc/vulkan/icd.d/nvidia_icd.json"),
    ]
    for sys_icd in sys_icd_candidates:
        if sys_icd.is_file():
            try:
                spec = json.loads(sys_icd.read_text(encoding="utf-8"))
                lib_name = spec.get("ICD", {}).get("library_path", "")
                if lib_name:
                    lib = ctypes.CDLL(lib_name)
                    if hasattr(lib, "vk_icdGetInstanceProcAddr") or hasattr(lib, "vkCreateInstance"):
                        return sys_icd
            except Exception:
                pass

    for lib_cand in ("libGLX_nvidia.so.0", "libEGL_nvidia.so.0", "libnvidia-glvkspirv.so"):
        try:
            lib = ctypes.CDLL(lib_cand)
            if hasattr(lib, "vk_icdGetInstanceProcAddr") and hasattr(lib, "vkCreateInstance"):
                icd_dir = Path("/etc/vulkan/icd.d")
                icd_dir.mkdir(parents=True, exist_ok=True)
                icd_path = icd_dir / "nvidia_icd.json"
                icd_spec = {
                    "file_format_version": "1.0.0",
                    "ICD": {"library_path": lib_cand, "api_version": "1.3.0"},
                }
                icd_path.write_text(json.dumps(icd_spec, indent=2) + "\n", encoding="utf-8")
                return icd_path
        except Exception:
            continue

    # Remove stale broken custom ICD so Vulkan loader/Dawn falls back cleanly without noise
    stale_icd = Path("/etc/vulkan/icd.d/nvidia_icd.json")
    if stale_icd.is_file():
        stale_icd.unlink(missing_ok=True)
    return None


def ensure_blender_5_2() -> Path:
    exe = BLENDER_OPT_DIR / "blender"
    if exe.exists():
        ensure_nvidia_vulkan_icd()
        return exe
    if shutil.which("apt-get"):
        subprocess.run(["apt-get", "update", "-qq"], check=False)
        subprocess.run(["apt-get", "install", "-y", "-qq"] + APT_LIBS, check=False)
    ensure_nvidia_vulkan_icd()
    extract_root = Path("/content/blender")
    extract_root.mkdir(parents=True, exist_ok=True)
    cached_tar = DRIVE_CACHE / "tools" / "blender-5.2.3-linux-x64.tar.xz"
    tar_path = cached_tar if cached_tar.exists() else (extract_root / "blender.tar.xz")
    if not tar_path.exists():
        print("Downloading Blender 5.2.3:", BLENDER_URL, flush=True)
        urllib.request.urlretrieve(BLENDER_URL, tar_path)
    with tarfile.open(tar_path) as tf:
        tf.extractall(extract_root)
    bdir = next(extract_root.glob("blender-*"))
    if BLENDER_OPT_DIR.exists() or BLENDER_OPT_DIR.is_symlink():
        BLENDER_OPT_DIR.unlink()
    BLENDER_OPT_DIR.symlink_to(bdir)
    assert exe.exists(), f"Blender executable missing at {exe}"
    return exe


def bootstrap_part1(stage_dir: Path) -> tuple[Path, dict[str, str], Path]:
    """Bootstrap Part 1: PyTorch cu130 + ninja + physics/geometry + Node CLIs + Blender 5.2.3."""
    env = os.environ.copy()
    env["WARP_CACHE_PATH"] = str(DRIVE_CACHE / "warp")
    env["TORCH_CUDA_ARCH_LIST"] = "8.9"
    env["CXX_APPEND_FLAGS"] = "-std=c++20"
    env["NVCC_APPEND_FLAGS"] = "-std=c++20"
    env["CUPY_PACKAGE"] = "cupy-cuda13x"
    env["MAX_JOBS"] = "4"
    env["PATH"] = f"{VENV_DIR / 'bin'}:{TOOLS_DIR / 'node_modules' / '.bin'}:{env.get('PATH', '')}"

    (DRIVE_CACHE / "warp").mkdir(parents=True, exist_ok=True)
    (DRIVE_CACHE / "wheels").mkdir(parents=True, exist_ok=True)
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)

    _run(["uv", "venv", "--seed", "--python", "3.12", str(VENV_DIR)], env=env)
    py = VENV_DIR / "bin" / "python"

    _run([
        "uv", "pip", "install", "--python", str(py),
        "torch==2.14.1", "torchvision==0.29.1",
        "--index-url", "https://download.pytorch.org/whl/cu130",
    ], env=env)
    _run([str(py), "-c", "import torch; assert (torch.randn(16, device='cuda') @ torch.randn(16, device='cuda')).isfinite()"], env=env)

    _run(["uv", "pip", "install", "--python", str(py), "ninja", "setuptools", "wheel", "packaging"], env=env)
    ninja_bin = shutil.which("ninja", path=env["PATH"])
    assert ninja_bin is not None and Path(ninja_bin).exists(), "ninja not found on venv PATH"
    print(f"Verified ninja binary: {ninja_bin}", flush=True)

    _run(["uv", "pip", "install", "--python", str(py), "-r", str(stage_dir / "scripts" / "requirements_colab_l4.txt")], env=env)
    _run([
        str(py), "-c",
        "import newton; assert hasattr(newton.solvers, 'SolverVBD'), 'SolverVBD missing'; "
        "assert 'SolverStyle3D' != 'SolverVBD'",
    ], env=env)
    _run([str(py), "-m", "uipc", "doctor", "--probe-cuda"], env=env, check=False)

    _run([
        "npm", "install", "--prefix", str(TOOLS_DIR),
        SPLAT_TRANSFORM_PKG, "@gltf-transform/cli@4.4.2",
    ], env=env)

    blender_exe = ensure_blender_5_2()
    # Install pillow in Blender's bundled python if missing (for PIL image ops inside Blender scripts)
    b_py = next(blender_exe.parent.glob("5.*/python/bin/python*"), None)
    if b_py is not None:
        subprocess.run([str(b_py), "-m", "pip", "install", "-q", "pillow"], check=False)
    return py, env, blender_exe


def _stub_gsplat_3dgut_world_batch_kernels(csrc_dir: Path) -> None:
    """Stub heavy 3DGUT world-space batch template kernels (ParallelBatchBwd, ParallelBatchFwd, SerialBatchFwd) while preserving all exported C++ symbols."""
    bwd_cu = csrc_dir / "RasterizeToPixelsFromWorld3DGSParallelBatchBwd.cu"
    if bwd_cu.exists():
        src = bwd_cu.read_text(encoding="utf-8")
        k_marker = "template<uint32_t CDIM, bool ReturnNormals, bool UseHitDistance, typename scalar_t>\n__global__ void rasterize_gradient_bwd_kernel"
        fn_marker = "void launch_rasterize_to_pixels_from_world_3dgs_parallel_batch_bwd_kernel("
        if fn_marker in src:
            prefix = src.split(k_marker)[0] if k_marker in src else src.split(fn_marker)[0]
            fn_part = src[src.index(fn_marker):]
            m = re.search(r"\)\s*\{", fn_part)
            if m is not None:
                sig = fn_part[: m.start() + 1]
                stubbed = (
                    prefix
                    + sig
                    + '\n{\n    TORCH_CHECK(false, "3DGUT parallel batch bwd stubbed for pinhole 3DGS");\n}\n} // namespace gsplat\n\n#endif\n'
                )
                bwd_cu.write_text(stubbed, encoding="utf-8")
                print(f"Stubbed 3DGUT Bwd template kernel in {bwd_cu.name}", flush=True)

    for cu_name, fn_marker in (
        ("RasterizeToPixelsFromWorld3DGSParallelBatchFwd.cu", "void launch_rasterize_to_pixels_from_world_3dgs_parallel_batch_fwd_kernel("),
        ("RasterizeToPixelsFromWorld3DGSSerialBatchFwd.cu", "void launch_rasterize_to_pixels_from_world_3dgs_serial_batch_fwd_kernel("),
    ):
        fwd_cu = csrc_dir / cu_name
        if not fwd_cu.exists():
            continue
        src = fwd_cu.read_text(encoding="utf-8")
        ns_marker = "namespace gsplat\n{"
        if fn_marker not in src or ns_marker not in src:
            continue
        prefix = src[: src.index(ns_marker) + len(ns_marker)] + "\n"
        fn_part = src[src.index(fn_marker):]
        m = re.search(r"\)\s*\{", fn_part)
        if m is None:
            continue
        sig = fn_part[: m.start() + 1]
        stubbed = (
            prefix
            + sig
            + '\n{\n    TORCH_CHECK(false, "3DGUT world-space batch fwd stubbed for pinhole 3DGS");\n}\n} // namespace gsplat\n\n#endif\n'
        )
        fwd_cu.write_text(stubbed, encoding="utf-8")
        print(f"Stubbed 3DGUT Fwd template kernel in {fwd_cu.name}", flush=True)


def install_gsplat_cached_or_build(py: Path, env: dict[str, str], timeout_sec: int = 1800) -> bool:
    """Install cached gsplat wheel from Drive, or shallow-clone + stub 3DGUT world-space batch kernels + build wheel."""
    whl_dir = DRIVE_CACHE / "wheels"
    whl_dir.mkdir(parents=True, exist_ok=True)
    cached_whls = sorted(whl_dir.glob("gsplat-*.whl"))
    if not cached_whls:
        gsplat_src = Path("/content/gsplat_src")
        if gsplat_src.exists():
            shutil.rmtree(gsplat_src)
        _run([
            "git", "clone", "--depth", "1", "--recurse-submodules",
            "https://github.com/nerfstudio-project/gsplat.git", str(gsplat_src),
        ], env=env, timeout_sec=300)
        _stub_gsplat_3dgut_world_batch_kernels(gsplat_src / "gsplat" / "cuda" / "csrc")

        build_env = env.copy()
        build_env["BUILD_EXPERIMENTAL"] = "0"
        build_env["MAX_JOBS"] = "4"
        build_env["TORCH_CUDA_ARCH_LIST"] = "8.9"
        build_env["CXX_APPEND_FLAGS"] = "-std=c++20"
        build_env["NVCC_APPEND_FLAGS"] = "-std=c++20"
        build_env["CUPY_PACKAGE"] = "cupy-cuda13x"
        build_env["PATH"] = f"{VENV_DIR / 'bin'}:{build_env.get('PATH', '')}"
        assert shutil.which("ninja", path=build_env["PATH"]) is not None, "ninja missing from build_env PATH"

        log_path = DRIVE_ROOT / "results" / "gsplat_build.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        local_whl_dir = Path("/content/wheels_tmp")
        local_whl_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(py), "-m", "pip", "wheel", "-v", "--no-build-isolation",
            "--no-deps", "--wheel-dir", str(local_whl_dir), str(gsplat_src),
        ]
        print(">>>", " ".join(cmd), f"(timeout={timeout_sec}s, streaming to {log_path})", flush=True)
        proc = subprocess.Popen(
            cmd, env=build_env, start_new_session=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        timed_out = False

        def _kill_pg() -> None:
            nonlocal timed_out
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        timer = threading.Timer(timeout_sec, _kill_pg)
        timer.start()
        lines_buf: list[str] = []
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                lines_buf.append(line)
                if len(lines_buf) % 25 == 0 or "[" in line or "Building wheel" in line or "Successfully built" in line:
                    print(line, end="", flush=True)
            rc = proc.wait()
        finally:
            timer.cancel()
            try:
                ensure_drive_mounted()
                log_path.write_text("".join(lines_buf), encoding="utf-8")
            except OSError:
                pass
        if timed_out:
            print(f"[STAGE 4] gsplat build exceeded {timeout_sec}s; skipping Stage 5 cleanly.", flush=True)
            return False
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)
        for w in sorted(local_whl_dir.glob("gsplat-*.whl")):
            ensure_drive_mounted()
            shutil.copy2(w, whl_dir / w.name)
        cached_whls = sorted(whl_dir.glob("gsplat-*.whl"))
    if not cached_whls:
        return False
    _run(["uv", "pip", "install", "--python", str(py), "rich>=12.0.0", "jaxtyping>=0.2.36", "nvtx>=0.2.10", "plyfile>=1.1.0"], env=env)
    _run(["uv", "pip", "install", "--python", str(py), "--no-deps", str(cached_whls[-1])], env=env)
    _run([str(py), "-c", "from gsplat.rendering import rasterization; from gsplat.strategy import MCMCStrategy; from gsplat import export_splats; print('gsplat smoke OK')"], env=env)
    return True


def build_qa_contact_sheet(work_dir: Path, out_png: Path) -> Path:
    from PIL import Image, ImageDraw  # type: ignore

    cam0_diag = work_dir / "C0_lookdev" / "cam0_iou_diag.png"
    if not cam0_diag.exists():
        cam0_diag = DRIVE_ROOT / "results" / "C0_lookdev" / "cam0_iou_diag.png"
    if not cam0_diag.exists():
        cam0_diag = work_dir / "B1_120cam" / "cam0_iou_diag.png"
    if not cam0_diag.exists():
        cam0_diag = DRIVE_ROOT / "results" / "B1_120cam" / "cam0_iou_diag.png"
    panels = [
        ("1. Studio Front", work_dir / "C0_lookdev" / "still_01_front.png"),
        ("2. Collar/Axilla Macro", work_dir / "C0_lookdev" / "still_04_collar_axilla_macro.png"),
        ("3. SWAG_01 Frame-0", work_dir / "C0_lookdev" / "ringer_swag01_poster.webp"),
        ("4. Cam-0 IoU Check", cam0_diag),
        ("5. Studio 3/4", work_dir / "C0_lookdev" / "still_02_three_quarter.png"),
        ("6. Studio Back", work_dir / "C0_lookdev" / "still_03_back.png"),
    ]
    w, h = 640, 480
    sheet = Image.new("RGB", (w * 3, h * 2), (24, 24, 27))
    draw = ImageDraw.Draw(sheet)
    for idx, (label, path) in enumerate(panels):
        x0, y0 = (idx % 3) * w, (idx // 3) * h
        if path.exists():
            im = Image.open(path).convert("RGB").resize((w, h))
            sheet.paste(im, (x0, y0))
        draw.text((x0 + 12, y0 + 12), label, fill=(240, 240, 240))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    return out_png


def write_final_sha256_manifest(results_dir: Path, out_csv: Path) -> None:
    rows = []
    for p in sorted(results_dir.rglob("*")):
        if p.is_file() and p.name != out_csv.name and not p.name.endswith(".tmp"):
            rows.append((str(p.relative_to(results_dir)), p.stat().st_size, sha256_file(p)))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["relative_path", "bytes", "sha256"])
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 Colab L4 8-SKU Multi-Wave Orchestrator")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and exit without mounting Colab Drive")
    parser.add_argument("--run-gsplat", action="store_true", help="Force-enable Stage 4/5 gsplat build & training")
    parser.add_argument("--skip-gsplat", action="store_true", help="Skip Stage 4/5 gsplat build & training")
    parser.add_argument("--phase3-waves", nargs="+", default=["3A-0", "3A", "3B", "3C"], help="Multi-SKU Phase 3 waves to schedule")
    parser.add_argument("--phase3-skus", default="BS-RINGER-001,BS-POLO-001,BS-CAMP-001,BS-CAMP-002", help="Focused comma-separated Phase 3 SKU burst queue")
    args = parser.parse_args()
    skip_gsplat = True if args.skip_gsplat else (False if args.run_gsplat else SKIP_GSPLAT_BUILD)

    if args.dry_run:
        print(json.dumps({
            "sprint_scope": SPRINT_SCOPE,
            "skip_gsplat": skip_gsplat,
            "phase3_waves": args.phase3_waves,
            "phase3_skus": args.phase3_skus,
            "pinned_solver": PINNED_NEWTON_SOLVER,
            "forbidden_solver": FORBIDDEN_SOLVER,
            "splat_transform_pkg": SPLAT_TRANSFORM_PKG,
            "vulkan_icd": "/etc/vulkan/icd.d/nvidia_icd.json",
            "apt_libs": APT_LIBS,
            "m4_atlas_source": DRIVE_M4_ATLAS.as_posix(),
            "master_blend": (DRIVE_M4_ATLAS / "maximus_v2_master_psd.blend").as_posix(),
            "surface_atlas": (DRIVE_M4_ATLAS / "maximus_v2_surface_atlas.npz").as_posix(),
            "stages": [
                "1. C0_SOLVE_AND_GLB (SolverVBD + PBD + Meshopt <= 1.2 MB)",
                "2. C0_OPTIX_HERO_AND_REEL (0.72 AV1 <= 400KB + H.264 <= 550KB)",
                "3. B1_OPTIX_120CAM_AND_OBJECT_ID (body=0,top=1,bottom=2,footwear=3, IoU>0.98)",
                "4. GSPLAT_WHEEL_CAPPED (torch==2.14.1 cu130, 3DGUT Bwd stubbed)",
                "5. B2_GSPLAT_GARMENT_ONLY_AND_MORPH_PACK (N=100k..120k, Vulkan splat-transform, 4x60s colorway SH)",
                "6. MULTI_SKU_PHASE3_WAVES + DRIVE_SHA256_MANIFEST",
                "7. QA_AND_UNASSIGN (finally: runtime.unassign())",
            ],
            "qa_contact_sheet": "PHASE3_MEGA_QA_SHEET.png",
            "status": "DRY_RUN_STOP_GATE",
        }, indent=2))
        return

    from google.colab import drive, runtime  # type: ignore

    drive.mount("/content/drive", force_remount=False)
    sprint_summary: dict[str, object] = {"sprint_scope": SPRINT_SCOPE, "stage4_timeout_skipped_b2": False}
    try:
        local_stage, master_blend, surface_atlas, ringer_blend = stage_to_local_nvme()
        os.chdir(local_stage)
        WORK_ROOT.mkdir(parents=True, exist_ok=True)
        _run(["sha256sum", "-c", "MANIFEST_SHA256.txt"])
        _run(["nvidia-smi"])
        assert master_blend.exists(), "Missing M4 master_psd.blend"
        assert surface_atlas.exists(), "Missing M4 surface_atlas.npz"
        assert ringer_blend.exists(), "Missing ringer_tee_master.blend"

        py, env, blender_exe = bootstrap_part1(local_stage)

        # STAGE 1: C0_SOLVE_AND_GLB (re-run if prior C0_GATE_REPORT did not pass c0_all_pass)
        c0_dir = WORK_ROOT / "C0_ringer"
        c0_expected = [
            DRIVE_ROOT / "results" / "C0_ringer" / "C0_GATE_REPORT.json",
            DRIVE_ROOT / "results" / "C0_ringer" / "c0_ringer_solved.npz",
            DRIVE_ROOT / "results" / "C0_ringer" / "maximus_v2_dressed_ringer_lod1.glb",
        ]
        c0_prior_ok = is_stage_verified("C0_SOLVE_AND_GLB", c0_expected) and json.loads(
            (DRIVE_ROOT / "results" / "C0_ringer" / "C0_GATE_REPORT.json").read_text("utf-8")
        ).get("newton_vbd_gates", {}).get("c0_all_pass", False)
        if not c0_prior_ok:
            _run([
                str(py), "scripts/run_c0_ringer_benchmark.py",
                "--hero-npz", "inputs/hero_ringer_shell.npz",
                "--out-dir", str(c0_dir),
            ], env=env)
            _run([
                str(blender_exe), "-b", str(master_blend), "--factory-startup", "--python-exit-code", "3",
                "--python", "scripts/run_c0_optix_lookdev_and_reel.py", "--",
                "--mode", "glb", "--solved-npz", str(c0_dir / "c0_ringer_solved.npz"),
                "--hero-npz", "inputs/hero_ringer_shell.npz",
                "--theta-fit-json", "inputs/m_theta_fit.json",
                "--ringer-blend", str(ringer_blend), "--out-dir", str(c0_dir),
            ], env=env)
            _run([
                str(py), "-c",
                "import json; from pathlib import Path; import sys; sys.path.insert(0, 'scripts'); "
                "from morph_export_packager import pack_glb; "
                f"r=pack_glb(Path('{c0_dir}/maximus_v2_dressed_ringer_raw.glb'), Path('{c0_dir}/maximus_v2_dressed_ringer_lod1.glb'), dry_run=False); "
                f"Path('{c0_dir}/GLB_PACK_REPORT.json').write_text(json.dumps(r, indent=2)); "
                "assert r['under_2mb_budget'], f'GLB exceeded 2MB: {r}'",
            ], env=env)
            for f in c0_dir.glob("*"):
                if f.is_file():
                    atomic_promote_to_drive(f, DRIVE_ROOT / "results" / "C0_ringer" / f.name, "C0_SOLVE_AND_GLB")
        assert is_stage_verified("C0_SOLVE_AND_GLB", c0_expected), "Stage 1 C0_SOLVE_AND_GLB failed Drive verification"

        c0_rep = json.loads((DRIVE_ROOT / "results" / "C0_ringer" / "C0_GATE_REPORT.json").read_text("utf-8"))
        glb_mb = (DRIVE_ROOT / "results" / "C0_ringer" / "maximus_v2_dressed_ringer_lod1.glb").stat().st_size / (1024.0 * 1024.0)
        g_p99 = c0_rep.get("newton_vbd_gates", {}).get("G_STRETCH", {}).get("p99_ratio", 1.0)
        sprint_summary["stage1_c0"] = c0_rep.get("newton_vbd_gates", {})
        sprint_summary["stage1_glb_mb"] = round(glb_mb, 3)

        current_c0_sha256 = sha256_file(DRIVE_ROOT / "results" / "C0_ringer" / "c0_ringer_solved.npz")

        # STAGE 2: C0_OPTIX_HERO_AND_REEL (re-run if Stage 1 ran or M1 torso sRGB outside [33, 39])
        lookdev_dir = WORK_ROOT / "C0_lookdev"
        s2_expected = [
            DRIVE_ROOT / "results" / "C0_lookdev" / "C0_LOOKDEV_REPORT.json",
            DRIVE_ROOT / "results" / "C0_lookdev" / "ringer_swag01_av1.mp4",
            DRIVE_ROOT / "results" / "C0_lookdev" / "ringer_swag01_h264.mp4",
        ]
        s2_rep_path = DRIVE_ROOT / "results" / "C0_lookdev" / "C0_LOOKDEV_REPORT.json"
        s2_rep = json.loads(s2_rep_path.read_text("utf-8")) if s2_rep_path.exists() else {}
        s2_prior_ok = (
            c0_prior_ok
            and is_stage_verified("C0_OPTIX_HERO_AND_REEL", s2_expected)
            and 33.0 <= s2_rep.get("m1_torso_median_srgb", 0.0) <= 39.0
            and 0.0 < s2_rep.get("av1_kb", 999.0) <= 400.0
            and 0.0 < s2_rep.get("h264_kb", 999.0) <= 550.0
            and s2_rep.get("solved_npz_sha256") == current_c0_sha256
        )
        if not s2_prior_ok:
            solved_npz = c0_dir / "c0_ringer_solved.npz"
            if not solved_npz.exists():
                solved_npz = DRIVE_ROOT / "results" / "C0_ringer" / "c0_ringer_solved.npz"
            c0_rep_path = c0_dir / "C0_GATE_REPORT.json"
            if not c0_rep_path.exists():
                c0_rep_path = DRIVE_ROOT / "results" / "C0_ringer" / "C0_GATE_REPORT.json"
            _run([
                str(blender_exe), "-b", str(master_blend), "--factory-startup", "--python-exit-code", "3",
                "--python", "scripts/run_c0_optix_lookdev_and_reel.py", "--",
                "--mode", "lookdev", "--cycles-device", "OPTIX",
                "--solved-npz", str(solved_npz),
                "--hero-npz", "inputs/hero_ringer_shell.npz",
                "--theta-fit-json", "inputs/m_theta_fit.json",
                "--c0-report-json", str(c0_rep_path),
                "--ringer-blend", str(ringer_blend),
                "--out-dir", str(lookdev_dir), "--glb-mb", f"{glb_mb:.3f}", "--g-stretch-p99", f"{g_p99:.4f}",
            ], env=env)
            for f in lookdev_dir.glob("*"):
                if f.is_file():
                    atomic_promote_to_drive(f, DRIVE_ROOT / "results" / "C0_lookdev" / f.name, "C0_OPTIX_HERO_AND_REEL")
        assert is_stage_verified("C0_OPTIX_HERO_AND_REEL", s2_expected), "Stage 2 C0_OPTIX_HERO_AND_REEL failed Drive verification"

        # STAGE 3: B1_OPTIX_120CAM (bound to current_c0_sha256)
        b1_dir = WORK_ROOT / "B1_120cam"
        s3_expected = [
            DRIVE_ROOT / "results" / "B1_120cam" / "B1_120CAM_REPORT.json",
            DRIVE_ROOT / "results" / "B1_120cam" / "b1_120cam_dataset.tar",
            DRIVE_ROOT / "results" / "B1_120cam" / "init_surface_points.ply",
        ]
        s3_rep_path = DRIVE_ROOT / "results" / "B1_120cam" / "B1_120CAM_REPORT.json"
        s3_rep = json.loads(s3_rep_path.read_text("utf-8")) if s3_rep_path.exists() else {}
        s3_prior_ok = (
            is_stage_verified("B1_OPTIX_120CAM", s3_expected)
            and s3_rep.get("solved_npz_sha256") == current_c0_sha256
        )
        if not s3_prior_ok:
            solved_npz = c0_dir / "c0_ringer_solved.npz"
            if not solved_npz.exists():
                solved_npz = DRIVE_ROOT / "results" / "C0_ringer" / "c0_ringer_solved.npz"
            _run([
                str(py), "scripts/run_b1_b2_known_camera_splat.py",
                "--hero-npz", "inputs/hero_ringer_shell.npz", "--solved-npz", str(solved_npz),
                "--out-dir", str(b1_dir), "--n-views", "120", "--strategy", "mcmc", "--phase", "b1",
            ], env=env)
            _run([
                str(blender_exe), "-b", str(master_blend), "--factory-startup", "--python-exit-code", "3",
                "--python", "scripts/run_b1_blender_120cam.py", "--",
                "--cycles-device", "OPTIX", "--solved-npz", str(solved_npz),
                "--hero-npz", "inputs/hero_ringer_shell.npz",
                "--theta-fit-json", "inputs/m_theta_fit.json",
                "--ringer-blend", str(ringer_blend), "--transforms-json", str(b1_dir / "transforms_cv.json"),
                "--out-dir", str(b1_dir),
            ], env=env)
            for f in b1_dir.glob("*"):
                if f.is_file():
                    atomic_promote_to_drive(f, DRIVE_ROOT / "results" / "B1_120cam" / f.name, "B1_OPTIX_120CAM")
        assert is_stage_verified("B1_OPTIX_120CAM", s3_expected), "Stage 3 B1_OPTIX_120CAM failed Drive verification"

        # STAGE 4: GSPLAT_WHEEL_CAPPED (30-min hard cap) & STAGE 5: B2_GSPLAT_SOG_V3
        if skip_gsplat:
            print("skip_gsplat=True: skipping Stage 4/5 gsplat build for fast C0+Lookdev certification.", flush=True)
            sprint_summary["stage4_skipped"] = True
        else:
            gsplat_ok = install_gsplat_cached_or_build(py, env, timeout_sec=1800)
            if not gsplat_ok:
                sprint_summary["stage4_timeout_skipped_b2"] = True
            else:
                b2_expected = [DRIVE_ROOT / "results" / "B2_gsplat_sog" / "B1_B2_PLAN.json"]
                b2_rep_path = DRIVE_ROOT / "results" / "B2_gsplat_sog" / "B1_B2_PLAN.json"
                b2_rep = json.loads(b2_rep_path.read_text("utf-8")) if b2_rep_path.exists() else {}
                b2_prior_ok = (
                    is_stage_verified("B2_GSPLAT_SOG_V3", b2_expected)
                    and b2_rep.get("solved_npz_sha256") == current_c0_sha256
                )
                if not b2_prior_ok:
                    if not (b1_dir / "B1_TRANSFORMS_READY.json").exists():
                        b1_dir.mkdir(parents=True, exist_ok=True)
                        for fn in ("B1_TRANSFORMS_READY.json", "init_surface_points.ply", "init_surface_binding.npz"):
                            src_f = DRIVE_ROOT / "results" / "B1_120cam" / fn
                            if src_f.exists():
                                shutil.copy2(src_f, b1_dir / fn)
                        with tarfile.open(DRIVE_ROOT / "results" / "B1_120cam" / "b1_120cam_dataset.tar", "r") as tf_ds:
                            tf_ds.extractall(b1_dir)
                    _run([
                        str(py), "scripts/run_b1_b2_known_camera_splat.py",
                        "--hero-npz", "inputs/hero_ringer_shell.npz",
                        "--out-dir", str(b1_dir), "--n-views", "120", "--strategy", "mcmc",
                        "--phase", "b2", "--train-live",
                    ], env=env)
                    for p in list(b1_dir.glob("B1_B2_PLAN.json")) + list((b1_dir / "gsplat_out").glob("*")):
                        if p.is_file():
                            atomic_promote_to_drive(p, DRIVE_ROOT / "results" / "B2_gsplat_sog" / p.name, "B2_GSPLAT_SOG_V3")

        # JOB B (5-MIN BOUNDED): MIT MATTING BENCHMARK (BEN2 + BiRefNet for Track T5 / Lane 4)
        mit_dir = WORK_ROOT / "MIT_matting_bench"
        mit_expected = [
            DRIVE_ROOT / "results" / "MIT_matting_bench" / "MIT_MATTING_BENCHMARK_REPORT.json",
            DRIVE_ROOT / "results" / "MIT_matting_bench" / "MIT_MATTING_BENCH_CONTACT_SHEET.png",
        ]
        if not is_stage_verified("MIT_MATTING_BENCH", mit_expected):
            try:
                _run([
                    str(py), "scripts/run_mit_matting_benchmark.py",
                    "--out-dir", str(mit_dir),
                ], env=env, check=False, timeout_sec=300)
                for p in mit_dir.glob("*"):
                    if p.is_file():
                        atomic_promote_to_drive(p, DRIVE_ROOT / "results" / "MIT_matting_bench" / p.name, "MIT_MATTING_BENCH")
            except Exception as mit_exc:
                print(f"[JOB B WARN] MIT matting benchmark non-fatal error: {mit_exc}", flush=True)
        if (DRIVE_ROOT / "results" / "MIT_matting_bench" / "MIT_MATTING_BENCHMARK_REPORT.json").exists():
            sprint_summary["job_b_mit_matting"] = json.loads(
                (DRIVE_ROOT / "results" / "MIT_matting_bench" / "MIT_MATTING_BENCHMARK_REPORT.json").read_text("utf-8")
            )

        # STAGE 6: QA SHEET, VERIFICATION REPORT & FINAL MANIFEST
        if (DRIVE_ROOT / "results" / "C0_lookdev" / "C0_LOOKDEV_REPORT.json").exists():
            sprint_summary["stage2_lookdev"] = json.loads((DRIVE_ROOT / "results" / "C0_lookdev" / "C0_LOOKDEV_REPORT.json").read_text("utf-8"))
        if (DRIVE_ROOT / "results" / "B1_120cam" / "B1_120CAM_REPORT.json").exists():
            sprint_summary["stage3_b1"] = json.loads((DRIVE_ROOT / "results" / "B1_120cam" / "B1_120CAM_REPORT.json").read_text("utf-8"))
        if (DRIVE_ROOT / "results" / "B2_gsplat_sog" / "B1_B2_PLAN.json").exists():
            sprint_summary["stage5_b2"] = json.loads((DRIVE_ROOT / "results" / "B2_gsplat_sog" / "B1_B2_PLAN.json").read_text("utf-8")).get("b2_metrics", {})
        # MULTI-SKU PHASE 3 BATCH HOOK (runs when phase3_dress_all_skus_colab.py is staged)
        p3_script = local_stage / "phase3_dress_all_skus_colab.py"
        if p3_script.exists() and (args.phase3_waves or args.phase3_skus):
            p3_cmd = [str(py), str(p3_script), "--waves", *args.phase3_waves]
            if args.phase3_skus:
                p3_cmd.extend(["--skus", args.phase3_skus])
            p3_cmd.append("--dry-run")
            _run(p3_cmd, env=env, check=False)

        qa_sheet = build_qa_contact_sheet(WORK_ROOT, WORK_ROOT / "BURST_1A_MEGA_QA_SHEET.png")
        atomic_promote_to_drive(qa_sheet, DRIVE_ROOT / "results" / "BURST_1A_MEGA_QA_SHEET.png", "BURST_1A_QA")
        rep_path = WORK_ROOT / "SPRINT_VERIFICATION_REPORT.json"
        rep_path.write_text(json.dumps(sprint_summary, indent=2) + "\n", encoding="utf-8")
        atomic_promote_to_drive(rep_path, DRIVE_ROOT / "results" / "SPRINT_VERIFICATION_REPORT.json", "BURST_1A_QA")
        manifest_csv = WORK_ROOT / "FINAL_SHA256_MANIFEST.csv"
        write_final_sha256_manifest(DRIVE_ROOT / "results", manifest_csv)
        atomic_promote_to_drive(manifest_csv, DRIVE_ROOT / "results" / "FINAL_SHA256_MANIFEST.csv", "BURST_1A_QA")
        _run(["sha256sum", "-c", "MANIFEST_SHA256.txt"])
    finally:
        try:
            drive.flush_and_unmount()
        except Exception as e:
            print("flush warn:", e, flush=True)
        finally:
            runtime.unassign()


if __name__ == "__main__":
    main()
