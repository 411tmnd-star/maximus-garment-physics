"""03_gpu_cloth_solver/blender_cloth_fallback.py — Headless Blender 5.2.3 Cloth Benchmark Wrapper (External Process Only)."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def find_blender() -> str | None:
    for cand in (
        shutil.which("blender"),
        r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe",
    ):
        if cand and Path(cand).exists():
            return str(cand)
    return None


def run_blender_headless(script_path: Path, args: list[str] | None = None, dry_run: bool = True) -> dict:
    blender_exe = find_blender()
    cmd = [blender_exe or "blender", "-b", "--factory-startup", "-P", str(script_path)]
    if args:
        cmd += ["--", *args]
    if dry_run or blender_exe is None:
        return {"executed": False, "blender_exe": blender_exe, "cmd": cmd}
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return {
        "executed": True,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-1000:],
        "stderr_tail": proc.stderr[-1000:],
    }


if __name__ == "__main__":
    print(json.dumps(run_blender_headless(Path(__file__), dry_run=True), indent=2))
