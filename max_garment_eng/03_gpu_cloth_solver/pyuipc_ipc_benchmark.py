"""03_gpu_cloth_solver/pyuipc_ipc_benchmark.py — Exact-Contact IPC Benchmark (`spiriMirror/libuipc`).

Licence: the upstream dependency `spiriMirror/libuipc` is Apache-2.0; this wrapper script is
covered by this repository's LICENSE.

Targets SIGGRAPH 2025 Stiff-GIPC / AL-IPC (`91_pinned_cloth` & `95_anisotropic_cloth` patterns)
for multi-layer collars, cuffs, and rigid zipper hardware where zero interpenetration must hold strictly.
PyPI wheel `pyuipc==0.0.29` ships cp310-cp314 CUDA 12.8 `sm_89` SASS compatible with Colab L4 CUDA 13 drivers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def inspect_pyuipc_runtime() -> dict[str, Any]:
    out: dict[str, Any] = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": sys.platform,
        "pyuipc_installed": False,
        "pypi_verified_wheel": "pyuipc==0.0.29 (CUDA 12.8 sm_89 backward-compatible with L4 Driver 580 / CUDA 13)",
        "doctor_cmd": "python -m uipc doctor --probe-cuda",
    }
    try:
        import uipc  # type: ignore

        out["pyuipc_installed"] = True
        out["uipc_version"] = getattr(uipc, "__version__", "unknown")
    except Exception:
        try:
            import pyuipc  # type: ignore

            out["pyuipc_installed"] = True
            out["uipc_version"] = getattr(pyuipc, "__version__", "unknown")
        except Exception as exc:
            out["import_error"] = str(exc)
    return out


def solve_ipc_anisotropic_cloth(
    V_cloth: np.ndarray,
    F_cloth: np.ndarray,
    pinned_mask: np.ndarray,
    workspace: Path,
    n_frames: int = 30,
    dt: float = 1.0 / 60.0,
) -> dict[str, Any]:
    """Run libuipc exact-contact IPC shell solve on CUDA GPU (pattern from spiriMirror/libuipc-samples)."""
    from uipc import Engine, Scene, World, builtin, view  # type: ignore
    from uipc.constitution import (  # type: ignore
        DiscreteShellBending,
        ElasticModuli2D,
        StrainLimitingBaraffWitkinShell,
    )
    from uipc.geometry import label_surface, trimesh  # type: ignore

    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    engine = Engine("cuda", str(workspace))
    world = World(engine)

    config = Scene.default_config()
    config["dt"] = float(dt)
    scene = Scene(config)

    mesh = trimesh(np.asarray(V_cloth, dtype=np.float64), np.asarray(F_cloth, dtype=np.int32))
    label_surface(mesh)
    is_fixed = mesh.vertices().create(builtin.is_fixed, 0)
    view(is_fixed).reshape(-1)[np.asarray(pinned_mask, dtype=bool)] = 1

    bw = StrainLimitingBaraffWitkinShell()
    bend = DiscreteShellBending()
    moduli = ElasticModuli2D.youngs_modulus(1.0e5, 0.3)
    bw.apply_to(mesh, moduli)
    bend.apply_to(mesh, 1.0e-2)

    obj = scene.objects().create("garment_ipc")
    geo_slot, _ = obj.geometries().create(mesh)
    world.init(scene)

    for _ in range(int(n_frames)):
        world.advance()
        world.retrieve()

    V_out = np.asarray(geo_slot.geometry().positions().view(), dtype=np.float64).copy()
    return {
        "solver": "uipc.StrainLimitingBaraffWitkinShell+DiscreteShellBending",
        "n_frames": int(n_frames),
        "vertices_out": V_out,
    }


if __name__ == "__main__":
    print(json.dumps(inspect_pyuipc_runtime(), indent=2))
