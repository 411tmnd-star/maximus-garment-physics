"""02_alignment_and_collar: Libigl boundary_loop / heat_geodesic / ARAP + size-graded rest-edge adapters (PBD re-exported)."""
from __future__ import annotations

import importlib.metadata
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ENGINE_ROOT = Path(__file__).resolve().parents[1]
MAXIMUS_ROOT = ENGINE_ROOT.parent / "maximus_engine"
for _p in (MAXIMUS_ROOT, MAXIMUS_ROOT / "colab_stage"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import corpus_link  # noqa: E402
corpus_link.link(strict=os.environ.get("MAXIMUS_STRICT", "1") == "1")
import canon_geometry_kernels as K  # noqa: E402

try:
    import igl  # type: ignore
except ImportError:
    igl = None


def libigl_version() -> str | None:
    """Query libigl version via importlib.metadata (igl.__version__ was removed after 2.5.4.dev0)."""
    try:
        return importlib.metadata.version("libigl")
    except importlib.metadata.PackageNotFoundError:
        return None


def boundary_loop_vertices(F: np.ndarray) -> list[np.ndarray]:
    """Return ordered boundary loops; uses igl.boundary_loop_all (or all_boundary_loop), else half-edge walk."""
    F = np.asarray(F, dtype=np.int64)
    if igl is not None:
        if hasattr(igl, "boundary_loop_all"):
            return [np.asarray(lp, dtype=np.int64) for lp in igl.boundary_loop_all(F)]
        if hasattr(igl, "all_boundary_loop"):
            return [np.asarray(lp, dtype=np.int64) for lp in igl.all_boundary_loop(F)]

    edge_counts: dict[tuple[int, int], int] = {}
    directed: list[tuple[int, int]] = []
    for a, b, c in F:
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            directed.append((u, v))
            key = (u, v) if u < v else (v, u)
            edge_counts[key] = edge_counts.get(key, 0) + 1
    adj: dict[int, list[int]] = {}
    for u, v in directed:
        key = (u, v) if u < v else (v, u)
        if edge_counts[key] == 1:
            adj.setdefault(u, []).append(v)

    visited: set[int] = set()
    loops_out: list[np.ndarray] = []
    for start in sorted(adj):
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        cur = start
        while True:
            nxts = [w for w in adj.get(cur, []) if w not in visited]
            if not nxts:
                break
            cur = nxts[0]
            visited.add(cur)
            loop.append(cur)
        loops_out.append(np.asarray(loop, dtype=np.int64))
    return loops_out


def heat_geodesic(V: np.ndarray, F: np.ndarray, source_indices: np.ndarray, t: float | None = None) -> np.ndarray:
    """Geodesic distance (m) from source_indices via libigl heat_geodesic or clean-room MC-50."""
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    src = np.atleast_1d(np.asarray(source_indices, dtype=np.int64))
    if igl is not None and hasattr(igl, "heat_geodesic"):
        t_val = float(t) if t is not None else float(K._mean_edge_length(V, F) ** 2)
        return np.asarray(igl.heat_geodesic(V, F, t_val, src), dtype=np.float64)
    if igl is not None and hasattr(igl, "heat_geodesics_precompute"):
        t_val = float(t) if t is not None else float(K._mean_edge_length(V, F) ** 2)
        data = igl.heat_geodesics_precompute(V, F, t_val)
        return np.asarray(igl.heat_geodesics_solve(data, src), dtype=np.float64)
    return K.heat_geodesic_distance(V, F, src, t=t)


def collar_geodesic_pin_weights(
    V: np.ndarray,
    F: np.ndarray,
    neck_loop_ids: np.ndarray,
    r0_m: float = 0.005,
    r1_m: float = 0.035,
) -> np.ndarray:
    """C^2 smooth collar pinning weight: 1.0 within r0=5 mm, 0.0 beyond r1=35 mm."""
    d_geo = heat_geodesic(V, F, neck_loop_ids)
    u = np.clip((d_geo - r0_m) / max(r1_m - r0_m, 1e-9), 0.0, 1.0)
    return 1.0 - (10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5)


def grade_garment_rest_edges(
    V_garment_rest: np.ndarray,
    edges: np.ndarray,
    V_body_neutral: np.ndarray,
    V_body_size_target: np.ndarray,
    V_body_wearer: np.ndarray,
    k: int = 8,
    sigma_d: float = 0.03,
    eps_body_m: float = 0.003,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Separate garment size grading (L/XL/XXL) from wearer build (min_lean/neutral/max_stocky).

    Emits CONSTRICTED / SIZE_UP_REQUIRED metadata if wearer chest girth exceeds size-graded elastic capacity (1.12x).
    """
    from scipy.spatial import cKDTree  # type: ignore

    Vg = np.asarray(V_garment_rest, dtype=np.float64)
    E = np.asarray(edges, dtype=np.int64)
    Vb0 = np.asarray(V_body_neutral, dtype=np.float64)
    Vbs = np.asarray(V_body_size_target, dtype=np.float64)
    Vbw = np.asarray(V_body_wearer, dtype=np.float64)

    disp_size = Vbs - Vb0
    dists, nn = cKDTree(Vb0).query(Vg, k=min(k, len(Vb0)))
    w = np.exp(-0.5 * (dists / max(sigma_d, 1e-6)) ** 2)
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    X_grade = Vg + np.einsum("ij,ijk->ik", w, disp_size[nn])
    l_bar = np.linalg.norm(X_grade[E[:, 0]] - X_grade[E[:, 1]], axis=1)

    # Physical feasibility check on torso ring span
    wearer_span = float(np.ptp(Vbw[:, 0]) + np.ptp(Vbw[:, 1]))
    grade_span = float(np.ptp(X_grade[:, 0]) + np.ptp(X_grade[:, 1]))
    feasibility_ratio = (wearer_span + 2.0 * np.pi * eps_body_m / np.pi) / max(grade_span, 1e-6)
    constricted = bool(feasibility_ratio > 1.12)
    meta = {
        "feasibility_ratio": round(feasibility_ratio, 4),
        "status": "CONSTRICTED / SIZE_UP_REQUIRED" if constricted else "FEASIBLE",
        "constricted": constricted,
    }
    return l_bar, meta


# Single owner (spec §1): the PBD strain limiter lives in 03_gpu_cloth_solver/newton_cloth_template.py. The copy-pasted
# body that used to sit here is retired; this re-export keeps `alignment_adapters.pbd_strain_limit_and_tangent_clearance`
# working for existing callers.
_SOLVER_DIR = ENGINE_ROOT / "03_gpu_cloth_solver"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SOLVER_DIR))
from newton_cloth_template import LayerCollider, pbd_strain_limit_and_tangent_clearance  # noqa: E402,F401


def arap_relax(
    V_rest: np.ndarray,
    V_init: np.ndarray,
    F: np.ndarray,
    active_indices: np.ndarray,
    pinned_indices: np.ndarray | None = None,
    n_iters: int = 8,
) -> tuple[np.ndarray, list[float]]:
    """ARAP relaxation enforcing collar pin lock using MC-52 (or libigl ARAP when dense)."""
    return K.arap_pose_relaxation(V_rest, V_init, F, active_indices, pinned_indices=pinned_indices, n_iters=n_iters)


if __name__ == "__main__":
    tri_F = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int64)
    tri_V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
    loops = boundary_loop_vertices(tri_F)
    w_pin = collar_geodesic_pin_weights(tri_V, tri_F, np.array([0]))
    print(f"igl_version={libigl_version()} loops={[len(x) for x in loops]} pin_min={w_pin.min():.4f} pin_max={w_pin.max():.4f}")
