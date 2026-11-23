"""06_splat_binding_and_relight/zesch_triangle_binder.py — Clean-Room Per-Triangle Gaussian Deformation Binding.

Implements numerically stable square-root SVD covariance transport directly on M_k = F_i @ R_k @ S_k
(never eigendecomposing F Sigma F^T, avoiding float32 condition-number squaring on thin disk splats),
two-sided anisotropy clamping (r_max=10.0, s_cap=0.015 m), and polar decomposition R_{p,i} in SO(3)
for Wigner-D / rotated-view-direction SH parity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def compute_triangle_basis(V: np.ndarray, F: np.ndarray, area_scale_normal: bool = True) -> np.ndarray:
    """Build per-triangle 3x3 basis matrices D_i = [v1 - v0 | v2 - v0 | s_n * unit_normal]. Shape: (M, 3, 3)."""
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    n_raw = np.cross(e1, e2)
    n_len = np.linalg.norm(n_raw, axis=1, keepdims=True)
    n = n_raw / np.maximum(n_len, 1e-12)
    if area_scale_normal:
        tri_scale = np.sqrt(np.maximum(n_len, 1e-12))
    else:
        tri_scale = np.ones_like(n_len)
    return np.stack([e1, e2, n * tri_scale], axis=2)


def compute_deformation_gradients(
    V_rest: np.ndarray,
    V_deformed: np.ndarray,
    F: np.ndarray,
    max_singular_ratio: float = 2.0,
) -> np.ndarray:
    """Compute F_i = D_def * inv(D_rest) for each triangle, clamping extreme singular stretches."""
    D_m = compute_triangle_basis(V_rest, F)
    D_s = compute_triangle_basis(V_deformed, F)
    F_mats = np.empty_like(D_m)
    for i in range(len(D_m)):
        try:
            Fi = D_s[i] @ np.linalg.inv(D_m[i])
        except np.linalg.LinAlgError:
            Fi = np.eye(3, dtype=np.float64)
        if max_singular_ratio > 0:
            U, S, Vt = np.linalg.svd(Fi)
            s_min = max(float(S.min()), 1e-6)
            S = np.clip(S, s_min, s_min * max_singular_ratio)
            Fi = (U * S[None, :]) @ Vt
        F_mats[i] = Fi
    return F_mats


def polar_rotation_so3(F_mats: np.ndarray) -> np.ndarray:
    """Extract proper SO(3) local rotation R_{p,i} = U diag(1, 1, det(U V^T)) V^T via polar decomposition."""
    U, _, Vt = np.linalg.svd(F_mats)
    R = np.einsum("mij,mjk->mik", U, Vt)
    det = np.linalg.det(R)
    neg = det < 0.0
    if np.any(neg):
        U[neg, :, 2] *= -1.0
        R[neg] = np.einsum("mij,mjk->mik", U[neg], Vt[neg])
    return R


def wigner_d1_from_rotation(R_p: np.ndarray) -> np.ndarray:
    """Exact l=1 real Wigner D-matrix in gsplat/3DGS [-y, z, -x] basis: D^(1) = P R_p P^T."""
    P = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]], dtype=np.float64)
    return np.einsum("ab,mbc,dc->mad", P, R_p, P)


@dataclass
class BoundGarmentSplats:
    garment_id: int
    face_ids: np.ndarray        # (K,) int64
    bary: np.ndarray            # (K, 3) float64
    local_offsets: np.ndarray   # (K, 3) float64 r_k in rest world frame (clamped <= 3mm)
    rots_rest: np.ndarray       # (K, 3, 3) float64 R_k in SO(3)
    scales_rest: np.ndarray     # (K, 3) float64 (s_1, s_2, s_3)
    ids: np.ndarray             # (K,) int32 id(k) = garment_id


def bind_gaussians_to_triangles(
    means_rest: np.ndarray,
    covs_or_rots_rest: np.ndarray,
    V_rest: np.ndarray,
    F: np.ndarray,
    garment_id: int,
    scales_rest: np.ndarray | None = None,
    max_normal_offset_m: float = 0.003,
) -> BoundGarmentSplats:
    """Attach each Gaussian center to its nearest triangle centroid + clamped normal offset."""
    from scipy.spatial import cKDTree  # type: ignore

    means = np.asarray(means_rest, dtype=np.float64)
    V = np.asarray(V_rest, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)

    if scales_rest is None:
        # Decompose rest covariance matrices Sigma_k = R_k S_k^2 R_k^T into R_k, S_k
        eigvals, eigvecs = np.linalg.eigh(np.asarray(covs_or_rots_rest, dtype=np.float64))
        scales = np.sqrt(np.maximum(eigvals[:, ::-1], 1e-12))
        rots = eigvecs[:, :, ::-1]
        neg_det = np.linalg.det(rots) < 0.0
        rots[neg_det, :, 2] *= -1.0
    else:
        rots = np.asarray(covs_or_rots_rest, dtype=np.float64)
        scales = np.asarray(scales_rest, dtype=np.float64)

    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    centroids = (A + B + C) / 3.0
    _, face_ids = cKDTree(centroids).query(means)
    face_ids = np.asarray(face_ids, dtype=np.int64)

    a, b, c = A[face_ids], B[face_ids], C[face_ids]
    v0, v1, v2 = b - a, c - a, means - a
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    denom = np.maximum(d00 * d11 - d01 * d01, 1e-12)
    v_w = (d11 * d20 - d01 * d21) / denom
    w_w = (d00 * d21 - d01 * d20) / denom
    u_w = 1.0 - v_w - w_w
    bary = np.clip(np.stack([u_w, v_w, w_w], axis=1), -0.5, 1.5)
    bary /= np.maximum(bary.sum(axis=1, keepdims=True), 1e-12)

    anchor = bary[:, 0:1] * a + bary[:, 1:2] * b + bary[:, 2:3] * c
    r_k = means - anchor
    # Clamp normal offset to [-0.003, +0.003] m
    fn = np.cross(v0, v1)
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    h_k = np.einsum("ij,ij->i", r_k, fn)
    r_tang = r_k - h_k[:, None] * fn
    r_k = r_tang + np.clip(h_k, -max_normal_offset_m, max_normal_offset_m)[:, None] * fn

    ids = np.full(len(means), int(garment_id), dtype=np.int32)
    return BoundGarmentSplats(
        garment_id=int(garment_id),
        face_ids=face_ids,
        bary=bary,
        local_offsets=r_k,
        rots_rest=rots,
        scales_rest=scales,
        ids=ids,
    )


def transport_bound_gaussians(
    bound: BoundGarmentSplats,
    V_rest: np.ndarray,
    V_deformed: np.ndarray,
    F: np.ndarray,
    max_singular_ratio: float = 2.0,
    r_max: float = 10.0,
    s_cap: float = 0.015,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Square-root SVD transport on M_k = F_i R_k S_k with two-sided anisotropy clamping.

    Returns (means_new, rots_new, scales_new, R_polar_per_splat, garment_ids).
    """
    F_mats = compute_deformation_gradients(V_rest, V_deformed, F, max_singular_ratio=max_singular_ratio)
    R_polar_tris = polar_rotation_so3(F_mats)

    Fi = F_mats[bound.face_ids]
    R_p = R_polar_tris[bound.face_ids]
    tri_verts = V_deformed[F[bound.face_ids]]
    anchor_new = np.einsum("ka,kai->ki", bound.bary, tri_verts)
    means_new = anchor_new + np.einsum("kij,kj->ki", Fi, bound.local_offsets)

    # Square-root linear factor M_k = F_i @ R_k @ diag(S_k)
    RS = bound.rots_rest * bound.scales_rest[:, None, :]
    M = np.einsum("kij,kjl->kil", Fi, RS)
    U, S, _ = np.linalg.svd(M)
    det_u = np.linalg.det(U)
    neg = det_u < 0.0
    if np.any(neg):
        U[neg, :, 2] *= -1.0

    # Two-sided anisotropy clamping on ordered singular values s0 >= s1 >= s2
    s0 = np.minimum(S[:, 0], np.minimum(r_max * S[:, 1], s_cap))
    s1 = np.minimum(S[:, 1], s0)
    s2 = np.maximum(S[:, 2], s0 / r_max)
    scales_new = np.stack([s0, s1, s2], axis=1)
    return means_new, U, scales_new, R_p, bound.ids


if __name__ == "__main__":
    V0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    R90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    V1 = (V0 @ R90.T) * 1.05 + np.array([0.2, 0.3, 0.1])
    F_tri = np.array([[0, 1, 2]], dtype=np.int64)
    mu0 = np.array([[0.25, 0.25, 0.002]], dtype=np.float64)
    cov0 = np.array([np.diag([0.0001, 0.0002, 0.00005])], dtype=np.float64)
    bound = bind_gaussians_to_triangles(mu0, cov0, V0, F_tri, garment_id=3)
    mu1, rots1, scales1, R_p, gids = transport_bound_gaussians(bound, V0, V1, F_tri)
    expected_mu1 = (mu0 @ R90.T) * 1.05 + np.array([0.2, 0.3, 0.1])
    err = float(np.linalg.norm(mu1 - expected_mu1))
    d1 = wigner_d1_from_rotation(R_p)
    print(f"zesch_sqrt_svd_test err={err:.2e} id={int(gids[0])} scales1={scales1[0].round(6).tolist()} d1_det={np.linalg.det(d1[0]):.4f}")
    assert err < 1e-10
