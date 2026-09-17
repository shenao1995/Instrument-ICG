"""Regularized Newton step for ICG region (and optional texture) residuals."""
from __future__ import annotations

import numpy as np

from icg.kinematics import (ARM_DIM, STATE_DIM, apply_increment, full_point_jacobian,
                            pose_numpy, projection_jacobian)


def project_point(xyz, k):
    z = max(float(xyz[2]), 1e-6)
    return np.array([k[0, 0] * xyz[0] / z + k[0, 2],
                     k[1, 1] * xyz[1] / z + k[1, 2]], np.float64)


def assemble_normal(lines, rotations, translations, joints, k, pivot, shaft_offset,
                    optimize_joints, sigma, lambda_r, lambda_t, lambda_j, texture_lines=()):
    """Build gradient g and Hessian H of the ICG negative log-likelihood."""
    g = np.zeros(STATE_DIM, np.float64)
    h = np.zeros((STATE_DIM, STATE_DIM), np.float64)
    used = 0
    fx, fy = float(k[0, 0]), float(k[1, 1])
    inv_var = 1.0 / max(sigma * sigma, 1e-8)

    def add_residual(xyz, part_id, observed, normal, weight):
        nonlocal g, h, used
        if part_id is None:
            # Texture 3D points are in the previous camera frame; treat as wrist-attached
            # on the nearest instrument by x-coordinate of the projection.
            uv = project_point(xyz, k)
            part_id = 1 if uv[0] < k[0, 2] else 5  # left/right wrist
        jac_x = full_point_jacobian(xyz, part_id, rotations, translations, joints,
                                    pivot, shaft_offset, optimize_joints)
        jac_pi = projection_jacobian(xyz, fx, fy)
        predicted = project_point(xyz, k)
        if normal is None:
            residual = predicted - observed
            jac = jac_pi @ jac_x
        else:
            residual = np.array([normal @ (predicted - observed)])
            jac = (normal @ jac_pi)[None] @ jac_x
        scale = float(weight) * inv_var
        g += scale * (jac.T @ residual)
        h += scale * (jac.T @ jac)
        used += 1

    for line in lines:
        add_residual(line["xyz"], line["part_id"], line["observed"], line["n"], line["weight"])
    for line in texture_lines:
        add_residual(line["xyz"], line.get("part_id"), line["observed"], None, line["weight"])

    reg = np.zeros(STATE_DIM, np.float64)
    for arm in range(STATE_DIM // ARM_DIM):
        base = arm * ARM_DIM
        reg[base:base + 3] = lambda_r
        reg[base + 3:base + 6] = lambda_t
        reg[base + 6:base + 9] = lambda_j
    h += np.diag(reg)
    return g, h, used


def newton_step(g, h):
    try:
        delta = -np.linalg.solve(h, g)
    except np.linalg.LinAlgError:
        delta = -np.linalg.lstsq(h, g, rcond=None)[0]
    # Keep a single iteration inside the trust of small ICG increments.
    rot = np.linalg.norm(delta.reshape(-1, ARM_DIM)[:, :3], axis=1).max()
    trans = np.linalg.norm(delta.reshape(-1, ARM_DIM)[:, 3:6], axis=1).max()
    if rot > 0.35 or trans > 0.02:
        delta *= min(0.35 / max(rot, 1e-8), 0.02 / max(trans, 1e-8), 1.0)
    return delta


def update_pose_tensors(pose, delta, alpha_limit, jaw_limit):
    import torch
    r, t, j = pose_numpy(pose)
    r, t, j = apply_increment(r, t, j, delta, alpha_limit, jaw_limit)
    device, dtype = pose["R"].device, pose["R"].dtype
    pose = {
        "R": torch.as_tensor(r, device=device, dtype=dtype)[None],
        "t": torch.as_tensor(t, device=device, dtype=dtype)[None],
        "joints": torch.as_tensor(j, device=device, dtype=dtype)[None],
    }
    return pose
