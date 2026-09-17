"""Mb-ICG body Jacobians for the verified dual-instrument kinematic tree.

Each instrument is a tree rooted at the wrist (6-DoF in the OpenCV camera):
  wrist  --Ry(-alpha)--> shaft
  wrist  --Rz(+theta_left)--> left jaw  (origin at the jaw pivot)
  wrist  --Rz(-theta_right)--> right jaw

Incremental pose variation θ = (θr, θt) lives in the wrist/model frame as in
ICG: X_cam(θ) = R ((I + [θr]×) x + θt) + t.
Joint variations are added to (alpha, theta_left, theta_right).
Two instruments are independent trees, concatenated into an 18-vector.
"""
from __future__ import annotations

import numpy as np

N_ARMS = 2
ARM_DIM = 9
STATE_DIM = ARM_DIM * N_ARMS
PART_SHAFT, PART_WRIST, PART_JAW_L, PART_JAW_R = 0, 1, 2, 3


def skew(v):
    x, y, z = np.asarray(v, np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], np.float64)


def rotation_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], np.float64)


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], np.float64)


def d_rotation_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[-s, 0.0, c], [0.0, 0.0, 0.0], [-c, 0.0, -s]], np.float64)


def d_rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[-s, -c, 0.0], [c, -s, 0.0], [0.0, 0.0, 0.0]], np.float64)


def exp_so3(rotvec):
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3) + skew(rotvec)
    axis = np.asarray(rotvec, np.float64) / theta
    k = skew(axis)
    return np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def projection_jacobian(xyz, fx, fy):
    x, y, z = [float(v) for v in np.asarray(xyz, np.float64).reshape(3)]
    z = max(z, 1e-6)
    return np.array([[fx / z, 0.0, -fx * x / (z * z)],
                     [0.0, fy / z, -fy * y / (z * z)]], np.float64)


def _wrist_local(xyz, rotation, translation):
    return rotation.T @ (np.asarray(xyz, np.float64).reshape(3) - translation)


def point_jacobian(xyz, part_id, rotation, translation, joints, pivot, shaft_offset, optimize_joints):
    """Camera-space Jacobian of a rigid-body point wrt one instrument's 9-vector.

    Returns (3, 9). Columns: θr (3), θt (3), alpha, theta_left, theta_right.
    """
    xyz = np.asarray(xyz, np.float64).reshape(3)
    rotation = np.asarray(rotation, np.float64).reshape(3, 3)
    translation = np.asarray(translation, np.float64).reshape(3)
    alpha, theta_l, theta_r = [float(v) for v in np.asarray(joints, np.float64).reshape(3)]
    part = int(part_id) % 4
    wrist = _wrist_local(xyz, rotation, translation)
    jac = np.zeros((3, ARM_DIM), np.float64)

    if part == PART_SHAFT:
        ry = rotation_y(-alpha)
        x_shaft = ry.T @ wrist
        x_model = ry @ x_shaft
        jac[:, 6] = rotation @ (d_rotation_y(-alpha) * (-1.0) @ x_shaft)
    elif part == PART_WRIST:
        x_model = wrist
    elif part == PART_JAW_L:
        pivot_v = np.array([pivot, 0.0, 0.0])
        rz = rotation_z(theta_l)
        x_jaw = rz.T @ (wrist - pivot_v)
        x_model = rz @ x_jaw + pivot_v
        jac[:, 7] = rotation @ (d_rotation_z(theta_l) @ x_jaw)
    else:
        pivot_v = np.array([pivot, 0.0, 0.0])
        rz = rotation_z(-theta_r)
        x_jaw = rz.T @ (wrist - pivot_v)
        x_model = rz @ x_jaw + pivot_v
        jac[:, 8] = rotation @ (d_rotation_z(-theta_r) * (-1.0) @ x_jaw)

    jac[:, :3] = -rotation @ skew(x_model)
    jac[:, 3:6] = rotation
    if not optimize_joints:
        jac[:, 6:9] = 0.0
    return jac


def full_point_jacobian(xyz, part_id, rotations, translations, joints, pivot, shaft_offset,
                        optimize_joints=True):
    """Jacobian (3, 18) of a camera-space point wrt the dual-instrument state."""
    arm = int(part_id) // 4
    local = point_jacobian(xyz, part_id, rotations[arm], translations[arm], joints[arm],
                           pivot, shaft_offset, optimize_joints)
    full = np.zeros((3, STATE_DIM), np.float64)
    full[:, arm * ARM_DIM:(arm + 1) * ARM_DIM] = local
    return full


def apply_increment(rotations, translations, joints, delta, alpha_limit, jaw_limit):
    """Apply an ICG model-frame increment to both instruments. ``delta`` is (18,)."""
    delta = np.asarray(delta, np.float64).reshape(STATE_DIM)
    r_out, t_out, j_out = [], [], []
    for arm in range(N_ARMS):
        d = delta[arm * ARM_DIM:(arm + 1) * ARM_DIM]
        rotation = np.asarray(rotations[arm], np.float64)
        translation = np.asarray(translations[arm], np.float64)
        r_out.append(rotation @ exp_so3(d[:3]))
        t_out.append(translation + rotation @ d[3:6])
        joints_arm = np.asarray(joints[arm], np.float64).reshape(3) + d[6:9]
        joints_arm[0] = np.clip(joints_arm[0], -alpha_limit, alpha_limit)
        joints_arm[1] = np.clip(joints_arm[1], -jaw_limit, jaw_limit)
        joints_arm[2] = np.clip(joints_arm[2], -jaw_limit, jaw_limit)
        j_out.append(joints_arm)
    return np.stack(r_out), np.stack(t_out), np.stack(j_out)


def pose_numpy(pose):
    """Detach a batched pose dict to numpy arrays shaped [A,...]."""
    r = pose["R"].detach().cpu().numpy()
    t = pose["t"].detach().cpu().numpy()
    j = pose["joints"].detach().cpu().numpy()
    if r.ndim == 4:
        r, t, j = r[0], t[0], j[0]
    return r, t, j
