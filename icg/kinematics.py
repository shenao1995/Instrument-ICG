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


def camera_to_part_local(xyz, part_id, rotation, translation, joints, pivot, shaft_offset):
    """Inverse of the existing FK; coordinates are in the canonical rigid CAD part."""
    wrist = _wrist_local(xyz, np.asarray(rotation, np.float64),
                         np.asarray(translation, np.float64))
    part = int(part_id) % 4
    alpha, theta_l, theta_r = joints
    if part == PART_SHAFT:
        return rotation_y(-alpha).T @ wrist + np.array([shaft_offset, 0., 0.])
    if part == PART_WRIST:
        return wrist
    angle = theta_l if part == PART_JAW_L else -theta_r
    return rotation_z(angle).T @ (wrist - np.array([pivot, 0., 0.]))


def point_and_jacobian_from_local(local_xyz, part_id, rotation, translation, joints,
                                  pivot, shaft_offset, optimize_joints=True):
    """Return X_camera and its (3,9) model-frame tangent Jacobian at THIS pose.

    The supplied local surface point is fixed, including during LM retries.
    """
    x = np.asarray(local_xyz, np.float64).reshape(3)
    rotation = np.asarray(rotation, np.float64).reshape(3, 3)
    translation = np.asarray(translation, np.float64).reshape(3)
    alpha, theta_l, theta_r = joints
    part = int(part_id) % 4
    jac = np.zeros((3, ARM_DIM), np.float64)
    if part == PART_SHAFT:
        shifted = x - np.array([shaft_offset, 0., 0.])
        wrist = rotation_y(-alpha) @ shifted
        jac[:, 6] = rotation @ (-d_rotation_y(-alpha) @ shifted)
    elif part == PART_WRIST:
        wrist = x
    else:
        angle, sign, column = (theta_l, 1., 7) if part == PART_JAW_L else (-theta_r, -1., 8)
        wrist = rotation_z(angle) @ x + np.array([pivot, 0., 0.])
        jac[:, column] = rotation @ (sign * d_rotation_z(angle) @ x)
    jac[:, :3] = -rotation @ skew(wrist)
    jac[:, 3:6] = rotation
    if not optimize_joints:
        jac[:, 6:] = 0.
    return rotation @ wrist + translation, jac


def part_local_to_camera(local_xyz, part_id, rotation, translation, joints, pivot, shaft_offset):
    return point_and_jacobian_from_local(local_xyz, part_id, rotation, translation,
                                         joints, pivot, shaft_offset)[0]


def batch_points_and_jacobians_from_local(local_xyz, part_id, rotation, translation, joints,
                                           pivot, shaft_offset, optimize_joints=True):
    """Vectorized scalar FK/Jacobian for points on ONE rigid part; no convention change."""
    x = np.asarray(local_xyz, np.float64).reshape(-1, 3)
    rotation = np.asarray(rotation, np.float64)
    part = int(part_id) % 4
    alpha, theta_l, theta_r = joints
    jac = np.zeros((len(x), 3, ARM_DIM), np.float64)
    if part == PART_SHAFT:
        shifted = x - np.array([shaft_offset, 0., 0.])
        wrist = shifted @ rotation_y(-alpha).T
        jac[:, :, 6] = (shifted @ (-d_rotation_y(-alpha)).T) @ rotation.T
    elif part == PART_WRIST:
        wrist = x
    else:
        angle, sign, column = (theta_l, 1., 7) if part == PART_JAW_L else (-theta_r, -1., 8)
        wrist = x @ rotation_z(angle).T + np.array([pivot, 0., 0.])
        jac[:, :, column] = (x @ (sign * d_rotation_z(angle)).T) @ rotation.T
    skew_batch = np.zeros((len(x), 3, 3), np.float64)
    skew_batch[:, 0, 1], skew_batch[:, 0, 2] = -wrist[:, 2], wrist[:, 1]
    skew_batch[:, 1, 0], skew_batch[:, 1, 2] = wrist[:, 2], -wrist[:, 0]
    skew_batch[:, 2, 0], skew_batch[:, 2, 1] = -wrist[:, 1], wrist[:, 0]
    jac[:, :, :3] = -np.einsum('ij,njk->nik', rotation, skew_batch)
    jac[:, :, 3:6] = rotation
    if not optimize_joints:
        jac[:, :, 6:] = 0.
    return wrist @ rotation.T + np.asarray(translation), jac


def transform_part_point(local_xyz, part_id, rotations, translations, joints, pivot, shaft_offset,
                         optimize_joints=True):
    """Return a fixed surface point and its full (3,18) dual-arm Jacobian."""
    if not 0 <= int(part_id) < N_ARMS * 4:
        raise ValueError(f"Invalid part_id: {part_id}")
    arm = int(part_id) // 4
    xyz, block = point_and_jacobian_from_local(
        local_xyz, part_id, rotations[arm], translations[arm], joints[arm],
        pivot, shaft_offset, optimize_joints)
    jac = np.zeros((3, STATE_DIM), np.float64)
    jac[:, arm * ARM_DIM:(arm + 1) * ARM_DIM] = block
    return xyz, jac


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
