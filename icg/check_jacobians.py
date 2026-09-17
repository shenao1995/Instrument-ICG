"""Finite-difference check of Mb-ICG point Jacobians (no GPU)."""
from __future__ import annotations

import numpy as np

from icg.kinematics import apply_increment, full_point_jacobian, rotation_y, rotation_z


def _fk(part, x_local, R, t, joints, pivot, shaft_offset):
    alpha, th_l, th_r = joints
    if part == 0:
        x_m = rotation_y(-alpha) @ (x_local - np.array([shaft_offset, 0.0, 0.0]))
    elif part == 1:
        x_m = x_local
    elif part == 2:
        x_m = rotation_z(th_l) @ x_local + np.array([pivot, 0.0, 0.0])
    else:
        x_m = rotation_z(-th_r) @ x_local + np.array([pivot, 0.0, 0.0])
    return R @ x_m + t


def main():
    rng = np.random.default_rng(0)
    R = [np.eye(3), np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])]
    t = [np.array([0.01, -0.02, 0.07]), np.array([-0.03, 0.01, 0.08])]
    joints = [np.array([0.3, 0.2, 0.15]), np.array([-0.1, 0.4, 0.25])]
    pivot, shaft_offset = 0.0095, 0.0
    locals_ = {
        0: np.array([0.02, 0.001, 0.0]),
        1: np.array([0.0, 0.0, 0.0]),
        2: np.array([0.012, 0.002, 0.0]),
        3: np.array([0.012, -0.002, 0.0]),
    }
    eps = 1e-6
    worst = 0.0
    for arm in range(2):
        for part in range(4):
            x_local = locals_[part]
            xyz = _fk(part, x_local, R[arm], t[arm], joints[arm], pivot, shaft_offset)
            pid = arm * 4 + part
            jac = full_point_jacobian(xyz, pid, R, t, joints, pivot, shaft_offset, True)
            numeric = np.zeros((3, 18))
            for k in range(18):
                delta = np.zeros(18)
                delta[k] = eps
                Rp, tp, jp = apply_increment(R, t, joints, delta, np.pi, np.pi)
                xyz_p = _fk(part, x_local, Rp[arm], tp[arm], jp[arm], pivot, shaft_offset)
                delta[k] = -eps
                Rm, tm, jm = apply_increment(R, t, joints, delta, np.pi, np.pi)
                xyz_m = _fk(part, x_local, Rm[arm], tm[arm], jm[arm], pivot, shaft_offset)
                numeric[:, k] = (xyz_p - xyz_m) / (2 * eps)
            err = np.linalg.norm(jac - numeric)
            worst = max(worst, err)
            print(f"arm {arm} part {part}: jac error {err:.3e}")
            if err > 1e-4:
                print(" analytic\n", jac)
                print(" numeric\n", numeric)
                raise SystemExit(1)
    print(f"ok, worst {worst:.3e}")


if __name__ == "__main__":
    main()
