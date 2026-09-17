"""CPU check of complete Region residual Jacobians for all 18 tangent DoFs."""
from __future__ import annotations
import numpy as np
from icg.check_jacobians import _fk
from icg.kinematics import (apply_increment, camera_to_part_local, exp_so3,
                            part_local_to_camera, transform_part_point)
from icg.optimizer import project_point, region_residual_and_jacobian


def check_region_jacobians():
    rng = np.random.default_rng(731)
    k = np.array([[510., 0., 255.5], [0., 505., 143.5], [0., 0., 1.]])
    worst_abs = worst_scaled = worst_roundtrip = 0.
    coverage = np.zeros(18, dtype=bool)
    for shaft_offset in (0., .2159):
        for trial in range(4):
            state = (np.stack([exp_so3(rng.normal(0, .3, 3)) for _ in range(2)]),
                     np.array([[-.01, .003, .09], [.02, -.006, .12]]),
                     rng.uniform(-.5, .5, (2, 3)))
            pivot = .0095
            for pid in range(8):
                arm, part = divmod(pid, 4)
                local = rng.uniform(-.004, .004, 3)
                local[0] += shaft_offset if part == 0 else .008
                xyz = _fk(part, local, state[0][arm], state[1][arm], state[2][arm], pivot, shaft_offset)
                recovered = camera_to_part_local(xyz, pid, state[0][arm], state[1][arm], state[2][arm], pivot, shaft_offset)
                worst_roundtrip = max(worst_roundtrip, float(np.max(np.abs(recovered - local))))
                np.testing.assert_allclose(recovered, local, rtol=0, atol=1e-12)
                forward = part_local_to_camera(local, pid, state[0][arm], state[1][arm], state[2][arm], pivot, shaft_offset)
                np.testing.assert_allclose(forward, xyz, rtol=0, atol=1e-12)
                normal = rng.normal(size=2); normal /= np.linalg.norm(normal)
                line = dict(local_xyz=local, part_id=pid, n=normal,
                            observed=project_point(xyz, k) + rng.normal(size=2))
                # Evaluate away from the correspondence-establishment pose too.
                candidate_delta = rng.normal(size=18) * np.tile([.01]*3 + [.0002]*3 + [.01]*3, 2)
                candidate = apply_increment(*state, candidate_delta, np.pi, np.pi)
                _, analytic = region_residual_and_jacobian(line, candidate, k, pivot, shaft_offset)
                numeric = np.zeros(18)
                for axis in range(18):
                    eps = 1e-7 if axis % 9 in (3, 4, 5) else 1e-6
                    d = np.zeros(18); d[axis] = eps
                    plus = apply_increment(*candidate, d, np.pi, np.pi)
                    minus = apply_increment(*candidate, -d, np.pi, np.pi)
                    # Independent FK oracle (not the analytic implementation).
                    def residual(q):
                        x = _fk(part, local, q[0][arm], q[1][arm], q[2][arm], pivot, shaft_offset)
                        return normal @ (project_point(x, k) - line['observed'])
                    numeric[axis] = (residual(plus) - residual(minus)) / (2 * eps)
                error = np.abs(analytic - numeric)
                worst_abs = max(worst_abs, float(error.max()))
                worst_scaled = max(worst_scaled, float((error / (1 + np.abs(numeric))).max()))
                coverage |= np.abs(numeric) > 1e-5
                np.testing.assert_allclose(analytic, numeric, rtol=2e-6, atol=2e-5,
                                           err_msg=f'part={pid}, trial={trial}, offset={shaft_offset}')
                other_arm = slice((1-arm)*9, (2-arm)*9)
                np.testing.assert_array_equal(analytic[other_arm], 0.)
    assert coverage.all(), f'Ungraded DoFs: {np.where(~coverage)[0]}'
    print(f'Region Jacobian PASS: 18/18 DoFs, 64 points/candidate poses; '
          f'max abs={worst_abs:.3e}, scaled={worst_scaled:.3e}, roundtrip={worst_roundtrip:.3e}')


if __name__ == '__main__':
    check_region_jacobians()
