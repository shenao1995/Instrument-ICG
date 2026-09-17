"""CPU central-difference check of fixed-normal Depth residuals, all 18 DoFs."""
from __future__ import annotations
import numpy as np
from icg.check_jacobians import _fk
from icg.kinematics import apply_increment, exp_so3, batch_points_and_jacobians_from_local
from icg.optimizer import depth_residual_and_jacobian


def check_depth_jacobians():
    rng=np.random.default_rng(408)
    worst=0.
    coverage=np.zeros(18,dtype=bool)
    for offset in (0.,.2159):
        for trial in range(4):
            state=(np.stack([exp_so3(rng.normal(0,.3,3)) for _ in range(2)]),
                   np.array([[-.01,.003,.09],[.02,-.006,.12]]),rng.uniform(-.5,.5,(2,3)))
            for pid in range(8):
                arm,part=divmod(pid,4)
                local=rng.uniform(-.004,.004,3)
                local[0]+=offset if part==0 else .008
                normal=rng.normal(size=3);normal/=np.linalg.norm(normal)
                observed=_fk(part,local,state[0][arm],state[1][arm],state[2][arm],.0095,offset)+rng.normal(0,.001,3)
                line=dict(local_xyz=local,part_id=pid,normal=normal,observed_xyz=observed)
                d=rng.normal(size=18)*np.tile([.01]*3+[.0002]*3+[.01]*3,2)
                candidate=apply_increment(*state,d,np.pi,np.pi)
                _,analytic=depth_residual_and_jacobian(line,candidate,.0095,offset)
                batch_xyz,batch_jac=batch_points_and_jacobians_from_local(
                    local[None],pid,candidate[0][arm],candidate[1][arm],candidate[2][arm],.0095,offset)
                np.testing.assert_allclose(normal@batch_jac[0],analytic[arm*9:(arm+1)*9],atol=1e-12,rtol=1e-12)
                expected_xyz=_fk(part,local,candidate[0][arm],candidate[1][arm],candidate[2][arm],.0095,offset)
                np.testing.assert_allclose(batch_xyz[0],expected_xyz,atol=1e-12,rtol=1e-12)
                numeric=np.zeros(18)
                for axis in range(18):
                    eps=1e-7 if axis%9 in (3,4,5) else 1e-6
                    step=np.zeros(18);step[axis]=eps
                    plus=apply_increment(*candidate,step,np.pi,np.pi)
                    minus=apply_increment(*candidate,-step,np.pi,np.pi)
                    def residual(q):
                        xyz=_fk(part,local,q[0][arm],q[1][arm],q[2][arm],.0095,offset)
                        return normal@(xyz-observed)
                    numeric[axis]=(residual(plus)-residual(minus))/(2*eps)
                worst=max(worst,float(np.abs(numeric-analytic).max()))
                coverage|=np.abs(numeric)>1e-6
                np.testing.assert_allclose(analytic,numeric,rtol=2e-6,atol=1e-8,
                                           err_msg=f'pid={pid}, offset={offset}, trial={trial}')
                np.testing.assert_array_equal(analytic[(1-arm)*9:(2-arm)*9],0.)
    assert coverage.all(),f'Missing DoFs: {np.flatnonzero(~coverage)}'
    print(f'Depth Jacobian PASS: 18/18 DoFs, 64 fixed points/normals; max abs error={worst:.3e}')
    return worst


if __name__=='__main__':
    check_depth_jacobians()
