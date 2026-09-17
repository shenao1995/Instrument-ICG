"""Local optimizer invariants; synthetic observations, no dataset/GPU required."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

from icg.kinematics import apply_increment, exp_so3, part_local_to_camera
from icg.optimizer import (LMConfig, clip_step, evaluate_region, freeze_correspondences,
                           huber, lm_update, pose_tensors_from_numpy, project_point)
from icg.tracker import InstrumentTracker


def problem():
    rng = np.random.default_rng(17)
    state = (np.stack([exp_so3([.05, -.03, .02]), exp_so3([-.04, .03, -.02])]),
             np.array([[-.01, 0., .08], [.01, .003, .085]]),
             np.array([[.2, .15, .3], [-.1, .3, .2]]))
    delta = rng.normal(size=18) * np.tile([.004]*3 + [.0001]*3 + [.004]*3, 2)
    target = apply_increment(*state, delta, np.pi, np.pi)
    k = np.array([[500., 0., 32.], [0., 510., 32.], [0., 0., 1.]])
    lines = []
    for pid in range(8):
        arm = pid // 4
        for _ in range(10):
            local = rng.uniform(-.006, .006, 3)
            xyz = part_local_to_camera(local, pid, state[0][arm], state[1][arm], state[2][arm], .0095, 0.)
            obs_xyz = part_local_to_camera(local, pid, target[0][arm], target[1][arm], target[2][arm], .0095, 0.)
            normal = rng.normal(size=2); normal /= np.linalg.norm(normal)
            region = ('left_' if arm == 0 else 'right_') + ('shaft','wrist','grippers','grippers')[pid%4]
            lines.append(dict(xyz=xyz, c=project_point(xyz,k), observed=project_point(obs_xyz,k),
                              n=normal, part_id=pid, region_id=region, weight=.8))
    return state, k, lines


class RegionLMTests(unittest.TestCase):
    def setUp(self):
        self.state, self.k, self.lines = problem()
        self.fixed = freeze_correspondences(self.lines, self.state, .0095, 0.)
        self.config = LMConfig()

    def update(self, **kwargs):
        return lm_update(self.fixed, self.state, self.k, .0095, 0.,
                         kwargs.pop('config',self.config), .01, np.pi, np.pi, **kwargs)

    def test_huber_cost_and_irls_weight(self):
        cost,w = huber(np.array([-4.,-2.,0.,1.,3.]),2.)
        np.testing.assert_allclose(cost,[6.,2.,0.,.5,4.])
        np.testing.assert_allclose(w,[.5,1.,1.,1.,2/3])

    def test_frozen_points_and_monotonic_inner_steps(self):
        snapshot = [(v['local_xyz'].copy(),v['observed'].copy(),v['n'].copy()) for v in self.fixed]
        state, damping = self.state, .01
        for _ in range(5):
            state, damping, info = lm_update(self.fixed,state,self.k,.0095,0.,self.config,damping,np.pi,np.pi)
            if info['accepted']:
                self.assertLess(info['cost_after'],info['cost_before'])
                self.assertAlmostEqual(info['cost_after'],evaluate_region(self.fixed,state,self.k,.0095,0.)[2]['robust_cost'])
            else:
                self.assertEqual(info['cost_after'],info['cost_before'])
        for row,values in zip(self.fixed,snapshot):
            for key,value in zip(('local_xyz','observed','n'),values):
                np.testing.assert_array_equal(row[key],value)
                self.assertFalse(row[key].flags.writeable)
        # Camera XYZ is diagnostic only after freezing.
        poisoned = [dict(row,xyz=np.full(3,np.nan)) for row in self.fixed]
        self.assertEqual(evaluate_region(poisoned,state,self.k,.0095,0.)[2],
                         evaluate_region(self.fixed,state,self.k,.0095,0.)[2])

    def test_rejection_rolls_back_and_exhausts_retries(self):
        config = replace(self.config,max_rotation_step_deg=0.,max_translation_step_mm=0.,max_joint_step_deg=0.)
        state,damping,info = self.update(config=config)
        self.assertIs(state,self.state)
        self.assertFalse(info['accepted'])
        self.assertEqual(len(info['attempts']),6)
        self.assertEqual(info['lm_retries'],5)
        self.assertEqual(info['cost_after'],info['cost_before'])
        self.assertGreater(damping,.01)
        for arm in ('left','right'):
            self.assertTrue(all(v==0 for v in info[arm].values()))

    def test_invalid_candidate_is_rejected_without_dropping_points(self):
        bad=(self.state[0],self.state[1].copy(),self.state[2])
        bad[1][:,2]=-1.
        with patch('icg.optimizer.apply_increment',return_value=bad):
            state,_,info=self.update()
        self.assertIs(state,self.state)
        self.assertFalse(info['accepted'])
        self.assertTrue(all(a['candidate_cost'] is None for a in info['attempts']))

    def test_cost_increase_is_rejected_and_lambda_cap_is_respected(self):
        bad=(self.state[0],self.state[1].copy(),self.state[2])
        bad[1][:,0]+=.03
        config=replace(self.config,lm_lambda_max=.01)
        with patch('icg.optimizer.apply_increment',return_value=bad):
            state,damping,info=self.update(config=config)
        self.assertIs(state,self.state)
        self.assertFalse(info['accepted'])
        self.assertGreater(info['attempts'][0]['candidate_cost'],info['cost_before'])
        self.assertEqual(damping,.01)
        self.assertEqual(len(info['attempts']),1)

    def test_robust_gradient_matches_actual_cost_including_sigma(self):
        lines=[dict(row,observed=row['observed']+np.array([4.,-3.])) for row in self.fixed]
        g,_,_=evaluate_region(lines,self.state,self.k,.0095,0.,sigma_px=15.)
        numeric=np.zeros(18)
        for axis in range(18):
            eps=1e-7 if axis%9 in (3,4,5) else 1e-6
            delta=np.zeros(18); delta[axis]=eps
            plus=apply_increment(*self.state,delta,np.pi,np.pi)
            minus=apply_increment(*self.state,-delta,np.pi,np.pi)
            cp=evaluate_region(lines,plus,self.k,.0095,0.,sigma_px=15.)[2]['robust_cost']
            cm=evaluate_region(lines,minus,self.k,.0095,0.,sigma_px=15.)[2]['robust_cost']
            numeric[axis]=(cp-cm)/(2*eps)
        np.testing.assert_allclose(g,numeric,rtol=1e-5,atol=1e-6)

    def test_cli_alias_and_configuration_validation(self):
        from track_icg import build_parser
        parser=build_parser()
        self.assertEqual(parser.parse_args(['--iterations','4']).corr_iterations,4)
        self.assertEqual(parser.parse_args(['--corr-iterations','0']).corr_iterations,0)
        for bad in (dict(lm_lambda=0),dict(lm_lambda_min=2),dict(huber_delta_px=0),
                    dict(lm_max_retries=-1),dict(max_translation_step_mm=float('nan'))):
            with self.assertRaises(ValueError): LMConfig(**bad)

    def test_retry_can_accept_and_damping_remains_bounded(self):
        real=apply_increment
        count=0
        def first_invalid(*args):
            nonlocal count
            count+=1
            result=real(*args)
            if count==1: result[1][:,2]=-1.
            return result
        with patch('icg.optimizer.apply_increment',side_effect=first_invalid):
            _,damping,info=self.update()
        self.assertTrue(info['accepted'])
        self.assertEqual(info['lm_retries'],1)
        self.assertFalse(info['attempts'][0]['accepted'])
        self.assertAlmostEqual(damping,.05)

    def test_clipping_is_independent_per_arm_and_component(self):
        delta=np.zeros((2,9))
        delta[0]=[1.,0.,0.,.01,0.,0.,1.,-1.,.01]
        delta[1]=[.001,.002,0.,.0001,0.,0.,.003,.002,.001]
        result=clip_step(delta,self.config).reshape(2,9)
        np.testing.assert_array_equal(result[1],delta[1])
        self.assertAlmostEqual(np.linalg.norm(result[0,:3]),np.deg2rad(3))
        self.assertAlmostEqual(np.linalg.norm(result[0,3:6]),.002)
        np.testing.assert_allclose(result[0,6:],[np.deg2rad(3),-np.deg2rad(3),.01])

    def test_joint_freeze_and_report_actual_joint_clamp(self):
        state,_,info=self.update(optimize_joints=False)
        np.testing.assert_array_equal(state[2],self.state[2])
        self.assertTrue(info['accepted'])
        for arm in ('left','right'):
            self.assertEqual(info[arm]['alpha_step_deg'],0.)
        bounded = (self.state[0],self.state[1],np.zeros((2,3)))
        fixed=freeze_correspondences(self.lines,bounded,.0095,0.)
        state,_,info=lm_update(fixed,bounded,self.k,.0095,0.,self.config,.01,0.,0.)
        np.testing.assert_array_equal(state[2],0.)
        for arm in ('left','right'):
            for name in ('alpha_step_deg','theta_left_step_deg','theta_right_step_deg'):
                self.assertEqual(info[arm][name],0.)

    def test_tracker_freezes_inner_correspondences_and_never_reads_gt(self):
        class Frame:
            K=torch.tensor(self.k,dtype=torch.float32)
            rgb=torch.zeros(3,64,64)
            mask=torch.zeros(6,64,64)
            @property
            def pose(self): raise AssertionError('Optimizer accessed GT')
        mesh=SimpleNamespace(vertices=torch.zeros(1,3),convention=SimpleNamespace(pivot=.0095,shaft_offset=0.))
        rendered=dict(mask=torch.zeros(1,6,64,64))
        renderer=SimpleNamespace(mesh=mesh,render_icg=lambda pose,k:rendered)
        template={name:torch.tensor(value[None],dtype=torch.float64) for name,value in zip(('R','t','joints'),self.state)}
        tracker=InstrumentTracker(renderer,n_corr_iterations=1,n_update_iterations=2,scales=(9,7,5,2),
                                  early_stop_cost_rel=0,early_stop_rotation_deg=0,early_stop_translation_mm=0,early_stop_joint_deg=0)
        with patch.object(tracker,'_correspondences',return_value=(self.lines,None)) as sample:
            with patch('icg.tracker.lm_update',wraps=lm_update) as updates:
                result=tracker.refine(template,Frame())
        self.assertEqual(sample.call_count,1)
        self.assertEqual(updates.call_count,2)
        self.assertIs(updates.call_args_list[0].args[0],updates.call_args_list[1].args[0])
        self.assertEqual(result['history'][0]['scale'],1)

    def test_tracker_early_stop_and_low_correspondence_guard(self):
        mesh=SimpleNamespace(vertices=torch.zeros(1,3),convention=SimpleNamespace(pivot=.0095,shaft_offset=0.))
        renderer=SimpleNamespace(mesh=mesh,render_icg=lambda p,k:dict(mask=torch.zeros(1,6,64,64)))
        pose={name:torch.tensor(value[None],dtype=torch.float64) for name,value in zip(('R','t','joints'),self.state)}
        frame=SimpleNamespace(K=torch.tensor(self.k),rgb=torch.zeros(3,64,64),mask=torch.zeros(6,64,64))
        tracker=InstrumentTracker(renderer,n_corr_iterations=4,early_stop_cost_rel=2,early_stop_patience=1)
        with patch.object(tracker,'_correspondences',return_value=(self.lines,None)):
            result=tracker.refine(pose,frame)
        self.assertEqual(result['stop_reason'],'converged')
        self.assertEqual(len(result['history']),1)
        self.assertEqual(len(result['history'][0]['updates']),1)
        with patch.object(tracker,'_correspondences',return_value=(self.lines[:10],None)):
            result=tracker.refine(pose,frame)
        self.assertEqual(result['stop_reason'],'insufficient_correspondences')
        self.assertEqual(result['history'][0]['updates'],[])
        for name in pose: torch.testing.assert_close(result['pose'][name],pose[name])


if __name__=='__main__':
    unittest.main()
