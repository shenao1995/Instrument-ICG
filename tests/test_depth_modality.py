"""Sparse depth geometry, gating, units, and shared-LM regression tests."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
import unittest
import cv2
import numpy as np
import torch

from icg.depth_modality import (DepthConfig, PART_TO_SEMANTIC, backproject,
                               find_depth_match, find_depth_matches, sample_depth_correspondences)
from icg.kinematics import apply_increment, part_local_to_camera
from icg.optimizer import (LMConfig, depth_residual_and_jacobian, evaluate_depth,
                           evaluate_modalities, lm_update, project_point)
from icg.sim_data import load_depth, resize_depth_to_metres
from icg.tracker import InstrumentTracker
from icg.check_depth_jacobians import check_depth_jacobians


def synthetic_depth_system():
    rng=np.random.default_rng(3)
    state=(np.stack([np.eye(3),np.eye(3)]),np.array([[-.01,0.,.08],[.01,0.,.09]]),np.zeros((2,3)))
    d=np.zeros((2,9));d[:,3:6]=[[.0001,-.0001,.0008],[-.0001,.0001,.0006]]
    target=apply_increment(*state,d.ravel(),np.pi,np.pi)
    k=np.array([[500.,0.,32.],[0.,500.,32.],[0.,0.,1.]])
    depth,region=[],[]
    for pid in range(8):
        arm=pid//4
        for _ in range(10):
            local=rng.uniform(-.004,.004,3)
            xyz=part_local_to_camera(local,pid,state[0][arm],state[1][arm],state[2][arm],.0095,0.)
            obs=part_local_to_camera(local,pid,target[0][arm],target[1][arm],target[2][arm],.0095,0.)
            normal=rng.normal(size=3);normal/=np.linalg.norm(normal)
            depth.append(dict(local_xyz=local,observed_xyz=obs,normal=normal,part_id=pid,weight=1.))
            n=rng.normal(size=2);n/=np.linalg.norm(n)
            region.append(dict(local_xyz=local,observed=project_point(xyz,k),n=n,part_id=pid,weight=1.))
    return state,target,k,depth,region


class DepthTests(unittest.TestCase):
    def setUp(self):
        self.k=np.array([[100.,0.,5.],[0.,100.,5.],[0.,0.,1.]])
        self.depth=np.full((11,11),np.nan)
        self.semantic=np.zeros((6,11,11))
        self.zbuffer=np.ones((11,11))
        self.parts=np.zeros((11,11),np.int64)
        self.config=DepthConfig(max_distance_mm=100.,occlusion_threshold_mm=100.)
        self.xyz=np.array([0.,0.,1.])

    def match(self,config=None,pid=0):
        return find_depth_match(self.xyz,pid,self.depth,self.semantic,self.k,self.zbuffer,self.parts,config or self.config)

    def test_backprojection(self):
        np.testing.assert_allclose(backproject([5,6],[5,4],[1,2],self.k),[[0,0,1],[.02,-.02,2]])

    def test_semantic_gating_including_both_jaws(self):
        np.testing.assert_array_equal(PART_TO_SEMANTIC,[0,1,2,2,3,4,5,5])
        self.depth[5,5]=1.
        self.semantic[1,5,5]=1.
        self.assertIsNone(self.match()[0])
        for pid in range(8):
            self.semantic[:]=0
            self.semantic[PART_TO_SEMANTIC[pid],5,5]=1.
            self.parts[:]=pid
            self.assertIsNotNone(self.match(pid=pid)[0])

    def test_invalid_depth_rejection(self):
        self.semantic[0]=1.
        for value in (0.,np.nan,np.inf,-1.):
            self.depth[:]=value
            self.assertIsNone(self.match()[0])

    def test_search_radius(self):
        self.depth[5,7]=1.
        self.semantic[0,5,7]=1.
        self.assertIsNone(self.match(replace(self.config,search_radius_px=1))[0])
        self.assertIsNotNone(self.match(replace(self.config,search_radius_px=2))[0])

    def test_nearest_3d_instead_of_smallest_z_difference(self):
        self.depth[5,5]=1.005
        self.depth[5,9]=1.
        self.semantic[0,5,[5,9]]=1.
        point,_=self.match()
        np.testing.assert_array_equal(point['observed_pixel'],[5,5])
        self.assertAlmostEqual(point['distance_m'],.005)

    def test_visibility_rejection(self):
        self.depth[5,5]=1.
        self.semantic[0,5,5]=1.
        self.zbuffer[5,5]=.99
        self.assertEqual(self.match()[1],'rendered_occlusion')
        self.zbuffer[5,5]=1.
        self.parts[5,5]=4
        self.assertEqual(self.match()[1],'rendered_occlusion')

    def test_measured_occlusion_rejection(self):
        self.depth[5,5]=.98
        self.depth[5,6]=1.
        self.semantic[0,5,6]=1.
        self.assertEqual(self.match(replace(self.config,occlusion_threshold_mm=5.))[1],'measured_occlusion')

    def test_distance_gate(self):
        self.depth[5,5]=1.01
        self.semantic[0,5,5]=1.
        self.assertEqual(self.match(replace(self.config,max_distance_mm=5.))[1],'distance_gate')

    def test_batched_search_matches_scalar_reference(self):
        rng=np.random.default_rng(73)
        self.depth=rng.uniform(.995,1.005,(11,11))
        self.depth[2,2]=np.nan
        self.depth[3,3]=0.
        self.parts=rng.integers(0,8,(11,11))
        self.semantic[:]=1.
        self.semantic[1,:3,:3]=0.
        yy,xx=np.mgrid[:11,:11]
        points=backproject(xx.ravel(),yy.ravel(),np.ones(121),self.k)
        ids=self.parts.ravel()
        for config in (self.config,replace(self.config,max_distance_mm=2.,search_radius_px=1,
                                           occlusion_threshold_mm=1.)):
            batched,reasons=find_depth_matches(points,ids,self.depth,self.semantic,self.k,self.zbuffer,self.parts,config)
            for i,(point,pid) in enumerate(zip(points,ids)):
                scalar,reason=find_depth_match(point,pid,self.depth,self.semantic,self.k,self.zbuffer,self.parts,config)
                self.assertEqual(reasons[i],reason)
                if scalar is not None:
                    np.testing.assert_allclose(batched[i]['observed_xyz'],scalar['observed_xyz'])
                    np.testing.assert_array_equal(batched[i]['observed_pixel'],scalar['observed_pixel'])

    def test_units_invalids_nearest_resize_and_camera_selection(self):
        raw=np.array([[100,0],[np.nan,-1]],dtype=np.float32)
        result=resize_depth_to_metres(raw,(4,4),.001)
        self.assertEqual(result.dtype,np.float32)
        np.testing.assert_allclose(result[:2,:2],.1)
        self.assertTrue(np.isnan(result[:2,2:]).all())
        self.assertTrue(np.isnan(result[2:]).all())
        with self.assertRaises(ValueError):resize_depth_to_metres(raw,(4,4),None)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name,value in [('depth',100),('depth_right',200)]:
                (root/name).mkdir()
                cv2.imwrite(str(root/name/'0000.png'),np.full((2,2),value,np.uint16))
            self.assertAlmostEqual(float(load_depth(root,0,'left',(2,2),.001)[0,0]),.1,places=6)
            self.assertAlmostEqual(float(load_depth(root,0,'right',(2,2),.001)[0,0]),.2,places=6)

    def test_residual_sign_and_fixed_normal(self):
        state=(np.stack([np.eye(3)]*2),np.array([[0.,0.,1.]]*2),np.zeros((2,3)))
        line=dict(local_xyz=np.zeros(3),observed_xyz=np.array([0.,0.,.99]),normal=np.array([0.,0.,1.]),part_id=1,weight=1.)
        r,j=depth_residual_and_jacobian(line,state,.0095,0.)
        self.assertAlmostEqual(r,.01)
        self.assertAlmostEqual(j[5],1.)
        line['normal']=-line['normal']
        r,j=depth_residual_and_jacobian(line,state,.0095,0.)
        self.assertAlmostEqual(r,-.01)
        self.assertAlmostEqual(j[5],-1.)

    def test_depth_jacobian_all_18_dof(self):
        self.assertLess(check_depth_jacobians(),1e-8)

    def test_fixed_candidate_cost_and_depth_only_lm(self):
        state,target,k,depth,_=synthetic_depth_system()
        before=evaluate_depth(depth,state,.0095,0.)[2]['robust_cost']
        self.assertLess(evaluate_depth(depth,target,.0095,0.)[2]['robust_cost'],before)
        snapshot=[{key:row[key].copy() for key in ('normal','local_xyz','observed_xyz')} for row in depth]
        with patch('icg.optimizer.evaluate_region',side_effect=AssertionError('Depth-only evaluated Region')):
            result,_,info=lm_update((),state,k,.0095,0.,LMConfig(),.01,np.pi,np.pi,
                                    depth_lines=depth,use_region=False,use_depth=True)
        self.assertTrue(info['accepted'])
        self.assertLess(info['total_cost_after'],info['total_cost_before'])
        self.assertEqual(info['region_cost_after'],0.)
        for row,old in zip(depth,snapshot):
            for key in old:np.testing.assert_array_equal(row[key],old[key])
        self.assertAlmostEqual(info['cost_after'],evaluate_depth(depth,result,.0095,0.)[2]['robust_cost'])

    def test_batched_depth_gradient_matches_scalar_and_finite_difference(self):
        state,_,_,depth,_=synthetic_depth_system()
        g,h,stats=evaluate_depth(depth,state,.0095,0.)
        residual,jac=zip(*(depth_residual_and_jacobian(line,state,.0095,0.) for line in depth))
        r,j=np.asarray(residual),np.asarray(jac)
        np.testing.assert_allclose(g,j.T@r/.002**2,atol=1e-8,rtol=1e-10)
        np.testing.assert_allclose(h,j.T@j/.002**2,atol=1e-7,rtol=1e-10)
        for axis in range(18):
            eps=1e-7 if axis%9 in (3,4,5) else 1e-6
            d=np.zeros(18);d[axis]=eps
            cp=evaluate_depth(depth,apply_increment(*state,d,np.pi,np.pi),.0095,0.)[2]['robust_cost']
            cm=evaluate_depth(depth,apply_increment(*state,-d,np.pi,np.pi),.0095,0.)[2]['robust_cost']
            self.assertAlmostEqual((cp-cm)/(2*eps),g[axis],delta=2e-6)

    def test_combined_acceptance_allows_one_cost_to_increase(self):
        state,target,k,depth,region=synthetic_depth_system()
        with patch('icg.optimizer.apply_increment',return_value=target):
            _,_,info=lm_update(region,state,k,.0095,0.,LMConfig(),.01,np.pi,np.pi,
                               depth_lines=depth,use_depth=True,region_weight=.001)
        self.assertTrue(info['accepted'])
        self.assertGreater(info['region_cost_after'],info['region_cost_before'])
        self.assertLess(info['depth_cost_after'],info['depth_cost_before'])
        self.assertLess(info['total_cost_after'],info['total_cost_before'])
        self.assertAlmostEqual(info['total_cost_after'],.001*info['region_cost_after']+info['depth_cost_after'])

    def test_depth_candidate_cannot_hide_invalid_points(self):
        state,target,k,depth,_=synthetic_depth_system()
        bad=(target[0],target[1].copy(),target[2])
        bad[1][:,2]=-1.
        with patch('icg.optimizer.apply_increment',return_value=bad):
            kept,_,info=lm_update((),state,k,.0095,0.,LMConfig(),.01,np.pi,np.pi,
                                 depth_lines=depth,use_region=False,use_depth=True)
        self.assertIs(kept,state)
        self.assertFalse(info['accepted'])
        self.assertTrue(all(row['candidate_cost'] is None for row in info['attempts']))

    def test_cli_requires_explicit_scale_and_supports_ablations(self):
        from track_icg import build_parser, main
        parser=build_parser()
        with self.assertRaisesRegex(ValueError,'depth-scale'):main(parser.parse_args([]))
        with self.assertRaisesRegex(ValueError,'at least one'):main(parser.parse_args(['--no-region','--no-depth']))
        for flags,expected in [(['--use-region','--no-depth'],(True,False)),
                               (['--no-region','--use-depth'],(False,True)),
                               (['--use-region','--use-depth'],(True,True))]:
            args=parser.parse_args(flags)
            self.assertEqual((args.use_region,args.use_depth),expected)

    def test_sampling_budgets_and_fixed_arrays(self):
        yy,xx=np.mgrid[:11,:11]
        xyz=backproject(xx,yy,np.ones((11,11)),self.k)
        normal=np.zeros_like(xyz);normal[:,:,2]=1.
        mask=np.zeros((1,6,11,11));mask[:,0]=1.
        rendered=dict(xyz=xyz.transpose(2,0,1)[None],normal=normal.transpose(2,0,1)[None],
                      part_id=self.parts[None,None],mask=mask)
        self.depth[:]=1.
        self.semantic[0]=1.
        state=(np.stack([np.eye(3)]*2),np.array([[0.,0.,1.]]*2),np.zeros((2,3)))
        lines,stats=sample_depth_correspondences(rendered,self.depth,self.semantic,self.k,state,.0095,0.,
                                               replace(self.config,max_points_per_region=10))
        self.assertEqual(len(lines),10)
        self.assertEqual(stats['sampled'],10)
        for row in lines:
            for key in ('normal','local_xyz','observed_xyz'):
                self.assertFalse(row[key].flags.writeable)
            self.assertEqual(row['region_id'],'left_shaft')

    def test_depth_only_tracker_no_gt_no_region_no_candidate_render(self):
        state,_,k,depth,_=synthetic_depth_system()
        for line in depth:line['region_id']='left_shaft' if line['part_id']<4 else 'right_shaft'
        class Frame:
            K=torch.tensor(k)
            rgb=torch.zeros(3,64,64)
            mask=torch.zeros(6,64,64)
            depth=torch.ones(64,64)
            @property
            def pose(self):raise AssertionError('Read GT pose')
        mesh=SimpleNamespace(vertices=torch.zeros(1,3),convention=SimpleNamespace(pivot=.0095,shaft_offset=0.))
        renderer=SimpleNamespace(mesh=mesh,render_icg=lambda p,k:dict(mask=torch.zeros(1,6,64,64)))
        pose={name:torch.tensor(value[None]) for name,value in zip(('R','t','joints'),state)}
        tracker=InstrumentTracker(renderer,use_region=False,use_depth=True,n_corr_iterations=1,n_update_iterations=2,
                                  early_stop_cost_rel=0.,early_stop_rotation_deg=0.,early_stop_translation_mm=0.,early_stop_joint_deg=0.)
        with patch.object(tracker,'_correspondences',side_effect=AssertionError('Region search in Depth-only')):
            with patch('icg.tracker.sample_depth_correspondences',return_value=(tuple(depth),{})) as sample:
                with patch.object(renderer,'render_icg',wraps=renderer.render_icg) as render:
                    result=tracker.refine(pose,Frame())
        self.assertEqual(sample.call_count,1)
        self.assertEqual(render.call_count,2) # one outer pass + final output
        self.assertEqual(len(result['history'][0]['updates']),2)
        self.assertEqual(result['region_correspondences'],0)
        self.assertEqual(result['depth_correspondences'],80)


if __name__=='__main__':unittest.main()
