"""Frame-to-frame ICG / ICG+ / Mb-ICG tracker for dual endoscopic instruments."""
from __future__ import annotations

import time

import numpy as np
import torch

from icg.correspondence import sample_correspondences
from icg.depth_modality import DepthConfig, sample_depth_correspondences
from icg.histograms import RegionHistogram, histogram_samples, mask_posterior
from icg.kinematics import pose_numpy
from icg.metrics import dice_dict
from icg.optimizer import (LMConfig, evaluate_modalities, freeze_correspondences, lm_update,
                           pose_tensors_from_numpy)
from icg.parameterization import pose_to_vector


REGION_NAMES = (
    "left_shaft", "left_wrist", "left_grippers",
    "right_shaft", "right_wrist", "right_grippers",
)


class InstrumentTracker:
    def __init__(self, renderer, method="mbicg", observation="mask", use_texture=False,
                 n_lines=180, iterations=None, scales=(1,),
                 sigma_r=(25., 15., 10.),
                 lambda_r=100.0, lambda_t=1000.0, lambda_j=50.0,
                 alpha_limit=np.pi / 2, jaw_limit=np.deg2rad(125.),
                 n_corr_iterations=4, n_update_iterations=2, lm_config=None,
                 min_correspondences=30, early_stop_cost_rel=1e-4,
                 early_stop_translation_mm=.05, early_stop_rotation_deg=.02,
                 early_stop_joint_deg=.02, early_stop_patience=2,
                 use_region=True, use_depth=False, region_weight=1., depth_weight=1., depth_config=None):
        if method not in ("icg", "icgplus", "mbicg"):
            raise ValueError(f"Unknown method {method}")
        if observation not in ("mask", "histogram"):
            raise ValueError("observation must be 'mask' or 'histogram'")
        if use_texture:
            raise ValueError("This optimizer supports Region/Depth only; --texture is not supported")
        if not all(np.isfinite(v) and v >= 0 for v in (region_weight,depth_weight)):
            raise ValueError('Modality weights must be finite and nonnegative')
        self.use_region = bool(use_region and region_weight > 0)
        self.use_depth = bool(use_depth and depth_weight > 0)
        if not (self.use_region or self.use_depth):
            raise ValueError('Enable at least one modality with positive weight')
        self.region_weight, self.depth_weight = float(region_weight),float(depth_weight)
        self.depth_config = depth_config or DepthConfig()
        self.renderer = renderer
        self.method = method
        self.observation = observation
        self.n_lines = int(n_lines)
        self.n_corr_iterations = int(n_corr_iterations if iterations is None else iterations)
        self.n_update_iterations = int(n_update_iterations)
        self.scales = list(scales)
        self.sigma_r = list(sigma_r)
        if not self.sigma_r or not all(np.isfinite(v) and v > 0 for v in self.sigma_r):
            raise ValueError('Region sigma values must be finite and positive')
        self.lm_config = lm_config or LMConfig(tikhonov_rotation=lambda_r,
                                               tikhonov_translation=lambda_t,
                                               tikhonov_joint=lambda_j)
        self.min_correspondences = int(min_correspondences)
        self.early_stop_cost_rel = float(early_stop_cost_rel)
        self.early_stop_translation_mm = float(early_stop_translation_mm)
        self.early_stop_rotation_deg = float(early_stop_rotation_deg)
        self.early_stop_joint_deg = float(early_stop_joint_deg)
        self.early_stop_patience = int(early_stop_patience)
        if self.n_corr_iterations < 0 or self.n_update_iterations < 1:
            raise ValueError('corr_iterations must be >= 0 and update_iterations >= 1')
        if self.min_correspondences < 1 or self.early_stop_patience < 1 or self.n_lines < 1:
            raise ValueError('correspondence counts and early_stop_patience must be positive')
        thresholds = (early_stop_cost_rel, early_stop_translation_mm,
                      early_stop_rotation_deg, early_stop_joint_deg)
        if not all(np.isfinite(v) and v >= 0 for v in thresholds):
            raise ValueError('Early stop thresholds must be finite and nonnegative')
        if not self.scales or any(s < 1 for s in self.scales):
            raise ValueError('scales must be nonempty and positive')
        self.alpha_limit = float(alpha_limit)
        self.jaw_limit = float(jaw_limit)
        self.optimize_joints = method == "mbicg"
        self.multi_region = method in ("icgplus", "mbicg")
        self.histograms = [RegionHistogram() for _ in REGION_NAMES]
        self.mesh = renderer.mesh
        self._histograms_seeded = False

    def _region_masks(self, semantic):
        """Predicted or observed masks used as ICG regions.

        ICG uses one silhouette per instrument. ICG+ / Mb-ICG use part regions
        (shaft, wrist, grippers) so internal boundaries constrain articulation.
        """
        if self.multi_region:
            return [semantic[i] for i in range(semantic.shape[0])]
        return [semantic[:3].max(0), semantic[3:6].max(0)]

    def _posteriors(self, rgb, target, predicted_regions):
        if self.observation == "mask":
            observed = self._region_masks(target)
            return [mask_posterior(m) for m in observed]
        rgb_hwc = np.transpose(rgb, (1, 2, 0))
        posts = []
        for i, region in enumerate(predicted_regions):
            hist = self.histograms[i if self.multi_region else (0 if i == 0 else 3)]
            posts.append(hist.posterior(rgb_hwc))
        return posts

    def _update_histograms(self, rgb, predicted):
        rgb_hwc = np.transpose(rgb, (1, 2, 0))
        regions = self._region_masks(predicted)
        for i, region in enumerate(regions):
            inner, outer = histogram_samples(region)
            idx = i if self.multi_region else (0 if i == 0 else 3)
            self.histograms[idx].update(rgb_hwc, inner, outer)

    def _correspondences(self, rendered, rgb, target, scale):
        pred = rendered["mask"][0].detach().cpu().numpy()
        xyz = rendered["xyz"][0].permute(1, 2, 0).detach().cpu().numpy()
        part_id = rendered["part_id"][0, 0].detach().cpu().numpy()
        target_np = target.detach().cpu().numpy() if torch.is_tensor(target) else np.asarray(target)
        pred_regions = self._region_masks(pred)
        posts = self._posteriors(rgb, target_np, pred_regions)
        n_per = max(40, self.n_lines // max(1, len(pred_regions)))
        lines = []
        for region, posterior in zip(pred_regions, posts):
            lines.extend(sample_correspondences(
                region, xyz, part_id, posterior, n_lines=n_per, scale=scale))
        return lines, pred

    def _small_step(self, update):
        return all(
            update[arm]['translation_step_mm'] < self.early_stop_translation_mm and
            update[arm]['rotation_step_deg'] < self.early_stop_rotation_deg and
            max(update[arm][key] for key in ('alpha_step_deg', 'theta_left_step_deg',
                                             'theta_right_step_deg')) < self.early_stop_joint_deg
            for arm in ('left', 'right'))

    @torch.no_grad()
    def refine(self, pose, frame):
        # No GT pose access: only K/rgb/mask/depth observations enter optimization.
        device = self.mesh.vertices.device
        def sync():
            if device.type == 'cuda': torch.cuda.synchronize(device)
        sync()
        started = time.perf_counter()
        timing = dict(render_s=0.,correspondence_s=0.,solve_s=0.,render_calls=0)
        k = frame.K.to(device)
        k_numpy = k.detach().cpu().numpy()
        rgb = frame.rgb.detach().cpu().numpy()
        if self.use_depth and getattr(frame,'depth',None) is None:
            raise ValueError('Depth modality requires a depth observation with explicit depth scale')
        if self.use_region and self.observation == "histogram" and not self._histograms_seeded:
            self._update_histograms(rgb, frame.mask.detach().cpu().numpy())
            self._histograms_seeded = True
        state = tuple(np.array(v,dtype=np.float64,copy=True) for v in pose_numpy(pose))
        pivot, offset = self.mesh.convention.pivot, self.mesh.convention.shaft_offset
        lm_lambda = self.lm_config.lm_lambda
        history, initial_dice = [], None
        consecutive_small = 0
        counts_region = dict(total=0,**{name:0 for name in REGION_NAMES})
        counts_depth = dict(counts_region)
        stop_reason = 'iteration_limit' if self.n_corr_iterations else 'no_optimization'

        def render(current):
            sync()
            t0=time.perf_counter()
            output=self.renderer.render_icg(current,k)
            sync()
            timing['render_s']+=time.perf_counter()-t0
            timing['render_calls']+=1
            return output

        def counts(lines):
            result=dict(total=len(lines),**{name:0 for name in REGION_NAMES})
            for line in lines: result[line['region_id']]+=1
            return result

        for corr_iteration in range(self.n_corr_iterations):
            scale = 1 if self.observation == 'mask' else self.scales[min(corr_iteration,len(self.scales)-1)]
            sigma = self.sigma_r[min(corr_iteration,len(self.sigma_r)-1)]
            rendered_pose=pose_tensors_from_numpy(state,pose)
            rendered=render(rendered_pose)
            if initial_dice is None:
                initial_dice=dice_dict(rendered['mask'],frame.mask.to(device)[None])
            t0=time.perf_counter()
            rendered_state=pose_numpy(rendered_pose)
            fixed,depth_fixed=(),()
            if self.use_region:
                lines,_=self._correspondences(rendered,rgb,frame.mask,scale)
                fixed=freeze_correspondences(lines,rendered_state,pivot,offset)
            depth_search={}
            if self.use_depth:
                depth_fixed,depth_search=sample_depth_correspondences(
                    rendered,frame.depth,frame.mask,k_numpy,rendered_state,pivot,offset,self.depth_config)
            counts_region,counts_depth=counts(fixed),counts(depth_fixed)
            timing['correspondence_s']+=time.perf_counter()-t0
            modality_args=dict(depth_lines=depth_fixed,use_region=self.use_region,use_depth=self.use_depth,
                               region_weight=self.region_weight,depth_weight=self.depth_weight,
                               depth_sigma_m=self.depth_config.sigma_mm/1000.,
                               depth_huber_delta_m=self.depth_config.huber_delta_mm/1000.)
            t0=time.perf_counter()
            _,_,before=evaluate_modalities(fixed,depth_fixed,state,k_numpy,pivot,offset,self.optimize_joints,
                                           self.lm_config.huber_delta_px,sigma,
                                           modality_args['depth_sigma_m'],modality_args['depth_huber_delta_m'],
                                           self.use_region,self.use_depth,self.region_weight,self.depth_weight)
            timing['solve_s']+=time.perf_counter()-t0
            outer=dict(corr_iteration=corr_iteration,scale=scale,residual_sigma_px=sigma,
                       correspondences=counts_region,region_correspondences=counts_region,
                       depth_correspondences=counts_depth,depth_search=depth_search,
                       region_residual_before=before['region'],depth_residual_before=before['depth'],updates=[])
            history.append(outer)
            if ((self.use_region and len(fixed)<self.min_correspondences) or
                    (self.use_depth and len(depth_fixed)<self.depth_config.min_correspondences)):
                stop_reason=outer['stop_reason']='insufficient_correspondences'
                outer['insufficient_modalities']=[name for name,failed in (
                    ('region',self.use_region and len(fixed)<self.min_correspondences),
                    ('depth',self.use_depth and len(depth_fixed)<self.depth_config.min_correspondences)) if failed]
                break
            stop=False
            for update_iteration in range(self.n_update_iterations):
                t0=time.perf_counter()
                state,lm_lambda,update=lm_update(fixed,state,k_numpy,pivot,offset,self.lm_config,lm_lambda,
                                                self.alpha_limit,self.jaw_limit,self.optimize_joints,sigma,**modality_args)
                timing['solve_s']+=time.perf_counter()-t0
                update['update_iteration']=update_iteration
                outer['updates'].append(update)
                if not update['accepted']:
                    stop_reason=outer['stop_reason']='no_decreasing_step'
                    stop=True
                    break
                reduction=(update['cost_before']-update['cost_after'])/max(update['cost_before'],1e-12)
                update['relative_cost_reduction']=reduction
                small=self._small_step(update) or reduction<self.early_stop_cost_rel
                consecutive_small=consecutive_small+1 if small else 0
                update['early_stop_streak']=consecutive_small
                if consecutive_small>=self.early_stop_patience:
                    stop_reason=outer['stop_reason']='converged'
                    stop=True
                    break
            if stop: break
        pose=pose_tensors_from_numpy(state,pose)
        rendered=render(pose)  # final evaluation/output, never an LM-candidate render
        pred=rendered['mask']
        if self.use_region and self.observation=='histogram':
            self._update_histograms(rgb,pred[0].detach().cpu().numpy())
        dice=dice_dict(pred,frame.mask.to(device)[None])
        if history:
            last=history[-1]
            region_residual=last['updates'][-1]['region_residual_after'] if last['updates'] else last['region_residual_before']
            depth_residual=last['updates'][-1]['depth_residual_after'] if last['updates'] else last['depth_residual_before']
        else:
            region_residual=depth_residual={}
        sync()
        timing['total_s']=time.perf_counter()-started
        return dict(pose=pose,x=pose_to_vector(pose),result=rendered,dice=dice,
                    initial_dice=initial_dice or dice,history=history,stop_reason=stop_reason,
                    correspondences=counts_region['total']+counts_depth['total'],
                    region_correspondences=counts_region['total'],depth_correspondences=counts_depth['total'],
                    region_residual=region_residual,depth_residual=depth_residual,timing=timing)
