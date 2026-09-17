"""Frame-to-frame ICG / ICG+ / Mb-ICG tracker for dual endoscopic instruments."""
from __future__ import annotations

import numpy as np
import torch

from icg.correspondence import sample_correspondences
from icg.histograms import RegionHistogram, histogram_samples, mask_posterior
from icg.kinematics import pose_numpy
from icg.metrics import dice_dict
from icg.optimizer import (LMConfig, evaluate_region, freeze_correspondences, lm_update,
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
                 early_stop_joint_deg=.02, early_stop_patience=2):
        if method not in ("icg", "icgplus", "mbicg"):
            raise ValueError(f"Unknown method {method}")
        if observation not in ("mask", "histogram"):
            raise ValueError("observation must be 'mask' or 'histogram'")
        if use_texture:
            raise ValueError("This optimizer is Region-only; --texture is not supported in this version")
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
        # This function reads only K/rgb/mask from frame; GT pose is evaluation-only
        # in track_icg.py, outside the optimizer and all acceptance/stopping logic.
        device = self.mesh.vertices.device
        k = frame.K.to(device)
        k_numpy = k.detach().cpu().numpy()
        rgb = frame.rgb.detach().cpu().numpy()
        if self.observation == "histogram" and not self._histograms_seeded:
            self._update_histograms(rgb, frame.mask.detach().cpu().numpy())
            self._histograms_seeded = True
        state = tuple(np.array(v, dtype=np.float64, copy=True) for v in pose_numpy(pose))
        pivot, offset = self.mesh.convention.pivot, self.mesh.convention.shaft_offset
        lm_lambda = self.lm_config.lm_lambda  # reset between frames, persist within frame
        history, initial_dice = [], None
        consecutive_small = 0
        stop_reason = 'iteration_limit' if self.n_corr_iterations else 'no_optimization'
        for corr_iteration in range(self.n_corr_iterations):
            scale = 1 if self.observation == "mask" else self.scales[min(corr_iteration, len(self.scales)-1)]
            sigma = self.sigma_r[min(corr_iteration, len(self.sigma_r)-1)]
            rendered = self.renderer.render_icg(pose_tensors_from_numpy(state, pose), k)
            if initial_dice is None:
                initial_dice = dice_dict(rendered['mask'], frame.mask.to(device)[None])
            lines, _ = self._correspondences(rendered, rgb, frame.mask, scale)
            # Surface XYZ is interpolated from the float32 pose actually rendered.
            # Invert that same pose once; subsequent evaluations use the fixed local point.
            rendered_state = pose_numpy(pose_tensors_from_numpy(state, pose))
            fixed = freeze_correspondences(lines, rendered_state, pivot, offset)
            counts = dict(total=len(fixed), **{name: 0 for name in REGION_NAMES})
            for line in fixed:
                counts[line['region_id']] += 1
            _, _, before = evaluate_region(fixed, state, k_numpy, pivot, offset,
                                            self.optimize_joints, self.lm_config.huber_delta_px, sigma)
            outer = dict(corr_iteration=corr_iteration, scale=scale, residual_sigma_px=sigma,
                         correspondences=counts, region_residual_before=before, updates=[])
            history.append(outer)
            if len(fixed) < self.min_correspondences:
                stop_reason = outer['stop_reason'] = 'insufficient_correspondences'
                break
            stop = False
            for update_iteration in range(self.n_update_iterations):
                state, lm_lambda, update = lm_update(
                    fixed, state, k_numpy, pivot, offset, self.lm_config, lm_lambda,
                    self.alpha_limit, self.jaw_limit, self.optimize_joints, sigma)
                update['update_iteration'] = update_iteration
                outer['updates'].append(update)
                if not update['accepted']:
                    stop_reason = outer['stop_reason'] = 'no_decreasing_step'
                    stop = True
                    break
                reduction = (update['cost_before'] - update['cost_after']) / max(update['cost_before'], 1e-12)
                update['relative_cost_reduction'] = reduction
                small = self._small_step(update) or reduction < self.early_stop_cost_rel
                consecutive_small = consecutive_small + 1 if small else 0
                update['early_stop_streak'] = consecutive_small
                if consecutive_small >= self.early_stop_patience:
                    stop_reason = outer['stop_reason'] = 'converged'
                    stop = True
                    break
            if stop:
                break
        pose = pose_tensors_from_numpy(state, pose)
        rendered = self.renderer.render_icg(pose, k)
        pred = rendered["mask"]
        if self.observation == 'histogram':
            self._update_histograms(rgb, pred[0].detach().cpu().numpy())
        dice = dice_dict(pred, frame.mask.to(device)[None])
        return {
            "pose": pose, "x": pose_to_vector(pose), "result": rendered,
            "dice": dice, "initial_dice": initial_dice or dice, "history": history,
            "stop_reason": stop_reason,
            "correspondences": history[-1]["correspondences"]['total'] if history else 0,
        }
