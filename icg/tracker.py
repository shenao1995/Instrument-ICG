"""Frame-to-frame ICG / ICG+ / Mb-ICG tracker for dual endoscopic instruments."""
from __future__ import annotations

import numpy as np
import torch

from icg.correspondence import sample_correspondences
from icg.histograms import RegionHistogram, histogram_samples, mask_posterior
from icg.kinematics import pose_numpy
from icg.metrics import dice_dict, pose_error_dict
from icg.optimizer import assemble_normal, newton_step, update_pose_tensors
from icg.parameterization import pose_to_vector
from icg.texture import TextureModality


REGION_NAMES = (
    "left_shaft", "left_wrist", "left_grippers",
    "right_shaft", "right_wrist", "right_grippers",
)


class InstrumentTracker:
    def __init__(self, renderer, method="mbicg", observation="mask", use_texture=False,
                 n_lines=180, iterations=6, scales=(9, 7, 5, 2),
                 sigma_r=(25.0, 15.0, 10.0), lambda_r=100.0, lambda_t=1000.0, lambda_j=50.0,
                 alpha_limit=np.pi / 2, jaw_limit=np.deg2rad(125.)):
        if method not in ("icg", "icgplus", "mbicg"):
            raise ValueError(f"Unknown method {method}")
        if observation not in ("mask", "histogram"):
            raise ValueError("observation must be 'mask' or 'histogram'")
        self.renderer = renderer
        self.method = method
        self.observation = observation
        self.use_texture = bool(use_texture and method != "icg")
        self.n_lines = int(n_lines)
        self.iterations = int(iterations)
        self.scales = list(scales)
        self.sigma_r = list(sigma_r)
        self.lambda_r = float(lambda_r)
        self.lambda_t = float(lambda_t)
        self.lambda_j = float(lambda_j)
        self.alpha_limit = float(alpha_limit)
        self.jaw_limit = float(jaw_limit)
        self.optimize_joints = method == "mbicg"
        self.multi_region = method in ("icgplus", "mbicg")
        self.histograms = [RegionHistogram() for _ in REGION_NAMES]
        self.texture = TextureModality() if self.use_texture else None
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
        texture_lines = []
        if self.texture is not None:
            texture_lines = self.texture.correspondences(
                np.transpose(rgb, (1, 2, 0)), pred.max(0), xyz)
        return lines, texture_lines, pred

    def refine(self, pose, frame):
        device = self.mesh.vertices.device
        k = frame.K.to(device)
        rgb = frame.rgb.detach().cpu().numpy()
        if self.observation == "histogram" and not self._histograms_seeded:
            self._update_histograms(rgb, frame.mask.detach().cpu().numpy())
            self._histograms_seeded = True
        history = []
        for step in range(self.iterations):
            scale = self.scales[min(step, len(self.scales) - 1)]
            sigma = self.sigma_r[min(step, len(self.sigma_r) - 1)]
            rendered = self.renderer.render_icg(pose, k)
            lines, texture_lines, pred = self._correspondences(rendered, rgb, frame.mask, scale)
            rotations, translations, joints = pose_numpy(pose)
            g, h, used = assemble_normal(
                lines, rotations, translations, joints, k.detach().cpu().numpy(),
                self.mesh.convention.pivot, self.mesh.convention.shaft_offset,
                self.optimize_joints, sigma, self.lambda_r, self.lambda_t, self.lambda_j,
                texture_lines=texture_lines)
            delta = newton_step(g, h) if used else np.zeros(18)
            pose = update_pose_tensors(pose, delta, self.alpha_limit, self.jaw_limit)
            history.append({
                "iteration": step, "correspondences": used,
                "texture": len(texture_lines), "scale": scale,
                "delta_norm": float(np.linalg.norm(delta)),
            })
            if used == 0 or np.linalg.norm(delta) < 1e-5:
                break
        rendered = self.renderer.render_icg(pose, k)
        pred = rendered["mask"]
        self._update_histograms(rgb, pred[0].detach().cpu().numpy())
        dice = dice_dict(pred, frame.mask.to(device)[None])
        errors = pose_error_dict(pose, {key: value.to(device) for key, value in frame.pose.items()},
                                 self.mesh.convention.pivot, self.mesh.convention.shaft_offset)
        return {
            "pose": pose, "x": pose_to_vector(pose), "result": rendered,
            "dice": dice, "pose_error": errors, "history": history,
            "correspondences": history[-1]["correspondences"] if history else 0,
        }
