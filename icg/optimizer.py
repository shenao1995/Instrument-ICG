"""Robust Region LM on fixed part-local correspondences (no GT dependencies)."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from icg.kinematics import (ARM_DIM, STATE_DIM, apply_increment, camera_to_part_local,
                            projection_jacobian, transform_part_point, batch_points_and_jacobians_from_local)


@dataclass(frozen=True)
class LMConfig:
    lm_lambda: float = 1e-2
    lm_lambda_min: float = 1e-8
    lm_lambda_max: float = 1e8
    lm_max_retries: int = 5  # retries AFTER the first solve
    huber_delta_px: float = 2.0
    max_rotation_step_deg: float = 3.0
    max_translation_step_mm: float = 2.0
    max_joint_step_deg: float = 3.0
    # Tikhonov penalties on the incremental step, NOT a target-pose prior.
    tikhonov_rotation: float = 100.0
    tikhonov_translation: float = 1000.0
    tikhonov_joint: float = 50.0

    def __post_init__(self):
        for key, value in vars(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{key} must be finite and nonnegative')
        if not 0 < self.lm_lambda_min <= self.lm_lambda <= self.lm_lambda_max:
            raise ValueError('Require 0 < lm_lambda_min <= lm_lambda <= lm_lambda_max')
        if self.huber_delta_px <= 0:
            raise ValueError('huber_delta_px must be positive')
        if int(self.lm_max_retries) != self.lm_max_retries:
            raise ValueError('lm_max_retries must be an integer')


def project_point(xyz, k):
    if not np.all(np.isfinite(xyz)) or xyz[2] <= 1e-6:
        raise ValueError('Surface point is behind/too close to the camera or non-finite')
    return np.array([k[0, 0] * xyz[0] / xyz[2] + k[0, 2],
                     k[1, 1] * xyz[1] / xyz[2] + k[1, 2]], np.float64)


def freeze_correspondences(lines, state, pivot, shaft_offset):
    """Copy and validate observations ONCE, and attach an immutable CAD-local point.

    Candidates never drop/reweight observations based on visibility. An invalid
    candidate projection invalidates its entire cost, rather than hiding a point.
    """
    rotations, translations, joints = state
    fixed = []
    for line in lines:
        pid = int(line['part_id'])
        if not 0 <= pid < 8:
            continue
        xyz = np.asarray(line['xyz'], np.float64)
        normal = np.asarray(line['n'], np.float64)
        observed = np.asarray(line['observed'], np.float64)
        weight = float(line['weight'])
        if (not all(np.all(np.isfinite(v)) for v in (xyz, normal, observed)) or
                not np.isfinite(weight) or weight <= 0 or xyz[2] <= 1e-6 or
                np.linalg.norm(normal) < 1e-12):
            continue
        arm = pid // 4
        local = camera_to_part_local(xyz, pid, rotations[arm], translations[arm],
                                      joints[arm], pivot, shaft_offset)
        row = dict(line, local_xyz=local, weight=weight)
        for name in ('xyz', 'local_xyz', 'observed', 'n', 'c'):
            if name in row:
                row[name] = np.array(row[name], dtype=np.float64, copy=True)
                row[name].setflags(write=False)
        fixed.append(row)
    return tuple(fixed)


def region_residual_and_jacobian(line, state, k, pivot, shaft_offset, optimize_joints=True):
    """Full n.T @ J_projection(X(q)) @ J_kinematics(local, q), in pixels."""
    xyz, jac_x = transform_part_point(line['local_xyz'], line['part_id'], *state,
                                      pivot, shaft_offset, optimize_joints)
    residual = float(line['n'] @ (project_point(xyz, k) - line['observed']))
    jac = line['n'] @ projection_jacobian(xyz, k[0, 0], k[1, 1]) @ jac_x
    return residual, jac


def huber(residuals, delta):
    abs_r = np.abs(residuals)
    weights = np.ones_like(abs_r)
    outside = abs_r > delta
    weights[outside] = delta / abs_r[outside]
    cost = np.where(outside, delta * (abs_r - .5 * delta), .5 * residuals ** 2)
    return cost, weights


def evaluate_region(lines, state, k, pivot, shaft_offset, optimize_joints=True,
                    huber_delta_px=2.0, sigma_px=1.0):
    if not np.isfinite(sigma_px) or sigma_px <= 0:
        raise ValueError('sigma_px must be finite and positive')
    residuals, jacobians = [], []
    for line in lines:
        residual, jac = region_residual_and_jacobian(
            line, state, k, pivot, shaft_offset, optimize_joints)
        residuals.append(residual)
        jacobians.append(jac)
    residuals = np.asarray(residuals, np.float64)
    jacobians = np.asarray(jacobians, np.float64).reshape(-1, STATE_DIM)
    # Keep the old Region likelihood variance, so the existing Tikhonov
    # coefficients retain their strength. Huber threshold remains in pixels.
    weights = np.array([line['weight'] for line in lines], np.float64) / sigma_px ** 2
    rho, robust_weights = huber(residuals, huber_delta_px)
    abs_r = np.abs(residuals)
    stats = {
        'raw_cost': float(.5 * np.sum(weights * residuals ** 2)),
        'robust_cost': float(np.sum(weights * rho)),
        'mean_abs_px': float(abs_r.mean()) if len(abs_r) else None,
        'median_abs_px': float(np.median(abs_r)) if len(abs_r) else None,
        'p90_abs_px': float(np.percentile(abs_r, 90)) if len(abs_r) else None,
        'max_abs_px': float(abs_r.max()) if len(abs_r) else None,
    }
    combined = weights * robust_weights
    g = jacobians.T @ (combined * residuals)
    h = jacobians.T @ (combined[:, None] * jacobians)
    if not np.all(np.isfinite(g)) or not np.all(np.isfinite(h)) or not np.isfinite(stats['robust_cost']):
        raise ValueError('Non-finite Region system')
    return g, h, stats



def depth_residual_and_jacobian(line, state, pivot, shaft_offset, optimize_joints=True):
    """Point-to-plane residual in metres; observed XYZ and camera normal are fixed."""
    xyz, jac_x = transform_part_point(line['local_xyz'], line['part_id'], *state,
                                      pivot, shaft_offset, optimize_joints)
    if not np.all(np.isfinite(xyz)) or xyz[2] <= 1e-6:
        raise ValueError('Invalid candidate depth projection')
    return float(line['normal'] @ (xyz - line['observed_xyz'])), line['normal'] @ jac_x


def evaluate_depth(lines, state, pivot, shaft_offset, optimize_joints=True,
                   sigma_m=.002, huber_delta_m=.002):
    if not all(np.isfinite(v) and v > 0 for v in (sigma_m,huber_delta_m)):
        raise ValueError('Depth sigma and Huber delta must be positive metres')
    r=np.zeros(len(lines),np.float64)
    j=np.zeros((len(lines),STATE_DIM),np.float64)
    ids=np.array([line['part_id'] for line in lines],np.int64)
    if np.any((ids<0)|(ids>=8)):
        raise ValueError('Invalid depth part_id')
    for pid in np.unique(ids):
        selected=np.flatnonzero(ids==pid)
        local=np.stack([lines[i]['local_xyz'] for i in selected])
        observed=np.stack([lines[i]['observed_xyz'] for i in selected])
        normal=np.stack([lines[i]['normal'] for i in selected])
        arm=int(pid)//4
        xyz,jac=batch_points_and_jacobians_from_local(local,pid,state[0][arm],state[1][arm],state[2][arm],
                                                     pivot,shaft_offset,optimize_joints)
        if not np.all(np.isfinite(xyz)) or np.any(xyz[:,2]<=1e-6):
            raise ValueError('Invalid candidate depth projection')
        r[selected]=np.einsum('ni,ni->n',normal,xyz-observed)
        j[selected,arm*ARM_DIM:(arm+1)*ARM_DIM]=np.einsum('ni,nij->nj',normal,jac)
    weights=np.array([line['weight'] for line in lines],np.float64)/sigma_m**2
    rho,irls=huber(r,huber_delta_m)
    absolute_mm=np.abs(r)*1000.
    stats=dict(raw_cost=float(.5*np.sum(weights*r*r)),robust_cost=float(np.sum(weights*rho)),
               mean_abs_mm=float(absolute_mm.mean()) if len(r) else None,
               median_abs_mm=float(np.median(absolute_mm)) if len(r) else None,
               p90_abs_mm=float(np.percentile(absolute_mm,90)) if len(r) else None,
               max_abs_mm=float(absolute_mm.max()) if len(r) else None)
    weighted=weights*irls
    g=j.T@(weighted*r)
    h=j.T@(weighted[:,None]*j)
    if not np.all(np.isfinite(g)) or not np.all(np.isfinite(h)) or not np.isfinite(stats['robust_cost']):
        raise ValueError('Non-finite Depth system')
    return g,h,stats


def empty_residual_stats(unit):
    return dict(raw_cost=0.,robust_cost=0.,**{f'{name}_abs_{unit}':None for name in ('mean','median','p90','max')})


def evaluate_modalities(lines, depth_lines, state, k, pivot, shaft_offset, optimize_joints=True,
                        huber_delta_px=2., sigma_px=1., depth_sigma_m=.002, depth_huber_delta_m=.002,
                        use_region=True, use_depth=False, region_weight=1., depth_weight=1.):
    """Single shared normal equation and acceptance objective; disabled terms are not evaluated."""
    if not all(np.isfinite(v) and v>=0 for v in (region_weight,depth_weight)):
        raise ValueError('Modality weights must be finite and nonnegative')
    if not ((use_region and region_weight>0) or (use_depth and depth_weight>0)):
        raise ValueError('At least one enabled modality must have positive weight')
    g=np.zeros(STATE_DIM)
    h=np.zeros((STATE_DIM,STATE_DIM))
    region,depth=empty_residual_stats('px'),empty_residual_stats('mm')
    if use_region and region_weight>0:
        gr,hr,region=evaluate_region(lines,state,k,pivot,shaft_offset,optimize_joints,huber_delta_px,sigma_px)
        g+=region_weight*gr
        h+=region_weight*hr
    if use_depth and depth_weight>0:
        gd,hd,depth=evaluate_depth(depth_lines,state,pivot,shaft_offset,optimize_joints,depth_sigma_m,depth_huber_delta_m)
        g+=depth_weight*gd
        h+=depth_weight*hd
    stats=dict(region=region,depth=depth,
               robust_cost=region_weight*region['robust_cost']+depth_weight*depth['robust_cost'],
               raw_cost=region_weight*region['raw_cost']+depth_weight*depth['raw_cost'])
    return g,h,stats


def modality_history(before, after):
    return dict(total_cost_before=before['robust_cost'],total_cost_after=after['robust_cost'],
                region_cost_before=before['region']['robust_cost'],region_cost_after=after['region']['robust_cost'],
                depth_cost_before=before['depth']['robust_cost'],depth_cost_after=after['depth']['robust_cost'],
                region_residual_before=before['region'],region_residual_after=after['region'],
                depth_residual_before=before['depth'],depth_residual_after=after['depth'],
                depth_residual=after['depth'])

def clip_step(delta, config):
    """Independent rotation/translation norm caps and scalar joint caps per arm."""
    blocks = np.array(delta, dtype=np.float64, copy=True).reshape(-1, ARM_DIM)
    for block in blocks:
        for part, cap in ((slice(0, 3), np.deg2rad(config.max_rotation_step_deg)),
                          (slice(3, 6), config.max_translation_step_mm / 1000.)):
            norm = np.linalg.norm(block[part])
            if norm > cap:
                block[part] *= cap / norm
        block[6:9] = np.clip(block[6:9], -np.deg2rad(config.max_joint_step_deg),
                            np.deg2rad(config.max_joint_step_deg))
    return blocks.reshape(STATE_DIM)


def step_statistics(delta):
    arms = []
    for block in np.asarray(delta).reshape(2, ARM_DIM):
        arms.append({
            'rotation_step_deg': float(np.rad2deg(np.linalg.norm(block[:3]))),
            'translation_step_mm': float(1000 * np.linalg.norm(block[3:6])),
            'alpha_step_deg': float(np.rad2deg(abs(block[6]))),
            'theta_left_step_deg': float(np.rad2deg(abs(block[7]))),
            'theta_right_step_deg': float(np.rad2deg(abs(block[8]))),
        })
    return dict(zip(('left', 'right'), arms))


def lm_update(lines, state, k, pivot, shaft_offset, config, lm_lambda,
              alpha_limit, jaw_limit, optimize_joints=True, sigma_px=1.0, *,
              depth_lines=(), use_region=True, use_depth=False, region_weight=1., depth_weight=1.,
              depth_sigma_m=.002, depth_huber_delta_m=.002):
    """One accepted inner update or a rollback; all attempts share observations.

    Cost is the weighted sum of enabled, sigma-normalized robust modalities.
    Region uses pixels; Depth uses metres. Tikhonov regularizes only the step.
    The diagonal LM damping is separate from that Tikhonov matrix.
    """
    def evaluate(candidate_state):
        return evaluate_modalities(lines,depth_lines,candidate_state,k,pivot,shaft_offset,
                                   optimize_joints,config.huber_delta_px,sigma_px,depth_sigma_m,depth_huber_delta_m,
                                   use_region,use_depth,region_weight,depth_weight)
    g, h, before = evaluate(state)
    diagonal = np.maximum(np.diag(h), 1e-12)
    tikhonov = np.tile([config.tikhonov_rotation] * 3 +
                       [config.tikhonov_translation] * 3 + [config.tikhonov_joint] * 3, 2)
    lm_lambda = float(np.clip(lm_lambda, config.lm_lambda_min, config.lm_lambda_max))
    info = dict(cost_before=before['robust_cost'], cost_after=before['robust_cost'],
                raw_cost_before=before['raw_cost'], raw_cost_after=before['raw_cost'],
                accepted=False, lm_lambda_before=lm_lambda, lm_lambda_after=lm_lambda,
                lm_retries=0, residual_before=before['region'], residual_after=before['region'], attempts=[],
                **modality_history(before,before),
                **step_statistics(np.zeros(STATE_DIM)))
    for attempt in range(config.lm_max_retries + 1):
        matrix = h + np.diag(lm_lambda * diagonal + tikhonov)
        try:
            delta = -np.linalg.solve(matrix, g)
        except np.linalg.LinAlgError:
            delta = -np.linalg.lstsq(matrix, g, rcond=None)[0]
        candidate_cost = float('inf')
        candidate_stats = None
        if np.all(np.isfinite(delta)):
            delta = clip_step(delta, config)
            if not optimize_joints:
                delta.reshape(2, ARM_DIM)[:, 6:] = 0.
            candidate = apply_increment(*state, delta, alpha_limit, jaw_limit)
            if not optimize_joints:
                candidate[2][:] = state[2]
            # Log the actual constrained step, including joint-bound clamping.
            delta.reshape(2, ARM_DIM)[:, 6:] = candidate[2] - state[2]
            try:
                _, _, candidate_stats = evaluate(candidate)
                candidate_cost = candidate_stats['robust_cost']
            except ValueError:
                pass  # invalid projection -> reject the entire candidate
        accepted = candidate_cost < before['robust_cost']
        info['attempts'].append(dict(lm_lambda=lm_lambda,
                                     candidate_cost=candidate_cost if np.isfinite(candidate_cost) else None,
                                     accepted=bool(accepted)))
        info['lm_retries'] = attempt
        if accepted:
            lm_lambda = max(config.lm_lambda_min, lm_lambda * .5)
            info.update(cost_after=candidate_cost, raw_cost_after=candidate_stats['raw_cost'],
                        accepted=True, lm_lambda_after=lm_lambda, residual_after=candidate_stats['region'],
                        **modality_history(before,candidate_stats),
                        **step_statistics(delta))
            return candidate, lm_lambda, info
        previous_lambda = lm_lambda
        lm_lambda = min(config.lm_lambda_max, lm_lambda * 10.)
        info['lm_lambda_after'] = lm_lambda
        if previous_lambda == config.lm_lambda_max:
            break
    info['stop_reason'] = 'no_decreasing_step'
    return state, lm_lambda, info


def pose_tensors_from_numpy(state, template):
    import torch
    return {name: torch.as_tensor(value, device=template[name].device,
                                  dtype=template[name].dtype)[None]
            for name, value in zip(('R', 't', 'joints'), state)}
