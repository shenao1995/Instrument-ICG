"""ICG / ICG+ / Mb-ICG tracking of dual endoscopic instruments.

A new repository. Geometry, camera, FK, mesh loading, stereo, and the
nvdiffrast clip transform are copied from Instrument-pose-opt and are not
modified there.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from icg.geometry import ROOT, InstrumentMesh, SemanticRenderer, instruments_json
from icg.metrics import CHANNEL_NAMES, pose_error_dict
from icg.optimizer import LMConfig
from icg.parameterization import initialize_pose
from icg.sim_data import CONVENTION, iter_frames, list_runs, PoseConverter
from icg.tracker import InstrumentTracker
from icg.visualization import mask_palette, validation_images

ARM_NAMES = ("left", "right")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-root", type=Path, default=Path("/data/data1/shena/data/simulated_data/data30"))
    p.add_argument("--instrument", type=Path, default=Path("/data/data1/shena/data/simulated_instrument"))
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--camera", choices=("left", "right"), default="left")
    p.add_argument("--baseline", type=float, default=.005)
    p.add_argument("--height", type=int, default=288)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument("--method", choices=("icg", "icgplus", "mbicg"), default="mbicg",
                   help="icg: single-region 6DoF, joints frozen; icgplus: multi-region; mbicg: kinematic 9DoF")
    p.add_argument("--observation", choices=("mask", "histogram"), default="mask",
                   help="mask uses GT semantics as the region posterior; histogram is classic ICG color stats")
    p.add_argument("--texture", action="store_true", help="Reserved legacy flag; unavailable in Region-only LM")
    p.add_argument("--corr-iterations", "--iterations", dest="corr_iterations", type=int, default=4,
                   help="Outer correspondence iterations; 0 evaluates the initial pose only")
    p.add_argument("--update-iterations", type=int, default=2, help="LM updates per fixed correspondence set")
    p.add_argument("--lm-lambda", type=float, default=1e-2)
    p.add_argument("--lm-lambda-min", type=float, default=1e-8)
    p.add_argument("--lm-lambda-max", type=float, default=1e8)
    p.add_argument("--lm-max-retries", type=int, default=5, help="Retries after the initial LM attempt")
    p.add_argument("--huber-delta-px", type=float, default=2.)
    p.add_argument("--region-sigma-px", type=float, nargs="+", default=[25., 15., 10.],
                   help="Existing Region likelihood sigma per outer iteration (last value repeated)")
    p.add_argument("--max-rotation-step-deg", type=float, default=3.)
    p.add_argument("--max-translation-step-mm", type=float, default=2.)
    p.add_argument("--max-joint-step-deg", type=float, default=3.)
    p.add_argument("--tikhonov-rotation", type=float, default=100.)
    p.add_argument("--tikhonov-translation", type=float, default=1000.)
    p.add_argument("--tikhonov-joint", type=float, default=50.)
    p.add_argument("--early-stop-cost-rel", type=float, default=1e-4)
    p.add_argument("--early-stop-translation-mm", type=float, default=.05)
    p.add_argument("--early-stop-rotation-deg", type=float, default=.02)
    p.add_argument("--early-stop-joint-deg", type=float, default=.02)
    p.add_argument("--early-stop-patience", type=int, default=2)
    p.add_argument("--min-correspondences", type=int, default=30)
    p.add_argument("--n-lines", type=int, default=180, help="Correspondence lines per frame (split across regions)")
    p.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=[1, 1, 1, 1],
        help="Histogram correspondence scales; mask observation always uses scale=1"
    )
    p.add_argument("--runs", nargs="*", default=())
    p.add_argument("--num-samples", "--max-runs", type=int, default=0, dest="max_runs")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--init", choices=("perturb-gt", "identity", "centroid", "gt"), default="perturb-gt")
    p.add_argument("--init-rotation-deg", type=float, default=10.)
    p.add_argument("--init-translation-mm", type=float, default=5.)
    p.add_argument("--init-joints-deg", type=float, default=8.)
    p.add_argument("--track", action="store_true",
                   help="After the first frame, initialize from the previous prediction")
    p.add_argument("--init-depth", type=float, default=.06)
    p.add_argument("--init-x-split", type=float, default=.04)
    p.add_argument("--min-fov-pixels", type=int, default=80)
    p.add_argument("--optimize-anyway", action="store_true")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--translation-min", type=float, nargs=3, default=(-.08, -.08, .025))
    p.add_argument("--translation-max", type=float, nargs=3, default=(.08, .08, .12))
    p.add_argument("--alpha-limit-deg", type=float, default=90.)
    p.add_argument("--joint-limit-deg", type=float, default=125.)
    p.add_argument("--no-overlays", action="store_true")
    p.add_argument("--save-history", action="store_true")
    return p


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def resolve_output(args, runs):
    if args.output is not None:
        return Path(args.output)
    if not runs:
        raise ValueError("No runs selected")
    tag = args.method
    if len(runs) == 1:
        return ROOT / "runs" / tag / runs[0].name
    return ROOT / "runs" / tag / Path(args.data_root).name


def save_overlay(path, frame, result):
    batch = {"rgb": frame.rgb[None], "mask": frame.mask[None],
             "tips": frame.tips[None], "tip_confidence": frame.tip_confidence[None]}
    cpu = {key: value.detach().cpu() for key, value in result.items() if torch.is_tensor(value)}
    if "rgb" not in cpu:
        palette = torch.as_tensor(mask_palette(cpu["mask"].shape[1]), dtype=cpu["mask"].dtype)
        cpu["rgb"] = torch.einsum("bchw,cd->bdhw", cpu["mask"].clamp(0, 1), palette).clamp(0, 1)
    panels = validation_images(batch, cpu)
    grid = panels["overlap_GT_left_prediction_right"]
    Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(path)


def mean_or_none(values):
    finite = [v for v in values if v is not None and v == v]
    return None if not finite else float(sum(finite) / len(finite))


def summarize(rows):
    summary = {
        "frames": len(rows),
        "optimized": sum(1 for row in rows if row.get("optimized")),
        "skipped_out_of_fov": sum(1 for row in rows if row.get("skipped") == "out_of_fov"),
        "mean_time_s": mean_or_none([row.get("time_s") for row in rows if not row.get("skipped")]),
        "mean_correspondences": mean_or_none([row.get("correspondences") for row in rows if not row.get("skipped")]),
    }
    for name in CHANNEL_NAMES + ("mean", "left_mean", "right_mean"):
        summary[f"dice_{name}"] = mean_or_none([row.get("dice", {}).get(name) for row in rows if not row.get("skipped")])
    for arm in ARM_NAMES:
        for key in (f"{arm}_translation_error_mm", f"{arm}_rotation_error_deg", f"{arm}_joints_mae_deg"):
            summary[key] = mean_or_none([row.get("pose_error", {}).get(key) for row in rows if not row.get("skipped")])
    summary['initial_dice_mean'] = mean_or_none([row.get('initial_dice', {}).get('mean') for row in rows])
    for arm in ARM_NAMES:
        for suffix in ('translation_error_mm', 'rotation_error_deg', 'joints_mae_deg'):
            key = f'{arm}_{suffix}'
            summary[f'initial_{key}'] = mean_or_none([row.get('initial_pose_error', {}).get(key) for row in rows])
    return summary


def write_csv(path, rows):
    keys = ["run", "frame", "optimized", "skipped", "init", "time_s", "correspondences",
            "dice_mean", "dice_left_mean", "dice_right_mean",
            *[f"dice_{name}" for name in CHANNEL_NAMES],
            "left_translation_error_mm", "left_rotation_error_deg",
            "right_translation_error_mm", "right_rotation_error_deg"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(keys) + "\n")
        for row in rows:
            dice, err = row.get("dice") or {}, row.get("pose_error") or {}
            values = {
                "run": row.get("run"), "frame": row.get("frame"),
                "optimized": int(bool(row.get("optimized"))),
                "skipped": row.get("skipped") or "",
                "init": (row.get("init") or {}).get("init", ""),
                "time_s": row.get("time_s"), "correspondences": row.get("correspondences"),
                "dice_mean": dice.get("mean"), "dice_left_mean": dice.get("left_mean"),
                "dice_right_mean": dice.get("right_mean"),
                **{f"dice_{name}": dice.get(name) for name in CHANNEL_NAMES},
                "left_translation_error_mm": err.get("left_translation_error_mm"),
                "left_rotation_error_deg": err.get("left_rotation_error_deg"),
                "right_translation_error_mm": err.get("right_translation_error_mm"),
                "right_rotation_error_deg": err.get("right_rotation_error_deg"),
            }
            handle.write(",".join("" if values[k] is None else str(values[k]) for k in keys) + "\n")


def main(args):
    if min(args.height, args.width) < 32 or args.height % 2 or args.width % 2:
        raise ValueError("height/width must be even and >= 32")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("nvdiffrast ICG rendering requires --device cuda")
    size = (args.height, args.width)
    runs = list_runs(args.data_root)
    if args.runs:
        wanted = set(args.runs)
        runs = [run for run in runs if run.name in wanted]
        missing = wanted - {run.name for run in runs}
        if missing:
            raise ValueError(f"Unknown runs: {sorted(missing)}")
    if args.max_runs:
        runs = runs[:args.max_runs]
    output = resolve_output(args, runs)
    output.mkdir(parents=True, exist_ok=True)
    per_run_dirs = len(runs) > 1

    mesh = InstrumentMesh(args.instrument, 0, load_appearance=False,
                          convention=CONVENTION, arms=2, part_albedo=None).to(device)
    renderer = SemanticRenderer(mesh, size, "nvdiffrast", supersample=1, render_rgb=False)
    converter = PoseConverter(CONVENTION, args.camera, args.baseline)
    alpha_limit, jaw_limit = np.deg2rad(args.alpha_limit_deg), np.deg2rad(args.joint_limit_deg)
    tracker = InstrumentTracker(
        renderer,
        method=args.method,
        observation=args.observation,
        use_texture=args.texture,
        n_lines=args.n_lines,
        n_corr_iterations=args.corr_iterations,
        n_update_iterations=args.update_iterations,
        sigma_r=args.region_sigma_px,
        lm_config=LMConfig(**{name: getattr(args, name) for name in LMConfig.__dataclass_fields__}),
        min_correspondences=args.min_correspondences,
        early_stop_cost_rel=args.early_stop_cost_rel,
        early_stop_translation_mm=args.early_stop_translation_mm,
        early_stop_rotation_deg=args.early_stop_rotation_deg,
        early_stop_joint_deg=args.early_stop_joint_deg,
        early_stop_patience=args.early_stop_patience,
        scales=tuple(args.scales),
        alpha_limit=alpha_limit,
        jaw_limit=jaw_limit,
    )
    config = {**vars(args), "output": str(output), "per_run_dirs": per_run_dirs,
              "mesh_sha256": mesh.hashes, "renderer": renderer.configuration(),
              "geometry_convention": CONVENTION, "state_dim": 18}
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    (output / "config.json").write_text(json.dumps(jsonable(config), indent=2), encoding="utf-8")
    print(f"method={args.method}  observation={args.observation}  output={output}", flush=True)

    rows = []
    started = time.perf_counter()
    batch_jsonl = None if per_run_dirs else (output / "frames.jsonl").open("w", encoding="utf-8")
    try:
        for run_index, run in enumerate(runs):
            print(f"[{run_index + 1}/{len(runs)}] {run.name}", flush=True)
            run_dir = output / run.name if per_run_dirs else output
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "overlays").mkdir(exist_ok=True)
            (run_dir / "history").mkdir(exist_ok=True)
            run_rows = []
            jsonl = (run_dir / "frames.jsonl").open("w", encoding="utf-8") if per_run_dirs else batch_jsonl
            previous_vector = None
            first_frame = True
            try:
                for frame in iter_frames(run, converter, args.camera, size, args.instrument,
                                         frame_stride=args.frame_stride, limit=args.limit,
                                         start=args.start_frame):
                    synchronize(device)
                    frame_start = time.perf_counter()
                    use_previous = args.track and args.init == "perturb-gt" and not first_frame and previous_vector is not None
                    pose, vector, init_info = initialize_pose(
                        args.init, frame, renderer, args.init_depth, args.init_x_split,
                        args.min_fov_pixels, alpha_limit, jaw_limit,
                        translation_min=args.translation_min, translation_max=args.translation_max,
                        rotation_degrees=args.init_rotation_deg, translation_mm=args.init_translation_mm,
                        joints_degrees=args.init_joints_deg,
                        seed=args.seed + run_index * 1_000_003 + frame.index,
                        previous_vector=previous_vector if use_previous else None,
                        first_frame=first_frame or not use_previous)
                    record = {"run": frame.run, "frame": frame.index, "init": init_info, "optimized": False}
                    overlay_stem = f"{frame.index:04d}" if per_run_dirs else f"{frame.run}_{frame.index:04d}"
                    if not init_info["visible"] and not args.optimize_anyway:
                        record.update(skipped="out_of_fov", time_s=time.perf_counter() - frame_start)
                        run_rows.append(record)
                        rows.append(record)
                        jsonl.write(json.dumps(jsonable(record)) + "\n")
                        jsonl.flush()
                        print(f"  frame {frame.index:04d}: skip, init {init_info['init']}", flush=True)
                        previous_vector = None
                        first_frame = False
                        continue
                    # GT is read only for evaluation; it never enters LM acceptance or stopping.
                    gt_pose = {key: value.to(device) for key, value in frame.pose.items()}
                    initial_errors = pose_error_dict(pose, gt_pose, mesh.convention.pivot, mesh.convention.shaft_offset)
                    best = tracker.refine(pose, frame)
                    best['pose_error'] = pose_error_dict(best['pose'], gt_pose, mesh.convention.pivot, mesh.convention.shaft_offset)
                    synchronize(device)
                    if not args.no_overlays:
                        save_overlay(run_dir / "overlays" / f"{overlay_stem}.png", frame, best["result"])
                    if args.save_history:
                        (run_dir / "history" / f"{overlay_stem}.json").write_text(
                            json.dumps(jsonable(best["history"]), indent=2), encoding="utf-8")
                    record.update(
                        optimized=any(u['accepted'] for c in best['history'] for u in c['updates']),
                        skipped=None, time_s=time.perf_counter() - frame_start,
                        initial_dice=best['initial_dice'], initial_pose_error=initial_errors,
                        stop_reason=best['stop_reason'],
                        method=args.method, correspondences=best["correspondences"],
                        dice=best["dice"], pose_error=best["pose_error"],
                        predicted=instruments_json(best["pose"]),
                        gt=instruments_json({key: value.to(device) for key, value in frame.pose.items()}),
                    )
                    run_rows.append(record)
                    rows.append(record)
                    jsonl.write(json.dumps(jsonable(record)) + "\n")
                    jsonl.flush()
                    dice, errors = best["dice"], best["pose_error"]
                    print(
                        f"  frame {frame.index:04d}: {record['time_s']:.2f}s  "
                        f"corr {best['correspondences']}  "
                        f"dice L {dice.get('left_mean')} R {dice.get('right_mean')}  "
                        f"t {errors['left_translation_error_mm']:.1f}/{errors['right_translation_error_mm']:.1f} mm  "
                        f"R {errors['left_rotation_error_deg']:.1f}/{errors['right_rotation_error_deg']:.1f} deg  "
                        f"method {args.method}",
                        flush=True,
                    )
                    print(f"  initial/final Dice: {best['initial_dice']['mean']:.6f} -> {dice['mean']:.6f}; "
                          f"stop={best['stop_reason']}", flush=True)
                    for arm in ARM_NAMES:
                        pairs = [f"{label}: {initial_errors[arm + '_' + key]:.6f} -> {errors[arm + '_' + key]:.6f}"
                                 for key, label in (("translation_error_mm", "t_mm"),
                                                    ("rotation_error_deg", "R_deg"), ("joints_mae_deg", "joint_deg"))]
                        print(f"  {arm}: " + "; ".join(pairs), flush=True)
                    previous_vector = best["x"]
                    first_frame = False
            finally:
                if per_run_dirs:
                    jsonl.close()
            if per_run_dirs:
                run_summary = summarize(run_rows)
                run_summary.update(method=args.method, output=str(run_dir))
                (run_dir / "summary.json").write_text(json.dumps(jsonable(run_summary), indent=2), encoding="utf-8")
                write_csv(run_dir / "summary.csv", run_rows)
    finally:
        if batch_jsonl is not None:
            batch_jsonl.close()

    summary = summarize(rows)
    summary.update(elapsed_s=time.perf_counter() - started, method=args.method,
                   output=str(output), runs=[run.name for run in runs])
    (output / "summary.json").write_text(json.dumps(jsonable(summary), indent=2), encoding="utf-8")
    write_csv(output / "summary.csv", rows)
    print("Summary: " + json.dumps(jsonable(summary), indent=2), flush=True)
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
