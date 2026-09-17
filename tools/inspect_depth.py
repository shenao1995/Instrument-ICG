"""Inspect raw PNG units against GT CAD Z; reports candidates, never selects a scale."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from icg.geometry import InstrumentMesh, SemanticRenderer
from icg.sim_data import CONVENTION, PoseConverter, iter_frames, list_runs


def depth_statistics(values):
    a = np.asarray(values)
    finite = a[np.isfinite(a)]
    valid = finite[finite > 0]
    return dict(dtype=str(a.dtype), shape=list(a.shape),
                min=float(finite.min()) if finite.size else None,
                max=float(finite.max()) if finite.size else None,
                nonzero_min=float(valid.min()) if valid.size else None,
                nonzero_max=float(valid.max()) if valid.size else None,
                median=float(np.median(valid)) if valid.size else None,
                percentiles={f'p{p}': float(np.percentile(valid,p)) if valid.size else None for p in (1,10,50,90,99)},
                valid_pixels=int(valid.size), invalid_pixels=int(a.size-valid.size))


def compare_depth(depth_m, rendered, observed_mask):
    """Evaluation only: same-pixel CAD-Z comparison on instrument overlap."""
    xyz = rendered['xyz'][0].detach().cpu().numpy()
    pred = rendered['mask'][0].detach().cpu().numpy().max(0) > .5
    observed = observed_mask.detach().cpu().numpy() if torch.is_tensor(observed_mask) else np.asarray(observed_mask)
    overlap = (observed.max(0) > .5) & pred & np.isfinite(depth_m) & (depth_m > 0)
    overlap &= np.isfinite(xyz[2]) & (xyz[2] > 0)
    error = np.abs(np.asarray(depth_m)[overlap] - xyz[2][overlap])
    return dict(valid_pixels=int(error.size),
                median_abs_depth_error_m=float(np.median(error)) if error.size else None,
                mean_abs_depth_error_m=float(error.mean()) if error.size else None,
                p90_abs_depth_error_m=float(np.percentile(error,90)) if error.size else None)


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--instrument',type=Path,required=True)
    p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--frame',type=int,default=0)
    p.add_argument('--camera',choices=('left','right'),default='left')
    p.add_argument('--height',type=int,default=288)
    p.add_argument('--width',type=int,default=512)
    p.add_argument('--baseline',type=float,default=.005)
    p.add_argument('--candidate-scales',type=float,nargs='+',default=[1.,1e-1,1e-2,1e-3,1e-4,1e-5])
    p.add_argument('--output',type=Path,default=Path('depth-inspection.json'))
    return p


@torch.no_grad()
def main(args):
    if args.frame < 0 or not all(np.isfinite(s) and s > 0 for s in args.candidate_scales):
        raise ValueError('frame must be nonnegative and candidate scales finite and positive')
    available={p.name:p for p in list_runs(args.data_root)}
    unknown=set(args.runs)-available.keys()
    if unknown: raise ValueError(f'Unknown runs: {sorted(unknown)}')
    size=(args.height,args.width)
    mesh=InstrumentMesh(args.instrument,0,load_appearance=False,convention=CONVENTION,arms=2).to('cuda')
    renderer=SemanticRenderer(mesh,size,'nvdiffrast',supersample=1,render_rgb=False)
    results=[]
    for name in args.runs:
        run=available[name]
        source=run/('depth' if args.camera=='left' else 'depth_right')/f'{args.frame:04d}.png'
        raw=cv2.imread(str(source),cv2.IMREAD_UNCHANGED)
        if raw is None: raise FileNotFoundError(source)
        if raw.ndim!=2: raise ValueError(f'Expected single-channel depth: {source}')
        frames=iter_frames(run,PoseConverter(CONVENTION,args.camera,args.baseline),args.camera,size,args.instrument,
                           start=args.frame,limit=1)
        try: frame=next(frames)
        except StopIteration: raise ValueError(f'Frame {args.frame} not found in {run}')
        finally: frames.close()
        if raw.shape!=frame.source_size:
            raise ValueError(f'Depth shape {raw.shape} does not match camera {frame.source_size}')
        pose={key:value.to('cuda') for key,value in frame.pose.items()}
        rendered=renderer.render_icg(pose,frame.K.to('cuda'))
        resized=cv2.resize(raw.astype(np.float32),size[::-1],interpolation=cv2.INTER_NEAREST)
        result=dict(run=name,frame=args.frame,camera=args.camera,source=str(source),raw=depth_statistics(raw),candidates=[])
        print(json.dumps({k:v for k,v in result.items() if k!='candidates'},indent=2))
        print('scale        median_abs_error_m   mean_abs_error_m   valid_pixels')
        for scale in args.candidate_scales:
            row=dict(scale=scale,**compare_depth(resized.astype(np.float64)*scale,rendered,frame.mask))
            result['candidates'].append(row)
            print(f"{scale:<12g} {str(row['median_abs_depth_error_m']):<20} {str(row['mean_abs_depth_error_m']):<18} {row['valid_pixels']}")
        results.append(result)
    report=dict(selected_scale=None,notice='No scale selected or persisted. Choose --depth-scale explicitly from these results.',results=results)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(report['notice'])
    print(f'Saved {args.output}')


if __name__=='__main__':
    main(build_parser().parse_args())
