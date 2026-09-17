"""Sparse, semantic-gated model-to-depth point-to-plane correspondences."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from icg.kinematics import camera_to_part_local

REGION_NAMES = ('left_shaft','left_wrist','left_grippers','right_shaft','right_wrist','right_grippers')
PART_TO_SEMANTIC = np.array([0,1,2,2,3,4,5,5], dtype=np.int64)


@dataclass(frozen=True)
class DepthConfig:
    max_points_per_region: int = 100
    search_radius_px: int = 5
    max_distance_mm: float = 5.
    sigma_mm: float = 2.
    huber_delta_mm: float = 2.
    visibility_tol_mm: float = 1.
    min_correspondences: int = 30
    occlusion_threshold_mm: float = 5.

    def __post_init__(self):
        for name, value in vars(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'Depth {name} must be finite and nonnegative')
        for name in ('max_points_per_region','search_radius_px','min_correspondences'):
            if int(getattr(self,name)) != getattr(self,name):
                raise ValueError(f'Depth {name} must be an integer')
        for name in ('max_points_per_region','max_distance_mm','sigma_mm','huber_delta_mm','visibility_tol_mm','min_correspondences'):
            if getattr(self,name) <= 0: raise ValueError(f'Depth {name} must be positive')


def backproject(u, v, z, k):
    """Pixel-centre OpenCV coordinates and axial depth in metres -> camera XYZ."""
    u,v,z = np.broadcast_arrays(np.asarray(u,np.float64),np.asarray(v,np.float64),np.asarray(z,np.float64))
    return np.stack(((u-k[0,2])*z/k[0,0], (v-k[1,2])*z/k[1,1], z),axis=-1)


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)


def rendered_arrays(rendered):
    return (_numpy(rendered['xyz'])[0].transpose(1,2,0),
            _numpy(rendered['normal'])[0].transpose(1,2,0),
            np.rint(_numpy(rendered['part_id'])[0,0]).astype(np.int64),
            _numpy(rendered['mask'])[0].max(0) > .5)


def find_depth_match(xyz, part_id, depth, semantic, k, rendered_z, rendered_part, config):
    """Nearest Euclidean 3D match inside a local pixel window, with two occlusion gates."""
    h,w = depth.shape
    if not np.all(np.isfinite(xyz)) or xyz[2] <= 0:
        return None, 'invalid_model'
    uv = np.array([k[0,0]*xyz[0]/xyz[2]+k[0,2], k[1,1]*xyz[1]/xyz[2]+k[1,2]])
    cx,cy = np.rint(uv).astype(int)
    if not (0 <= cx < w and 0 <= cy < h): return None, 'outside_image'
    z_buffer = rendered_z[cy,cx]
    if (not np.isfinite(z_buffer) or z_buffer <= 0 or
            abs(xyz[2]-z_buffer) >= config.visibility_tol_mm/1000. or
            rendered_part[cy,cx] != part_id):
        return None, 'rendered_occlusion'
    centre_depth = depth[cy,cx]
    occlusion = config.occlusion_threshold_mm/1000.
    # A measured foreground occluder may have another semantic label.
    if np.isfinite(centre_depth) and centre_depth > 0 and xyz[2]-centre_depth > occlusion:
        return None, 'measured_occlusion'
    radius = config.search_radius_px
    x0,x1 = max(0,cx-radius),min(w,cx+radius+1)
    y0,y1 = max(0,cy-radius),min(h,cy+radius+1)
    z = depth[y0:y1,x0:x1]
    channel = int(PART_TO_SEMANTIC[part_id])
    valid = np.isfinite(z) & (z>0) & (semantic[channel,y0:y1,x0:x1]>.5)
    valid &= xyz[2]-z <= occlusion
    yy,xx = np.where(valid)
    if not len(xx): return None, 'no_valid_semantic_depth'
    xx,yy = xx+x0,yy+y0
    points = backproject(xx,yy,depth[yy,xx],k)
    distance2 = np.sum((points-xyz)**2,axis=1)
    selected = int(np.argmin(distance2))
    if distance2[selected] >= (config.max_distance_mm/1000.)**2:
        return None, 'distance_gate'
    return dict(observed_xyz=points[selected],observed_pixel=np.array([xx[selected],yy[selected]]),
                distance_m=float(np.sqrt(distance2[selected]))), None



def find_depth_matches(points, ids, depth, semantic, k, rendered_z, rendered_part, config):
    """Batched equivalent of find_depth_match over sparse local windows only."""
    points=np.asarray(points,np.float64).reshape(-1,3)
    ids=np.asarray(ids,np.int64)
    if np.any((ids<0)|(ids>=8)) or len(ids)!=len(points): raise ValueError('Invalid depth part IDs')
    n=len(points)
    if not n:return [],[]
    h,w=depth.shape
    reasons=np.full(n,'',dtype=object)
    bad=~np.isfinite(points).all(1)|(points[:,2]<=0)
    reasons[bad]='invalid_model'
    safe=np.where(bad[:,None],np.array([0.,0.,1.]),points)
    uv=np.column_stack((k[0,0]*safe[:,0]/safe[:,2]+k[0,2],k[1,1]*safe[:,1]/safe[:,2]+k[1,2]))
    centres=np.rint(np.clip(uv,-1e9,1e9)).astype(np.int64)
    cx,cy=centres.T
    def reject(condition,reason):
        reasons[(reasons=='')&condition]=reason
    reject((cx<0)|(cx>=w)|(cy<0)|(cy>=h),'outside_image')
    sx,sy=np.clip(cx,0,w-1),np.clip(cy,0,h-1)
    buffer=rendered_z[sy,sx]
    reject(~np.isfinite(buffer)|(buffer<=0)|(np.abs(safe[:,2]-buffer)>=config.visibility_tol_mm/1000.)|
           (rendered_part[sy,sx]!=ids),'rendered_occlusion')
    centre=depth[sy,sx]
    occ=config.occlusion_threshold_mm/1000.
    reject(np.isfinite(centre)&(centre>0)&(safe[:,2]-centre>occ),'measured_occlusion')
    radius=config.search_radius_px
    dy,dx=np.mgrid[-radius:radius+1,-radius:radius+1]
    xx,yy=cx[:,None]+dx.ravel(),cy[:,None]+dy.ravel()
    in_image=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
    sx,sy=np.clip(xx,0,w-1),np.clip(yy,0,h-1)
    z=depth[sy,sx]
    valid=in_image&np.isfinite(z)&(z>0)&(semantic[PART_TO_SEMANTIC[ids,None],sy,sx]>.5)
    valid &= safe[:,2,None]-z<=occ
    reject(~valid.any(1),'no_valid_semantic_depth')
    candidates=backproject(xx,yy,np.where(valid,z,0.),k)
    distance2=np.sum((candidates-safe[:,None,:])**2,axis=2)
    distance2=np.where(valid,distance2,np.inf)
    selected=np.argmin(distance2,axis=1)
    nearest=distance2[np.arange(n),selected]
    reject(nearest>=(config.max_distance_mm/1000.)**2,'distance_gate')
    matches=[]
    for i in range(n):
        pick=selected[i]
        matches.append(None if reasons[i] else dict(observed_xyz=candidates[i,pick].copy(),
                       observed_pixel=np.array([xx[i,pick],yy[i,pick]]),distance_m=float(np.sqrt(nearest[i]))))
    return matches,[str(reason) if reason else None for reason in reasons]


def sample_depth_correspondences(rendered, depth, semantic, k, state, pivot, shaft_offset, config):
    """Up to max_points_per_region, split between visible rigid parts of each class.

    One existing raster pass supplies points and normals. Inner LM cannot mutate
    the copied local points, measured XYZ, or camera-space normals.
    """
    xyz,normals,part_ids,visible=rendered_arrays(rendered)
    depth,semantic,k=_numpy(depth),_numpy(semantic),_numpy(k)
    if depth.shape!=visible.shape or semantic.shape!=(6,*visible.shape):
        raise ValueError('Depth, semantics, and renderer must share working resolution')
    selected_pixels=[]
    for channel in range(6):
        parts=[pid for pid in range(8) if PART_TO_SEMANTIC[pid]==channel]
        pools={pid:np.argwhere(visible&(part_ids==pid)) for pid in parts}
        parts=[pid for pid in parts if len(pools[pid])]
        for number,pid in enumerate(parts):
            budget=config.max_points_per_region//len(parts)+(number<config.max_points_per_region%len(parts))
            pixels=pools[pid]
            indices=np.linspace(0,len(pixels)-1,min(budget,len(pixels)),dtype=int)
            selected_pixels.extend((int(y),int(x),pid) for y,x in pixels[indices])
    fixed=[]
    stats=dict(sampled=len(selected_pixels),accepted=0,rejected={})
    if not selected_pixels:return (),stats
    ys,xs,ids=np.asarray(selected_pixels).T
    points=xyz[ys,xs].astype(np.float64)
    normal=normals[ys,xs].astype(np.float64)
    norm=np.linalg.norm(normal,axis=1)
    good=np.isfinite(norm)&(norm>=1e-8)
    if not good.all():stats['rejected']['invalid_normal']=int((~good).sum())
    points,ids,normal,norm=points[good],ids[good],normal[good],norm[good]
    matches,reasons=find_depth_matches(points,ids,depth,semantic,k,xyz[:,:,2],part_ids,config)
    for i,(match,reason) in enumerate(zip(matches,reasons)):
        if match is None:
            stats['rejected'][reason]=stats['rejected'].get(reason,0)+1
            continue
        pid=int(ids[i]);arm=pid//4
        local=camera_to_part_local(points[i],pid,state[0][arm],state[1][arm],state[2][arm],pivot,shaft_offset)
        row=dict(local_xyz=local,observed_xyz=match['observed_xyz'],normal=normal[i]/norm[i],
                 part_id=pid,region_id=REGION_NAMES[PART_TO_SEMANTIC[pid]],weight=1.,
                 observed_pixel=match['observed_pixel'],distance_m=match['distance_m'])
        for key in ('local_xyz','observed_xyz','normal','observed_pixel'):
            row[key]=np.array(row[key],copy=True)
            row[key].setflags(write=False)
        fixed.append(row)
    stats['accepted']=len(fixed)
    return tuple(fixed),stats


def depth_alignment_statistics(depth_m, rendered, semantic):
    """Evaluation-only same-pixel axial-depth error on semantic/CAD overlap."""
    depth=_numpy(depth_m)
    observed=_numpy(semantic)
    xyz=_numpy(rendered['xyz'])[0]
    visible=_numpy(rendered['mask'])[0].max(0)>.5
    keep=visible & (observed.max(0)>.5) & np.isfinite(depth) & (depth>0)
    keep &= np.isfinite(xyz[2]) & (xyz[2]>0)
    error=np.abs(depth[keep]-xyz[2][keep])
    return dict(valid_pixels=int(error.size),
                median_abs_depth_error_m=float(np.median(error)) if error.size else None,
                mean_abs_depth_error_m=float(error.mean()) if error.size else None,
                p90_abs_depth_error_m=float(np.percentile(error,90)) if error.size else None)
