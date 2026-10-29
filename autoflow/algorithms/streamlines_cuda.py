"""Optional CUDA RK4 integration over the existing sampled VTK voxel domain.

VTK still prepares the masked field. CUDA advances one seed/direction per
thread; double precision positions avoid accumulating float32 spatial error.
Importing AutoFlow does not import CuPy or initialize a CUDA context.
"""

from functools import lru_cache

import numpy as np
import pyvista as pv

from ..task_control import check_cancelled


_SOURCE = r'''
struct Field {
    const float *v; const unsigned char *mask;
    const double *ax, *ay, *az;
    int nx, ny, nz;
};

__device__ int locate(const double *axis, int n, double x) {
    if (!isfinite(x) || x < axis[0] || x > axis[n]) return -1;
    int lo=0, hi=n;
    while (lo+1 < hi) {
        int mid=(lo+hi)/2;
        if (x < axis[mid]) hi=mid; else lo=mid;
    }
    return lo;
}

__device__ bool sample(Field f, const double *p, double *v, double &length) {
    int i=locate(f.ax,f.nx,p[0]), j=locate(f.ay,f.ny,p[1]), k=locate(f.az,f.nz,p[2]);
    if (i<0 || j<0 || k<0) return false;
    if (!f.mask[(i*f.ny+j)*f.nz+k]) {
        // Closed voxel faces belong to either adjacent active cell. This
        // also permits seeds placed exactly on the segmentation boundary.
        int ii=i,jj=j,kk=k; bool found=false;
        for (int a=0;a<2 && !found;a++) for (int b=0;b<2 && !found;b++) for (int c=0;c<2 && !found;c++) {
            if ((a && (i==0 || fabs(p[0]-f.ax[i])>1e-10)) ||
                (b && (j==0 || fabs(p[1]-f.ay[j])>1e-10)) ||
                (c && (k==0 || fabs(p[2]-f.az[k])>1e-10))) continue;
            if (f.mask[((i-a)*f.ny+j-b)*f.nz+k-c]) { ii=i-a;jj=j-b;kk=k-c;found=true; }
        }
        if (!found) return false;
        i=ii;j=jj;k=kk;
    }
    double dx=f.ax[i+1]-f.ax[i], dy=f.ay[j+1]-f.ay[j], dz=f.az[k+1]-f.az[k];
    double x=(p[0]-f.ax[i])/dx, y=(p[1]-f.ay[j])/dy, z=(p[2]-f.az[k])/dz;
    length=sqrt(dx*dx+dy*dy+dz*dz);
    v[0]=v[1]=v[2]=0.;
    // vtkVoxel point order matters near low-speed fixed points: retain the
    // same accumulation order instead of introducing different roundoff.
    for (int c=0;c<2;c++) for (int b=0;b<2;b++) for (int a=0;a<2;a++) {
        double w=(a?x:1-x)*(b?y:1-y)*(c?z:1-z);
        long long at=(((long long)(i+a)*(f.ny+1)+j+b)*(f.nz+1)+k+c)*3;
        for (int d=0;d<3;d++) v[d]+=w*f.v[at+d];
    }
    return isfinite(v[0]) && isfinite(v[1]) && isfinite(v[2]);
}

__device__ double norm(const double *v) {
    return sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
}

__device__ bool direction(Field f, const double *p, double *v) {
    double length;
    if (!sample(f,p,v,length)) return false;
    double speed=norm(v);
    if (speed>0.) for (int d=0;d<3;d++) v[d]/=speed;
    return true;
}

extern "C" __global__ void trace(
    const float *vectors, const unsigned char *mask,
    const double *ax, const double *ay, const double *az,
    int nx, int ny, int nz, const double *seeds, int nseeds,
    int offset, int tracks, int max_steps, double step_mm,
    double terminal, double max_length,
    float *points, float *out_vectors, double *times, int *counts, int *reasons) {
    int tid=blockDim.x*blockIdx.x+threadIdx.x;
    if (tid>=tracks) return;
    int track=offset+tid, seed=track%nseeds;
    double sign=track<nseeds?1.:-1.;
    Field f={vectors,mask,ax,ay,az,nx,ny,nz};
    double p[3], v[3], length;
    for (int d=0;d<3;d++) p[d]=seeds[seed*3+d];
    int count=0, reason=4;
    double elapsed=0., propagation=0.;
    long long base=(long long)tid*(max_steps+2);
    if (!sample(f,p,v,length)) { counts[tid]=0; reasons[tid]=1; return; }
    for (int d=0;d<3;d++) { points[base*3+d]=(float)p[d]; out_vectors[base*3+d]=(float)v[d]; }
    times[base]=0.; count=1;
    double speed=norm(v);
    for (int step=0;propagation<max_length;step++) {
        if (step>max_steps) { reason=5; break; }
        if (speed==0. || speed<=terminal) { reason=6; break; }
        double h=sign*fmin(step_mm>0.?step_mm:.5*length,max_length-propagation);
        double a[3],b[3],c[3],d[3],q[3],next[3],nv[3],next_length;
        for (int t=0;t<3;t++) a[t]=v[t]/speed;
        for (int t=0;t<3;t++) q[t]=p[t]+h*.5*a[t];
        if (!direction(f,q,b)) { reason=1; break; }
        for (int t=0;t<3;t++) q[t]=p[t]+h*.5*b[t];
        if (!direction(f,q,c)) { reason=1; break; }
        for (int t=0;t<3;t++) q[t]=p[t]+h*c[t];
        if (!direction(f,q,d)) { reason=1; break; }
        for (int t=0;t<3;t++) next[t]=p[t]+h*(a[t]/6.+b[t]/3.+c[t]/3.+d[t]/6.);
        if (!sample(f,next,nv,next_length)) { reason=1; break; }
        double next_speed=norm(nv);
        if ((speed+next_speed)*.5<=terminal) { reason=6; break; }
        elapsed+=h/speed; propagation+=fabs(h);
        bool different=false;
        for (int t=0;t<3;t++) if (points[(base+count-1)*3+t]!=(float)next[t]) different=true;
        if (different) {
            for (int t=0;t<3;t++) {
                points[(base+count)*3+t]=(float)next[t];
                out_vectors[(base+count)*3+t]=(float)nv[t];
            }
            times[base+count]=elapsed; count++;
        }
        for (int t=0;t<3;t++) { p[t]=next[t]; v[t]=nv[t]; }
        speed=next_speed; length=next_length;
        if (speed==0. || speed<=terminal) { reason=6; break; }
    }
    counts[tid]=count; reasons[tid]=reason;
}
'''


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("No CUDA device available")
    return cp.RawKernel(_SOURCE, "trace", options=("--std=c++11", "--fmad=false"))


def trace_sampled_volume(volume, seeds, shape, spacing, origin, *, max_steps,
                         terminal_speed, integration_step_mm=None):
    """Return VTK-compatible polylines or raise for the caller's CPU fallback."""
    import cupy as cp
    kernel = _kernel()
    shape = tuple(int(x) for x in shape)
    axes = [np.asarray(float(origin[d]) + np.arange(shape[d]+1)*float(spacing[d]),
                       dtype=volume.points.dtype).astype(np.float64) for d in range(3)]
    if any(np.any(np.diff(axis) <= 0) for axis in axes):
        raise RuntimeError("Voxel coordinates cannot be represented by a CUDA regular grid")
    # Preserve VTK's sampled field, including its cell-to-point boundary rules.
    # Reading raw velocity voxels here would silently change interpolation.
    indices = np.rint((np.asarray(volume.points, float)-origin)/spacing).astype(np.int64)
    if np.any(indices < 0) or np.any(indices > np.asarray(shape)):
        raise RuntimeError("Sampled points are outside the voxel grid")
    vectors = np.zeros(tuple(n+1 for n in shape)+(3,), np.float32)
    vectors[tuple(indices.T)] = volume.point_data["vector"]
    mask = np.zeros(shape, np.uint8)
    centers = np.asarray(volume.cell_centers().points)
    cells = np.rint((centers-origin)/spacing-.5).astype(np.int64)
    visible = np.ones(volume.n_cells, bool)
    if "vtkGhostType" in volume.cell_data:
        visible &= (np.asarray(volume.cell_data["vtkGhostType"]) & 32) == 0
    mask[tuple(cells[visible].T)] = 1
    if not np.all(np.isfinite(vectors)):
        raise RuntimeError("CUDA integration requires finite sampled vectors")
    source = np.ascontiguousarray(seeds, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3 or not np.all(np.isfinite(source)):
        raise ValueError("CUDA seeds must be a finite (N, 3) array")
    max_steps = int(max_steps)
    if max_steps < 0 or max_steps > 1_000_000:
        raise ValueError("CUDA maximum steps must be between 0 and 1000000")
    capacity = max_steps+2
    nseeds = len(source)
    # Bound temporary trajectory storage to 32 MiB and a fraction of free VRAM.
    free, _ = cp.cuda.runtime.memGetInfo()
    batch = min(2048, max(1, min(32*1024**2, int(free)//8)//(capacity*32)))
    outputs, lengths, stop_reasons = [], [], []
    with cp.cuda.Stream(non_blocking=True) as stream:
        field = [cp.asarray(x) for x in (vectors,mask,*axes,source)]
        for offset in range(0, nseeds*2, batch):
            check_cancelled()
            tracks = min(batch, nseeds*2-offset)
            points = cp.empty((tracks,capacity,3),cp.float32)
            values = cp.empty_like(points)
            times = cp.empty((tracks,capacity),cp.float64)
            counts = cp.empty(tracks,cp.int32)
            reasons = cp.empty_like(counts)
            kernel(((tracks+127)//128,),(128,),(
                *field[:5], *[np.int32(n) for n in shape], field[5], np.int32(nseeds),
                np.int32(offset),np.int32(tracks),np.int32(max_steps),
                np.float64(integration_step_mm or 0.), np.float64(terminal_speed),
                np.float64(4.*volume.length), points,values,times,counts,reasons))
            keep = (cp.arange(capacity)[None,:] < counts[:,None]).ravel()
            outputs.append((cp.asnumpy(points.reshape(-1,3)[keep]),
                            cp.asnumpy(values.reshape(-1,3)[keep]),
                            cp.asnumpy(times.ravel()[keep])))
            lengths.append(cp.asnumpy(counts))
            stop_reasons.append(cp.asnumpy(reasons))
        stream.synchronize()
    counts = np.concatenate(lengths)
    starts = np.r_[0,np.cumsum(counts,dtype=np.int64)]
    valid = np.flatnonzero(counts>1)
    mesh = pv.PolyData()
    mesh.points = np.concatenate([x[0] for x in outputs])
    if not mesh.n_points:
        return None
    offsets = np.r_[0,np.cumsum(counts[valid],dtype=np.int64)]
    # Points from one-point traces remain in the point arrays, as in VTK.
    connectivity = np.concatenate([np.arange(starts[i],starts[i+1],dtype=np.int64) for i in valid]) if len(valid) else np.empty(0,np.int64)
    mesh.lines = pv.CellArray.from_arrays(offsets,connectivity)
    vectors = np.concatenate([x[1] for x in outputs])
    mesh.point_data["vector"] = vectors
    for component,name in enumerate(("u","v","w")):
        mesh.point_data[name] = vectors[:,component]
    mesh.point_data["Velocity"] = np.linalg.norm(vectors,axis=1).astype(np.float32)
    mesh.point_data["IntegrationTime"] = np.concatenate([x[2] for x in outputs])
    if len(valid):
        mesh.cell_data["SeedIds"] = (valid%nseeds).astype(np.int32)
        mesh.cell_data["ReasonForTermination"] = np.concatenate(stop_reasons)[valid]
    mesh.set_active_scalars("Velocity")
    return mesh
