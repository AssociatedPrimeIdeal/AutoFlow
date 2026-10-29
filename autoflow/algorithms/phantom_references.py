"""Independent, resolution-matched discrete references for synthetic inputs.

This module does not use VTK, AutoFlow metric estimators or production label
resampling. Native-grid plane truth is the exact cut area of occupied voxel
boxes, weighted by their supplied velocity samples. It is a declared discrete
measurement model, distinct from the underlying continuous analytic tube.
"""

from itertools import product
import numpy as np


def trilinear_reference(field, points, spacing, origin, *, return_valid=False):
    """Evaluate a supplied sample-centred array with independent eight-corner weights."""
    field = np.asarray(field)
    xyz = (np.asarray(points, dtype=float)-np.asarray(origin))/np.asarray(spacing)
    shape = np.asarray(field.shape[:3])
    valid = np.all((xyz >= 0.) & (xyz <= shape-1), axis=1)
    xyz = np.clip(xyz, 0., shape-1)
    low = np.floor(xyz).astype(int)
    fraction = xyz-low
    result = np.zeros((len(xyz),)+field.shape[3:], dtype=float)
    tail = (1,)*(field.ndim-3)
    for corner in product((0, 1), repeat=3):
        offset = np.asarray(corner)
        index = np.minimum(low+offset, shape-1)
        weight = np.prod(np.where(offset, fraction, 1.-fraction), axis=1)
        result += weight.reshape((-1,)+tail)*field[tuple(index.T)]
    return (result, valid) if return_valid else result


def canonical_binary_reference(mask, spacing, target=1.):
    """Reference for binary linear-indicator argmax, with background winning ties."""
    mask = np.asarray(mask, dtype=bool)
    spacing = np.asarray(spacing, dtype=float)
    steps = (np.asarray(mask.shape)-1)*spacing/target
    tolerance = 8*np.finfo(np.float32).eps*np.maximum(1.,abs(steps))
    steps = np.where(abs(steps-np.rint(steps)) <= tolerance, np.rint(steps), steps)
    shape = tuple((np.ceil(steps).astype(int)+1).tolist())
    points = np.moveaxis(np.indices(shape,dtype=float),0,-1).reshape(-1,3)*target
    foreground,valid = trilinear_reference(mask.astype(np.float32),points,spacing,(0,0,0),return_valid=True)
    background = trilinear_reference((~mask).astype(np.float32),points,spacing,(0,0,0))
    # Production scores are rounded to float32 before classification.
    return (valid & (foreground.astype(np.float32)>background.astype(np.float32))).reshape(shape)


def _polygon_distance(points, query):
    """Distance in the plane; zero when the convex polygon contains the query."""
    a,b = points,np.roll(points,-1,axis=0)
    edge = b-a
    cross = edge[:,0]*(query[1]-a[:,1])-edge[:,1]*(query[0]-a[:,0])
    if np.all(cross >= -1e-9) or np.all(cross <= 1e-9):
        return 0.
    alpha = np.clip(np.sum((query-a)*edge,axis=1)/np.maximum(np.sum(edge*edge,axis=1),1e-30),0.,1.)
    return float(np.min(np.linalg.norm(a+alpha[:,None]*edge-query,axis=1)))


def voxel_plane_cut(mask, spacing, origin, center_world, normal, *, select_connected=True):
    """Intersect native occupied boxes with a plane using explicit convex polygons."""
    mask = np.asarray(mask,dtype=bool)
    spacing,origin,center = map(lambda x:np.asarray(x,dtype=float), (spacing,origin,center_world))
    normal = np.array(normal,dtype=float,copy=True)
    normal /= np.linalg.norm(normal)
    indices = np.argwhere(mask)
    centers = origin+indices*spacing
    distance = (centers-center)@normal
    reach = float(np.sum(abs(normal)*spacing)/2)
    keep = abs(distance) <= reach+1e-9
    indices,centers = indices[keep],centers[keep]
    corners = np.array(list(product((-.5,.5),repeat=3)))*spacing
    edges = [(i,j) for i in range(8) for j in range(i+1,8)
             if np.count_nonzero(corners[i]!=corners[j]) == 1]
    axis_u = np.cross(normal,np.eye(3)[np.argmin(abs(normal))]);axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(normal,axis_u)
    polygons,areas,cut_indices = [],[],[]
    for index,cell_center in zip(indices,centers):
        vertices = cell_center+corners
        signed = (vertices-center)@normal
        # Half-open boxes count a plane-coincident shared face once.
        if np.max(signed) <= 1e-10 and np.min(signed) < -1e-10:
            continue
        points = [vertices[k] for k in range(8) if abs(signed[k]) <= 1e-10]
        for left,right in edges:
            if signed[left]*signed[right] < -1e-20:
                fraction = signed[left]/(signed[left]-signed[right])
                points.append(vertices[left]+fraction*(vertices[right]-vertices[left]))
        if len(points)<3:
            continue
        points = np.unique(np.round(points,11),axis=0)
        local = points-center
        uv = np.column_stack((local@axis_u,local@axis_v))
        middle = uv.mean(axis=0)
        order = np.argsort(np.arctan2(uv[:,1]-middle[1],uv[:,0]-middle[0]))
        points,uv = points[order],uv[order]
        area = abs(np.sum(uv[:,0]*np.roll(uv[:,1],-1)-uv[:,1]*np.roll(uv[:,0],-1)))/2
        if area <= 1e-12:
            continue
        polygons.append((points,uv));areas.append(area);cut_indices.append(index)
    if not areas:
        return dict(indices=np.empty((0,3),int),areas=np.empty(0),polygons=[])
    if select_connected:
        parent = list(range(len(areas)))
        def find(i):
            while parent[i]!=i:
                parent[i] = parent[parent[i]];i = parent[i]
            return i
        owners = {}
        for i,(points,_uv) in enumerate(polygons):
            for point in points:
                key = tuple(np.round(point,8))
                if key in owners:
                    parent[find(i)] = find(owners[key])
                else:
                    owners[key] = i
        nearest = min(range(len(areas)),key=lambda i:(_polygon_distance(polygons[i][1],np.zeros(2)),i))
        component = find(nearest)
        chosen = [i for i in range(len(areas)) if find(i)==component]
    else:
        chosen = range(len(areas))
    return dict(indices=np.array([cut_indices[i] for i in chosen],int),
                areas=np.array([areas[i] for i in chosen]),polygons=[polygons[i][0] for i in chosen])


def native_plane_reference(mask, flow, spacing, origin, center_world, normal):
    """Exact cut-cell sums for the input resolution's cell-sample convention."""
    cut = voxel_plane_cut(mask,spacing,origin,center_world,normal)
    direction = np.asarray(normal,dtype=float)/np.linalg.norm(normal)
    indices,area = cut['indices'],cut['areas']
    velocity = np.asarray(flow)[tuple(indices.T)].astype(float)
    projected = velocity@direction
    flux = np.sum(projected*area[:,None],axis=0)/100.  # cm/s * mm² -> mL/s
    return dict(area_mm2=float(area.sum()),flowrate_ml_s=flux,cell_count=len(area))


def pressure_gradient_reference(flow, spacing, rr_ms, rho=1060., viscosity_mpa_s=4.):
    """Centered spatial/periodic temporal Navier-Stokes operator on supplied samples.

    The returned boundary values are not a validity claim. Compare only on
    complete spatial and temporal stencils declared by the metric stage.
    """
    velocity = np.asarray(flow,dtype=float)/100.
    dt = float(rr_ms)/1000/velocity.shape[3]
    temporal = (np.roll(velocity,-1,axis=3)-np.roll(velocity,1,axis=3))/(2*dt)
    advective,laplacian = np.zeros_like(velocity),np.zeros_like(velocity)
    for axis,step_mm in enumerate(spacing):
        h = float(step_mm)/1000.
        plus,minus = np.roll(velocity,-1,axis=axis),np.roll(velocity,1,axis=axis)
        advective += velocity[...,axis,None]*(plus-minus)/(2*h)
        laplacian += (plus-2*velocity+minus)/(h*h)
    return -float(rho)*(temporal+advective)+(float(viscosity_mpa_s)/1000.)*laplacian


def wss_operator_reference(flow, spacing, origin, wall_points, inward_normals, distance_mm,
                           viscosity_mpa_s=4.):
    """Independent two-probe tangential operator, conditional on the supplied wall.

    This checks interpolation, projection, units and derivative arithmetic;
    it does not certify that a reconstructed wall matches a physical vessel.
    """
    points,normals = np.asarray(wall_points,dtype=float),np.asarray(inward_normals,dtype=float)
    normals = normals/np.linalg.norm(normals,axis=1,keepdims=True)
    first,valid1 = trilinear_reference(flow,points+distance_mm*normals,spacing,origin,return_valid=True)
    second,valid2 = trilinear_reference(flow,points+2*distance_mm*normals,spacing,origin,return_valid=True)
    first,second = first/100.,second/100.
    tangent1 = first-np.sum(first*normals[:,None,:],axis=-1)[...,None]*normals[:,None,:]
    tangent2 = second-np.sum(second*normals[:,None,:],axis=-1)[...,None]*normals[:,None,:]
    result = viscosity_mpa_s*(4*tangent1-tangent2)/(2*distance_mm)
    result[~(valid1&valid2)] = np.nan
    return result


def add_discrete_references(data):
    """Attach each input's own reference, never a separately generated finer case."""
    params,truth = data['parameters'],data['truth']
    reference = native_plane_reference(data['segmentation'][...,0]>0,data['flow'],data['resolution'],
        data['origin'],params['center_world_mm'],params['direction_unit'])
    truth.update(discrete_native_plane_area_mm2=np.array(reference['area_mm2']),
                 discrete_native_plane_flowrate_ml_s=reference['flowrate_ml_s'],
                 discrete_native_plane_cell_count=np.array(reference['cell_count']),
                 discrete_canonical_mask=canonical_binary_reference(data['segmentation'][...,0]>0,data['resolution']))
    if np.allclose(data['resolution'],1.):
        truth['discrete_pressure_gradient_pa_m'] = pressure_gradient_reference(
            data['flow'],data['resolution'],data['rr'],params['rho_kg_m3'],params['mu_pa_s']*1000.)
    params.update(primary_reference='resolution_matched_discrete_input',
        native_plane_reference='occupied_native_voxel_box_cut_areas_times_native_velocity_samples',
        continuous_analytic_reference='secondary_physical_sampling_and_model_deviation_only')
    return data
