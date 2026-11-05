"""Local contact geometry, independent of object identity and simulation truth."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class SurfaceObservation:
    points: np.ndarray
    normals: np.ndarray
    valid: np.ndarray
    camera: np.ndarray | None = None


def estimate_surface(points, camera=None, *, radius=.008, neighbors=12):
    """Fit each view separately. A line, isolated point or mixed corner is unknown."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('finite Nx3 observation required')
    if len(points) < 8:
        return SurfaceObservation(points, np.zeros_like(points), np.zeros(len(points), bool), camera)
    distances, indices = cKDTree(points).query(points, k=min(neighbors, len(points)), distance_upper_bound=radius)
    mask = np.isfinite(distances)
    local = points[np.minimum(indices, len(points)-1)]
    count = mask.sum(axis=1)
    mean = (local*mask[..., None]).sum(axis=1)/np.maximum(count, 1)[:, None]
    centered = (local-mean[:, None, :])*mask[..., None]
    covariance = np.einsum('nki,nkj->nij', centered, centered)/np.maximum(count, 1)[:, None, None]
    values, vectors = np.linalg.eigh(covariance)
    normals = vectors[:, :, 0]
    valid = ((count >= 8) & (values[:, 1] >= .00005**2)
             & (values[:, 0] <= .08*np.maximum(values.sum(axis=1), 1e-20))
             & (values[:, 0] <= .3*np.maximum(values[:, 1], 1e-20))
             & (values[:, 1] >= .08*values[:, 2]))
    if camera is not None:
        camera = np.asarray(camera, dtype=float)
        if camera.shape != (3,) or not np.isfinite(camera).all():
            raise ValueError('finite camera origin required')
        flip = np.einsum('ni,ni->n', normals, camera-points) < 0
        normals[flip] *= -1
    normals[~valid] = 0
    return SurfaceObservation(points, normals, valid, camera)


def nearest_surface(query, normals, observation, *, max_distance, cosine):
    """Return nearest compatible points; unknown normals never supply evidence.

    View-oriented normals are signed. Cloud-only fixtures use unoriented normals;
    their arbitrary eigenvector signs must not change a geometric test result.
    """
    query = np.asarray(query)
    squared = ((query[:, None, :]-observation.points[None, :, :])**2).sum(axis=2)
    alignment = normals @ observation.normals.T
    compatible = (alignment >= cosine if observation.camera is not None else abs(alignment) >= cosine)
    compatible &= observation.valid[None, :]
    squared[~compatible] = np.inf
    indices = np.argmin(squared, axis=1)
    distances = np.sqrt(squared[np.arange(len(query)), indices])
    keep = distances <= max_distance
    return distances, observation.points[indices], keep


def sample_contact_patch(model, surface, *, cosine, edge_margin=.0015):
    """Sample actual triangle interiors, excluding sharp/open mesh boundaries.

    Coincident OBJ vertices are welded for edge topology only. Internal coplanar
    triangulation edges are retained. No vertices on an ambiguous edge are used.
    """
    triangles = model.vertices_object_m[model.triangles_object]
    cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    normals = cross/np.maximum(lengths[:, None], 1e-20)
    face_ok = (normals @ surface.normal_object >= cosine) & (lengths > 1e-14)
    slab = float(np.clip(model.height_m*.055, .0035, .0065))
    plane_vertices = abs((triangles-surface.center_object_m) @ surface.normal_object)
    face_ok &= plane_vertices.min(axis=1) <= slab
    if not face_ok.any():
        raise RuntimeError('抓取 ROI CAD 没有朝向一致的面：'+surface.region_id)
    generator = np.random.default_rng(20261013)
    areas = lengths*face_ok
    indices = generator.choice(len(triangles), 6000, p=areas/areas.sum())
    u, v = generator.random((2, len(indices)))
    root = np.sqrt(u)
    samples = (triangles[indices]*np.column_stack((1-root, root*(1-v), root*v))[:, :, None]).sum(axis=1)
    delta = samples-surface.center_object_m
    normal_distance = abs(delta @ surface.normal_object)
    tangent = np.linalg.norm(delta-np.outer(delta @ surface.normal_object, surface.normal_object), axis=1)
    radius = surface.patch_radius_m or float(np.clip(model.height_m*.38, .024, .045))
    mask = face_ok[indices] & (normal_distance <= slab) & (tangent <= radius)
    samples, selected_normals = samples[mask], normals[indices[mask]]
    edges = {}
    for face, triangle in enumerate(triangles):
        for i, j in ((0,1), (1,2), (2,0)):
            key = tuple(sorted((tuple(np.round(triangle[i],9)), tuple(np.round(triangle[j],9)))))
            edges.setdefault(key, []).append(face)
    boundaries = []
    for key, faces in edges.items():
        if not face_ok[faces].any():
            continue
        sharp = len(faces) != 2 or float(normals[faces[0]] @ normals[faces[1]]) < cosine
        if sharp:
            boundaries.append(key)
    if len(samples) and boundaries:
        segments = np.asarray(boundaries)
        a, vector = segments[:, 0], segments[:, 1]-segments[:, 0]
        relative = samples[:, None, :]-a[None, :, :]
        t = np.clip(np.einsum('nei,ei->ne',relative,vector)/np.maximum((vector*vector).sum(axis=1),1e-20),0,1)
        distance = np.linalg.norm(relative-t[..., None]*vector, axis=2)
        interior = distance.min(axis=1) >= edge_margin
        samples, selected_normals = samples[interior], selected_normals[interior]
    if len(samples) < 16:
        raise RuntimeError(f'抓取 ROI CAD 面内部采样不足：{surface.region_id} 仅有 {len(samples)} 个点')
    step = max(1, len(samples)//180)
    return samples[::step], selected_normals[::step]
