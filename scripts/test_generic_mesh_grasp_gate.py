#!/usr/bin/env python3
"""Offline regression: mesh matching must reject a mug without handle evidence."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "vision_moveit_demo"))
from vision_moveit_demo.cad_matching import GenericMeshCadMatcher, ObjectCatalog, RuleVlmAdapter  # noqa: E402


def main() -> None:
    model = ObjectCatalog(ROOT).find(RuleVlmAdapter().infer("抓取黄色马克杯把手并放到托盘"))
    model = replace(model, geometry_type="mesh")
    matcher = GenericMeshCadMatcher()
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(model.vertices_object_m),
        o3d.utility.Vector3iVector(model.triangles_object.astype(np.int32)),
    )
    points = np.asarray(mesh.sample_points_uniformly(number_of_points=7000).points)
    transform = np.eye(4)
    transform[:3, 3] = (0.45, -0.12, 0.445)
    points = points @ transform[:3, :3].T + transform[:3, 3]

    # Simulate two saved wrist RGB-D observations that both miss the handle.
    critical = matcher._critical_vertices(model)
    assert critical is not None
    critical = critical @ transform[:3, :3].T + transform[:3, 3]
    critical_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(critical))
    point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    distances = np.asarray(point_cloud.compute_point_cloud_distance(critical_cloud))
    body_only = points[distances > 0.045]
    observations = [body_only[::2], body_only[1::2]]
    try:
        matcher._match_points(model, np.concatenate(observations), observations, 1.0)
    except RuntimeError as error:
        if "抓取关键区证据不足" not in str(error):
            raise SystemExit(f"expected grasp-gate rejection, got: {error}")
        print("generic mesh grasp gate: correctly rejected absent handle evidence")
        return
    raise SystemExit("generic mesh grasp gate accepted a pose without handle evidence")


if __name__ == "__main__":
    main()
