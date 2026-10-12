#!/usr/bin/env python3
"""Deterministic synthetic regression for the generic 6D CAD matcher."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "vision_moveit_demo"))
from vision_moveit_demo.cad_matching import GenericMeshCadMatcher, ObjectCatalog, RuleVlmAdapter  # noqa: E402


def rotation_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    return np.array([[cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
                     [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
                     [-sy, cy * sx, cy * cx]])


def angle_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    value = np.clip((np.trace(first.T @ second) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(value)))


def main() -> None:
    generator = np.random.default_rng(20261010)
    model = ObjectCatalog(ROOT).find(RuleVlmAdapter().infer("抓取黄色马克杯把手并放到托盘"))
    # This test measures geometric registration only.  It retains the handle
    # annotation as a pose-disambiguation cue, but disables the production
    # safety gate because a random 35% occlusion may deliberately hide it.
    model = replace(model, geometry_type="mesh")
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(model.vertices_object_m),
        o3d.utility.Vector3iVector(model.triangles_object.astype(np.int32)),
    )
    base_points = np.asarray(mesh.sample_points_uniformly(number_of_points=5500).points)
    matcher = GenericMeshCadMatcher()
    failures = []
    for trial in range(20):
        rotation = rotation_xyz(*np.deg2rad(generator.uniform([-20.0, -20.0, -180.0], [20.0, 20.0, 180.0])))
        translation = np.array([0.45, -0.12, 0.445]) + generator.uniform(-0.04, 0.04, size=3)
        points = base_points @ rotation.T + translation
        points = points[generator.random(len(points)) > 0.35]
        points += generator.normal(0.0, 0.001, size=points.shape)
        outliers = generator.uniform(translation - 0.08, translation + 0.08, size=(max(1, len(points) // 50), 3))
        points = np.concatenate((points, outliers))
        try:
            estimate = matcher._match_points(model, points, [points], 1.0, enforce_grasp_evidence=False)
            position_error = float(np.linalg.norm(estimate.position_base_m - translation))
            rotation_error = angle_error_deg(estimate.transform_base_object[:3, :3], rotation)
            if position_error > 0.003 or rotation_error > 3.0:
                failures.append((trial, position_error, rotation_error))
        except RuntimeError as error:
            failures.append((trial, str(error)))
    if failures:
        raise SystemExit(f"generic mesh synthetic regression failed: {failures}")
    print("generic mesh synthetic regression: 20/20 passed")


if __name__ == "__main__":
    main()
