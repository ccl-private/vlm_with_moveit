#!/usr/bin/env python3
"""Deterministic regression for the separate CAD grasp-ROI matcher."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "vision_moveit_demo"))
from vision_moveit_demo.cad_matching import (  # noqa: E402
    CadPoseEstimate,
    GenericMeshCadMatcher,
    GraspRoiMatcher,
    ObjectCatalog,
    RuleVlmAdapter,
)


def angle_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    value = np.clip((np.trace(first.T @ second) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(value)))


def pose(transform: np.ndarray) -> CadPoseEstimate:
    return CadPoseEstimate(
        model_id="mug_with_handle_v1",
        position_base_m=transform[:3, 3],
        orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        transform_base_object=transform,
        surface_residual_m=0.004,
        inlier_ratio=0.8,
        segmentation_confidence=1.0,
        geometry_confidence=0.9,
        axial_yaw_observable=True,
        cylinder_axis_base=np.array([0.0, 0.0, 1.0]),
        point_count=1000,
        matching_method="generic_mesh_fpfh_ransac_multiscale_icp_visible_surface_refinement",
        object_yaw_rad=0.0,
    )


def main() -> None:
    generator = np.random.default_rng(20261012)
    model = ObjectCatalog(ROOT).find(RuleVlmAdapter().infer("抓取黄色马克杯把手并放到托盘"))
    template = model.grasp_templates[0]
    matcher = GraspRoiMatcher()
    positive = matcher._roi_vertices(model, template.opposing_contact_pair.positive_surface)
    negative = matcher._roi_vertices(model, template.opposing_contact_pair.negative_surface)
    transform = np.eye(4)
    transform[:3, :3] = GenericMeshCadMatcher._rotation_z(np.deg2rad(45.0))
    transform[:3, 3] = (0.45, -0.12, 0.445)
    expected = np.concatenate((positive, negative)) @ transform[:3, :3].T + transform[:3, 3]
    # Two independent wrist baselines see both required contact surfaces; add
    # depth-scale noise and unrelated yellow points to exercise robust matching.
    clouds = []
    for _ in range(2):
        visible = expected[generator.random(len(expected)) > 0.12]
        visible = visible + generator.normal(0.0, 0.0008, size=visible.shape)
        outliers = generator.uniform((0.39, -0.18, 0.39), (0.51, -0.06, 0.50), size=(32, 3))
        clouds.append(np.concatenate((visible, outliers)))
    global_seed = np.eye(4)
    global_seed[:3, 3] = transform[:3, 3] + np.array((0.003, -0.002, 0.001))
    estimate = matcher.refine_clouds(model, template, pose(global_seed), clouds)
    position_error = float(np.linalg.norm(estimate.pose.position_base_m - transform[:3, 3]))
    rotation_error = angle_error_deg(estimate.pose.transform_base_object[:3, :3], transform[:3, :3])
    if position_error > 0.003 or rotation_error > 3.0:
        raise SystemExit(f"ROI pose error too large: {position_error:.4f}m, {rotation_error:.2f}deg")
    if estimate.positive_support < 0.16 or estimate.negative_support < 0.16:
        raise SystemExit(f"ROI support too weak: {estimate.positive_support:.3f}, {estimate.negative_support:.3f}")
    if 0.0 not in estimate.reachability_validated_yaw_offsets_rad:
        raise SystemExit("ROI result did not retain its own pose as a reachability-validated candidate")

    # Keeping only the cup body / background must never produce a grasp pose.
    body_only = [cloud[cloud[:, 1] < transform[1, 3] + 0.030] for cloud in clouds]
    try:
        matcher.refine_clouds(model, template, pose(global_seed), body_only)
    except RuntimeError as error:
        if "抓取 ROI" not in str(error):
            raise SystemExit(f"wrong missing-ROI failure: {error}")
    else:
        raise SystemExit("ROI matcher accepted observations without handle contact evidence")
    print("grasp ROI matcher regression: matched precise contacts and rejected missing handle")


if __name__ == "__main__":
    main()
