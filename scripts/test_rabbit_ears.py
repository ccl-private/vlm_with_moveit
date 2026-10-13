#!/usr/bin/env python3
"""兔子耳部 CAD/抓取 ROI 回归；含缺失耳部观测的拒绝测试。"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "vision_moveit_demo"))
from vision_moveit_demo.cad_matching import (  # noqa: E402
    CadPoseEstimate, GenericMeshCadMatcher, GraspRoiMatcher, ObjectCatalog, RuleVlmAdapter,
)


def main() -> None:
    model = ObjectCatalog(ROOT).find(RuleVlmAdapter().infer("抓取橙色兔子耳朵并放到托盘"))
    assert model.geometry_type == "mesh"
    template = model.grasp_templates[0]
    assert template.contact_collision_geometries == ("orange_rabbit_long_ear_collision",)
    matcher = GraspRoiMatcher()
    pair = template.opposing_contact_pair
    patches = [matcher._roi_vertices(model, surface) for surface in (pair.positive_surface, pair.negative_surface)]
    generator = np.random.default_rng(20261013)
    transform = np.eye(4)
    transform[:3, 3] = [0.28, -0.40, 0.505]
    seed = transform.copy()
    seed[:3, 3] += [0.002, -0.001, 0.001]
    pose = CadPoseEstimate(
        model_id=model.model_id, position_base_m=seed[:3, 3],
        orientation_xyzw=np.array([0., 0., 0., 1.]), transform_base_object=seed,
        surface_residual_m=0.002, inlier_ratio=1., segmentation_confidence=1.,
        geometry_confidence=0.9, axial_yaw_observable=True, cylinder_axis_base=seed[:3, 2],
        point_count=1000, matching_method="generic_mesh_test", object_yaw_rad=0.,
    )
    observed = np.concatenate(patches) + transform[:3, 3]
    clouds = [observed + generator.normal(0., 0.0005, observed.shape) for _ in range(2)]
    result = matcher.refine_clouds(model, template, pose, clouds)
    assert np.linalg.norm(result.pose.position_base_m - transform[:3, 3]) < 0.003
    assert GenericMeshCadMatcher._quaternion_xyzw(result.pose.transform_base_object[:3, :3]).shape == (4,)
    body = model.vertices_object_m[model.vertices_object_m[:, 2] < 0.03] + transform[:3, 3]
    try:
        matcher.refine_clouds(model, template, pose, [body, body])
    except RuntimeError as error:
        assert "抓取 ROI" in str(error)
    else:
        raise AssertionError("缺少耳部观测时错误放行抓取")
    print("rabbit ears ROI: precise positive case passed; missing-ear case rejected")


if __name__ == "__main__":
    main()
