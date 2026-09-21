"""阶段 3 闭环：语义选模、CAD 匹配驱动的抓取放置。

控制目标只来自规则语义、RGB 分割、深度点云和预载 CAD 模型。MuJoCo 真值仅在
回合末尾做评测，不参与 CAD 配准、抓取目标生成或 MoveIt 场景坐标。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import rclpy

from .cad_matching import (
    ColorThresholdSegmenter,
    CadMatcherDispatcher,
    ObjectCatalog,
    RuleVlmAdapter,
    grasp_pose_candidates,
)
from .stage2_executor import MujocoTaskExecutor
from .truth_baseline import MoveItTrajectoryClient, _execute_trajectory
from .unified_scene import UnifiedPandaCupSimulation


def _write_pgm(path: Path, mask: np.ndarray) -> None:
    image = np.where(mask, 255, 0).astype(np.uint8)
    with path.open("wb") as output:
        output.write(f"P5\n{image.shape[1]} {image.shape[0]}\n255\n".encode())
        output.write(image.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段 3：语义选模 CAD 匹配抓取基线")
    parser.add_argument("--instruction", default="抓取红色杯子并放到托盘")
    parser.add_argument("--root", type=Path, default=Path(os.environ["MOVEIT_EXPERIMENT_ROOT"]))
    parser.add_argument("--vnc", action="store_true", help="在 VNC 中显示 CAD 匹配驱动的执行过程。")
    arguments = parser.parse_args()

    simulation = UnifiedPandaCupSimulation(arguments.root, create_renderers=False)
    viewer = None
    if arguments.vnc:
        from .unified_preview import VncOverlayViewer

        viewer = VncOverlayViewer(simulation, title="MoveIt 阶段 3 CAD 模型匹配")
        simulation.initialize_renderers()
        viewer.glfw.make_context_current(viewer.window)
        def render_frame(state: str) -> bool:
            return viewer.render_once(state)

    else:
        simulation.initialize_renderers()
        render_frame = None

    try:
        intent = RuleVlmAdapter().infer(arguments.instruction)
        catalog = ObjectCatalog(arguments.root)
        model = catalog.find(intent)
        target_object_id = model.scene_object_id
        fixed_frame = simulation.cameras()["fixed"]
        segmentation = ColorThresholdSegmenter().segment(fixed_frame.rgb, intent)
        pose = CadMatcherDispatcher().match(model, segmentation, fixed_frame)
        candidates = grasp_pose_candidates(pose, model.grasp_templates)
        # 正常档以仿真时间一倍速运行；VNC 只显示该节拍，不通过渲染 sleep 改变它。
        executor = MujocoTaskExecutor(simulation, frame_callback=render_frame, realtime_factor=1.0)
        executor.set_display_state(f"CAD match complete: {target_object_id} / {model.model_id}")
        rclpy.init()
        client = MoveItTrajectoryClient()
        try:
            # 当前工装位置来自对象目录；目标杯位置仅来自本次 CAD 配准。
            time.sleep(3.0)
            client.publish_scene_positions(catalog.fixture_collision_positions(), grasp_target=target_object_id)
            time.sleep(1.0)
            motion_wall_start = time.monotonic()
            executor.set_gripper(opened=True)
            selected = None
            rejected_candidates: list[dict[str, str]] = []
            for candidate_template, candidate_position, candidate_orientation in candidates:
                pregrasp = candidate_position + np.array([0.0, 0.0, candidate_template.approach_clearance_m])
                executor.set_display_state(f"CAD candidate / MoveIt: {candidate_template.template_id}")
                client.publish_joint_state(executor.joint_positions())
                try:
                    pregrasp_trajectory = client.request(pregrasp, candidate_orientation)
                except TimeoutError as error:
                    rejected_candidates.append({"frame_id": candidate_template.template_id, "reason": str(error)})
                    continue
                selected = (candidate_template, candidate_position, candidate_orientation, pregrasp_trajectory)
                break
            if selected is None:
                raise RuntimeError(f"MoveIt 未找到可执行的 CAD 抓取标注候选：{rejected_candidates}")
            template, grasp_position, grasp_orientation, pregrasp_trajectory = selected
            placement = catalog.physical_placement_targets(model, template)
            tray_center = placement.tray_center_base_m
            targets = [
                ("approach", grasp_position),
                ("lift", grasp_position + np.array([0.0, 0.0, template.lift_clearance_m])),
                ("place_above", placement.gripper_above_base_m),
                ("place_descend", placement.gripper_release_base_m),
            ]
            executor.set_display_state("CAD grasp / MoveIt: pregrasp")
            _execute_trajectory(executor, pregrasp_trajectory)
            executor._event("moveit_pregrasp_complete")
            executor.set_display_state("CAD grasp / MoveIt: approach")
            client.publish_joint_state(executor.joint_positions())
            _execute_trajectory(executor, client.request(grasp_position, grasp_orientation))
            executor._event(
                "moveit_approach_complete",
                hand_x=float(executor.body_position("hand")[0]),
                hand_y=float(executor.body_position("hand")[1]),
                hand_z=float(executor.body_position("hand")[2]),
                object_x=float(executor.body_position(target_object_id)[0]),
                object_y=float(executor.body_position(target_object_id)[1]),
                object_z=float(executor.body_position(target_object_id)[2]),
            )
            executor.set_display_state("Gripper: close and stable attach CAD target")
            executor.set_gripper(opened=False)
            closing_axis_base = pose.transform_base_object[:3, :3] @ template.jaw_closing_axis_object
            pair = template.opposing_contact_pair
            executor.attach(
                target_object_id,
                closing_axis_base,
                (
                    pair.positive_surface.center_object_m,
                    pair.positive_surface.normal_object,
                    pair.negative_surface.center_object_m,
                    pair.negative_surface.normal_object,
                ),
            )
            for stage, target in targets[1:]:
                executor.set_display_state(f"CAD grasp / MoveIt: {stage} with {target_object_id}")
                client.publish_joint_state(executor.joint_positions())
                _execute_trajectory(executor, client.request(target, grasp_orientation))
                executor._event(
                    f"moveit_{stage}_complete",
                    hand_x=float(executor.body_position("hand")[0]),
                    hand_y=float(executor.body_position("hand")[1]),
                    hand_z=float(executor.body_position("hand")[2]),
                    object_x=float(executor.body_position(target_object_id)[0]),
                    object_y=float(executor.body_position(target_object_id)[1]),
                    object_z=float(executor.body_position(target_object_id)[2]),
                )
            executor.set_display_state("Gripper: physical release above tray")
            executor.set_gripper(opened=True)
            physical_release = executor.release_physical()
            # 仅评测：不会将这个真值结果反馈给任何下一步控制决策。
            evaluation_only_success = executor.object_in_tray(target_object_id, tray_center)
            motion_wall_duration_s = time.monotonic() - motion_wall_start
            output = arguments.root / "logs" / "episodes" / "cad_matching_latest"
            output.mkdir(parents=True, exist_ok=True)
            _write_pgm(output / "target_mask.pgm", segmentation.mask)
            payload = {
                "instruction": arguments.instruction,
                "stage": "cad_model_matching_pick_place_baseline",
                "semantic_result": intent.as_dict(),
                "target_object_id": target_object_id,
                "segmentation": segmentation.as_dict(),
                "cad_model": {
                    "model_id": model.model_id,
                    "geometry_type": model.geometry_type,
                    "scene_object_id": model.scene_object_id,
                    "mesh": str(model.mesh_path.relative_to(arguments.root)),
                    "symmetry": model.symmetry_type,
                    "selected_grasp_template": template.template_id,
                },
                "pose_estimate": pose.as_dict(),
                "grasp_position_base_m": grasp_position.tolist(),
                "grasp_orientation_base_xyzw": grasp_orientation.tolist(),
                "grasp_annotation": {
                    "frame_id": template.template_id,
                    "contact_regions": list(template.contact_regions),
                    "jaw_closing_axis_object": template.jaw_closing_axis_object.tolist(),
                    "opposing_contact_pair": {
                        "positive_surface": {
                            "region_id": template.opposing_contact_pair.positive_surface.region_id,
                            "normal_object": template.opposing_contact_pair.positive_surface.normal_object.tolist(),
                            "center_object_m": template.opposing_contact_pair.positive_surface.center_object_m.tolist(),
                        },
                        "negative_surface": {
                            "region_id": template.opposing_contact_pair.negative_surface.region_id,
                            "normal_object": template.opposing_contact_pair.negative_surface.normal_object.tolist(),
                            "center_object_m": template.opposing_contact_pair.negative_surface.center_object_m.tolist(),
                        },
                    },
                    "quality": template.quality,
                },
                "grasp_candidate_selection": {
                    "candidate_count": len(candidates),
                    "selected_frame_id": template.template_id,
                    "rejected_candidates": rejected_candidates,
                },
                "fixture_tray_center_base_m": tray_center.tolist(),
                "physical_placement": placement.as_dict(),
                "physical_release": physical_release.as_dict(),
                "execution": {
                    "profile": "normal",
                    "realtime_factor": executor.realtime_factor,
                    "trajectory_summaries": [summary.as_dict() for summary in executor.trajectory_summaries],
                    "rendered_frame_count": executor.rendered_frame_count,
                    "pure_motion_duration_s": float(executor.events[-1].timestamp - executor.events[0].timestamp),
                    "wall_motion_duration_s": motion_wall_duration_s,
                },
                "events": [
                    {"name": event.name, "timestamp_s": event.timestamp, "details": event.details}
                    for event in executor.events
                ],
                "evaluation_only_success": evaluation_only_success,
            }
            (output / "episode.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            if not evaluation_only_success:
                raise RuntimeError(f"评测发现 {target_object_id} 未落入托盘")
            if viewer is not None:
                print("CAD 模型匹配回合成功。VNC 主窗口按 Esc 退出。", flush=True)
                while viewer.render_once(f"Success: CAD matched {target_object_id} in tray"):
                    time.sleep(1.0 / 30.0)
        finally:
            client.destroy_node()
            rclpy.shutdown()
    finally:
        if viewer is not None:
            viewer.close()
        simulation.close()


if __name__ == "__main__":
    main()
