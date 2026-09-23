"""阶段 3 闭环：语义选模、CAD 匹配驱动的抓取放置。

控制目标只来自规则语义、RGB 分割、深度点云和预载 CAD 模型。MuJoCo 真值仅在
回合末尾做评测，不参与 CAD 配准、抓取目标生成或 MoveIt 场景坐标。
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Pose

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
from .planning_scene_policy import PlanningPhase


def _write_pgm(path: Path, mask: np.ndarray) -> None:
    image = np.where(mask, 255, 0).astype(np.uint8)
    with path.open("wb") as output:
        output.write(f"P5\n{image.shape[1]} {image.shape[0]}\n255\n".encode())
        output.write(image.tobytes())


def _quaternion_to_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion_xyzw
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def _matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
            w = (rotation[2, 1] - rotation[1, 2]) / scale
        elif index == 1:
            scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
            w = (rotation[0, 2] - rotation[2, 0]) / scale
        else:
            scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale
            w = (rotation[1, 0] - rotation[0, 1]) / scale
    return np.array([x, y, z, w]) / np.linalg.norm([x, y, z, w])


def _rotate_vector_by_quaternion(vector: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    rotation = _quaternion_to_matrix(quaternion_xyzw)
    return rotation @ vector


def _attached_pose_in_hand(
    object_position_base: np.ndarray,
    object_rotation_base: np.ndarray,
    hand_position_base: np.ndarray,
    hand_orientation_xyzw: np.ndarray,
) -> Pose:
    hand_rotation = _quaternion_to_matrix(hand_orientation_xyzw)
    relative_rotation = hand_rotation.T @ object_rotation_base
    relative_position = hand_rotation.T @ (object_position_base - hand_position_base)
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = map(float, relative_position)
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(
        float, _matrix_to_quaternion(relative_rotation)
    )
    return pose


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段 3：语义选模 CAD 匹配抓取基线")
    parser.add_argument("--instruction", default="抓取红色杯子并放到托盘")
    parser.add_argument("--root", type=Path, default=Path(os.environ["MOVEIT_EXPERIMENT_ROOT"]))
    parser.add_argument("--vnc", action="store_true", help="在 VNC 中显示 CAD 匹配驱动的执行过程。")
    parser.add_argument("--video-dir", type=Path, help="保存 VNC 主视角和腕部相机 MP4 的目录。")
    parser.add_argument(
        "--force-matcher-type",
        choices=("mesh",),
        help="仅用于回归诊断：临时以指定匹配器覆盖目录 geometry_type，不改对象目录。",
    )
    parser.add_argument(
        "--attached-speed-scale",
        type=float,
        default=6.0,
        help="夹住物体后的轨迹时长倍率，默认 6.0；数值越小越快。",
    )
    arguments = parser.parse_args()

    simulation = UnifiedPandaCupSimulation(arguments.root, create_renderers=False)
    viewer = None
    if arguments.vnc:
        from .unified_preview import VncOverlayViewer

        viewer = VncOverlayViewer(
            simulation,
            title="MoveIt 阶段 3 CAD 模型匹配",
            video_dir=arguments.video_dir,
        )
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
        catalog_geometry_type = model.geometry_type
        if arguments.force_matcher_type is not None:
            model = replace(model, geometry_type=arguments.force_matcher_type)
            print(
                f"[匹配器诊断] 目录类型={catalog_geometry_type}，本回合临时覆盖为 {model.geometry_type}；"
                "抓取标注和场景实例保持不变。",
                flush=True,
            )
        target_object_id = model.scene_object_id
        fixed_frame = simulation.cameras()["fixed"]
        segmentation = ColorThresholdSegmenter().segment(fixed_frame.rgb, intent)
        pose = CadMatcherDispatcher().match(model, segmentation, fixed_frame)
        templates = tuple(
            template for template in model.grasp_templates
            if intent.grasp_region is None or template.grasp_region == intent.grasp_region
        )
        if not templates:
            raise RuntimeError(f"CAD 模型没有与语义抓取区域匹配的抓取框：{intent.grasp_region}")
        candidates = grasp_pose_candidates(pose, templates)
        # 横向把手抓取框（手腕绕工具 Z 旋转 90°）下，MoveIt ``panda_hand``
        # 参考点相对 MuJoCo 指尖工作点低约 82 mm，故目标参考点须向上补偿。
        # 该外参来自闭爪前记录的真实 pad 位姿，避免把手上方/下方的伪接触。
        # CAD 标注与接触判定始终使用真实指尖工作点坐标系。
        print(
            "[视觉 CAD 配准] "
            f"目标={target_object_id}，实例像素={segmentation.pixel_count}，"
            f"物体位置=({pose.position_base_m[0]:.4f}, {pose.position_base_m[1]:.4f}, {pose.position_base_m[2]:.4f})，"
            f"yaw={pose.object_yaw_rad!r}，方法={pose.matching_method}",
            flush=True,
        )
        for candidate_template, candidate_position, candidate_orientation in candidates:
            print(
                "[CAD 抓取候选] "
                f"标注={candidate_template.template_id}，"
                f"抓取位=({candidate_position[0]:.4f}, {candidate_position[1]:.4f}, {candidate_position[2]:.4f})，"
                f"四元数_xyzw=({candidate_orientation[0]:.4f}, {candidate_orientation[1]:.4f}, "
                f"{candidate_orientation[2]:.4f}, {candidate_orientation[3]:.4f})，"
                f"预抓取净空={candidate_template.approach_clearance_m:.3f}m",
                flush=True,
            )
        # 正常档以仿真时间一倍速运行；VNC 只显示该节拍，不通过渲染 sleep 改变它。
        executor = MujocoTaskExecutor(
            simulation,
            frame_callback=render_frame,
            realtime_factor=1.0,
            attached_speed_scale=arguments.attached_speed_scale,
        )
        executor.set_display_state(f"CAD match complete: {target_object_id} / {model.model_id}")
        rclpy.init()
        client = MoveItTrajectoryClient()
        try:
            # 当前工装位置来自对象目录；目标杯位置仅来自本次 CAD 配准。
            time.sleep(3.0)
            # 所有目标都直接位于桌面；预抓取阶段保留桌面与其它物体的碰撞约束。
            client.publish_task_scene(
                catalog.fixture_collision_positions(), target_object_id, PlanningPhase.PREGRASP
            )
            time.sleep(1.0)
            motion_wall_start = time.monotonic()
            executor.set_gripper(opened=True)
            selected = None
            rejected_candidates: list[dict[str, str]] = []
            for candidate_template, candidate_position, candidate_orientation in candidates:
                pregrasp = candidate_position + np.array([0.0, 0.0, candidate_template.approach_clearance_m])
                # 横向对称夹取的最终中心相对安全通道向 -X 偏 5 mm。先到无偏移
                # 的高位（已验证可达），再在目标上方横移，避免 OMPL 在带杯身
                # 碰撞体时拒绝最终抓取中心的整段预抓取路径。
                safe_pregrasp = pregrasp + np.array([0.005, 0.0, 0.0])
                tcp_offset = _rotate_vector_by_quaternion(
                    np.array([0.0, 0.0, -0.082]), candidate_orientation
                )
                moveit_pregrasp = safe_pregrasp + tcp_offset
                executor.set_display_state(f"CAD candidate / MoveIt: {candidate_template.template_id}")
                client.publish_joint_state(executor.joint_positions())
                # ROS 2 话题是异步的。必须让规划桥先消费当前 MuJoCo 关节快照，
                # 否则 VNC 实时渲染下可能按上一阶段状态反复规划，出现“轨迹成功但手没移动”。
                time.sleep(0.25)
                try:
                    pregrasp_trajectory = client.request(
                        moveit_pregrasp, candidate_orientation
                    )
                except TimeoutError as error:
                    print(f"[MoveIt 预抓取失败] 标注={candidate_template.template_id}；原因={error}", flush=True)
                    rejected_candidates.append({"frame_id": candidate_template.template_id, "reason": str(error)})
                    continue
                selected = (candidate_template, candidate_position, candidate_orientation, pregrasp_trajectory)
                break
            if selected is None:
                raise RuntimeError(f"MoveIt 未找到可执行的 CAD 抓取标注候选：{rejected_candidates}")
            template, grasp_position, grasp_orientation, pregrasp_trajectory = selected
            tcp_offset = _rotate_vector_by_quaternion(
                np.array([0.0, 0.0, -0.082]), grasp_orientation
            )
            moveit_grasp_position = grasp_position + tcp_offset
            placement = catalog.physical_placement_targets(
                model, template, pose.transform_base_object[:3, :3]
            )
            tray_center = placement.tray_center_base_m
            targets = [
                ("approach", moveit_grasp_position),
                ("lift", grasp_position + np.array([0.0, 0.0, template.lift_clearance_m]) + tcp_offset),
                ("place_above", placement.gripper_above_base_m + tcp_offset),
                ("place_descend", placement.gripper_release_base_m + tcp_offset),
            ]
            executor.set_display_state("CAD grasp / MoveIt: pregrasp")
            _execute_trajectory(executor, pregrasp_trajectory)
            executor._event("moveit_pregrasp_complete")
            executor.set_display_state("CAD grasp / MoveIt: approach")
            if target_object_id == "yellow_mug":
                # 结束场景静置约束；从此处开始杯子只受真实接触动力学与夹持约束影响。
                simulation.data.eq_active[simulation.model.equality("yellow_mug_rest").id] = 0
                import mujoco
                mujoco.mj_forward(simulation.model, simulation.data)
                print("[场景静置] 已解除 yellow_mug_rest，开始真实物理接近", flush=True)
            # 已到达上方安全位后才移除目标碰撞盒，让末段接近能够建立真实把手接触。
            client.publish_task_scene(
                catalog.fixture_collision_positions(), target_object_id, PlanningPhase.APPROACH
            )
            grasp_confirmed = False
            for attempt in range(1, 13):
                if attempt > 1:
                    executor.set_gripper(opened=True)
                    executor.set_display_state(f"CAD grasp retry {attempt}: pregrasp")
                    client.publish_joint_state(executor.joint_positions())
                    time.sleep(0.25)
                    try:
                        retry_pregrasp = client.request(
                            moveit_pregrasp,
                            candidate_orientation,
                        )
                        _execute_trajectory(executor, retry_pregrasp)
                    except TimeoutError as error:
                        print(f"[抓取重试] 第 {attempt} 次预抓取规划失败：{error}", flush=True)
                        continue
                    executor.set_display_state(f"CAD grasp retry {attempt}: approach")
                time.sleep(0.5)
                client.publish_joint_state(executor.joint_positions())
                time.sleep(0.25)
                try:
                    # 抓取框定义了物体相对手爪的偏置。即使只要求末端位置到达，
                    # 自由的腕部姿态也会旋转这段偏置，令长方体在托盘外侧松开；
                    # 因此接近、抬升和放置全程保持 CAD 的完整抓取姿态。
                    _execute_trajectory(
                        executor,
                        client.request(
                            moveit_grasp_position,
                            grasp_orientation,
                        ),
                    )
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
                    executor.close_until_dual_contact(
                        target_object_id,
                        required_object_geometries=template.contact_collision_geometries,
                    )
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
                        required_object_geometries=template.contact_collision_geometries,
                    )
                    grasp_confirmed = True
                    break
                except (RuntimeError, TimeoutError) as error:
                    if target_object_id != "magenta_block" or attempt == 12:
                        raise
                    print(f"[抓取重试] 第 {attempt} 次未形成双侧接触：{error}", flush=True)
            if not grasp_confirmed:
                raise RuntimeError("洋红色长方体在 12 次真实抓取重试后仍未形成双侧接触")
            if target_object_id == "magenta_block":
                simulation.data.eq_active[simulation.model.equality("magenta_block_rest").id] = 0
                import mujoco
                mujoco.mj_forward(simulation.model, simulation.data)
                print("[场景静置] 已解除 magenta_block_rest，双侧接触已确认", flush=True)
            attached_pose = _attached_pose_in_hand(
                pose.position_base_m,
                pose.transform_base_object[:3, :3],
                moveit_grasp_position,
                grasp_orientation,
            )
            for stage, target in targets[1:]:
                executor.set_display_state(f"CAD grasp / MoveIt: {stage} with {target_object_id}")
                phase = PlanningPhase.PLACE if stage.startswith("place") else PlanningPhase.TRANSPORT
                client.publish_task_scene(
                    catalog.fixture_collision_positions(),
                    target_object_id,
                    phase,
                    attached_pose=attached_pose,
                )
                client.publish_joint_state(executor.joint_positions())
                time.sleep(0.25)
                try:
                    _execute_trajectory(
                        executor,
                        client.request(
                            target,
                            grasp_orientation,
                        ),
                    )
                except TimeoutError as error:
                    print(f"[MoveIt 阶段失败] 阶段={stage}；原因={error}", flush=True)
                    raise
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
                    "catalog_geometry_type": catalog_geometry_type,
                    "matcher_override": arguments.force_matcher_type,
                    "scene_object_id": model.scene_object_id,
                    "mesh": str(model.mesh_path.relative_to(arguments.root)),
                    "symmetry": model.symmetry_type,
                    "selected_grasp_template": template.template_id,
                    "selected_grasp_region": template.grasp_region,
                },
                "pose_estimate": pose.as_dict(),
                "grasp_position_base_m": grasp_position.tolist(),
                "grasp_orientation_base_xyzw": grasp_orientation.tolist(),
                "grasp_annotation": {
                    "frame_id": template.template_id,
                    "grasp_region": template.grasp_region,
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
