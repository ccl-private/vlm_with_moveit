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
from .active_perception import generate_observation_views
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


def _base_from_body(simulation: UnifiedPandaCupSimulation, body_name: str) -> np.ndarray:
    """读取当前 MuJoCo 刚体位姿；只用于已知手眼标定，不读取目标真值。"""
    body_id = simulation.model.body(body_name).id
    transform = np.eye(4)
    transform[:3, :3] = simulation.data.xmat[body_id].reshape(3, 3)
    transform[:3, 3] = simulation.data.xpos[body_id]
    return transform


def _contact_geometry_base(pose, templates) -> tuple[np.ndarray, np.ndarray]:
    """将 CAD 标注的相对接触面变换到基座系，供观察位规划使用。"""
    rotation = pose.transform_base_object[:3, :3]
    centers, normals = [], []
    for template in templates:
        pair = template.opposing_contact_pair
        for surface in (pair.positive_surface, pair.negative_surface):
            centers.append(pose.position_base_m + rotation @ surface.center_object_m)
            normals.append(rotation @ surface.normal_object)
    return np.asarray(centers), np.asarray(normals)


def _retreat_observation_trajectory(
    executor: MujocoTaskExecutor,
    trajectory,
    start_joint_positions: dict[str, float],
) -> None:
    """沿已验证的观察轨迹反向退出，避免以观察位作为下一次规划的起始构型。

    腕部观察位的目标是取得图像，而不是作为抓取规划的中间状态。反向复用刚刚
    由 MoveIt 生成的轨迹，既不引入新的 IK 求解，也不以直线插值穿过未知障碍物。
    """
    names = trajectory.joint_trajectory.joint_names
    points = list(trajectory.joint_trajectory.points)
    if not points:
        raise RuntimeError("主动观察返回失败：MoveIt 轨迹没有关节路点")
    times = [float(point.time_from_start.sec) + float(point.time_from_start.nanosec) * 1e-9 for point in points]
    duration_s = max(times[-1], 1e-3)
    retreat_times: list[float] = []
    retreat_positions: list[dict[str, float]] = []
    for point, point_time_s in zip(reversed(points), reversed(times)):
        target = dict(start_joint_positions)
        for name, value in zip(names, point.positions):
            if name.startswith("panda_joint"):
                target[name.replace("panda_", "")] = float(value)
        retreat_positions.append(target)
        retreat_times.append(duration_s - point_time_s)
    # 轨迹的第一个路点通常就是起始状态；若规划器省略了它，明确补上从而回到
    # 本次观察前的已验证状态。
    if any(abs(retreat_positions[-1][name] - value) > 1e-4 for name, value in start_joint_positions.items()):
        retreat_positions.append(dict(start_joint_positions))
        retreat_times.append(duration_s + 0.05)
    executor.execute_timed_joint_trajectory(retreat_times, retreat_positions)


def _yaw_distance_rad(first: float | None, second: float | None) -> float:
    """计算桌面 yaw 的最短角差；不可观测 yaw 不允许作为主动观察的一致性证据。"""
    if first is None or second is None:
        return float("inf")
    return float(abs(np.arctan2(np.sin(first - second), np.cos(first - second))))


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
        "--active-observation",
        action="store_true",
        help="抓取前以腕部 RGB-D 在粗路径末端进行局部闭环重观测。",
    )
    parser.add_argument(
        "--max-wrist-observations",
        type=int,
        default=3,
        help="主动观察最多执行的腕部视角数；on_path 最多使用 2 帧，sweep 使用该值，默认 3。",
    )
    parser.add_argument(
        "--active-observation-mode",
        choices=("on_path", "sweep"),
        default="on_path",
        help="on_path：沿粗预抓取路径近距闭环校正（默认）；sweep：旧的环绕多视角诊断模式。",
    )
    parser.add_argument(
        "--active-observation-debug-dir",
        type=Path,
        help="可选：保存每个成功到达腕部观察位的 RGB-D、掩码和标定，用于离线诊断；不参与控制。",
    )
    parser.add_argument(
        "--attached-speed-scale",
        type=float,
        default=6.0,
        help="夹住物体后的轨迹时长倍率，默认 6.0；数值越小越快。",
    )
    arguments = parser.parse_args()
    if arguments.max_wrist_observations <= 0:
        raise ValueError("--max-wrist-observations 必须为正数")

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
        matcher_dispatcher = CadMatcherDispatcher()
        pose = matcher_dispatcher.match(model, segmentation, fixed_frame)
        templates = tuple(
            template for template in model.grasp_templates
            if intent.grasp_region is None or template.grasp_region == intent.grasp_region
        )
        if not templates:
            raise RuntimeError(f"CAD 模型没有与语义抓取区域匹配的抓取框：{intent.grasp_region}")
        print(
            "[视觉 CAD 配准] "
            f"目标={target_object_id}，实例像素={segmentation.pixel_count}，"
            f"物体位置=({pose.position_base_m[0]:.4f}, {pose.position_base_m[1]:.4f}, {pose.position_base_m[2]:.4f})，"
            f"yaw={pose.object_yaw_rad!r}，方法={pose.matching_method}",
            flush=True,
        )
        # on_path 模式始终以固定相机结果作为腕部局部匹配的先验，不会把任何
        # MuJoCo 物体真值带入控制回路。
        coarse_pose = pose
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
        active_observation: dict[str, object] = {
            "enabled": arguments.active_observation,
            "mode": arguments.active_observation_mode,
            "max_wrist_observations": arguments.max_wrist_observations,
            "views": [],
            "selected_source": "fixed",
        }
        observation_debug_dir = arguments.active_observation_debug_dir
        if observation_debug_dir is not None:
            observation_debug_dir.mkdir(parents=True, exist_ok=True)
        try:
            # 当前工装位置来自对象目录；目标杯位置仅来自本次 CAD 配准。
            time.sleep(3.0)
            # 所有目标都直接位于桌面；预抓取阶段保留桌面与其它物体的碰撞约束。
            client.publish_task_scene(
                catalog.fixture_collision_positions(), target_object_id, PlanningPhase.PREGRASP
            )
            time.sleep(1.0)
            if arguments.active_observation and arguments.active_observation_mode == "sweep":
                # 相机相对手掌的外参从当前已标定 MuJoCo 链读取；它是固定刚体关系，
                # 与任何物体真值无关。观察位围绕粗定位位置均匀展开，避免粗 yaw
                # 错误时只朝“猜测的把手方向”移动。
                wrist_frame = simulation.cameras(("wrist",))["wrist"]
                hand_from_camera = np.linalg.inv(_base_from_body(simulation, "hand")) @ wrist_frame.base_from_camera
                contact_centers, contact_normals = _contact_geometry_base(pose, templates)
                observation_views = generate_observation_views(
                    pose.position_base_m,
                    contact_centers,
                    contact_normals,
                    hand_from_camera,
                )
                successful_observations = 0
                matched_observations: list[tuple[object, object, object, dict[str, object]]] = []
                successful_camera_positions: list[np.ndarray] = []
                # ``max_wrist_observations`` 限制成功采集的帧数；不可达/碰撞候选
                # 必须继续尝试其它方位，不能耗尽重观测预算。
                for observation_view in observation_views:
                    if successful_observations >= arguments.max_wrist_observations:
                        break
                    executor.set_display_state(f"Active observe / MoveIt: {observation_view.view_id}")
                    observation_start_joint_positions = executor.joint_positions()
                    client.publish_joint_state(observation_start_joint_positions)
                    time.sleep(0.25)
                    entry: dict[str, object] = {
                        "view_id": observation_view.view_id,
                        "expected_contact_visibility": observation_view.expected_contact_visibility,
                        "visible_contact_indices": list(observation_view.visible_contact_indices),
                        "camera_position_base_m": observation_view.camera_position_base_m.tolist(),
                    }
                    try:
                        trajectory = client.request(
                            observation_view.hand_position_base_m,
                            observation_view.hand_orientation_base_xyzw,
                        )
                        _execute_trajectory(executor, trajectory)
                        executor._event("active_observation_view_complete", view_id=observation_view.view_id)
                        wrist_frame = simulation.cameras(("wrist",))["wrist"]
                        # 固定相机只产生粗 pose；腕部精看时以该粗位置的 3D ROI
                        # 排除颜色相近的桌面与工装，同时保留受高光影响的杯身深度点。
                        lateral_radius = 0.5 * float(np.linalg.norm(model.dimensions_m[:2])) + 0.025
                        wrist_segmentation = ColorThresholdSegmenter().segment(
                            wrist_frame.rgb,
                            intent,
                            frame=wrist_frame,
                            expected_position_base_m=pose.position_base_m,
                            lateral_radius_m=lateral_radius,
                            z_min_m=float(pose.position_base_m[2] - model.height_m / 2.0 + 0.004),
                            z_max_m=float(pose.position_base_m[2] + model.height_m / 2.0 + 0.025),
                        )
                        if observation_debug_dir is not None:
                            debug_path = observation_debug_dir / f"{observation_view.view_id}.npz"
                            np.savez_compressed(
                                debug_path,
                                rgb=wrist_frame.rgb,
                                depth=wrist_frame.depth,
                                mask=wrist_segmentation.mask,
                                intrinsic=wrist_frame.intrinsic,
                                base_from_camera=wrist_frame.base_from_camera,
                                expected_contact_visibility=np.float64(observation_view.expected_contact_visibility),
                            )
                            entry["debug_observation"] = str(debug_path)
                        wrist_pose = None
                        try:
                            wrist_pose = matcher_dispatcher.match(model, wrist_segmentation, wrist_frame)
                        except RuntimeError as matching_error:
                            if model.geometry_type != "mug_handle":
                                raise
                            # 单帧不够区分杯身近似对称带来的 yaw 歧义是预期情况；
                            # 分割已经成功的 RGB-D 帧仍是多视角 CAD 融合的有效证据。
                            # 不能用“单帧先定准朝向”这个前置条件把它悄悄丢掉。
                            entry.update(
                                status="segmented_for_fusion",
                                reason=f"单帧 yaw 未定，保留点云给多视角关键区验证：{matching_error}",
                                pixel_count=wrist_segmentation.pixel_count,
                                score=0.10 * observation_view.expected_contact_visibility,
                            )
                        if wrist_pose is not None:
                            score = wrist_pose.geometry_confidence + 0.10 * observation_view.expected_contact_visibility
                            entry.update(
                                status="matched",
                                pixel_count=wrist_segmentation.pixel_count,
                                matching_method=wrist_pose.matching_method,
                                geometry_confidence=wrist_pose.geometry_confidence,
                                surface_residual_m=wrist_pose.surface_residual_m,
                                score=score,
                                position_base_m=wrist_pose.position_base_m.tolist(),
                                object_yaw_rad=wrist_pose.object_yaw_rad,
                            )
                        # 同一名义相机位置的腕部滚转不增加几何基线，因而不能占用
                        # 观察预算；但它会改变夹爪/前臂对相机的遮挡，仍可能暴露
                        # 上一帧看不到的把手区域，必须保留到融合点云中。
                        if any(
                            np.linalg.norm(observation_view.camera_position_base_m - previous) <= 0.005
                            for previous in successful_camera_positions
                        ):
                            entry.update(
                                status="redundant_camera_position",
                                reason="同一相机位置的另一腕部滚转不占观察预算，但其去遮挡点云参与融合",
                                contributes_new_camera_baseline=False,
                            )
                            print(
                                f"[主动观察融合] 视角={observation_view.view_id}；原因=复用相机位置但保留不同腕部遮挡下的点云",
                                flush=True,
                            )
                            executor.set_display_state(f"Active observe retreat: {observation_view.view_id}")
                            _retreat_observation_trajectory(
                                executor, trajectory, observation_start_joint_positions
                            )
                            executor._event("active_observation_return_complete", view_id=observation_view.view_id)
                            matched_observations.append((wrist_pose, wrist_segmentation, wrist_frame, entry))
                            active_observation["views"].append(entry)
                            continue
                        if wrist_pose is None:
                            print(
                                f"[主动观察帧] 视角={observation_view.view_id}，"
                                f"分割点={wrist_segmentation.pixel_count}；单帧 yaw 未定，保留给融合",
                                flush=True,
                            )
                        else:
                            print(
                                f"[主动观察帧] 视角={observation_view.view_id}，"
                                f"点={wrist_pose.point_count}，位置=({wrist_pose.position_base_m[0]:.4f}, "
                                f"{wrist_pose.position_base_m[1]:.4f}, {wrist_pose.position_base_m[2]:.4f})，"
                                f"yaw={wrist_pose.object_yaw_rad!r}，置信度={wrist_pose.geometry_confidence:.3f}",
                                flush=True,
                            )
                        executor.set_display_state(f"Active observe retreat: {observation_view.view_id}")
                        _retreat_observation_trajectory(executor, trajectory, observation_start_joint_positions)
                        executor._event("active_observation_return_complete", view_id=observation_view.view_id)
                        successful_observations += 1
                        successful_camera_positions.append(observation_view.camera_position_base_m)
                        entry["contributes_new_camera_baseline"] = True
                        matched_observations.append((wrist_pose, wrist_segmentation, wrist_frame, entry))
                    except (RuntimeError, TimeoutError) as error:
                        entry.update(status="rejected", reason=str(error))
                        print(f"[主动观察拒绝] 视角={observation_view.view_id}；原因={error}", flush=True)
                    active_observation["views"].append(entry)
                if successful_observations < 2:
                    active_observation["failure_reason"] = "reobserve_exhausted"
                    raise RuntimeError(
                        "reobserve_exhausted：少于两个腕部观察位完成匹配，不能用单帧低约束结果进入抓取"
                    )
                if model.geometry_type in {"mesh", "mug_handle"}:
                    # 通用 mesh 与把手杯的单帧匹配都可能在杯身等重复局部极值之间
                    # 切换；因此不能拿这些单帧 pose 投票。合并各帧基座系点云后
                    # 只做一次 CAD 精配准，才是真正的多视角几何约束。
                    pose = matcher_dispatcher.match_multiview(
                        model,
                        [
                            (
                                item[1],
                                item[2],
                                float(item[3]["expected_contact_visibility"]),
                            )
                            if model.geometry_type == "mug_handle"
                            else (item[1], item[2])
                            for item in matched_observations
                        ],
                        yaw_prior_rad=pose.object_yaw_rad if model.geometry_type == "mug_handle" else None,
                    )
                    if pose.object_yaw_rad is None:
                        active_observation["failure_reason"] = "fused_yaw_unobservable"
                        raise RuntimeError("reobserve_exhausted：多视角 CAD 融合未能辨识 yaw，拒绝生成抓取姿态")
                    print(
                        f"[主动观察融合候选] 位置=({pose.position_base_m[0]:.4f}, "
                        f"{pose.position_base_m[1]:.4f}, {pose.position_base_m[2]:.4f})，"
                        f"yaw={pose.object_yaw_rad!r}，置信度={pose.geometry_confidence:.3f}",
                        flush=True,
                    )
                    if model.geometry_type == "mesh":
                        fused_support_count = sum(
                            np.linalg.norm(item[0].position_base_m - pose.position_base_m) <= 0.030
                            and _yaw_distance_rad(item[0].object_yaw_rad, pose.object_yaw_rad) <= 0.75
                            for item in matched_observations
                        )
                        if fused_support_count < 2:
                            active_observation["failure_reason"] = "fused_pose_insufficient_independent_support"
                            active_observation["fused_support_count"] = fused_support_count
                            raise RuntimeError(
                                "reobserve_exhausted：融合 mesh pose 未获得至少两个独立腕部视角支持，拒绝猜测抓取姿态"
                            )
                    else:
                        # 杯身近似旋转对称，单帧的全局 mesh yaw 没有可投票的意义。
                        # MugHandleCadMatcher 已在内部要求被抓取的把手外竖条得到
                        # 足量单视角支持，并与相隔 20° 以上的替代朝向拉开差距。
                        # 这里复用该可审计的关键区域门禁，不能再用错误的单帧 yaw
                        # 反向否决多视角证据。
                        fused_support_count = sum(
                            item[3].get("contributes_new_camera_baseline", False) for item in matched_observations
                        )
                        active_observation["critical_region_support"] = pose.critical_region_support
                        active_observation["critical_region_margin"] = pose.critical_region_margin
                    segmentation = max(matched_observations, key=lambda item: item[1].confidence)[1]
                    active_observation["selected_source"] = "fused:" + ",".join(
                        str(item[3]["view_id"]) for item in matched_observations
                    )
                    active_observation["consensus_count"] = len(matched_observations)
                    active_observation["fused_support_count"] = fused_support_count
                    active_observation["fused_pose"] = {
                        "position_base_m": pose.position_base_m.tolist(),
                        "object_yaw_rad": pose.object_yaw_rad,
                        "geometry_confidence": pose.geometry_confidence,
                        "surface_residual_m": pose.surface_residual_m,
                        "point_count": pose.point_count,
                    }
                    consensus = matched_observations
                else:
                    consensus_groups = []
                    for seed_pose, _, _, _ in matched_observations:
                        group = [
                            item
                            for item in matched_observations
                            if np.linalg.norm(item[0].position_base_m - seed_pose.position_base_m) <= 0.025
                            and _yaw_distance_rad(item[0].object_yaw_rad, seed_pose.object_yaw_rad) <= 0.25
                        ]
                        consensus_groups.append(group)
                    consensus = max(consensus_groups, key=len)
                    if len(consensus) < 2:
                        active_observation["failure_reason"] = "pose_hypotheses_disagree"
                        raise RuntimeError(
                            "reobserve_exhausted：腕部观察位未在 25 mm / 0.25 rad 门槛内形成一致 pose，拒绝猜测抓取姿态"
                        )
                    pose, segmentation, _, selected_entry = max(consensus, key=lambda item: item[0].geometry_confidence)
                    active_observation["selected_source"] = selected_entry["view_id"]
                    active_observation["consensus_count"] = len(consensus)
                print(
                    f"[主动观察] 已完成 {successful_observations} 个腕部视角，其中 {len(consensus)} 个 pose 一致；"
                    f"选用={active_observation['selected_source']}，位置=({pose.position_base_m[0]:.4f}, "
                    f"{pose.position_base_m[1]:.4f}, {pose.position_base_m[2]:.4f})，"
                    f"yaw={pose.object_yaw_rad!r}，几何置信度={pose.geometry_confidence:.3f}",
                    flush=True,
                )
            motion_wall_start = time.monotonic()
            executor.set_gripper(opened=True)
            candidates = grasp_pose_candidates(pose, templates)
            # 横向把手抓取框（手腕绕工具 Z 旋转 90°）下，MoveIt ``panda_hand``
            # 参考点相对 MuJoCo 指尖工作点低约 82 mm，故目标参考点须向上补偿。
            # 该外参来自闭爪前记录的真实 pad 位姿，避免把手上方/下方的伪接触。
            # CAD 标注与接触判定始终使用真实指尖工作点坐标系。
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
            selected = None
            rejected_candidates: list[dict[str, str]] = []
            # CAD 位姿和手眼标定都可能留下几毫米的残差。不能因为名义预抓取位
            # 正好落在 Panda 的 IK/碰撞边界外就放弃一个已经可靠匹配的把手；但也
            # 不能在工作空间里任意搜索。只枚举桌面平面内 4 mm 的有限邻域，完整
            # 保持 CAD 标注的夹爪朝向，并先让 MoveIt 验证每个预抓取候选。
            reachability_offsets_base = (
                np.array([0.0, 0.0, 0.0]),
                np.array([-0.004, 0.0, 0.0]),
                np.array([0.0, 0.004, 0.0]),
                np.array([-0.004, 0.004, 0.0]),
                np.array([0.004, 0.0, 0.0]),
                np.array([0.0, -0.004, 0.0]),
                np.array([0.004, -0.004, 0.0]),
                np.array([-0.004, -0.004, 0.0]),
                np.array([0.004, 0.004, 0.0]),
            )
            for candidate_template, candidate_position, candidate_orientation in candidates:
                for reachability_offset in reachability_offsets_base:
                    adjusted_position = candidate_position + reachability_offset
                    pregrasp = adjusted_position + np.array([0.0, 0.0, candidate_template.approach_clearance_m])
                    # 横向对称夹取的最终中心相对安全通道向 -X 偏 5 mm。先到无偏移
                    # 的高位（已验证可达），再在目标上方横移，避免 OMPL 在带杯身
                    # 碰撞体时拒绝最终抓取中心的整段预抓取路径。
                    safe_pregrasp = pregrasp + np.array([0.005, 0.0, 0.0])
                    tcp_offset = _rotate_vector_by_quaternion(
                        np.array([0.0, 0.0, -0.082]), candidate_orientation
                    )
                    moveit_pregrasp = safe_pregrasp + tcp_offset
                    offset_text = (
                        f"({reachability_offset[0]:+.3f}, {reachability_offset[1]:+.3f}, "
                        f"{reachability_offset[2]:+.3f})m"
                    )
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
                        print(
                            f"[MoveIt 预抓取失败] 标注={candidate_template.template_id}；"
                            f"CAD 平面偏移={offset_text}；原因={error}",
                            flush=True,
                        )
                        rejected_candidates.append(
                            {
                                "frame_id": candidate_template.template_id,
                                "offset_base_m": offset_text,
                                "reason": str(error),
                            }
                        )
                        continue
                    print(
                        f"[MoveIt 预抓取候选] 标注={candidate_template.template_id}；"
                        f"CAD 平面偏移={offset_text}；已通过可达性验证",
                        flush=True,
                    )
                    selected = (
                        candidate_template,
                        adjusted_position,
                        candidate_orientation,
                        pregrasp_trajectory,
                        reachability_offset,
                    )
                    break
                if selected is not None:
                    break
            if selected is None:
                raise RuntimeError(f"MoveIt 未找到可执行的 CAD 抓取标注候选：{rejected_candidates}")
            template, grasp_position, grasp_orientation, pregrasp_trajectory, reachability_offset = selected
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
            if arguments.active_observation and arguments.active_observation_mode == "on_path":
                # 不再先做环绕扫描：粗预抓取本来就是必经安全走廊。到达后直接用
                # 腕部 RGB-D 做局部校正；相机必须主动注视把手，而不能假定抓取
                # 姿态下腕部光轴天然朝向目标。若关键区证据不足，最多换一个侧面。
                # 每次移动都从当前关节状态重新由 MoveIt 规划，不能在线篡改已批准
                # 的 IK 轨迹。
                local_observations: list[tuple[object, object, float]] = []
                refined_pose = None
                local_failure = None
                wrist_frame = simulation.cameras(("wrist",))["wrist"]
                hand_from_camera = np.linalg.inv(_base_from_body(simulation, "hand")) @ wrist_frame.base_from_camera
                contact_centers, contact_normals = _contact_geometry_base(coarse_pose, (template,))
                # 使用已在本场景验证可达的 26 cm 半径、30 cm 高度俯视位：相机
                # 会停在粗预抓取旁约 20 cm 的安全走廊，而不是贴近桌面进入腕部
                # 奇异/跟踪裕量不足的区域。排序优先把手可见性，再看当前手掌的
                # 位移距离。
                on_path_views = generate_observation_views(
                    coarse_pose.position_base_m,
                    contact_centers,
                    contact_normals,
                    hand_from_camera,
                    radius_m=0.26,
                    elevation_m=0.30,
                )
                # 观察位必须仍处于粗预抓取末端的局部工作空间。这个上限不是
                # MoveIt 的碰撞约束替代品，而是防止“为了看一眼”选到桌子另一侧
                # 的可达解；没有合格近视角时宁可安全拒绝，再由 sweep 诊断模式
                # 排障，也不把默认执行退化成大绕行。
                maximum_observation_hand_travel_m = 0.25
                selected_camera_positions: list[np.ndarray] = []
                for observation_index in range(min(2, arguments.max_wrist_observations)):
                    current_hand_position = executor.body_position("hand")
                    observation_view = None
                    observation_trajectory = None
                    observation_hand_travel_m = None
                    for candidate_view in sorted(
                        on_path_views,
                        key=lambda view: (
                            -view.expected_contact_visibility,
                            float(np.linalg.norm(view.hand_position_base_m - current_hand_position)),
                        ),
                    ):
                        if any(
                            np.linalg.norm(candidate_view.camera_position_base_m - previous) < 0.06
                            for previous in selected_camera_positions
                        ):
                            continue
                        candidate_hand_travel_m = float(
                            np.linalg.norm(candidate_view.hand_position_base_m - current_hand_position)
                        )
                        if candidate_hand_travel_m > maximum_observation_hand_travel_m:
                            continue
                        client.publish_joint_state(executor.joint_positions())
                        time.sleep(0.25)
                        try:
                            observation_trajectory = client.request(
                                candidate_view.hand_position_base_m,
                                candidate_view.hand_orientation_base_xyzw,
                            )
                        except TimeoutError:
                            continue
                        observation_view = candidate_view
                        observation_hand_travel_m = candidate_hand_travel_m
                        break
                    if observation_view is None or observation_trajectory is None:
                        raise RuntimeError("on_path_reobserve_exhausted：没有可达的近距把手观察位")
                    executor.set_display_state(f"On-path wrist observation {observation_index + 1}")
                    # 观察位会有明显的腕部转向；采用低于常规 1.7 倍的温和加速，
                    # 在保持跟踪裕量的同时避免为一次局部观测等待完整原始轨迹时长。
                    _execute_trajectory(executor, observation_trajectory, speed_scale=1.25)
                    executor._event(
                        "on_path_wrist_observation_complete",
                        view_id=observation_view.view_id,
                        expected_contact_visibility=observation_view.expected_contact_visibility,
                        hand_travel_m=observation_hand_travel_m,
                    )
                    selected_camera_positions.append(observation_view.camera_position_base_m)
                    wrist_frame = simulation.cameras(("wrist",))["wrist"]
                    lateral_radius = 0.5 * float(np.linalg.norm(model.dimensions_m[:2])) + 0.025
                    wrist_segmentation = ColorThresholdSegmenter().segment(
                        wrist_frame.rgb,
                        intent,
                        frame=wrist_frame,
                        expected_position_base_m=coarse_pose.position_base_m,
                        lateral_radius_m=lateral_radius,
                        z_min_m=float(coarse_pose.position_base_m[2] - model.height_m / 2.0 + 0.004),
                        z_max_m=float(coarse_pose.position_base_m[2] + model.height_m / 2.0 + 0.025),
                    )
                    entry: dict[str, object] = {
                        "view_id": f"on_path_{observation_index}:{observation_view.view_id}",
                        "camera_position_base_m": wrist_frame.base_from_camera[:3, 3].tolist(),
                        "expected_contact_visibility": observation_view.expected_contact_visibility,
                        "hand_travel_m": observation_hand_travel_m,
                        "pixel_count": wrist_segmentation.pixel_count,
                        "contributes_new_camera_baseline": True,
                    }
                    if observation_debug_dir is not None:
                        debug_path = observation_debug_dir / f"on_path_{observation_index}.npz"
                        np.savez_compressed(
                            debug_path,
                            rgb=wrist_frame.rgb,
                            depth=wrist_frame.depth,
                            mask=wrist_segmentation.mask,
                            intrinsic=wrist_frame.intrinsic,
                            base_from_camera=wrist_frame.base_from_camera,
                            expected_contact_visibility=np.float64(observation_view.expected_contact_visibility),
                        )
                        entry["debug_observation"] = str(debug_path)
                    local_observations.append(
                        (wrist_segmentation, wrist_frame, observation_view.expected_contact_visibility)
                    )
                    try:
                        if model.geometry_type in {"mesh", "mug_handle"}:
                            # 固定相机只提供低权重粗先验；最终把手证据来自近距腕部
                            # 帧。它使首个腕部停靠位即可形成几何基线，避免环绕扫描。
                            matching_observations = (
                                [(segmentation, fixed_frame, 0.15)] + local_observations
                                if model.geometry_type == "mug_handle"
                                else [(segmentation, fixed_frame)] + [item[:2] for item in local_observations]
                            )
                            refined_pose = matcher_dispatcher.match_multiview(
                                model,
                                matching_observations,
                                yaw_prior_rad=coarse_pose.object_yaw_rad
                                if model.geometry_type == "mug_handle"
                                else None,
                            )
                        else:
                            refined_pose = matcher_dispatcher.match(model, wrist_segmentation, wrist_frame)
                        entry.update(
                            status="matched",
                            geometry_confidence=refined_pose.geometry_confidence,
                            matching_method=refined_pose.matching_method,
                        )
                        active_observation["views"].append(entry)
                        break
                    except RuntimeError as error:
                        local_failure = str(error)
                        entry.update(status="insufficient_evidence", reason=local_failure)
                        active_observation["views"].append(entry)
                        if observation_index + 1 >= min(2, arguments.max_wrist_observations):
                            raise RuntimeError(
                                f"on_path_reobserve_exhausted：近距腕部校正仍缺少抓取证据：{local_failure}"
                            ) from error
                        print(
                            f"[路径内腕部校正] 第 {observation_index + 1} 帧关键区证据不足，"
                            "只尝试一个不同侧面的近距观察位。",
                            flush=True,
                        )
                if refined_pose is None:
                    raise RuntimeError(f"on_path_reobserve_exhausted：{local_failure}")
                pose = refined_pose
                segmentation = local_observations[-1][0]
                active_observation.update(
                    selected_source="on_path:fixed_plus_wrist",
                    consensus_count=len(local_observations) + 1,
                    fused_support_count=len(local_observations),
                    critical_region_support=pose.critical_region_support,
                    critical_region_margin=pose.critical_region_margin,
                    fused_pose={
                        "position_base_m": pose.position_base_m.tolist(),
                        "object_yaw_rad": pose.object_yaw_rad,
                        "geometry_confidence": pose.geometry_confidence,
                        "surface_residual_m": pose.surface_residual_m,
                        "point_count": pose.point_count,
                    },
                )
                refined_candidates = grasp_pose_candidates(pose, (template,))
                if not refined_candidates:
                    raise RuntimeError("on_path_reobserve_exhausted：精匹配后未生成原 CAD 抓取框")
                template, refined_grasp_position, grasp_orientation = refined_candidates[0]
                grasp_position = refined_grasp_position + reachability_offset
                tcp_offset = _rotate_vector_by_quaternion(np.array([0.0, 0.0, -0.082]), grasp_orientation)
                moveit_pregrasp = (
                    grasp_position
                    + np.array([0.005, 0.0, template.approach_clearance_m])
                    + tcp_offset
                )
                # 从腕部观测的当前真实关节状态规划剩余短路径，而不是继续复用粗
                # 位姿下生成的长轨迹。
                client.publish_joint_state(executor.joint_positions())
                time.sleep(0.25)
                executor.set_display_state("On-path corrected pregrasp / MoveIt")
                _execute_trajectory(executor, client.request(moveit_pregrasp, grasp_orientation))
                executor._event(
                    "on_path_pose_correction_complete",
                    correction_position_m=(pose.position_base_m - coarse_pose.position_base_m).tolist(),
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
                print(
                    f"[路径内腕部校正] 观察帧={len(local_observations)}，"
                    f"位置修正={np.linalg.norm(pose.position_base_m - coarse_pose.position_base_m):.4f}m，"
                    f"置信度={pose.geometry_confidence:.3f}",
                    flush=True,
                )
            executor.set_display_state("CAD grasp / MoveIt: approach")
            # 已到达上方安全位后才移除目标碰撞盒，让末段接近能够建立真实把手接触。
            client.publish_task_scene(
                catalog.fixture_collision_positions(), target_object_id, PlanningPhase.APPROACH
            )
            grasp_confirmed = False
            # 视觉 pose 的末端不确定度会集中放大到细把手的接触条上。对该类
            # 抓取，成熟产线会在 CAD 局部接触面附近进行受限的力/接触验证搜索，
            # 而不是把第一次闭合失败直接解释为“视觉可以任意猜”。偏移均不超过
            # 4 mm，且仅沿已标注的夹爪闭合轴和把手径向轴；每一次都必须通过同一
            # 对真实碰撞面的双侧接触门禁。
            if target_object_id == "yellow_mug":
                grasp_search_offsets_object = (
                    np.array([0.0, 0.0, 0.0]),
                    np.array([0.0, 0.004, 0.0]),
                    np.array([-0.004, 0.004, 0.0]),
                    np.array([-0.004, 0.0, 0.0]),
                    np.array([0.004, 0.004, 0.0]),
                    np.array([0.0, -0.004, 0.0]),
                    np.array([0.004, 0.0, 0.0]),
                    np.array([-0.004, -0.004, 0.0]),
                    np.array([0.004, -0.004, 0.0]),
                )
            else:
                grasp_search_offsets_object = (np.zeros(3),)
            max_grasp_attempts = 12 if target_object_id == "magenta_block" else len(grasp_search_offsets_object)
            for attempt in range(1, max_grasp_attempts + 1):
                offset_object = grasp_search_offsets_object[min(attempt - 1, len(grasp_search_offsets_object) - 1)]
                attempt_grasp_position = grasp_position + pose.transform_base_object[:3, :3] @ offset_object
                attempt_moveit_grasp_position = attempt_grasp_position + tcp_offset
                attempt_pregrasp = attempt_grasp_position + np.array([0.0, 0.0, template.approach_clearance_m])
                attempt_moveit_pregrasp = attempt_pregrasp + np.array([0.005, 0.0, 0.0]) + tcp_offset
                if attempt > 1:
                    executor.set_gripper(opened=True)
                    executor.set_display_state(f"CAD grasp retry {attempt}: pregrasp")
                    client.publish_joint_state(executor.joint_positions())
                    time.sleep(0.25)
                    try:
                        retry_pregrasp = client.request(
                            attempt_moveit_pregrasp,
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
                            attempt_moveit_grasp_position,
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
                    if target_object_id == "yellow_mug":
                        # ``attach`` 已确认不只是接触而且两指分别处于 CAD 标注的
                        # 相对面，至此才可解除静置。若该验证失败，杯子必须保持在
                        # 原位供下一次毫米级受限搜索使用。
                        simulation.data.eq_active[simulation.model.equality("yellow_mug_rest").id] = 0
                        import mujoco
                        mujoco.mj_forward(simulation.model, simulation.data)
                        print("[场景静置] 双侧接触与相对面验证已确认，解除 yellow_mug_rest", flush=True)
                    # 后续抬升、放置和物体在手坐标系中的相对位姿必须使用真正
                    # 通过接触验证的微调抓取位，而不是初始视觉候选。
                    grasp_position = attempt_grasp_position
                    moveit_grasp_position = attempt_moveit_grasp_position
                    grasp_confirmed = True
                    break
                except (RuntimeError, TimeoutError) as error:
                    if target_object_id not in {"magenta_block", "yellow_mug"} or attempt == max_grasp_attempts:
                        raise
                    print(
                        f"[抓取重试] 第 {attempt} 次未形成双侧接触；"
                        f"CAD 局部偏移=({offset_object[0]:.3f}, {offset_object[1]:.3f}, {offset_object[2]:.3f})m；原因={error}",
                        flush=True,
                    )
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
                    "selected_reachability_offset_base_m": reachability_offset.tolist(),
                    "rejected_candidates": rejected_candidates,
                },
                "active_observation": active_observation,
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
                # 成功已经由 MuJoCo 的托盘物理验证确认。只补一帧带成功状态的
                # 画面后立即返回 finally 关闭录制器，避免 VNC 回合在终态无限录制。
                viewer.render_once(f"Success: CAD matched {target_object_id} in tray")
                print("CAD 模型匹配回合成功；正在自动关闭 VNC 并保存 MP4。", flush=True)
        finally:
            client.destroy_node()
            rclpy.shutdown()
    finally:
        if viewer is not None:
            viewer.close()
        simulation.close()


if __name__ == "__main__":
    main()
