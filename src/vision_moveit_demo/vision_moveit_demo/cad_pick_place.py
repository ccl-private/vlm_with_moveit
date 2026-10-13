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
    GraspRoiMatcher,
    ObjectCatalog,
    RuleVlmAdapter,
    grasp_pose_candidates,
)
from .active_perception import (
    ObservationView,
    generate_local_observation_views,
    generate_observation_views,
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


def _pose_with_local_z_yaw(pose, yaw_offset_rad: float):
    """Return a CAD pose rotated around its own upright Z axis.

    This is deliberately a small, ROI-validated adjustment rather than a
    workspace search.  ``GraspRoiMatcher`` supplies the offsets only after it
    has independently rechecked both annotated contact surfaces in wrist RGB-D.
    """
    cosine, sine = np.cos(yaw_offset_rad), np.sin(yaw_offset_rad)
    local_rotation = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    transform = pose.transform_base_object.copy()
    transform[:3, :3] = transform[:3, :3] @ local_rotation
    diagnostics = dict(pose.registration_diagnostics or {})
    diagnostics["reachability_selected_local_yaw_offset_rad"] = float(yaw_offset_rad)
    return replace(
        pose,
        transform_base_object=transform,
        orientation_xyzw=_matrix_to_quaternion(transform[:3, :3]),
        object_yaw_rad=float(np.arctan2(transform[1, 0], transform[0, 0])),
        registration_diagnostics=diagnostics,
    )


def _attached_pose_in_hand(
    object_position_base: np.ndarray,
    object_rotation_base: np.ndarray,
    hand_position_base: np.ndarray,
    hand_orientation_xyzw: np.ndarray,
) -> Pose:
    # 请求朝向属于 panda_link8；附着碰撞体却表达在 panda_hand。
    # 两者存在固定 -45° 手爪安装转角，必须与真实 MuJoCo 抓取关系一致。
    angle = -np.pi / 4.0
    hand_mount_rotation = np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    hand_rotation = _quaternion_to_matrix(hand_orientation_xyzw) @ hand_mount_rotation
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


def _rank_wrist_observation_views(
    views: list[ObservationView],
    selected_camera_positions: list[np.ndarray],
    covered_contact_indices: set[int],
    current_hand_position: np.ndarray,
) -> list[ObservationView]:
    """按预期新增夹持区、相机基线和运动代价选择下一次腕部观察。

    第一帧优先直接看到更多 CAD 接触面；后续帧优先补充尚未覆盖的面并扩大相机
    基线。固定相机的 yaw 不参与该排序，因此即使粗朝向错了，也不会让腕部沿着
    一个错误把手方向来回观察。
    """
    def score(view: ObservationView) -> tuple[float, float, float, float, str]:
        new_contacts = len(set(view.visible_contact_indices) - covered_contact_indices)
        if selected_camera_positions:
            baseline_m = min(
                float(np.linalg.norm(view.camera_position_base_m - previous))
                for previous in selected_camera_positions
            )
        else:
            baseline_m = 0.0
        hand_travel_m = float(np.linalg.norm(view.hand_position_base_m - current_hand_position))
        # 当两片接触面都已由不同方位覆盖时，优先选更可能看清把手的同位不同
        # 腕部滚转，去除夹爪/前臂遮挡；不能为了凑第三条几何基线跑到一个预测
        # 可见性接近零的方位。仍将基线作为同等可见性候选间的次级排序项。
        return (-float(new_contacts), -view.expected_contact_visibility, -baseline_m, hand_travel_m, view.view_id)

    return sorted(views, key=score)


def _trajectory_endpoint_joint_positions(trajectory, fallback: dict[str, float]) -> dict[str, float]:
    """提取 MoveIt 轨迹终点，供不执行动作的下一段可达性预检使用。"""
    positions = dict(fallback)
    points = list(trajectory.joint_trajectory.points)
    if not points:
        raise RuntimeError("MoveIt 预检轨迹没有关节路点")
    for name, value in zip(trajectory.joint_trajectory.joint_names, points[-1].positions):
        if name.startswith("panda_joint"):
            positions[name.replace("panda_", "")] = float(value)
    return positions


def _validate_full_grasp_path(
    client: MoveItTrajectoryClient,
    executor: MujocoTaskExecutor,
    catalog: ObjectCatalog,
    target_object_id: str,
    moveit_pregrasp: np.ndarray,
    moveit_grasp: np.ndarray,
    orientation_xyzw: np.ndarray,
) -> None:
    """预检从当前状态到预抓取、再到最终接近的完整 MoveIt 可达性。

    只验证，不执行。第二段规划临时向 MoveIt 发布第一段轨迹的末端关节状态，
    再立即恢复 MuJoCo 的真实当前状态和预抓取碰撞场景。这样 ``GOAL_STATE_INVALID``
    不会被误报为“闭爪未形成双侧接触”。
    """
    current_joint_positions = executor.joint_positions()
    client.publish_task_scene(
        catalog.fixture_collision_positions(), target_object_id, PlanningPhase.PREGRASP
    )
    client.publish_joint_state(current_joint_positions)
    time.sleep(0.25)
    try:
        pregrasp_trajectory = client.request(moveit_pregrasp, orientation_xyzw)
        endpoint_positions = _trajectory_endpoint_joint_positions(pregrasp_trajectory, current_joint_positions)
        client.publish_task_scene(
            catalog.fixture_collision_positions(), target_object_id, PlanningPhase.APPROACH
        )
        client.publish_joint_state(endpoint_positions)
        time.sleep(0.25)
        client.request(moveit_grasp, orientation_xyzw)
    finally:
        client.publish_task_scene(
            catalog.fixture_collision_positions(), target_object_id, PlanningPhase.PREGRASP
        )
        client.publish_joint_state(current_joint_positions)
        time.sleep(0.25)


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段 3：语义选模 CAD 匹配抓取基线")
    parser.add_argument("--instruction", default="抓取红色杯子并放到托盘")
    parser.add_argument("--root", type=Path, default=Path(os.environ["MOVEIT_EXPERIMENT_ROOT"]))
    parser.add_argument("--vnc", action="store_true", help="在 VNC 中显示 CAD 匹配驱动的执行过程。")
    parser.add_argument("--video-dir", type=Path, help="保存 VNC 主视角和腕部相机 MP4 的目录。")
    parser.add_argument(
        "--episode-dir", type=Path,
        help="回合 JSON 与掩码输出目录；默认 logs/episodes/cad_matching_latest。",
    )
    parser.add_argument(
        "--randomize-target", action="store_true",
        help="在任务前对黄色马克杯做可复现的随机桌面平移和 yaw 编辑。",
    )
    parser.add_argument("--random-seed", type=int, help="--randomize-target 的随机种子。")
    parser.add_argument(
        "--interactive-edit",
        action="store_true",
        help="任务开始前进入场景编辑：在 VNC 中移动工件，按 Enter 后从新位置重新感知和规划。",
    )
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
        help="主动观察最多执行的腕部视角数；on_path 在两帧成功后立即融合，证据不足时最多补到 3 帧，默认 3。",
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
        default=2.0,
        help="夹住物体后的轨迹时长倍率，默认 2.0；数值越小越快。",
    )
    arguments = parser.parse_args()
    if arguments.max_wrist_observations <= 0:
        raise ValueError("--max-wrist-observations 必须为正数")
    if arguments.interactive_edit and not arguments.vnc:
        raise ValueError("--interactive-edit 需要 --vnc，因为它通过 VNC 主窗口接收键盘输入")
    if arguments.randomize_target and arguments.random_seed is None:
        raise ValueError("--randomize-target 需要 --random-seed")
    if arguments.randomize_target and arguments.interactive_edit:
        raise ValueError("随机回归与手动 --interactive-edit 不能在同一回合混用")

    intent = RuleVlmAdapter().infer(arguments.instruction)
    if arguments.randomize_target and intent.target_model_id != "mug_with_handle_v1":
        raise ValueError("--randomize-target 当前仅支持黄色马克杯 mug_with_handle_v1")
    simulation = UnifiedPandaCupSimulation(arguments.root, create_renderers=False)
    scenario: dict[str, object] = {"randomized": False}
    if arguments.randomize_target:
        generator = np.random.default_rng(arguments.random_seed)
        delta_xy = generator.uniform(-0.07, 0.07, size=2)
        yaw_delta_rad = float(np.deg2rad(generator.uniform(-60.0, 60.0)))
        accepted, reason = simulation.move_editable_object("yellow_mug", tuple(delta_xy), yaw_delta_rad)
        scenario = {
            "randomized": True,
            "seed": arguments.random_seed,
            "requested_delta_xy_m": delta_xy.tolist(),
            "requested_yaw_delta_rad": yaw_delta_rad,
            "accepted": accepted,
            "result": reason,
        }
        if not accepted:
            raise RuntimeError(f"随机场景编辑被拒绝：{reason}")
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
        if arguments.interactive_edit:
            assert viewer is not None
            print("[场景编辑] 请在 VNC 主窗口完成编辑，按 Enter 后才会开始本轮感知和规划。", flush=True)
            if not viewer.edit_before_task():
                print("[场景编辑] 窗口已关闭；未启动任务。", flush=True)
                return
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
        # 全局 CAD matcher 只负责生成安全的近距观察工作区。对于强制 mesh
        # 诊断，闭爪前的抓取 pose 必须再经过接触 ROI 的独立验证与精修。
        grasp_roi_matcher = GraspRoiMatcher()
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
        # 固定相机 pose 只用于建立主动观察的粗 ROI、接近走廊和视角候选；最终
        # 抓取 pose 必须由腕部 RGB-D 多视角给出，不会把任何 MuJoCo 物体真值
        # 带入控制回路。
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
        # mesh sweep 会在 ROI 通过后填入“视觉已复核”的局部 yaw 候选；其余
        # 流程仍严格只使用一个 CAD pose。
        roi_reachability_poses = [pose]
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
                        if model.geometry_type == "mesh":
                            # 两阶段管线的近距帧只承担抓取 ROI：对每一帧重复
                            # FPFH/RANSAC 全物体搜索既不会参与最终 pose，又会让
                            # 主动观察被无关的全局模板计算阻塞。保留原始分割点云，
                            # 交给多视角 ROI 精配准一次性使用。
                            entry.update(
                                status="segmented_for_roi_fusion",
                                reason="mesh 腕部帧跳过单帧全局配准；仅用于双接触 ROI 多视角融合",
                                pixel_count=wrist_segmentation.pixel_count,
                                score=0.10 * observation_view.expected_contact_visibility,
                            )
                        else:
                            try:
                                wrist_pose = matcher_dispatcher.match(model, wrist_segmentation, wrist_frame)
                            except RuntimeError as matching_error:
                                if model.geometry_type not in {"mug_handle", "mesh"}:
                                    raise
                                # 单帧的局部遮挡会令杯把或通用表面模板缺少足够几何
                                # 证据；保留已经合格分割的 RGB-D 帧给多视角配准，而
                                # 不是把“单帧先定准 pose”变成隐式前置条件。
                                entry.update(
                                    status="segmented_for_fusion",
                                    reason=f"单帧通用配准未放行，保留点云给多视角验证：{matching_error}",
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
                                f"分割点={wrist_segmentation.pixel_count}；单帧未放行，保留给融合",
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
                    # 通用 mesh 不必等到穷举完所有候选才知道抓取区已被充分
                    # 解释。每增加一条独立相机基线就试一次 ROI；成功即停止
                    # sweep，证据不足时才继续寻找更互补的方位。
                    if (
                        model.geometry_type == "mesh"
                        and successful_observations >= 2
                        and len(templates) == 1
                    ):
                        try:
                            early_roi = grasp_roi_matcher.refine(
                                model,
                                templates[0],
                                coarse_pose,
                                [(item[1], item[2]) for item in matched_observations],
                            )
                        except RuntimeError:
                            pass
                        else:
                            entry["grasp_roi_preview"] = early_roi.as_dict()
                            active_observation["grasp_roi_preview"] = early_roi.as_dict()
                            print(
                                f"[主动观察融合] 已在 {successful_observations} 条独立基线满足抓取 ROI 门禁；"
                                "停止继续扫描。",
                                flush=True,
                            )
                            break
                if successful_observations < 2:
                    active_observation["failure_reason"] = "reobserve_exhausted"
                    raise RuntimeError(
                        "reobserve_exhausted：少于两个腕部观察位完成匹配，不能用单帧低约束结果进入抓取"
                    )
                if model.geometry_type in {"mesh", "mug_handle"}:
                    # 通用 mesh 与把手杯的单帧匹配都可能在杯身等重复局部极值之间
                    # 切换；因此不能拿这些单帧 pose 投票。合并各帧基座系点云后
                    # 只做一次 CAD 精配准，才是真正的多视角几何约束。
                    # 近距腕部帧刻意只覆盖抓取区，不能再强迫它们承担一次
                    # “解释整个 CAD”的 FPFH/RANSAC 任务。mesh 的全局工作区
                    # 初值来自固定相机 coarse_pose；腕部帧只交给 ROI 精配准。
                    if model.geometry_type == "mesh":
                        pose = coarse_pose
                    else:
                        pose = matcher_dispatcher.match_multiview(
                            model,
                            [
                                (item[1], item[2], float(item[3]["expected_contact_visibility"]))
                                for item in matched_observations
                            ],
                        )
                    if model.geometry_type == "mesh":
                        # sweep 同样遵循“整体 CAD 只负责近距、ROI 才能闭爪”的
                        # 约束。当前马克杯仅有一个与语义匹配的把手模板；若未来
                        # 有多个模板，应逐模板生成并验证候选，而不能跳过该门禁。
                        if len(templates) != 1:
                            raise RuntimeError("mesh 抓取 ROI 需要唯一的语义抓取模板")
                        roi_estimate = grasp_roi_matcher.refine(
                            model,
                            templates[0],
                            pose,
                            [(item[1], item[2]) for item in matched_observations],
                        )
                        pose = roi_estimate.pose
                        active_observation["grasp_roi"] = roi_estimate.as_dict()
                        roi_reachability_poses = [
                            _pose_with_local_z_yaw(pose, yaw_offset)
                            for yaw_offset in roi_estimate.reachability_validated_yaw_offsets_rad
                        ]
                        if not roi_reachability_poses:
                            # ``refine`` 已验证零偏移通过；此分支只作数据损坏
                            # 防护，绝不凭空添加未验证的 IK 候选。
                            raise RuntimeError("抓取 ROI 未留下可达性复核 pose 候选")
                        active_observation["reachability_validated_pose_count"] = len(roi_reachability_poses)
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
                        # 单帧 mesh pose 可以因可见杯身近似对称而被安全门禁拒绝；
                        # 它不是多视角融合的前置条件。这里验证的是两条独立相机
                        # 基线确实参与了融合；真正的接触区门禁已由上面的 ROI
                        # matcher 在每条腕部点云中独立验证。
                        fused_support_count = sum(
                            item[3].get("contributes_new_camera_baseline", False)
                            for item in matched_observations
                        )
                        if fused_support_count < 2:
                            active_observation["failure_reason"] = "fused_pose_insufficient_independent_support"
                            active_observation["fused_support_count"] = fused_support_count
                            raise RuntimeError(
                                "reobserve_exhausted：融合 mesh pose 未获得至少两个独立腕部视角支持，拒绝猜测抓取姿态"
                            )
                        active_observation["critical_region_support"] = pose.critical_region_support
                        active_observation["critical_region_margin"] = pose.critical_region_margin
                        active_observation["critical_region_baseline_count"] = pose.registration_diagnostics.get(
                            "critical_region_baseline_count"
                        ) if pose.registration_diagnostics else None
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
            if model.geometry_type == "mesh" and arguments.active_observation and arguments.active_observation_mode == "on_path":
                # 粗配准尚未判定把手 yaw，不能据此把腕相机送往“猜测的把手”。
                # 先到物体中心上方的中性安全观察位；ROI 在腕部画面中锁定两片
                # 接触面后，on-path 分支会重新生成真正的把手抓取位。
                candidates = [
                    (template, pose.position_base_m + np.array([0.0, 0.0, position[2] - pose.position_base_m[2]]), orientation)
                    for template, position, orientation in candidates
                ]
            # 对 mesh sweep，候选不是“任意偏转的 IK 搜索”：每个 pose 都已经
            # 在 GraspRoiMatcher 中重新满足了两片对向接触面的支持和残差门禁。
            candidate_records = [
                (candidate_template, candidate_position, candidate_orientation, candidate_pose)
                for candidate_pose in (
                    roi_reachability_poses
                    if model.geometry_type == "mesh" and arguments.active_observation_mode == "sweep"
                    else [pose]
                )
                for candidate_template, candidate_position, candidate_orientation
                in grasp_pose_candidates(candidate_pose, templates)
            ]
            if model.geometry_type == "mesh" and arguments.active_observation and arguments.active_observation_mode == "on_path":
                candidate_records = [
                    (candidate_template, pose.position_base_m + np.array([0.0, 0.0, candidate_position[2] - pose.position_base_m[2]]), candidate_orientation, pose)
                    for candidate_template, candidate_position, candidate_orientation, _candidate_pose in candidate_records
                ]
            # 横向把手抓取框（手腕绕工具 Z 旋转 90°）下，MoveIt ``panda_hand``
            # 参考点相对 MuJoCo 指尖工作点低约 82 mm，故目标参考点须向上补偿。
            # 该外参来自闭爪前记录的真实 pad 位姿，避免把手上方/下方的伪接触。
            # CAD 标注与接触判定始终使用真实指尖工作点坐标系。
            for candidate_template, candidate_position, candidate_orientation, _candidate_pose in candidate_records:
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
            for candidate_template, candidate_position, candidate_orientation, candidate_pose in candidate_records:
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
                        candidate_pose,
                    )
                    break
                if selected is not None:
                    break
            if selected is None:
                raise RuntimeError(f"MoveIt 未找到可执行的 CAD 抓取标注候选：{rejected_candidates}")
            template, grasp_position, grasp_orientation, pregrasp_trajectory, reachability_offset, selected_pose = selected
            pose = selected_pose
            # 备用项仅在已经到达安全预抓取位、首选最终接近位 IK 无解时才会
            # 尝试；它们均来自同一帧组的 ROI 复核，且仍要通过实际双侧接触。
            approach_fallbacks = [
                (candidate_template, candidate_position + reachability_offset, candidate_orientation, candidate_pose)
                for candidate_template, candidate_position, candidate_orientation, candidate_pose in candidate_records
                if not (
                    candidate_template.template_id == template.template_id
                    and np.allclose(candidate_position + reachability_offset, grasp_position)
                    and np.allclose(candidate_orientation, grasp_orientation)
                )
            ]
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
                # 不再先做大范围环绕扫描：粗预抓取本来就是必经安全走廊。到达后
                # 直接用腕部 RGB-D 做局部闭环；固定相机仅提供本段安全走廊和深度
                # ROI 的粗中心，不进入最终 pose 融合或 yaw 裁决。相机必须主动
                # 注视把手，而不能假定抓取姿态下腕部光轴天然朝向目标。每次移动
                # 都从当前关节状态重新由 MoveIt 规划，不能在线篡改已批准的 IK
                # 轨迹。
                local_observations: list[tuple[object, object, float]] = []
                refined_pose = None
                local_failure = None
                wrist_frame = simulation.cameras(("wrist",))["wrist"]
                hand_from_camera = np.linalg.inv(_base_from_body(simulation, "hand")) @ wrist_frame.base_from_camera
                contact_centers, contact_normals = _contact_geometry_base(coarse_pose, (template,))
                # 观察从粗预抓取附近开始；每一轮候选都以当前腕相机为中心，只作
                # 厘米级移动。这样“补看”是局部视觉伺服，而不是为了凑基线绕到
                # 桌子的另一侧。
                maximum_observation_hand_travel_m = 0.055
                selected_camera_positions: list[np.ndarray] = []
                covered_contact_indices: set[int] = set()
                maximum_wrist_observations = arguments.max_wrist_observations
                for observation_index in range(maximum_wrist_observations):
                    current_hand_position = executor.body_position("hand")
                    current_wrist_frame = simulation.cameras(("wrist",))["wrist"]
                    local_views = generate_local_observation_views(
                        coarse_pose.position_base_m,
                        contact_centers,
                        contact_normals,
                        current_wrist_frame.base_from_camera[:3, 3],
                        hand_from_camera,
                        current_wrist_frame.base_from_camera[:3, :3],
                    )
                    observation_view = None
                    observation_trajectory = None
                    observation_hand_travel_m = None
                    ranked_local_views = _rank_wrist_observation_views(
                        local_views,
                        selected_camera_positions,
                        covered_contact_indices,
                        current_hand_position,
                    )
                    if observation_index >= 2:
                        ranked_local_views.sort(key=lambda view: ("_roll_+15" not in view.view_id, view.view_id))
                    # 专用 matcher 已有可靠把手方向，粗预抓取位就是第一条安全
                    # 近距观察位。mesh 的粗 yaw 尚未可信，首帧便应开始横向扫描，
                    # 避免把 3 帧预算浪费在假想把手方向的原地画面。
                    if observation_index == 0 and model.geometry_type != "mesh":
                        ranked_local_views = sorted(
                            ranked_local_views,
                            key=lambda view: (view.view_id != "local_hold", view.view_id),
                        )
                    for candidate_view in ranked_local_views:
                        same_camera_position = any(
                            np.linalg.norm(candidate_view.camera_position_base_m - previous) < 0.006
                            for previous in selected_camera_positions
                        )
                        # 前两帧必须建立真实的空间基线；若它们仍不能分辨把手，
                        # 第三帧允许同一相机位置的另一腕部滚转来解除机器人自身
                        # 遮挡，而不是强行跑向低可见性方位。
                        if same_camera_position and observation_index < 2:
                            continue
                        candidate_hand_travel_m = float(
                            np.linalg.norm(candidate_view.hand_position_base_m - current_hand_position)
                        )
                        if candidate_hand_travel_m > maximum_observation_hand_travel_m:
                            continue
                        if candidate_hand_travel_m < 0.002:
                            observation_view = candidate_view
                            observation_hand_travel_m = candidate_hand_travel_m
                            break
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
                    if observation_view is None:
                        raise RuntimeError("on_path_reobserve_exhausted：没有可达的近距把手观察位")
                    executor.set_display_state(f"On-path wrist observation {observation_index + 1}")
                    # 观察位会有明显的腕部转向；采用低于常规 1.7 倍的温和加速，
                    # 在保持跟踪裕量的同时避免为一次局部观测等待完整原始轨迹时长。
                    if observation_trajectory is not None:
                        _execute_trajectory(executor, observation_trajectory, speed_scale=1.0)
                    executor._event(
                        "on_path_wrist_observation_complete",
                        view_id=observation_view.view_id,
                        expected_contact_visibility=observation_view.expected_contact_visibility,
                        hand_travel_m=observation_hand_travel_m,
                    )
                    selected_camera_positions.append(observation_view.camera_position_base_m)
                    covered_contact_indices.update(observation_view.visible_contact_indices)
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
                        # 分割面积越充分，越值得在多视角几何融合中占更高权重；这
                        # 是实时画面质量，而不是固定相机粗 pose 的置信度。
                        "fusion_weight": float(
                            observation_view.expected_contact_visibility
                            * (0.35 + 0.65 * wrist_segmentation.confidence)
                        ),
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
                        (wrist_segmentation, wrist_frame, float(entry["fusion_weight"]))
                    )
                    if len(local_observations) < 2:
                        entry.update(status="collected_for_wrist_fusion")
                        active_observation["views"].append(entry)
                        print(
                            "[路径内腕部校正] 已采集第 1 帧；固定相机不参与最终 pose，"
                            "继续选择互补腕部视角。",
                            flush=True,
                        )
                        continue
                    try:
                        if model.geometry_type in {"mesh", "mug_handle"}:
                            # 两帧及以上腕部 RGB-D 是最终 CAD pose 的唯一来源；
                            # 固定相机的粗结果不会作为观察、yaw 或置信度的投票者。
                            matching_observations = (
                                local_observations
                                if model.geometry_type == "mug_handle"
                                else [item[:2] for item in local_observations]
                            )
                            if model.geometry_type == "mesh":
                                # 两条近距腕部帧只看抓取区时，重新运行整体 FPFH 会
                                # 因为看不全杯身而退化。使用固定相机的 CAD 粗位姿
                                # 作为受限 ROI Kabsch 精修初值才是该阶段的职责边界。
                                roi_estimate = grasp_roi_matcher.refine(
                                    model,
                                    template,
                                    coarse_pose,
                                    [item[:2] for item in local_observations],
                                )
                                refined_pose = roi_estimate.pose
                                entry["grasp_roi"] = roi_estimate.as_dict()
                                active_observation["grasp_roi"] = roi_estimate.as_dict()
                            else:
                                refined_pose = matcher_dispatcher.match_multiview(model, matching_observations)
                        else:
                            refined_pose = matcher_dispatcher.match(model, wrist_segmentation, wrist_frame)
                        refined_candidates = grasp_pose_candidates(refined_pose, (template,))
                        if not refined_candidates:
                            raise RuntimeError("refined_grasp_missing：腕部 pose 未生成当前 CAD 抓取框")
                        _, candidate_grasp_position, candidate_orientation = refined_candidates[0]
                        candidate_grasp_position = candidate_grasp_position + reachability_offset
                        candidate_tcp_offset = _rotate_vector_by_quaternion(
                            np.array([0.0, 0.0, -0.082]), candidate_orientation
                        )
                        candidate_moveit_grasp = candidate_grasp_position + candidate_tcp_offset
                        candidate_moveit_pregrasp = (
                            candidate_grasp_position
                            + np.array([0.005, 0.0, template.approach_clearance_m])
                            + candidate_tcp_offset
                        )
                        # 接受 pose 前先预检“当前状态→预抓取→最终接近”。规划
                        # 不可达属于候选/工作空间问题，不能冒充夹爪接触失败。
                        _validate_full_grasp_path(
                            client,
                            executor,
                            catalog,
                            target_object_id,
                            candidate_moveit_pregrasp,
                            candidate_moveit_grasp,
                            candidate_orientation,
                        )
                        entry.update(
                            status="matched_and_reachable",
                            geometry_confidence=refined_pose.geometry_confidence,
                            matching_method=refined_pose.matching_method,
                            full_grasp_path="pregrasp_and_approach_validated",
                        )
                        active_observation["views"].append(entry)
                        break
                    except (RuntimeError, TimeoutError) as error:
                        local_failure = str(error)
                        entry.update(
                            status="goal_unreachable" if isinstance(error, TimeoutError) else "requires_local_reobserve",
                            reason=local_failure,
                        )
                        active_observation["views"].append(entry)
                        if observation_index + 1 >= maximum_wrist_observations:
                            raise RuntimeError(
                                "on_path_reobserve_exhausted：近距腕部校正未获得可达且一致的抓取 pose："
                                f"{local_failure}"
                            ) from error
                        print(
                            f"[路径内腕部校正] 第 {observation_index + 1} 帧未获得可执行抓取 pose，"
                            "仅小幅移动腕部到下一局部观察位后重试。",
                            flush=True,
                        )
                if refined_pose is None:
                    raise RuntimeError(f"on_path_reobserve_exhausted：{local_failure}")
                pose = refined_pose
                segmentation = local_observations[-1][0]
                active_observation.update(
                    selected_source="on_path:wrist_multiview",
                    consensus_count=len(local_observations),
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
                # 最后 10 cm 不盲从刚才的静态 plan：在已校正的预抓取位读取新的
                # 腕部 RGB-D，若它与前两帧腕部融合后的 pose 只存在小幅、可信的
                # 漂移，就以该测量重新生成末段目标。大跳变不以固定相机兜底，而
                # 是保留已通过多腕部视角验证的 pose，避免单帧噪声突然扭转夹爪。
                servo_entry: dict[str, object] = {"view_id": "on_path_servo_checkpoint"}
                try:
                    servo_frame = simulation.cameras(("wrist",))["wrist"]
                    lateral_radius = 0.5 * float(np.linalg.norm(model.dimensions_m[:2])) + 0.025
                    servo_segmentation = ColorThresholdSegmenter().segment(
                        servo_frame.rgb,
                        intent,
                        frame=servo_frame,
                        expected_position_base_m=pose.position_base_m,
                        lateral_radius_m=lateral_radius,
                        z_min_m=float(pose.position_base_m[2] - model.height_m / 2.0 + 0.004),
                        z_max_m=float(pose.position_base_m[2] + model.height_m / 2.0 + 0.025),
                    )
                    servo_entry.update(
                        camera_position_base_m=servo_frame.base_from_camera[:3, 3].tolist(),
                        pixel_count=servo_segmentation.pixel_count,
                        fusion_weight=float(0.35 + 0.65 * servo_segmentation.confidence),
                    )
                    if model.geometry_type == "mug_handle":
                        servo_pose = matcher_dispatcher.match_multiview(
                            model,
                            local_observations
                            + [(servo_segmentation, servo_frame, float(servo_entry["fusion_weight"]))],
                        )
                    elif model.geometry_type == "mesh":
                        servo_roi_estimate = grasp_roi_matcher.refine(
                            model,
                            template,
                            pose,
                            [item[:2] for item in local_observations] + [(servo_segmentation, servo_frame)],
                        )
                        servo_pose = servo_roi_estimate.pose
                        servo_entry["grasp_roi"] = servo_roi_estimate.as_dict()
                        active_observation["grasp_roi"] = servo_roi_estimate.as_dict()
                    else:
                        servo_pose = matcher_dispatcher.match(model, servo_segmentation, servo_frame)
                    position_delta_m = float(np.linalg.norm(servo_pose.position_base_m - pose.position_base_m))
                    yaw_delta_rad = _yaw_distance_rad(servo_pose.object_yaw_rad, pose.object_yaw_rad)
                    servo_entry.update(
                        candidate_position_delta_m=position_delta_m,
                        candidate_yaw_delta_rad=yaw_delta_rad,
                        geometry_confidence=servo_pose.geometry_confidence,
                    )
                    # 这是最后一帧、单个观测的微调关口，不应把多视角融合后的
                    # 可达抓取姿态改成另一个实质不同的候选。较大的差异应由前面
                    # 的局部 NBV 重观测解决；这里仅允许传感器级的小漂移。
                    if position_delta_m > 0.003 or yaw_delta_rad > np.deg2rad(3.0):
                        servo_entry.update(
                            status="rejected_large_delta",
                            reason="末段单帧超出 3mm/3deg 微调范围，保持已验证的腕部多视角结果",
                        )
                    else:
                        refined_candidates = grasp_pose_candidates(servo_pose, (template,))
                        if not refined_candidates:
                            raise RuntimeError("腕部末段校正后未生成原 CAD 抓取框")
                        candidate_template, refined_grasp_position, candidate_orientation = refined_candidates[0]
                        candidate_grasp_position = refined_grasp_position + reachability_offset
                        candidate_tcp_offset = _rotate_vector_by_quaternion(
                            np.array([0.0, 0.0, -0.082]), candidate_orientation
                        )
                        candidate_moveit_grasp = candidate_grasp_position + candidate_tcp_offset
                        candidate_moveit_pregrasp = (
                            candidate_grasp_position
                            + np.array([0.005, 0.0, candidate_template.approach_clearance_m])
                            + candidate_tcp_offset
                        )
                        # 微调虽小，也必须重新验证完整接近段；不可达时仍保留前面
                        # 已验证的融合 pose，而不把规划问题混同为物理夹取失败。
                        _validate_full_grasp_path(
                            client,
                            executor,
                            catalog,
                            target_object_id,
                            candidate_moveit_pregrasp,
                            candidate_moveit_grasp,
                            candidate_orientation,
                        )
                        pose = servo_pose
                        template = candidate_template
                        grasp_orientation = candidate_orientation
                        grasp_position = candidate_grasp_position
                        tcp_offset = candidate_tcp_offset
                        servo_entry.update(status="accepted", matching_method=servo_pose.matching_method)
                        active_observation["selected_source"] = "on_path:wrist_multiview+servo_checkpoint"
                        active_observation["fused_pose"] = {
                            "position_base_m": pose.position_base_m.tolist(),
                            "object_yaw_rad": pose.object_yaw_rad,
                            "geometry_confidence": pose.geometry_confidence,
                            "surface_residual_m": pose.surface_residual_m,
                            "point_count": pose.point_count,
                        }
                        executor._event(
                            "on_path_visual_servo_correction_complete",
                            correction_position_m=position_delta_m,
                            correction_yaw_rad=yaw_delta_rad,
                        )
                except (RuntimeError, TimeoutError) as error:
                    servo_entry.update(status="unavailable", reason=str(error))
                active_observation["views"].append(servo_entry)
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
            # 而不是把第一次闭合失败直接解释为“视觉可以任意猜”。腕部重观测后
            # 仍可能留下约 1 cm 横向误差，故沿夹爪闭合轴覆盖 ±12 mm、沿把手
            # 径向覆盖 ±4 mm；每一次都必须通过同一对真实碰撞面的双侧接触门禁。
            closing_axis = template.jaw_closing_axis_object / np.linalg.norm(template.jaw_closing_axis_object)
            tangent_axis = np.cross(np.array([0.0, 0.0, 1.0]), closing_axis)
            if np.linalg.norm(tangent_axis) < 1e-6:
                tangent_axis = np.cross(np.array([1.0, 0.0, 0.0]), closing_axis)
            tangent_axis /= np.linalg.norm(tangent_axis)
            pair = template.opposing_contact_pair
            separation = float(np.linalg.norm(pair.positive_surface.center_object_m - pair.negative_surface.center_object_m))
            step = min(0.006, separation / 2.0)
            grasp_search_offsets_object = tuple(
                closing_offset * closing_axis + tangent_offset * tangent_axis
                for closing_offset in (0.0, -step, step, -2.0 * step, 2.0 * step)
                for tangent_offset in (0.0, 0.004, -0.004)
            )
            max_grasp_attempts = len(grasp_search_offsets_object)
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
                            grasp_orientation,
                        )
                        _execute_trajectory(executor, retry_pregrasp)
                    except TimeoutError as error:
                        executor._event(
                            "goal_unreachable",
                            stage="retry_pregrasp",
                            attempt=attempt,
                            reason=str(error),
                        )
                        if attempt < max_grasp_attempts:
                            # 这是同一经 ROI 验证的夹持框内、有限集合中的一个
                            # 微调点无 IK 解；保持在上一次安全预抓取位并继续验证
                            # 其它微调点，而非把一个局部不可达点错误升级为视觉失败。
                            print(
                                f"[抓取微调跳过] 第 {attempt} 个预抓取微调位不可达，继续下一个有限候选。",
                                flush=True,
                            )
                            continue
                        raise RuntimeError(
                            "goal_unreachable：接触微调的预抓取位不可达；"
                            "这不是双侧接触失败，不继续枚举夹持偏移"
                        ) from error
                    executor.set_display_state(f"CAD grasp retry {attempt}: approach")
                time.sleep(0.5)
                client.publish_joint_state(executor.joint_positions())
                time.sleep(0.25)
                approach_start_joint_positions = executor.joint_positions()
                approach_trajectory = None
                try:
                    # 抓取框定义了物体相对手爪的偏置。即使只要求末端位置到达，
                    # 自由的腕部姿态也会旋转这段偏置，令长方体在托盘外侧松开；
                    # 因此接近、抬升和放置全程保持 CAD 的完整抓取姿态。
                    try:
                        approach_trajectory = client.request(attempt_moveit_grasp_position, grasp_orientation)
                    except TimeoutError as primary_error:
                        # 预抓取已通过而末端接近的 goal state 无 IK 解时，不把
                        # 一个离散 ROI yaw 样本误当成视觉失败。仅在同一 ROI 证据
                        # 已复核的微小 yaw 邻域中查找可达替代项；不可达仍安全失败。
                        fallback_selected = False
                        if model.geometry_type == "mesh" and attempt == 1:
                            for (
                                fallback_template,
                                fallback_grasp_position,
                                fallback_orientation,
                                fallback_pose,
                            ) in approach_fallbacks:
                                fallback_tcp_offset = _rotate_vector_by_quaternion(
                                    np.array([0.0, 0.0, -0.082]), fallback_orientation
                                )
                                fallback_moveit_position = fallback_grasp_position + fallback_tcp_offset
                                client.publish_joint_state(executor.joint_positions())
                                time.sleep(0.25)
                                try:
                                    fallback_trajectory = client.request(
                                        fallback_moveit_position, fallback_orientation
                                    )
                                except TimeoutError:
                                    continue
                                template = fallback_template
                                pose = fallback_pose
                                grasp_position = fallback_grasp_position
                                grasp_orientation = fallback_orientation
                                tcp_offset = fallback_tcp_offset
                                moveit_grasp_position = fallback_moveit_position
                                attempt_grasp_position = fallback_grasp_position
                                attempt_moveit_grasp_position = fallback_moveit_position
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
                                active_observation["reachability_selected_local_yaw_offset_rad"] = (
                                    pose.registration_diagnostics or {}
                                ).get("reachability_selected_local_yaw_offset_rad", 0.0)
                                executor._event(
                                    "roi_reachability_fallback_selected",
                                    local_yaw_offset_rad=active_observation[
                                        "reachability_selected_local_yaw_offset_rad"
                                    ],
                                )
                                print(
                                    "[ROI/MoveIt 联合选择] 首选接近位不可达；已切换到"
                                    "同一 ROI 证据支持的局部朝向。",
                                    flush=True,
                                )
                                approach_trajectory = fallback_trajectory
                                fallback_selected = True
                                break
                        if not fallback_selected:
                            raise primary_error
                    _execute_trajectory(executor, approach_trajectory)
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
                    simulation.release_object_rest(target_object_id)
                    print("[场景静置] 双侧接触与 CAD 相对面验证已确认，解除目标静置约束", flush=True)
                    # 后续抬升、放置和物体在手坐标系中的相对位姿必须使用真正
                    # 通过接触验证的微调抓取位，而不是初始视觉候选。
                    grasp_position = attempt_grasp_position
                    moveit_grasp_position = attempt_moveit_grasp_position
                    grasp_confirmed = True
                    break
                except TimeoutError as error:
                    executor._event(
                        "goal_unreachable",
                        stage="approach",
                        attempt=attempt,
                        reason=str(error),
                    )
                    if attempt < max_grasp_attempts:
                        # 与预抓取相同：不重采样场景、不放宽 ROI，只跳过这个
                        # MoveIt 已明确拒绝的毫米级接触候选。
                        print(
                            f"[抓取微调跳过] 第 {attempt} 个末段接近位不可达，继续下一个有限候选。",
                            flush=True,
                        )
                        continue
                    raise RuntimeError(
                        "goal_unreachable：最终接近位不可达；"
                        "这不是双侧接触失败，不继续枚举夹持偏移"
                    ) from error
                except RuntimeError as error:
                    if attempt == max_grasp_attempts:
                        raise
                    if approach_trajectory is not None:
                        # 闭合失败后夹爪可能正与目标接触。先沿刚刚验证过的轨迹
                        # 反向退出到安全预抓取构型，再为下一局部偏移重新规划；否则
                        # MoveIt 会把接触中的当前构型判为无效起点并让所有重试超时。
                        executor.set_gripper(opened=True)
                        executor.set_display_state(f"CAD grasp retry {attempt}: safe retreat")
                        _retreat_observation_trajectory(
                            executor, approach_trajectory, approach_start_joint_positions
                        )
                    print(
                        f"[抓取重试] 第 {attempt} 次未形成双侧接触；"
                        f"CAD 局部偏移=({offset_object[0]:.3f}, {offset_object[1]:.3f}, {offset_object[2]:.3f})m；原因={error}",
                        flush=True,
                    )
            if not grasp_confirmed:
                raise RuntimeError(f"{target_object_id} 在 {max_grasp_attempts} 次真实抓取重试后仍未形成双侧接触")
            attached_pose = _attached_pose_in_hand(
                pose.position_base_m,
                pose.transform_base_object[:3, :3],
                moveit_grasp_position,
                grasp_orientation,
            )
            object_grasp_offset = pose.transform_base_object[:3, :3].T @ (grasp_position - pose.position_base_m)
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
                transport_trajectory = None
                last_transport_error = None
                # 物体已夹持后可整体小幅转向；物体与夹爪的相对抓取关系保持
                # 固定。所有 CAD 使用同一组有限候选，逐项做完整碰撞规划。
                for yaw_delta_deg in (0., -5., 5., -10., 10.):
                    candidate_pose = _pose_with_local_z_yaw(pose, np.deg2rad(yaw_delta_deg))
                    _, _, candidate_orientation = grasp_pose_candidates(candidate_pose, (template,))[0]
                    candidate_tcp = _rotate_vector_by_quaternion(np.array([0., 0., -0.082]), candidate_orientation)
                    candidate_placement = catalog.physical_placement_targets(model, template, candidate_pose.transform_base_object[:3, :3])
                    if stage == "lift":
                        object_center = pose.position_base_m + np.array([0., 0., template.lift_clearance_m])
                    elif stage == "place_above":
                        object_center = candidate_placement.object_release_center_base_m + np.array([0., 0., candidate_placement.gripper_above_base_m[2] - candidate_placement.gripper_release_base_m[2]])
                    else:
                        object_center = candidate_placement.object_release_center_base_m
                    candidate_target = object_center + candidate_pose.transform_base_object[:3, :3] @ object_grasp_offset + candidate_tcp
                    try:
                        transport_trajectory = client.request(candidate_target, candidate_orientation)
                    except TimeoutError as error:
                        last_transport_error = error
                        continue
                    pose = candidate_pose
                    grasp_orientation = candidate_orientation
                    placement = candidate_placement
                    executor._event("transport_orientation_selected", stage=stage, local_yaw_offset_deg=yaw_delta_deg)
                    break
                if transport_trajectory is None:
                    raise RuntimeError(f"通用搬运阶段 {stage} 的全部姿态候选不可达：{last_transport_error}")
                _execute_trajectory(executor, transport_trajectory)
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
            output = arguments.episode_dir or (arguments.root / "logs" / "episodes" / "cad_matching_latest")
            if not output.is_absolute():
                output = arguments.root / output
            output.mkdir(parents=True, exist_ok=True)
            _write_pgm(output / "target_mask.pgm", segmentation.mask)
            payload = {
                "instruction": arguments.instruction,
                "stage": "cad_model_matching_pick_place_baseline",
                "semantic_result": intent.as_dict(),
                "target_object_id": target_object_id,
                "scenario": scenario,
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
