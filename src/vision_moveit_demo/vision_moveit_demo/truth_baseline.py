"""阶段 2 真值任务基线：MoveIt 分段规划，MuJoCo 执行与抓夹事件验证。"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, PlanningScene, RobotTrajectory
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive

from .stage2_executor import MujocoTaskExecutor
from .unified_scene import UnifiedPandaCupSimulation
from .planning_scene_policy import PlanningPhase, TaskPlanningScenePolicy


class MoveItTrajectoryClient(Node):
    def __init__(self) -> None:
        super().__init__("truth_baseline_client")
        self.trajectory_publisher = self.create_publisher(PoseStamped, "/vision_moveit/target_pose", 10)
        self.scene_publisher = self.create_publisher(PlanningScene, "/vision_moveit/planning_scene_diff", 10)
        self.joint_publisher = self.create_publisher(JointState, "/vision_moveit/sim_joint_states", 10)
        self.trajectories: list[RobotTrajectory] = []
        self._published_world_object_ids: set[str] = set()
        self._published_attached_object_ids: set[str] = set()
        self.create_subscription(RobotTrajectory, "/vision_moveit/planned_trajectory", self.trajectories.append, 10)

    def publish_joint_state(self, positions: dict[str, float]) -> None:
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = [f"panda_joint{index}" for index in range(1, 8)]
        message.position = [positions[f"joint{index}"] for index in range(1, 8)]
        self.joint_publisher.publish(message)

    def publish_scene(self, snapshot, grasp_target: str | None = None) -> None:
        positions = {name: value["position_base_m"] for name, value in snapshot.planning_scene_objects.items()}
        positions["table"] = np.array([0.725, 0.0, 0.36])
        self.publish_scene_positions(positions, grasp_target=grasp_target)

    def publish_scene_positions(
        self,
        positions: dict[str, np.ndarray],
        grasp_target: str | None = None,
        temporary_exclusions: tuple[str, ...] = (),
        attached_object: str | None = None,
        attached_pose: Pose | None = None,
    ) -> None:
        """发布显式场景坐标；视觉闭环调用方不得传入 MuJoCo 真值快照。"""
        scene = PlanningScene(is_diff=True)
        dimensions = {
            # Panda 基座安装在桌面边缘。MoveIt 场景仅放入基座前方的作业台面，
            # 否则完整桌板会与其固定底座/立柱相交，使起始状态非法。
            "table": (SolidPrimitive.BOX, [0.85, 0.96, 0.07]),
            "tray": (SolidPrimitive.BOX, [0.26, 0.22, 0.094]),
            "red_cylinder": (SolidPrimitive.CYLINDER, [0.14, 0.038]),
            "green_cylinder": (SolidPrimitive.CYLINDER, [0.14, 0.038]),
            "blue_cylinder": (SolidPrimitive.CYLINDER, [0.14, 0.038]),
            "purple_cube": (SolidPrimitive.BOX, [0.07, 0.07, 0.07]),
            "magenta_block": (SolidPrimitive.BOX, [0.14, 0.055, 0.06]),
            # 预抓取阶段只需阻止路径横穿杯身。用实际杯身圆柱而非包住把手
            # 与空腔的实心大盒，避免在把手上方的合法预抓取位被误判为碰撞。
            "yellow_mug": (SolidPrimitive.CYLINDER, [0.10, 0.033]),
        }
        # 任务脚本每回合启动独立的规划服务。这里直接省略目标和临时排除件即可；
        # 不能对一个尚不存在的 CollisionObject 发 REMOVE，否则 MoveIt 会拒绝
        # 整个 PlanningScene diff，表面上看像是 IK/可达性故障。
        excluded = set(temporary_exclusions)
        if grasp_target is not None:
            if grasp_target not in dimensions:
                raise ValueError(f"未知抓取目标：{grasp_target}")
            excluded.add(grasp_target)
        for excluded_name in sorted(excluded):
            if excluded_name not in dimensions:
                raise ValueError(f"未知的临时碰撞排除对象：{excluded_name}")
        desired_world_names = set(dimensions) - excluded
        desired_world_ids = {f"stage2_{name}" for name in desired_world_names}
        for stale_id in sorted(self._published_world_object_ids - desired_world_ids):
            object_message = CollisionObject()
            object_message.id = stale_id
            object_message.operation = CollisionObject.REMOVE
            scene.world.collision_objects.append(object_message)
        desired_attached_id = f"stage2_{attached_object}" if attached_object is not None else None
        if desired_attached_id is None:
            stale_attached_ids = self._published_attached_object_ids
        else:
            stale_attached_ids = self._published_attached_object_ids - {desired_attached_id}
        for stale_id in sorted(stale_attached_ids):
            attached = AttachedCollisionObject()
            attached.link_name = "panda_hand"
            attached.object.id = stale_id
            attached.object.operation = CollisionObject.REMOVE
            scene.robot_state.attached_collision_objects.append(attached)
        for name, (shape_type, shape_dimensions) in dimensions.items():
            # 接近阶段允许末端与目标杯建立接触；其它杯子、桌面和托盘仍进入碰撞场景。
            if name in excluded:
                continue
            object_message = CollisionObject()
            object_message.id = f"stage2_{name}"
            object_message.header.frame_id = "panda_link0"
            primitive = SolidPrimitive(type=shape_type, dimensions=shape_dimensions)
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = map(float, positions[name])
            pose.orientation.w = 1.0
            object_message.primitives = [primitive]
            object_message.primitive_poses = [pose]
            object_message.operation = CollisionObject.ADD
            scene.world.collision_objects.append(object_message)
        if attached_object is not None:
            if attached_object not in dimensions:
                raise ValueError(f"未知的附着对象：{attached_object}")
            if attached_pose is None:
                raise ValueError("附着对象必须提供相对于 panda_hand 的位姿")
            shape_type, shape_dimensions = dimensions[attached_object]
            attached = CollisionObject()
            attached.id = f"stage2_{attached_object}"
            attached.header.frame_id = "panda_hand"
            attached.primitives = [SolidPrimitive(type=shape_type, dimensions=shape_dimensions)]
            attached.primitive_poses = [attached_pose]
            attached.operation = CollisionObject.ADD
            attached_message = AttachedCollisionObject()
            attached_message.link_name = "panda_hand"
            attached_message.object = attached
            scene.robot_state.attached_collision_objects.append(attached_message)
        self.scene_publisher.publish(scene)
        self._published_world_object_ids = desired_world_ids
        self._published_attached_object_ids = (
            {desired_attached_id} if desired_attached_id is not None else set()
        )

    def publish_task_scene(
        self,
        positions: dict[str, np.ndarray],
        target_object: str,
        phase: PlanningPhase,
        attached_pose: Pose | None = None,
    ) -> None:
        """按任务目标和阶段发布碰撞场景，不由调用方手写颜色分支。"""
        policy = TaskPlanningScenePolicy(target_object=target_object, phase=phase)
        self.publish_scene_positions(
            positions,
            grasp_target=None,
            temporary_exclusions=tuple(sorted(policy.world_exclusions())),
            attached_object=policy.attached_object(),
            attached_pose=attached_pose,
        )

    def request(
        self,
        position: np.ndarray,
        orientation_xyzw: np.ndarray | None = None,
        timeout_s: float = 15.0,
        position_only: bool = False,
    ) -> RobotTrajectory:
        target = PoseStamped()
        target.header.frame_id = "panda_link0_position_only" if position_only else "panda_link0"
        target.header.stamp = self.get_clock().now().to_msg()
        target.pose.position.x, target.pose.position.y, target.pose.position.z = map(float, position)
        # 未指定时保持历史基线的向下夹爪；CAD 抓取标注可传入完整 6D 姿态。
        orientation = np.array([1.0, 0.0, 0.0, 0.0]) if orientation_xyzw is None else orientation_xyzw
        target.pose.orientation.x, target.pose.orientation.y, target.pose.orientation.z, target.pose.orientation.w = map(float, orientation)
        print(
            "[MoveIt 请求] "
            f"位置=({position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f})，"
            f"四元数_xyzw=({orientation[0]:.4f}, {orientation[1]:.4f}, {orientation[2]:.4f}, {orientation[3]:.4f})，"
            f"模式={'仅位置' if position_only else '完整姿态'}，超时={timeout_s:.1f}s",
            flush=True,
        )
        # 串行任务在发送前清空已处理轨迹；部分 ROS 发行版会重写 trajectory header
        # 的时间戳，故不能把它作为唯一关联键。
        self.trajectories.clear()
        self.trajectory_publisher.publish(target)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.trajectories:
                return self.trajectories[-1]
        raise TimeoutError(
            "等待 MoveIt 规划轨迹超时；请查看同一终端中紧邻的 [MoveIt 请求] 目标位姿，"
            "以及 VNC 启动脚本随后打印的 MoveIt 服务末尾日志（其中包含 GOAL_STATE_INVALID、碰撞或 IK 原因）"
        )


def _execute_trajectory(executor: MujocoTaskExecutor, trajectory: RobotTrajectory) -> None:
    names = trajectory.joint_trajectory.joint_names
    waypoint_times_s: list[float] = []
    waypoint_positions: list[dict[str, float]] = []
    for point in trajectory.joint_trajectory.points:
        target = executor.joint_positions()
        for name, value in zip(names, point.positions):
            if name.startswith("panda_joint"):
                target[name.replace("panda_", "")] = float(value)
        waypoint_positions.append(target)
        waypoint_times_s.append(float(point.time_from_start.sec) + float(point.time_from_start.nanosec) * 1e-9)
    executor.execute_timed_joint_trajectory(waypoint_times_s, waypoint_positions)


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段 2：真值抓取放置基线")
    parser.add_argument("--instruction", default="抓取红色杯子并放到托盘")
    parser.add_argument("--root", type=Path, default=Path(os.environ["MOVEIT_EXPERIMENT_ROOT"]))
    parser.add_argument("--vnc", action="store_true", help="在 VNC 中实时显示 MuJoCo、固定相机和腕部相机。")
    arguments = parser.parse_args()
    if "红色" not in arguments.instruction:
        raise ValueError("阶段 2 首个基线固定验证红色杯子")
    simulation = UnifiedPandaCupSimulation(arguments.root, create_renderers=False)
    viewer = None
    if arguments.vnc:
        from .unified_preview import VncOverlayViewer

        viewer = VncOverlayViewer(simulation, title="MoveIt 阶段 2 真值抓放")
        simulation.initialize_renderers()
        viewer.glfw.make_context_current(viewer.window)
        last_frame_time = 0.0

        def render_frame(state: str) -> bool:
            nonlocal last_frame_time
            remaining = 1.0 / 30.0 - (time.monotonic() - last_frame_time)
            if remaining > 0.0:
                time.sleep(remaining)
            last_frame_time = time.monotonic()
            return viewer.render_once(state)

    else:
        render_frame = None
    executor = MujocoTaskExecutor(simulation, frame_callback=render_frame)
    rclpy.init()
    client = MoveItTrajectoryClient()
    try:
        # 等待桥接节点建立订阅，并同步同一快照的碰撞场景与起始关节。
        # ROS 2 默认话题为易失 QoS；先等待发现完成，避免首个场景/目标消息丢失。
        time.sleep(3.0)
        if os.environ.get("STAGE2_SKIP_COLLISION_SCENE") != "1":
            client.publish_scene(simulation.synchronization_snapshot(), grasp_target="red_cylinder")
            time.sleep(1.0)
        cylinder = simulation.evaluation_only_truth("red_cylinder")
        tray = simulation.evaluation_only_truth("tray")
        # 阶段 2 允许读取托盘真值作单元测试，但实际放置仍必须在当前位置物理松爪，
        # 不能把物体自由关节改写到托盘中。
        release_hand = tray + np.array([0.0, 0.0, 0.194])
        targets = [
            ("pregrasp", np.array([0.45, 0.0, 0.55])),
            ("approach", cylinder + np.array([0.0, 0.0, 0.10])),
            ("lift", cylinder + np.array([0.0, 0.0, 0.30])),
            ("place_above", release_hand + np.array([0.0, 0.0, 0.12])),
            ("place_descend", release_hand),
        ]
        executor.set_gripper(opened=True)
        for stage, target in targets[:2]:
            executor.set_display_state(f"MoveIt: {stage}")
            client.publish_joint_state(executor.joint_positions())
            _execute_trajectory(executor, client.request(target))
            executor._event(f"moveit_{stage}_complete")
        executor.set_display_state("Gripper: close and attach red cylinder")
        executor.set_gripper(opened=False)
        executor.attach("red_cylinder")
        for stage, target in targets[2:]:
            executor.set_display_state(f"MoveIt: {stage} with red cylinder")
            client.publish_joint_state(executor.joint_positions())
            _execute_trajectory(executor, client.request(target))
            executor._event(f"moveit_{stage}_complete")
        executor.set_display_state("Gripper: physical release above tray")
        executor.set_gripper(opened=True)
        executor.release_physical()
        success = executor.object_in_tray("red_cylinder", tray)
        payload = {
            "instruction": arguments.instruction,
            "stage": "truth_task_baseline",
            "success": success,
            "events": [{"name": item.name, "timestamp_s": item.timestamp, "details": item.details} for item in executor.events],
        }
        output = arguments.root / "logs" / "episodes" / "truth_baseline_latest"
        output.mkdir(parents=True, exist_ok=True)
        (output / "episode.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if not success:
            raise RuntimeError("杯子未落入托盘")
        if viewer is not None:
            print("任务成功。VNC 中按 Esc 关闭两个窗口后程序退出。", flush=True)
            while viewer.render_once("Success: red cylinder in tray"):
                time.sleep(1.0 / 30.0)
    finally:
        client.destroy_node()
        rclpy.shutdown()
        if viewer is not None:
            viewer.close()
        simulation.close()


if __name__ == "__main__":
    main()
