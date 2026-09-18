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
from moveit_msgs.msg import CollisionObject, PlanningScene, RobotTrajectory
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive

from .stage2_executor import MujocoTaskExecutor
from .unified_scene import UnifiedPandaCupSimulation


class MoveItTrajectoryClient(Node):
    def __init__(self) -> None:
        super().__init__("truth_baseline_client")
        self.trajectory_publisher = self.create_publisher(PoseStamped, "/vision_moveit/target_pose", 10)
        self.scene_publisher = self.create_publisher(PlanningScene, "/vision_moveit/planning_scene_diff", 10)
        self.joint_publisher = self.create_publisher(JointState, "/vision_moveit/sim_joint_states", 10)
        self.trajectories: list[RobotTrajectory] = []
        self.create_subscription(RobotTrajectory, "/vision_moveit/planned_trajectory", self.trajectories.append, 10)

    def publish_joint_state(self, positions: dict[str, float]) -> None:
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = [f"panda_joint{index}" for index in range(1, 8)]
        message.position = [positions[f"joint{index}"] for index in range(1, 8)]
        self.joint_publisher.publish(message)

    def publish_scene(self, snapshot, grasp_target: str | None = None) -> None:
        scene = PlanningScene(is_diff=True)
        dimensions = {
            # Panda 基座安装在桌面边缘。MoveIt 场景仅放入基座前方的作业台面，
            # 否则完整桌板会与其固定底座/立柱相交，使起始状态非法。
            "table": (SolidPrimitive.BOX, [0.85, 0.96, 0.07]),
            "tray": (SolidPrimitive.BOX, [0.26, 0.22, 0.094]),
            "red_cup": (SolidPrimitive.CYLINDER, [0.14, 0.038]),
            "green_cup": (SolidPrimitive.CYLINDER, [0.14, 0.038]),
            "blue_cup": (SolidPrimitive.CYLINDER, [0.14, 0.038]),
        }
        positions = {name: value["position_base_m"] for name, value in snapshot.planning_scene_objects.items()}
        positions["table"] = np.array([0.725, 0.0, 0.36])
        for name, (shape_type, shape_dimensions) in dimensions.items():
            # 接近阶段允许末端与目标杯建立接触；其它杯子、桌面和托盘仍进入碰撞场景。
            if name == grasp_target:
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
        self.scene_publisher.publish(scene)

    def request(self, position: np.ndarray, timeout_s: float = 15.0) -> RobotTrajectory:
        previous_count = len(self.trajectories)
        target = PoseStamped()
        target.header.frame_id = "panda_link0"
        target.header.stamp = self.get_clock().now().to_msg()
        target.pose.position.x, target.pose.position.y, target.pose.position.z = map(float, position)
        # Panda 手朝下，夹爪朝向桌面。
        target.pose.orientation.x = 1.0
        target.pose.orientation.w = 0.0
        self.trajectory_publisher.publish(target)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if len(self.trajectories) > previous_count:
                return self.trajectories[-1]
        raise TimeoutError("等待 MoveIt 规划轨迹超时")


def _execute_trajectory(executor: MujocoTaskExecutor, trajectory: RobotTrajectory) -> None:
    names = trajectory.joint_trajectory.joint_names
    for point in trajectory.joint_trajectory.points:
        target = executor.joint_positions()
        for name, value in zip(names, point.positions):
            if name.startswith("panda_joint"):
                target[name.replace("panda_", "")] = float(value)
        current = executor.joint_positions()
        # MoveIt 的时间参数基于理想 ros2_control 执行器；MuJoCo 的后三级
        # Panda 关节力矩更低，尤其 joint7 不能套用前三级的速度。按各轴保守
        # 速度上限计算此点所需时间，避免 OMPL 选到另一冗余姿态时回放失稳。
        max_speed_rad_s = {
            "joint1": 1.2,
            "joint2": 1.2,
            "joint3": 1.2,
            "joint4": 1.0,
            "joint5": 0.65,
            "joint6": 0.65,
            "joint7": 0.35,
        }
        required_duration = max(
            abs(target[name] - current[name]) / max_speed_rad_s[name] for name in executor.arm_joint_names
        )
        executor.execute_joint_target(target, max(0.35, 1.25 * required_duration))


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
            client.publish_scene(simulation.synchronization_snapshot(), grasp_target="red_cup")
            time.sleep(1.0)
        cup = simulation.evaluation_only_truth("red_cup")
        tray = simulation.evaluation_only_truth("tray")
        targets = [
            ("pregrasp", np.array([0.45, 0.0, 0.55])),
            ("approach", cup + np.array([0.0, 0.0, 0.10])),
            ("lift", cup + np.array([0.0, 0.0, 0.30])),
            # 托盘位于工作空间边缘；先到其内侧上方的安全放置预位，
            # 再由夹取执行器沿竖直方向释放到托盘落点。
            ("place", np.array([0.54, -0.18, 0.75])),
        ]
        executor.set_gripper(opened=True)
        for stage, target in targets[:2]:
            executor.set_display_state(f"MoveIt: {stage}")
            client.publish_joint_state(executor.joint_positions())
            _execute_trajectory(executor, client.request(target))
            executor._event(f"moveit_{stage}_complete")
        executor.set_display_state("Gripper: close and attach red cup")
        executor.set_gripper(opened=False)
        executor.attach("red_cup")
        for stage, target in targets[2:]:
            executor.set_display_state(f"MoveIt: {stage} with red cup")
            client.publish_joint_state(executor.joint_positions())
            _execute_trajectory(executor, client.request(target))
            executor._event(f"moveit_{stage}_complete")
        executor.set_display_state("Gripper: release into tray")
        executor.set_gripper(opened=True)
        executor.release_to_tray(tray)
        success = executor.object_in_tray("red_cup", tray)
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
            while viewer.render_once("Success: red cup in tray"):
                time.sleep(1.0 / 30.0)
    finally:
        client.destroy_node()
        rclpy.shutdown()
        simulation.close()


if __name__ == "__main__":
    main()
