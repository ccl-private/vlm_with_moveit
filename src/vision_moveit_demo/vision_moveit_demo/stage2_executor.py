"""阶段 2：MuJoCo 轨迹回放、夹取附着与放置事件。

此模块不读取相机或 VLM 输出；目标物体仅由真值任务基线调用方提供。
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import mujoco
import numpy as np

from .unified_scene import UnifiedPandaCupSimulation


@dataclass(frozen=True)
class SimulationEvent:
    name: str
    timestamp: float
    details: dict[str, str | float]


@dataclass(frozen=True)
class TrajectoryExecutionSummary:
    """一条 MoveIt 时间轨迹在 MuJoCo 中的执行遥测。"""

    profile: str
    waypoint_count: int
    planner_duration_s: float
    commanded_duration_s: float
    simulated_duration_s: float
    final_tracking_error_rad: float
    max_tracking_error_rad: float

    def as_dict(self) -> dict[str, str | int | float]:
        return {
            "profile": self.profile,
            "waypoint_count": self.waypoint_count,
            "planner_duration_s": self.planner_duration_s,
            "commanded_duration_s": self.commanded_duration_s,
            "simulated_duration_s": self.simulated_duration_s,
            "final_tracking_error_rad": self.final_tracking_error_rad,
            "max_tracking_error_rad": self.max_tracking_error_rad,
        }


class MujocoTaskExecutor:
    """阶段 2 的唯一 MuJoCo 执行入口；夹取附着与事件在此原子处理。"""

    arm_joint_names = tuple(f"joint{index}" for index in range(1, 8))
    cup_names = ("red_cup", "green_cup", "blue_cup")

    def __init__(
        self,
        simulation: UnifiedPandaCupSimulation,
        frame_callback: Callable[[str], bool] | None = None,
        realtime_factor: float | None = None,
    ) -> None:
        if realtime_factor is not None and realtime_factor <= 0.0:
            raise ValueError("实时回放倍率必须为正数或 None")
        self.simulation = simulation
        self.attached_object: str | None = None
        self._grasp_relative_position: np.ndarray | None = None
        self._grasp_relative_rotation: np.ndarray | None = None
        self.events: list[SimulationEvent] = []
        self.frame_callback = frame_callback
        self.display_state = "Preparing"
        self._render_interval_steps = max(1, int(round(1.0 / (30.0 * simulation.model.opt.timestep))))
        self.realtime_factor = realtime_factor
        self._realtime_wall_origin: float | None = None
        self._realtime_simulation_origin: float | None = None
        self._step_count = 0
        self.rendered_frame_count = 0
        self.trajectory_summaries: list[TrajectoryExecutionSummary] = []

    def _event(self, name: str, **details: str | float) -> None:
        self.events.append(SimulationEvent(name, float(self.simulation.data.time), details))

    def set_display_state(self, state: str) -> None:
        self.display_state = state
        self._render_frame()

    def _render_frame(self) -> None:
        if self.frame_callback is not None and not self.frame_callback(self.display_state):
            raise RuntimeError("VNC 窗口已关闭，任务已停止")
        if self.frame_callback is not None:
            self.rendered_frame_count += 1

    def _pace_realtime(self) -> None:
        """以仿真时间为基准限速，避免 VNC 渲染函数决定控制节拍。"""
        if self.realtime_factor is None:
            return
        simulation_time = float(self.simulation.data.time)
        if self._realtime_wall_origin is None:
            self._realtime_wall_origin = time.monotonic()
            self._realtime_simulation_origin = simulation_time
            return
        assert self._realtime_simulation_origin is not None
        deadline = self._realtime_wall_origin + (simulation_time - self._realtime_simulation_origin) / self.realtime_factor
        remaining = deadline - time.monotonic()
        if remaining > 0.0:
            time.sleep(remaining)

    def _advance_one_step(self) -> None:
        """推进一个物理步，并按统一策略处理附着、显示和真实时间节拍。"""
        mujoco.mj_step(self.simulation.model, self.simulation.data)
        self._hold_attached_object()
        self._step_count += 1
        self._pace_realtime()
        if self._step_count % self._render_interval_steps == 0:
            self._render_frame()

    def _hold_attached_object(self) -> None:
        """以末端相对位姿稳定保持已确认夹取的杯子。

        Panda 资产中的自由杯与 weld equality 在启用瞬间会产生较大约束冲量，
        不适合这一阶段的确定性抓取基线。这里采用混合抓取模型：路径仍由
        MoveIt 规划、夹爪事件仍完整记录，而被确认夹取的物体以运动学方式
        随末端移动；释放后立即恢复普通刚体与托盘接触。
        """
        if self.attached_object is None:
            return
        if self._grasp_relative_position is None or self._grasp_relative_rotation is None:
            raise RuntimeError("夹取相对位姿缺失")
        model, data = self.simulation.model, self.simulation.data
        hand_id = model.body("hand").id
        hand_rotation = data.xmat[hand_id].reshape(3, 3)
        world_position = data.xpos[hand_id] + hand_rotation @ self._grasp_relative_position
        world_rotation = hand_rotation @ self._grasp_relative_rotation
        world_quaternion = np.empty(4)
        mujoco.mju_mat2Quat(world_quaternion, world_rotation.ravel())
        joint_id = model.joint(f"{self.attached_object}_freejoint").id
        qpos_address = model.jnt_qposadr[joint_id]
        dof_address = model.jnt_dofadr[joint_id]
        data.qpos[qpos_address : qpos_address + 3] = world_position
        data.qpos[qpos_address + 3 : qpos_address + 7] = world_quaternion
        data.qvel[dof_address : dof_address + 6] = 0.0
        mujoco.mj_forward(model, data)

    def joint_positions(self) -> dict[str, float]:
        model, data = self.simulation.model, self.simulation.data
        return {
            name: float(data.qpos[model.jnt_qposadr[model.joint(name).id]])
            for name in self.arm_joint_names
        }

    def execute_joint_target(self, target: dict[str, float], duration_s: float) -> None:
        """以位置控制器回放一段 MoveIt 关节目标；执行器不生成关节轨迹。"""
        model, data = self.simulation.model, self.simulation.data
        start = np.array([data.qpos[model.jnt_qposadr[model.joint(name).id]] for name in self.arm_joint_names])
        goal = np.array([target[name] for name in self.arm_joint_names])
        steps = max(1, int(np.ceil(duration_s / model.opt.timestep)))
        for index in range(1, steps + 1):
            fraction = index / steps
            data.ctrl[:7] = start + fraction * (goal - start)
            self._advance_one_step()
        actual = np.array(list(self.joint_positions().values()))
        tracking_error = float(np.max(np.abs(goal - actual)))
        self._event("trajectory_segment_complete", duration_s=float(duration_s), max_joint_error_rad=tracking_error)
        if tracking_error > 0.08:
            raise RuntimeError(
                "MuJoCo 轨迹跟踪误差过大："
                f"{tracking_error:.3f} rad；目标={np.round(goal, 3).tolist()}；"
                f"实际={np.round(actual, 3).tolist()}"
            )

    def execute_timed_joint_trajectory(
        self,
        waypoint_times_s: list[float],
        waypoint_positions: list[dict[str, float]],
        *,
        profile: str = "normal",
        speed_scale: float = 1.35,
    ) -> TrajectoryExecutionSummary:
        """连续执行 MoveIt 的整条时间参数化关节轨迹。

        MoveIt 的 ``time_from_start`` 负责定义每个路点的时间关系。正常档只在不超过
        集中配置的关节速度上限时整体加速，绝不再为每个离散点附加固定等待时间。
        """
        if profile != "normal":
            raise ValueError(f"不支持的执行速度档：{profile}")
        if speed_scale <= 0.0:
            raise ValueError("速度倍率必须为正数")
        if not waypoint_times_s or len(waypoint_times_s) != len(waypoint_positions):
            raise ValueError("时间路点与关节路点必须一一对应且非空")
        if any(later < earlier for earlier, later in zip(waypoint_times_s, waypoint_times_s[1:])):
            raise ValueError("MoveIt 时间路点必须单调递增")

        # 这是仿真正常档的硬速度上限（rad/s），而非真机参数。它只作为对 MoveIt
        # 时间参数化的二次保护；当前 1.35 倍加速后的轨迹远低于这些限制。
        normal_max_speed_rad_s = {
            "joint1": 2.0,
            "joint2": 2.0,
            "joint3": 2.0,
            "joint4": 2.0,
            "joint5": 1.5,
            "joint6": 1.5,
            "joint7": 1.5,
        }
        model, data = self.simulation.model, self.simulation.data
        start_simulation_time = float(data.time)
        previous_positions = np.array([self.joint_positions()[name] for name in self.arm_joint_names])
        previous_time_s = 0.0
        commanded_duration_s = 0.0
        max_tracking_error = 0.0

        for waypoint_time_s, waypoint in zip(waypoint_times_s, waypoint_positions):
            target_positions = np.array([waypoint[name] for name in self.arm_joint_names])
            raw_duration_s = max(0.0, waypoint_time_s - previous_time_s)
            speed_limited_duration_s = max(
                abs(target - current) / normal_max_speed_rad_s[name]
                for name, target, current in zip(self.arm_joint_names, target_positions, previous_positions)
            )
            duration_s = max(raw_duration_s / speed_scale, speed_limited_duration_s)
            # 首个 MoveIt 路点通常是 t=0 的当前姿态；若不是，也至少经过一个物理步。
            steps = max(1, int(np.ceil(duration_s / model.opt.timestep)))
            effective_duration_s = steps * model.opt.timestep
            for index in range(1, steps + 1):
                fraction = index / steps
                desired_positions = previous_positions + fraction * (target_positions - previous_positions)
                data.ctrl[:7] = desired_positions
                self._advance_one_step()
                actual_positions = np.array([self.joint_positions()[name] for name in self.arm_joint_names])
                max_tracking_error = max(max_tracking_error, float(np.max(np.abs(desired_positions - actual_positions))))
            commanded_duration_s += effective_duration_s
            previous_positions = target_positions
            previous_time_s = waypoint_time_s

        final_positions = np.array([self.joint_positions()[name] for name in self.arm_joint_names])
        final_tracking_error = float(np.max(np.abs(previous_positions - final_positions)))
        summary = TrajectoryExecutionSummary(
            profile=profile,
            waypoint_count=len(waypoint_positions),
            planner_duration_s=float(waypoint_times_s[-1]),
            commanded_duration_s=float(commanded_duration_s),
            simulated_duration_s=float(data.time - start_simulation_time),
            final_tracking_error_rad=final_tracking_error,
            max_tracking_error_rad=max_tracking_error,
        )
        self.trajectory_summaries.append(summary)
        self._event(
            "timed_trajectory_complete",
            planner_duration_s=summary.planner_duration_s,
            commanded_duration_s=summary.commanded_duration_s,
            final_tracking_error_rad=summary.final_tracking_error_rad,
            max_tracking_error_rad=summary.max_tracking_error_rad,
        )
        if final_tracking_error > 0.08:
            raise RuntimeError(
                "MuJoCo 连续轨迹终点跟踪误差过大："
                f"{final_tracking_error:.3f} rad；目标={np.round(previous_positions, 3).tolist()}；"
                f"实际={np.round(final_positions, 3).tolist()}"
            )
        return summary

    def set_gripper(self, opened: bool, settle_s: float = 0.25) -> None:
        """控制 Panda 手指；0 为闭合，255 为张开（上游 Panda MJCF 的控制范围）。"""
        self.simulation.data.ctrl[7] = 255.0 if opened else 0.0
        steps = max(1, int(np.ceil(settle_s / self.simulation.model.opt.timestep)))
        for _ in range(steps):
            self._advance_one_step()
        self._event("gripper_open" if opened else "gripper_close")

    def attach(self, object_name: str) -> None:
        """确认夹取后记录稳定的末端—物体相对位姿。"""
        if object_name not in self.cup_names:
            raise ValueError(f"不支持附着的对象：{object_name}")
        if self.attached_object is not None:
            raise RuntimeError(f"已有附着对象：{self.attached_object}")
        model, data = self.simulation.model, self.simulation.data
        hand_id, object_id = model.body("hand").id, model.body(object_name).id
        hand_rotation = data.xmat[hand_id].reshape(3, 3)
        relative_position = hand_rotation.T @ (data.xpos[object_id] - data.xpos[hand_id])
        relative_rotation = hand_rotation.T @ data.xmat[object_id].reshape(3, 3)
        self.attached_object = object_name
        self._grasp_relative_position = relative_position.copy()
        self._grasp_relative_rotation = relative_rotation.copy()
        self._hold_attached_object()
        self._event("grasp_attached", object_id=object_name, method="kinematic_hold")

    def release_to_tray(self, tray_center_base: np.ndarray) -> None:
        """解除约束并在托盘内部放置杯子，供物理稳定与放置事件复核。"""
        if self.attached_object is None:
            raise RuntimeError("没有附着对象可释放")
        model, data = self.simulation.model, self.simulation.data
        object_name = self.attached_object
        joint_id = model.joint(f"{object_name}_freejoint").id
        qpos_address = model.jnt_qposadr[joint_id]
        # 托盘底部顶面为 z=0.462，圆柱杯半高为 0.07。
        data.qpos[qpos_address : qpos_address + 3] = np.asarray(tray_center_base) + np.array([0.0, 0.0, 0.082])
        data.qpos[qpos_address + 3 : qpos_address + 7] = (1.0, 0.0, 0.0, 0.0)
        data.qvel[model.jnt_dofadr[joint_id] : model.jnt_dofadr[joint_id] + 6] = 0.0
        mujoco.mj_forward(model, data)
        self._event("grasp_released", object_id=object_name)
        self.attached_object = None
        self._grasp_relative_position = None
        self._grasp_relative_rotation = None
        for _ in range(max(1, int(np.ceil(0.35 / model.opt.timestep)))):
            self._advance_one_step()

    def object_in_tray(self, object_name: str, tray_center_base: np.ndarray) -> bool:
        position = self.simulation.evaluation_only_truth(object_name)
        delta = np.abs(position[:2] - np.asarray(tray_center_base)[:2])
        inside = bool(delta[0] < 0.095 and delta[1] < 0.075 and position[2] > 0.45)
        self._event("tray_verification", object_id=object_name, result="passed" if inside else "failed")
        return inside
