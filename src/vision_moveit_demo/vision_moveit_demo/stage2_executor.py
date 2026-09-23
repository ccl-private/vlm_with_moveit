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


@dataclass(frozen=True)
class PhysicalReleaseSummary:
    """真实松爪后的物理状态证据；放置过程绝不重写物体自由关节状态。"""

    object_id: str
    release_position_base_m: np.ndarray
    release_quaternion_wxyz: np.ndarray
    position_after_settle_base_m: np.ndarray
    linear_velocity_after_settle_mps: np.ndarray
    settle_duration_s: float

    def as_dict(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "release_position_base_m": self.release_position_base_m.tolist(),
            "release_quaternion_wxyz": self.release_quaternion_wxyz.tolist(),
            "position_after_settle_base_m": self.position_after_settle_base_m.tolist(),
            "linear_velocity_after_settle_mps": self.linear_velocity_after_settle_mps.tolist(),
            "settle_duration_s": self.settle_duration_s,
            "qpos_reset_at_release": False,
            "qvel_reset_at_release": False,
            "placement_method": "physical_detach_and_mujoco_settle",
        }


class MujocoTaskExecutor:
    """阶段 2 的唯一 MuJoCo 执行入口；夹取附着与事件在此原子处理。"""

    arm_joint_names = tuple(f"joint{index}" for index in range(1, 8))
    graspable_object_names = ("red_cylinder", "green_cylinder", "blue_cylinder", "purple_cube", "magenta_block", "yellow_mug")

    def __init__(
        self,
        simulation: UnifiedPandaCupSimulation,
        frame_callback: Callable[[str], bool] | None = None,
        realtime_factor: float | None = None,
        attached_speed_scale: float = 6.0,
    ) -> None:
        if realtime_factor is not None and realtime_factor <= 0.0:
            raise ValueError("实时回放倍率必须为正数或 None")
        if attached_speed_scale <= 0.0:
            raise ValueError("携物阶段降速系数必须为正数")
        self.simulation = simulation
        self.attached_object: str | None = None
        self.events: list[SimulationEvent] = []
        self.frame_callback = frame_callback
        self.display_state = "Preparing"
        self._render_interval_steps = max(1, int(round(1.0 / (30.0 * simulation.model.opt.timestep))))
        self.realtime_factor = realtime_factor
        self.attached_speed_scale = attached_speed_scale
        self._realtime_wall_origin: float | None = None
        self._realtime_simulation_origin: float | None = None
        self._step_count = 0
        self.rendered_frame_count = 0
        self.trajectory_summaries: list[TrajectoryExecutionSummary] = []
        self.physical_release_summaries: list[PhysicalReleaseSummary] = []

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
        """推进一个物理步；抓取物体仅由接触确认后启用的 MuJoCo 约束保持。"""
        mujoco.mj_step(self.simulation.model, self.simulation.data)
        self._step_count += 1
        self._pace_realtime()
        if self._step_count % self._render_interval_steps == 0:
            self._render_frame()

    def joint_positions(self) -> dict[str, float]:
        model, data = self.simulation.model, self.simulation.data
        return {
            name: float(data.qpos[model.jnt_qposadr[model.joint(name).id]])
            for name in self.arm_joint_names
        }

    def body_position(self, body_name: str) -> np.ndarray:
        """返回当前仿真机体位置，仅用于执行遥测，绝不参与控制决策。"""
        return self.simulation.data.xpos[self.simulation.model.body(body_name).id].copy()

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
        speed_scale: float = 1.70,
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
        # 时间参数化的二次保护；当前 1.70 倍加速后的轨迹仍低于这些限制。
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
            # 已经通过双侧接触确认抓取后，额外载荷与约束会显著提高腕部惯性；
            # 降速而非放宽跟踪误差门限，保持物理夹持阶段的稳定性与可审计性。
            if self.attached_object is not None:
                duration_s *= self.attached_speed_scale
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

    def set_gripper(self, opened: bool, settle_s: float = 0.80) -> None:
        """控制 Panda 手指；0 为闭合，255 为张开（上游 Panda MJCF 的控制范围）。"""
        self.simulation.data.ctrl[7] = 255.0 if opened else 0.0
        steps = max(1, int(np.ceil(settle_s / self.simulation.model.opt.timestep)))
        for _ in range(steps):
            self._advance_one_step()
        self._event("gripper_open" if opened else "gripper_close")

    def close_until_dual_contact(
        self,
        object_name: str,
        timeout_s: float = 5.0,
        required_object_geometries: tuple[str, ...] = (),
    ) -> None:
        """闭爪时逐步检测双侧接触，首次形成稳定夹持即停止继续挤压。

        对薄把手而言，“完全闭合后再看接触”会将目标推出夹爪。本方法只负责
        时序控制；随后 ``attach`` 仍会校验 CAD 相对接触面和闭合轴。
        """
        if object_name not in self.graspable_object_names:
            raise ValueError(f"不支持附着的对象：{object_name}")
        model, data = self.simulation.model, self.simulation.data
        object_id = model.body(object_name).id
        left_id, right_id = model.body("left_finger").id, model.body("right_finger").id
        data.ctrl[7] = 0.0
        steps = max(1, int(np.ceil(timeout_s / model.opt.timestep)))
        for _ in range(steps):
            self._advance_one_step()
            left_contact = right_contact = False
            for contact in data.contact[: data.ncon]:
                contact_names = {model.geom(contact.geom1).name, model.geom(contact.geom2).name}
                if required_object_geometries and not contact_names.intersection(required_object_geometries):
                    continue
                bodies = {int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])}
                if object_id in bodies:
                    left_contact = left_contact or left_id in bodies
                    right_contact = right_contact or right_id in bodies
            if left_contact and right_contact:
                self._event("gripper_dual_contact", object_name=object_name)
                return
        finger_geoms = []
        for geom_id, body_id in enumerate(model.geom_bodyid):
            if int(body_id) in {left_id, right_id} and model.geom_contype[geom_id] != 0:
                finger_geoms.append(
                    {
                        "body": model.body(int(body_id)).name,
                        "pos": np.round(data.geom_xpos[geom_id], 4).tolist(),
                        "size": np.round(model.geom_size[geom_id], 4).tolist(),
                    }
                )
        finger_joint_positions = np.round(
            [data.qpos[model.jnt_qposadr[model.joint("finger_joint1").id]], data.qpos[model.jnt_qposadr[model.joint("finger_joint2").id]]], 5
        ).tolist()
        all_contacts = [
            f"{model.geom(contact.geom1).name}/{model.geom(contact.geom2).name}"
            for contact in data.contact[: data.ncon]
        ]
        raise RuntimeError(
            f"夹爪在 {timeout_s:.1f}s 内未与 {object_name} 形成双侧真实接触；"
            f"要求的 CAD 碰撞面={list(required_object_geometries)}；"
            f"object={np.round(data.xpos[object_id], 4).tolist()}，finger_geoms={finger_geoms}"
            f"，finger_qpos={finger_joint_positions}，all_contacts={all_contacts}"
        )

    def attach(
        self,
        object_name: str,
        expected_closing_axis_base: np.ndarray | None = None,
        opposing_contact_pair_object: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
        required_object_geometries: tuple[str, ...] = (),
    ) -> None:
        """仅在双侧真实接触、轴对齐且（CAD 路径中）落在相对面后确认物理夹持。"""
        if object_name not in self.graspable_object_names:
            raise ValueError(f"不支持附着的对象：{object_name}")
        if self.attached_object is not None:
            raise RuntimeError(f"已有附着对象：{self.attached_object}")
        model, data = self.simulation.model, self.simulation.data
        hand_id, object_id = model.body("hand").id, model.body(object_name).id
        hand_rotation = data.xmat[hand_id].reshape(3, 3)
        actual_axis = hand_rotation[:, 1]
        alignment = float("nan")
        if expected_closing_axis_base is not None:
            expected_axis = np.asarray(expected_closing_axis_base, dtype=np.float64)
            expected_axis /= np.linalg.norm(expected_axis)
            alignment = abs(float(np.dot(actual_axis, expected_axis)))
            if alignment < 0.95:
                raise RuntimeError(f"夹爪闭合轴未与 CAD 标注面法向对齐：|dot|={alignment:.3f}")
        left_id, right_id = model.body("left_finger").id, model.body("right_finger").id
        left_contact = right_contact = 0
        object_rotation = data.xmat[object_id].reshape(3, 3)
        left_contact_points_object: list[np.ndarray] = []
        right_contact_points_object: list[np.ndarray] = []
        contact_geometry_names: list[str] = []
        for index in range(data.ncon):
            contact = data.contact[index]
            contact_names = {model.geom(contact.geom1).name, model.geom(contact.geom2).name}
            if required_object_geometries and not contact_names.intersection(required_object_geometries):
                continue
            body_a, body_b = model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]
            bodies = {int(body_a), int(body_b)}
            if object_id not in bodies:
                continue
            point_object = object_rotation.T @ (contact.pos - data.xpos[object_id])
            if left_id in bodies:
                left_contact += 1
                left_contact_points_object.append(point_object)
                contact_geometry_names.append("/".join(sorted(contact_names)))
            if right_id in bodies:
                right_contact += 1
                right_contact_points_object.append(point_object)
                contact_geometry_names.append("/".join(sorted(contact_names)))
        if left_contact == 0 or right_contact == 0:
            hand_position = np.round(data.xpos[hand_id], 4).tolist()
            object_position = np.round(data.xpos[object_id], 4).tolist()
            left_position = np.round(data.xpos[left_id], 4).tolist()
            right_position = np.round(data.xpos[right_id], 4).tolist()
            finger_joint_positions = np.round(
                [data.qpos[model.jnt_qposadr[model.joint("finger_joint1").id]], data.qpos[model.jnt_qposadr[model.joint("finger_joint2").id]]], 4
            ).tolist()
            contact_names = [
                f"{model.geom(contact.geom1).name}/{model.geom(contact.geom2).name}"
                for contact in data.contact[: data.ncon]
            ]
            raise RuntimeError(
                f"夹爪未形成双侧真实接触：left={left_contact}，right={right_contact}；拒绝伪附着；"
                f"hand={hand_position}，object={object_position}，left_finger={left_position}，right_finger={right_position}，"
                f"finger_qpos={finger_joint_positions}，all_contacts={contact_names}"
            )
        matched_opposing_faces = "not_checked"
        if opposing_contact_pair_object is not None:
            positive_center, positive_normal, negative_center, negative_normal = (
                np.asarray(value, dtype=np.float64) for value in opposing_contact_pair_object
            )

            def touches_surface(points: list[np.ndarray], center: np.ndarray, normal: np.ndarray) -> bool:
                # 仅允许落在 CAD 标注面附近 3 mm；此前 15 mm 容差会把同一侧
                # 把手接触误判为两侧接触，无法保证真正的左右对称夹握。
                return any(abs(float(np.dot(point - center, normal))) <= 0.003 for point in points)

            left_positive = touches_surface(left_contact_points_object, positive_center, positive_normal)
            left_negative = touches_surface(left_contact_points_object, negative_center, negative_normal)
            right_positive = touches_surface(right_contact_points_object, positive_center, positive_normal)
            right_negative = touches_surface(right_contact_points_object, negative_center, negative_normal)
            if not ((left_positive and right_negative) or (left_negative and right_positive)):
                raise RuntimeError(
                    "左右指尖接触没有分别落在 CAD 标注的相对面："
                    f"left(+/-)=({left_positive}/{left_negative})，"
                    f"right(+/-)=({right_positive}/{right_negative})；"
                    f"left_points_object={np.round(left_contact_points_object, 4).tolist()}；"
                    f"right_points_object={np.round(right_contact_points_object, 4).tolist()}"
                )
            matched_opposing_faces = "passed"
        relative_position = hand_rotation.T @ (data.xpos[object_id] - data.xpos[hand_id])
        relative_rotation = hand_rotation.T @ data.xmat[object_id].reshape(3, 3)
        relative_quaternion = np.empty(4)
        mujoco.mju_mat2Quat(relative_quaternion, relative_rotation.ravel())
        equality_id = model.equality(f"grasp_{object_name}").id
        # 每次夹取都以已验证的当前接触相对位姿初始化 weld；不重写自由关节状态。
        # MuJoCo weld 的 11 个 eq_data 字段依次为：body2 锚点、body1 锚点、
        # body2 相对 body1 的四元数、torquescale。以物体原点作为 body2 锚点，
        # 并把同一点在 hand 局部系的位置写入 body1 锚点；旧实现误把四元数写进
        # 锚点槽，导致物体在抬升中被拉向错误位置。
        model.eq_data[equality_id, :3] = 0.0
        model.eq_data[equality_id, 3:6] = relative_position
        model.eq_data[equality_id, 6:10] = relative_quaternion
        # 0.04 m 近似两侧指尖形成的有效夹持接触面尺度：保留姿态保持，且避免
        # 过大的角约束力矩干扰腕部轨迹。
        model.eq_data[equality_id, 10] = 0.04
        data.eq_active[equality_id] = 1
        mujoco.mj_forward(model, data)
        self.attached_object = object_name
        self._event(
            "grasp_attached", object_id=object_name, method="physical_fingertip_contact_and_weld",
            left_contacts=float(left_contact), right_contacts=float(right_contact), closing_axis_alignment=alignment,
            opposing_contact_pair=matched_opposing_faces,
            required_object_geometries=list(required_object_geometries),
            verified_contact_geometries=sorted(set(contact_geometry_names)),
            # 记录到 CAD 局部系，便于离线直接核验接触是否分别位于把手两面，
            # 而不是只依赖 VNC 画面观感。
            left_contact_points_object=np.round(left_contact_points_object, 6).tolist(),
            right_contact_points_object=np.round(right_contact_points_object, 6).tolist(),
        )

    def release_physical(self, settle_s: float = 0.75) -> PhysicalReleaseSummary:
        """在当前夹爪位置解除附着，再完全交由 MuJoCo 接触动力学放置。"""
        if self.attached_object is None:
            raise RuntimeError("没有附着对象可释放")
        if settle_s <= 0.0:
            raise ValueError("物理稳定时间必须为正数")
        model, data = self.simulation.model, self.simulation.data
        object_name = self.attached_object
        equality_id = model.equality(f"grasp_{object_name}").id
        joint_id = model.joint(f"{object_name}_freejoint").id
        qpos_address = model.jnt_qposadr[joint_id]
        dof_address = model.jnt_dofadr[joint_id]
        release_position = data.qpos[qpos_address : qpos_address + 3].copy()
        release_quaternion = data.qpos[qpos_address + 3 : qpos_address + 7].copy()
        # 禁用已验证接触后启用的 MuJoCo weld；这里故意不写 qpos/qvel。
        data.eq_active[equality_id] = 0
        mujoco.mj_forward(model, data)
        self._event("grasp_released_physical", object_id=object_name, qpos_reset=0.0, qvel_reset=0.0)
        self.attached_object = None
        for _ in range(max(1, int(np.ceil(settle_s / model.opt.timestep)))):
            self._advance_one_step()
        summary = PhysicalReleaseSummary(
            object_id=object_name,
            release_position_base_m=release_position,
            release_quaternion_wxyz=release_quaternion,
            position_after_settle_base_m=data.qpos[qpos_address : qpos_address + 3].copy(),
            linear_velocity_after_settle_mps=data.qvel[dof_address : dof_address + 3].copy(),
            settle_duration_s=settle_s,
        )
        self.physical_release_summaries.append(summary)
        self._event("physical_settle_complete", object_id=object_name, duration_s=settle_s)
        return summary

    def object_in_tray(self, object_name: str, tray_center_base: np.ndarray) -> bool:
        position = self.simulation.evaluation_only_truth(object_name)
        delta = np.abs(position[:2] - np.asarray(tray_center_base)[:2])
        # 托盘已直接落在桌面：其底板上表面是中心 z 加 12 mm。旧的固定
        # ``z > 0.45`` 只适用于垫高托盘，会把底面刚好落在托盘内的长方体
        # （中心约 z=0.449）误报为失败。
        tray_floor_z = float(np.asarray(tray_center_base)[2] + 0.012)
        inside = bool(
            delta[0] < 0.095
            and delta[1] < 0.075
            and position[2] >= tray_floor_z + 0.005
            and position[2] <= tray_floor_z + 0.13
        )
        self._event("tray_verification", object_id=object_name, result="passed" if inside else "failed")
        return inside
