"""阶段 1 统一 MuJoCo 场景：Panda、工位与双 RGB-D 相机。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402


@dataclass(frozen=True)
class UnifiedCameraFrame:
    """带标定和时间戳的相机帧；控制路径不含物体真值。"""

    name: str
    rgb: np.ndarray
    depth: np.ndarray
    intrinsic: np.ndarray
    world_from_camera: np.ndarray
    base_from_camera: np.ndarray
    timestamp: float


@dataclass(frozen=True)
class UnifiedSceneSnapshot:
    """同一物理时刻的仿真同步快照。

    该对象只供后续 MuJoCo 执行适配器与 MoveIt 碰撞场景同步使用。它不是
    视觉控制路径的输入；评测物体真值仍须通过 ``evaluation_only_truth`` 获取。
    """

    timestamp: float
    joint_positions: dict[str, float]
    planning_scene_objects: dict[str, dict[str, np.ndarray]]


class UnifiedPandaCupSimulation:
    """阶段 1 场景运行时；后续阶段在此基础上接入状态机和执行适配器。"""

    # 这些是带 freejoint、可在任务开始前由交互编辑器移动的工件。托盘是固定
    # 工装，故意不在此列表中：抓放任务的放置基准不应在一轮任务中被悄悄改动。
    EDITABLE_OBJECTS = (
        "red_cylinder",
        "green_cylinder",
        "blue_cylinder",
        "purple_cube",
        "magenta_block",
        "yellow_mug",
    )
    _REST_EQUALITIES = {
        "magenta_block": "magenta_block_rest",
        "yellow_mug": "yellow_mug_rest",
    }

    def __init__(self, root: Path, width: int = 640, height: int = 480, create_renderers: bool = True) -> None:
        self.root = root
        scene = root / "assets" / "panda_mjcf" / "unified_panda_objects_scene.xml"
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self._initial_rest_equality_data = {
            equality_name: self.model.eq_data[self.model.equality(equality_name).id].copy()
            for equality_name in self._REST_EQUALITIES.values()
        }
        self.width, self.height = width, height
        self._renderers: dict[str, mujoco.Renderer] = {}
        if create_renderers:
            self.initialize_renderers()
        self.reset()

    def initialize_renderers(self) -> None:
        """创建 RGB-D 渲染器；VNC 模式必须在 GLFW 上下文创建后调用。"""
        if self._renderers:
            return
        self._renderers = {
            name: mujoco.Renderer(self.model, height=self.height, width=self.width)
            for name in ("camera_global", "camera_fixed", "camera_wrist")
        }

    def reset(self) -> None:
        """恢复官方 Panda home 关键帧和场景默认物体位置。"""
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        for equality_name, equality_data in self._initial_rest_equality_data.items():
            equality_id = self.model.equality(equality_name).id
            self.model.eq_data[equality_id] = equality_data
            self.data.eq_active[equality_id] = 1
        # Panda 的官方 keyframe 只定义了机械臂关节；场景后来加入的 freejoint
        # 必须在这里显式复位，否则会继承为零位姿，导致杯子落到世界原点。
        graspable_object_poses = {
            "red_cylinder_freejoint": ((0.30, 0.30, 0.465), (1.0, 0.0, 0.0, 0.0)),
            "green_cylinder_freejoint": ((0.22, 0.08, 0.465), (1.0, 0.0, 0.0, 0.0)),
            "blue_cylinder_freejoint": ((0.30, -0.24, 0.465), (1.0, 0.0, 0.0, 0.0)),
            "purple_cube_freejoint": ((0.35, -0.08, 0.43), (1.0, 0.0, 0.0, 0.0)),
            "magenta_block_freejoint": ((0.42, -0.28, 0.425), (0.962425, 0.0, 0.0, 0.271547)),
            # MuJoCo freejoint 四元数为 w,x,y,z；绕 Z 180° 让把手面向相机/作业侧。
            # 该姿态不作为感知真值输入。
            "yellow_mug_freejoint": ((0.45, -0.12, 0.445), (0.0, 0.0, 0.0, 1.0)),
        }
        for joint_name, (position, quaternion) in graspable_object_poses.items():
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qpos_address = self.model.jnt_qposadr[joint_id]
            dof_address = self.model.jnt_dofadr[joint_id]
            self.data.qpos[qpos_address : qpos_address + 3] = position
            self.data.qpos[qpos_address + 3 : qpos_address + 7] = quaternion
            self.data.qvel[dof_address : dof_address + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def hold_home(self, steps: int = 1) -> None:
        """阶段 1 仅稳定保持初始姿态；不承担任务轨迹执行。"""
        self.data.ctrl[:] = self.model.key_ctrl[0]
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

    def move_editable_object(
        self,
        object_name: str,
        delta_xy_m: tuple[float, float],
        yaw_delta_rad: float = 0.0,
    ) -> tuple[bool, str]:
        """在任务开始前移动一个工件，并将其贴合到桌面碰撞面。"""
        if object_name not in self.EDITABLE_OBJECTS:
            raise ValueError(f"不允许交互移动的场景对象：{object_name}")
        joint = self.model.joint(f"{object_name}_freejoint")
        qpos_address = self.model.jnt_qposadr[joint.id]
        dof_address = self.model.jnt_dofadr[joint.id]
        original_qpos = self.data.qpos[qpos_address : qpos_address + 7].copy()
        original_qvel = self.data.qvel[dof_address : dof_address + 6].copy()

        table_geom = self.model.geom("table_top")
        table_center = self.data.geom_xpos[table_geom.id]
        table_half_size = self.model.geom_size[table_geom.id]
        footprint_radius = self._body_horizontal_radius(self.model.body(object_name).id)
        candidate_xy = original_qpos[:2] + np.asarray(delta_xy_m, dtype=np.float64)
        lower_xy = table_center[:2] - table_half_size[:2] + footprint_radius
        upper_xy = table_center[:2] + table_half_size[:2] - footprint_radius
        if np.any(candidate_xy < lower_xy) or np.any(candidate_xy > upper_xy):
            return False, "已拒绝：物体会超出桌面边界"

        self.data.qpos[qpos_address : qpos_address + 2] = candidate_xy
        # 桌面顶面高度来自 MuJoCo 几何，而不是写死的 XML 数字；工件的最低
        # 碰撞几何也由模型计算，因而修改桌面/工件尺寸后仍可正确贴合。
        table_top_z = float(table_center[2] + table_half_size[2])
        # 先在 z=0 计算所有碰撞几何的真实世界最低点。杯把的 capsule 是倾斜
        # 的，不能仅用其局部 z 尺寸，否则会把杯子错误垫高。
        self.data.qpos[qpos_address + 2] = 0.0
        if yaw_delta_rad:
            half_angle = yaw_delta_rad / 2.0
            yaw_quaternion = np.array([np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)])
            current_quaternion = self.data.qpos[qpos_address + 3 : qpos_address + 7].copy()
            self.data.qpos[qpos_address + 3 : qpos_address + 7] = self._quaternion_multiply(
                yaw_quaternion, current_quaternion
            )
        self.data.qvel[dof_address : dof_address + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.data.qpos[qpos_address + 2] = table_top_z - self._body_lowest_collision_world_z(object_name)
        mujoco.mj_forward(self.model, self.data)
        if self._has_forbidden_edit_contact(object_name):
            self.data.qpos[qpos_address : qpos_address + 7] = original_qpos
            self.data.qvel[dof_address : dof_address + 6] = original_qvel
            mujoco.mj_forward(self.model, self.data)
            return False, "已拒绝：物体会与其它场景物体或机械臂相交"
        self._synchronize_rest_equality(object_name)
        position = self.data.xpos[self.model.body(object_name).id]
        return True, f"{object_name} 已移动到 ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}) m"

    def _synchronize_rest_equality(self, object_name: str) -> None:
        """让可编辑工件的静置 weld 固定在编辑后的位姿，而非 XML 初始位姿。"""
        equality_name = self._REST_EQUALITIES.get(object_name)
        if equality_name is None:
            return
        equality_id = self.model.equality(equality_name).id
        table_id = self.model.body("table").id
        object_id = self.model.body(object_name).id
        table_rotation = self.data.xmat[table_id].reshape(3, 3)
        object_rotation = self.data.xmat[object_id].reshape(3, 3)
        relative_rotation = table_rotation.T @ object_rotation
        relative_position = table_rotation.T @ (self.data.xpos[object_id] - self.data.xpos[table_id])
        relative_quaternion_wxyz = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(relative_quaternion_wxyz, relative_rotation.ravel())
        # weld 的 eq_data 为 anchor1、anchor2、relpose(xyz+wxyz)、torquescale；
        # 本场景默认使用原点 anchor，只需同步 body2 相对 table 的 relpose。
        self.model.eq_data[equality_id, 3:6] = relative_position
        self.model.eq_data[equality_id, 6:10] = relative_quaternion_wxyz
        self.data.eq_active[equality_id] = 1

    def _body_horizontal_radius(self, body_id: int) -> float:
        geom_address = self.model.body_geomadr[body_id]
        geom_count = self.model.body_geomnum[body_id]
        radii = [
            float(np.linalg.norm(self.model.geom_pos[geom_id][:2]) + self.model.geom_rbound[geom_id])
            for geom_id in range(geom_address, geom_address + geom_count)
            if self.model.geom_contype[geom_id] != 0
        ]
        if not radii:
            raise RuntimeError("可编辑工件没有碰撞几何")
        return max(radii)

    def _body_lowest_collision_world_z(self, object_name: str) -> float:
        body_id = self.model.body(object_name).id
        geom_address = self.model.body_geomadr[body_id]
        geom_count = self.model.body_geomnum[body_id]
        lowest = float("inf")
        for geom_id in range(geom_address, geom_address + geom_count):
            if self.model.geom_contype[geom_id] == 0:
                continue
            geom_type = self.model.geom_type[geom_id]
            size = self.model.geom_size[geom_id]
            rotation = self.data.geom_xmat[geom_id].reshape(3, 3)
            vertical_axis = rotation[2, 2]
            if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                half_height = float(np.abs(rotation[2]) @ size)
            elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                half_height = float(size[0] * np.sqrt(max(0.0, 1.0 - vertical_axis**2)) + size[1] * abs(vertical_axis))
            elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                # capsule 的两个端点都是半径为 r 的球，故横向/纵向投影都由 r
                # 覆盖；size[1] 是两端球心相对中心的半长。
                half_height = float(size[0] + size[1] * abs(vertical_axis))
            elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                half_height = size[0]
            else:
                # 当前场景的最低点由 box/cylinder 决定；对其它碰撞类型保守
                # 使用包围半径，避免模型扩展后把物体嵌进桌面。
                half_height = self.model.geom_rbound[geom_id]
            lowest = min(lowest, float(self.data.geom_xpos[geom_id][2] - half_height))
        if not np.isfinite(lowest):
            raise RuntimeError("可编辑工件没有碰撞几何")
        return lowest

    @staticmethod
    def _quaternion_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        """MuJoCo w,x,y,z 四元数乘法。"""
        w1, x1, y1, z1 = first
        w2, x2, y2, z2 = second
        result = np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )
        return result / np.linalg.norm(result)

    def _has_forbidden_edit_contact(self, object_name: str) -> bool:
        object_body = self.model.body(object_name).id
        table_body = self.model.body("table").id
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            first_body = self.model.geom_bodyid[contact.geom1]
            second_body = self.model.geom_bodyid[contact.geom2]
            if object_body not in (first_body, second_body):
                continue
            other_body = second_body if first_body == object_body else first_body
            if other_body != table_body:
                return True
        return False

    def render_camera(self, name: str, include_depth: bool = True) -> UnifiedCameraFrame:
        """渲染一帧相机画面。

        VNC 预览只需 RGB，跳过深度渲染可避免一半离屏渲染开销；感知、标定与
        离线验收则保持默认的 RGB-D 输出。
        """
        if not self._renderers:
            raise RuntimeError("相机渲染器尚未初始化")
        renderer = self._renderers[name]
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        renderer.update_scene(self.data, camera=name)
        rgb = renderer.render().copy()
        if include_depth:
            renderer.enable_depth_rendering()
            depth = renderer.render().copy()
            renderer.disable_depth_rendering()
        else:
            depth = np.empty((0, 0), dtype=np.float32)
        fovy = np.deg2rad(self.model.cam_fovy[camera_id])
        focal = (self.height / 2.0) / np.tan(fovy / 2.0)
        intrinsic = np.array(
            [[focal, 0.0, (self.width - 1) / 2.0], [0.0, focal, (self.height - 1) / 2.0], [0.0, 0.0, 1.0]]
        )
        world_from_camera = np.eye(4)
        world_from_camera[:3, :3] = self.data.cam_xmat[camera_id].reshape(3, 3)
        world_from_camera[:3, 3] = self.data.cam_xpos[camera_id]
        base_id = self.model.body("link0").id
        base_from_world = np.eye(4)
        base_from_world[:3, :3] = self.data.xmat[base_id].reshape(3, 3).T
        base_from_world[:3, 3] = -base_from_world[:3, :3] @ self.data.xpos[base_id]
        return UnifiedCameraFrame(
            name=name,
            rgb=rgb,
            depth=depth,
            intrinsic=intrinsic,
            world_from_camera=world_from_camera,
            base_from_camera=base_from_world @ world_from_camera,
            timestamp=float(self.data.time),
        )

    def cameras(
        self,
        names: tuple[str, ...] = ("global", "fixed", "wrist"),
        include_depth: bool = True,
    ) -> dict[str, UnifiedCameraFrame]:
        """按需取得相机帧，名称为 ``global``、``fixed`` 或 ``wrist``。"""
        valid_names = {"global", "fixed", "wrist"}
        unknown = set(names) - valid_names
        if unknown:
            raise ValueError(f"未知相机名称：{sorted(unknown)}")
        return {name: self.render_camera(f"camera_{name}", include_depth=include_depth) for name in names}

    def synchronization_snapshot(self) -> UnifiedSceneSnapshot:
        """生成阶段 2 使用的原子场景同步输入，全部位姿表达在 ``panda_link0``。"""
        base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link0")
        base_from_world = np.eye(4)
        base_from_world[:3, :3] = self.data.xmat[base_id].reshape(3, 3).T
        base_from_world[:3, 3] = -base_from_world[:3, :3] @ self.data.xpos[base_id]
        joint_names = (
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
            "finger_joint1",
            "finger_joint2",
        )
        joint_positions = {
            name: float(self.data.qpos[self.model.jnt_qposadr[self.model.joint(name).id]])
            for name in joint_names
        }
        objects: dict[str, dict[str, np.ndarray]] = {}
        for object_name in ("red_cylinder", "green_cylinder", "blue_cylinder", "purple_cube", "magenta_block", "yellow_mug", "tray"):
            body_id = self.model.body(object_name).id
            world_from_object = np.eye(4)
            world_from_object[:3, :3] = self.data.xmat[body_id].reshape(3, 3)
            world_from_object[:3, 3] = self.data.xpos[body_id]
            base_from_object = base_from_world @ world_from_object
            objects[object_name] = {
                "position_base_m": base_from_object[:3, 3].copy(),
                "rotation_base": base_from_object[:3, :3].copy(),
            }
        return UnifiedSceneSnapshot(float(self.data.time), joint_positions, objects)

    def evaluation_only_truth(self, object_name: str) -> np.ndarray:
        """仅评测通路的物体位置；阶段 3 起控制代码不得调用。"""
        body_id = self.model.body(object_name).id
        return self.data.xpos[body_id].copy()

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
