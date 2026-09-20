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

    def __init__(self, root: Path, width: int = 640, height: int = 480, create_renderers: bool = True) -> None:
        self.root = root
        scene = root / "assets" / "panda_mjcf" / "unified_panda_cup_scene.xml"
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
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
        # Panda 的官方 keyframe 只定义了机械臂关节；场景后来加入的 freejoint
        # 必须在这里显式复位，否则会继承为零位姿，导致杯子落到世界原点。
        cup_positions = {
            "red_cup_freejoint": (0.48, -0.16, 0.465),
            "green_cup_freejoint": (0.34, 0.06, 0.465),
            "blue_cup_freejoint": (0.30, -0.24, 0.465),
        }
        for joint_name, position in cup_positions.items():
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qpos_address = self.model.jnt_qposadr[joint_id]
            dof_address = self.model.jnt_dofadr[joint_id]
            self.data.qpos[qpos_address : qpos_address + 3] = position
            self.data.qpos[qpos_address + 3 : qpos_address + 7] = (1.0, 0.0, 0.0, 0.0)
            self.data.qvel[dof_address : dof_address + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def hold_home(self, steps: int = 1) -> None:
        """阶段 1 仅稳定保持初始姿态；不承担任务轨迹执行。"""
        self.data.ctrl[:] = self.model.key_ctrl[0]
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

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
        for object_name in ("red_cup", "green_cup", "blue_cup", "tray"):
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
