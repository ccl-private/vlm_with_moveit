"""MuJoCo 模拟相机；真值接口只供评测读取。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402


@dataclass(frozen=True)
class CameraFrame:
    rgb: np.ndarray
    depth: np.ndarray
    intrinsic: np.ndarray
    world_from_camera: np.ndarray
    timestamp: float


class CupTabletopSimulation:
    """控制路径只可获得 render_camera；真值入口单独命名为 evaluation_truth。"""

    def __init__(self, scene_path: Path, width: int = 640, height: int = 480) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.width, self.height = width, height
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.camera_name = "camera_fixed"
        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        mujoco.mj_forward(self.model, self.data)

    def randomize_cups(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        occupied: list[np.ndarray] = []
        for body_name in ("red_cup", "blue_cup", "green_cup"):
            while True:
                xy = rng.uniform(low=(-0.23, -0.18), high=(0.18, 0.13), size=2)
                if all(np.linalg.norm(xy - other) > 0.12 for other in occupied):
                    occupied.append(xy)
                    break
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            joint_id = self.model.body_jntadr[body_id]
            address = self.model.jnt_qposadr[joint_id]
            self.data.qpos[address : address + 3] = (xy[0], xy[1], 0.455)
            self.data.qpos[address + 3 : address + 7] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(self.model, self.data)

    def render_camera(self) -> CameraFrame:
        self.renderer.update_scene(self.data, camera=self.camera_name)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        fovy = np.deg2rad(self.model.cam_fovy[self.camera_id])
        focal = (self.height / 2.0) / np.tan(fovy / 2.0)
        intrinsic = np.array([[focal, 0.0, (self.width - 1) / 2.0], [0.0, focal, (self.height - 1) / 2.0], [0.0, 0.0, 1.0]])
        transform = np.eye(4)
        transform[:3, :3] = self.data.cam_xmat[self.camera_id].reshape(3, 3)
        transform[:3, 3] = self.data.cam_xpos[self.camera_id]
        return CameraFrame(rgb, depth, intrinsic, transform, float(self.data.time))

    def evaluation_truth(self, object_id: str) -> np.ndarray:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_id)
        return self.data.xpos[body_id].copy()
