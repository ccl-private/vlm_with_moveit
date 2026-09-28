"""主动腕部观察的几何规划基础。

本模块只根据 CAD 抓取标注、已估计的粗位姿与已标定的手眼变换生成观察位；
不读取仿真物体真值。MoveIt 负责在调用侧筛掉不可达或碰撞的候选。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObservationView:
    """一个待由 MoveIt 验证的腕部观察位。"""

    view_id: str
    camera_position_base_m: np.ndarray
    camera_orientation_base_xyzw: np.ndarray
    hand_position_base_m: np.ndarray
    hand_orientation_base_xyzw: np.ndarray
    expected_contact_visibility: float
    visible_contact_indices: tuple[int, ...]


def _normalize(vector: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        raise ValueError(f"{name} 不能为零向量")
    return vector / norm


def matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """将正交旋转矩阵转换为 xyzw 四元数。"""
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
            y = 0.25 * scale
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
            w = (rotation[0, 2] - rotation[2, 0]) / scale
        else:
            scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            z = 0.25 * scale
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            w = (rotation[1, 0] - rotation[0, 1]) / scale
    quaternion = np.array([x, y, z, w], dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


def camera_look_at_rotation(camera_position_base_m: np.ndarray, target_position_base_m: np.ndarray) -> np.ndarray:
    """生成 MuJoCo 相机的基座系旋转；其局部 -Z 轴指向目标。"""
    forward = _normalize(target_position_base_m - camera_position_base_m, "相机到目标方向")
    camera_z = -forward
    reference_up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(reference_up, camera_z))) > 0.97:
        reference_up = np.array([0.0, 1.0, 0.0])
    camera_x = _normalize(np.cross(reference_up, camera_z), "相机横向轴")
    camera_y = _normalize(np.cross(camera_z, camera_x), "相机纵向轴")
    return np.column_stack((camera_x, camera_y, camera_z))


def _camera_roll_rotation(angle_rad: float) -> np.ndarray:
    """相机局部绕光轴的滚转；不改变局部 -Z 的注视方向。"""
    cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def generate_observation_views(
    target_position_base_m: np.ndarray,
    contact_centers_base_m: np.ndarray,
    contact_normals_base: np.ndarray,
    hand_from_camera: np.ndarray,
    radius_m: float = 0.26,
    elevation_m: float = 0.30,
) -> list[ObservationView]:
    """生成围绕目标的有限 NBV 集，并以标注接触面的正面可见性排序。

    ``hand_from_camera`` 为当前标定的 ``T_hand_camera``。返回的 hand 位姿尚未
    通过 MoveIt；调用方必须做 IK、碰撞和轨迹筛选后才可执行。
    """
    target = np.asarray(target_position_base_m, dtype=np.float64)
    centers = np.asarray(contact_centers_base_m, dtype=np.float64)
    normals = np.asarray(contact_normals_base, dtype=np.float64)
    transform_hand_camera = np.asarray(hand_from_camera, dtype=np.float64)
    if target.shape != (3,) or centers.ndim != 2 or centers.shape[1:] != (3,) or normals.shape != centers.shape:
        raise ValueError("目标、接触中心和接触法向量的形状无效")
    if transform_hand_camera.shape != (4, 4):
        raise ValueError("hand_from_camera 必须为 4x4 变换")
    if radius_m <= 0.0 or elevation_m <= 0.0:
        raise ValueError("观察半径和抬高量必须为正数")

    # 四个侧向视角均独立于粗 pose 的 yaw，能在粗 yaw 错误时仍有机会看见真实把手。
    # 它们先保持足够高的桌面净空，再由腕相机俯视目标；额外高位视角负责维持
    # 目标整体点云覆盖。
    offset_specs: list[tuple[str, np.ndarray]] = [
        ("wrist_view_0", np.array([radius_m, 0.0, elevation_m])),
        ("wrist_view_1", np.array([0.0, radius_m, elevation_m])),
        ("wrist_view_2", np.array([-radius_m, 0.0, elevation_m])),
        ("wrist_view_3", np.array([0.0, -radius_m, elevation_m])),
        ("wrist_view_4", np.array([0.0, 0.0, radius_m + elevation_m])),
    ]
    camera_from_hand = np.linalg.inv(transform_hand_camera)
    views: list[ObservationView] = []
    for view_id, offset in offset_specs:
        camera_position = target + offset
        look_at_rotation = camera_look_at_rotation(camera_position, target)
        sight_vectors = camera_position[None, :] - centers
        sight_vectors /= np.linalg.norm(sight_vectors, axis=1, keepdims=True)
        # 物体表面法向量朝向相机才可直接观察；最多两片相对接触面会分别由不同
        # 侧视角覆盖，分数用来排序而非假定一次观察能看见所有面。
        visibility = np.clip((normals * sight_vectors).sum(axis=1), 0.0, 1.0)
        visible = tuple(int(item) for item in np.flatnonzero(visibility >= 0.20))
        # 手眼外参含有横向偏置：保持相机视线不变时，绕光轴滚转 180° 会把手掌
        # 移到另一侧。对近基座方向这常是从不可达到可达的关键，不应因固定的
        # 相机“正向”而丢失一个高价值观察位。
        rolls = (0.0, np.pi)
        for roll in rolls:
            camera_rotation = look_at_rotation @ _camera_roll_rotation(roll)
            camera_from_base = np.eye(4)
            camera_from_base[:3, :3] = camera_rotation
            camera_from_base[:3, 3] = camera_position
            hand_from_base = camera_from_base @ camera_from_hand
            suffix = "" if roll == 0.0 else "_roll_180"
            views.append(
                ObservationView(
                    view_id=f"{view_id}{suffix}",
                    camera_position_base_m=camera_position,
                    camera_orientation_base_xyzw=matrix_to_quaternion(camera_rotation),
                    hand_position_base_m=hand_from_base[:3, 3],
                    hand_orientation_base_xyzw=matrix_to_quaternion(hand_from_base[:3, :3]),
                    expected_contact_visibility=float(visibility.mean()),
                    visible_contact_indices=visible,
                )
            )
    return sorted(views, key=lambda view: (-view.expected_contact_visibility, view.view_id))
