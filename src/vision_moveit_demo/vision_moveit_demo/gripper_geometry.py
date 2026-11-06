"""Panda 夹持工作点与开度标定；只读取机器人模型，不读取工件状态。"""
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class PandaGripperGeometry:
    # 标注抓取中心对应两侧最大接触 pad 的中心中点，朝向沿用标注约定。
    tcp_hand_m: np.ndarray
    tcp_link8_m: np.ndarray
    minimum_gap_at_zero_m: float
    finger_travel_m: float

    @classmethod
    def from_model(cls, model):
        data = mujoco.MjData(model)
        for name in ("finger_joint1", "finger_joint2"):
            data.qpos[model.jnt_qposadr[model.joint(name).id]] = 0.0
        mujoco.mj_kinematics(model, data)
        hand = model.body("hand").id
        hand_rotation = data.xmat[hand].reshape(3, 3)
        centers, inner_faces = [], []
        for name, side in (("left_finger", 1), ("right_finger", -1)):
            body = model.body(name).id
            pads = [g for g in range(model.ngeom)
                    if model.geom_bodyid[g] == body
                    and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
                    and model.geom_contype[g] != 0]
            if not pads:
                raise ValueError(f"{name} 缺少接触 pad，不能标定夹持工作点")
            primary = max(pads, key=lambda g: model.geom_size[g, 0] * model.geom_size[g, 2])
            centers.append(hand_rotation.T @ (data.geom_xpos[primary] - data.xpos[hand]))
            faces = []
            for g in pads:
                center = hand_rotation.T @ (data.geom_xpos[g] - data.xpos[hand])
                rotation = hand_rotation.T @ data.geom_xmat[g].reshape(3, 3)
                extent_y = float(np.abs(rotation[1]) @ model.geom_size[g])
                faces.append(side * center[1] - extent_y)
            inner_faces.append(min(faces))
        tcp = np.mean(centers, axis=0)
        # hand 在 link8 上的安装只有固定旋转；该旋转也从模型取得。
        mount_rotation = np.empty(9)
        mujoco.mju_quat2Mat(mount_rotation, model.body_quat[hand])
        travel = float(model.jnt_range[model.joint("finger_joint1").id, 1])
        if not np.allclose(tcp[:2], 0.0, atol=1e-7):
            raise ValueError("当前 Panda 工作点必须位于工具中心轴")
        return cls(tcp, mount_rotation.reshape(3, 3) @ tcp,
                   float(sum(inner_faces)), travel)

    def finger_position_for_width(self, width_m: float) -> float:
        """标注宽度是预张开净空，不能静默截断为小于所需宽度。"""
        maximum = self.minimum_gap_at_zero_m + 2 * self.finger_travel_m
        if not np.isfinite(width_m) or not self.minimum_gap_at_zero_m <= width_m <= maximum:
            raise ValueError(f"夹爪净开度 {width_m} 不在模型范围内")
        return (float(width_m) - self.minimum_gap_at_zero_m) / 2

    def as_dict(self):
        return dict(tcp_hand_m=self.tcp_hand_m.tolist(), tcp_link8_m=self.tcp_link8_m.tolist(),
                    minimum_gap_at_zero_m=self.minimum_gap_at_zero_m,
                    finger_travel_m=self.finger_travel_m,
                    workpoint="midpoint_of_primary_contact_pad_centers")
