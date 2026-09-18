"""视觉输入到位姿的可替换接口；当前颜色基线不是 VLM。"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .simulator import CameraFrame


@dataclass(frozen=True)
class ObjectEstimate:
    object_id: str
    position_base: np.ndarray
    confidence: float
    source_frame: str = "camera_fixed"


class ColorSemanticAdapter:
    cup_height_m = 0.12
    colors = {
        # 还需限制 G/B 绝对亮度，避免暖色桌面或黄色托盘被误判为红杯。
        "红色": ("red_cup", lambda x: (x[..., 0] > 120) & (x[..., 1] < 100) & (x[..., 2] < 100)),
        "蓝色": ("blue_cup", lambda x: (x[..., 2] > 105) & (x[..., 2] > x[..., 0] * 1.25) & (x[..., 2] > x[..., 1] * 1.15)),
        "绿色": ("green_cup", lambda x: (x[..., 1] > 100) & (x[..., 1] > x[..., 0] * 1.3) & (x[..., 1] > x[..., 2] * 1.15)),
    }

    def estimate(self, instruction: str, frame: CameraFrame) -> ObjectEstimate:
        selected = [name for name in self.colors if name in instruction]
        if len(selected) != 1:
            raise ValueError("指令必须且只能包含一种支持的颜色：红色、蓝色或绿色")
        object_id, selector = self.colors[selected[0]]
        mask = selector(frame.rgb)
        pixels = int(mask.sum())
        if pixels < 40:
            raise RuntimeError(f"未从模拟相机中找到{selected[0]}杯子")
        rows, columns = np.nonzero(mask)
        row, column = int(np.median(rows)), int(np.median(columns))
        depth = float(np.median(frame.depth[mask]))
        if not np.isfinite(depth) or depth <= 0:
            raise RuntimeError("目标区域没有有效深度")
        fx, fy, cx, cy = frame.intrinsic[0, 0], frame.intrinsic[1, 1], frame.intrinsic[0, 2], frame.intrinsic[1, 2]
        point_camera = np.array([(column - cx) * depth / fx, -(row - cy) * depth / fy, -depth, 1.0])
        point_base = frame.world_from_camera @ point_camera
        # 深度落在杯口可见表面；根据类别先验的标准杯高换算到杯体几何中心。
        point_base[2] -= self.cup_height_m / 2.0
        return ObjectEstimate(object_id, point_base[:3], min(1.0, pixels / 700.0))
