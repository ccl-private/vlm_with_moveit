"""阶段 1 的统一场景离屏验收，不启动 MoveIt 或抓取执行。"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .unified_scene import UnifiedPandaCupSimulation


def _pixel_count(rgb: np.ndarray, color: str) -> int:
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    selectors = {
        "red_cylinder": (red > 120) & (green < 100) & (blue < 100),
        "green_cylinder": (green > 100) & (green > red * 1.3) & (green > blue * 1.15),
        "blue_cylinder": (blue > 105) & (blue > red * 1.25) & (blue > green * 1.15),
    }
    return int(selectors[color].sum())


def main() -> None:
    root = Path(os.environ["MOVEIT_EXPERIMENT_ROOT"])
    simulation = UnifiedPandaCupSimulation(root)
    try:
        frames = simulation.cameras()
        fixed = frames["fixed"]
        snapshot = simulation.synchronization_snapshot()
        visible_pixels = {name: _pixel_count(fixed.rgb, name) for name in ("red_cylinder", "green_cylinder", "blue_cylinder")}
        if any(count < 40 for count in visible_pixels.values()):
            raise RuntimeError(f"固定相机未清楚看到全部杯子：{visible_pixels}")
        if fixed.rgb.shape != (480, 640, 3) or fixed.depth.shape != (480, 640):
            raise RuntimeError(f"固定相机尺寸异常：RGB={fixed.rgb.shape}，深度={fixed.depth.shape}")
        if not all(np.isfinite(frame.depth).all() for frame in frames.values()):
            raise RuntimeError("至少一路相机含有无效深度值")
        if abs(snapshot.timestamp - fixed.timestamp) > 1e-9:
            raise RuntimeError("相机帧与同步快照时间戳不一致")
        camera_from_base = np.linalg.inv(fixed.base_from_camera)
        camera_points = {
            name: (camera_from_base @ np.append(simulation.evaluation_only_truth(name), 1.0))[:3]
            for name in visible_pixels
        }
        # MuJoCo 相机坐标采用视线 -Z；三只杯子在固定相机前方即 z<0。
        if not all(point[2] < 0.0 for point in camera_points.values()):
            raise RuntimeError(f"固定相机外参方向错误：{camera_points}")
        result = {
            "status": "passed",
            "timestamp_s": snapshot.timestamp,
            "fixed_camera_visible_pixels": visible_pixels,
            "fixed_camera_cup_positions_mujoco_camera": {
                name: point.round(6).tolist() for name, point in camera_points.items()
            },
            "snapshot_joint_count": len(snapshot.joint_positions),
            "snapshot_object_names": sorted(snapshot.planning_scene_objects),
            "note": "验证使用的物体真值仅用于阶段 1 坐标系验收，不是控制输入。",
        }
        output = root / "logs" / "unified_preview"
        output.mkdir(parents=True, exist_ok=True)
        (output / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        simulation.close()


if __name__ == "__main__":
    main()
