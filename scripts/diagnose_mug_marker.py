#!/usr/bin/env python3
"""打印固定相机中马克杯把手视觉标记的候选连通域。"""
from pathlib import Path
import os
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "vision_moveit_demo"))
os.environ.setdefault("MUJOCO_GL", "egl")

from vision_moveit_demo.cad_matching import _points_from_mask  # noqa: E402
from vision_moveit_demo.unified_scene import UnifiedPandaCupSimulation  # noqa: E402


def components(mask: np.ndarray) -> list[np.ndarray]:
    """四邻域连通域；无第三方图像库依赖。"""
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    output: list[np.ndarray] = []
    for start_y, start_x in zip(*np.nonzero(mask)):
        if seen[start_y, start_x]:
            continue
        queue = [(int(start_y), int(start_x))]
        seen[start_y, start_x] = True
        pixels: list[tuple[int, int]] = []
        while queue:
            y, x = queue.pop()
            pixels.append((y, x))
            for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= next_y < height and 0 <= next_x < width and mask[next_y, next_x] and not seen[next_y, next_x]:
                    seen[next_y, next_x] = True
                    queue.append((next_y, next_x))
        if len(pixels) >= 5:
            output.append(np.asarray(pixels, dtype=np.int32))
    return output


def main() -> None:
    simulation = UnifiedPandaCupSimulation(ROOT)
    try:
        frame = simulation.cameras()["fixed"]
        image_path = ROOT / "logs" / "stage3_cad_vnc" / "mug_marker_diagnostic.ppm"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(f"P6\n{frame.rgb.shape[1]} {frame.rgb.shape[0]}\n255\n".encode() + frame.rgb.tobytes())
        print(f"[固定相机图像] {image_path}")
        marker_id = simulation.model.geom("yellow_mug_handle_visual_marker").id
        marker_world = simulation.data.geom_xpos[marker_id].copy()
        camera_from_world = np.linalg.inv(frame.world_from_camera)
        marker_camera = camera_from_world[:3, :3] @ marker_world + camera_from_world[:3, 3]
        marker_pixel = frame.intrinsic @ (marker_camera / marker_camera[2])
        print(
            f"[标定片真渲染位置] world={np.round(marker_world, 4).tolist()}，"
            f"camera={np.round(marker_camera, 4).tolist()}，"
            f"pixel={np.round(marker_pixel[:2], 1).tolist()}"
        )
        pixel_x, pixel_y = np.round(marker_pixel[:2]).astype(int)
        patch = frame.rgb[max(0, pixel_y - 4) : pixel_y + 5, max(0, pixel_x - 4) : pixel_x + 5]
        print(f"[标定片像素邻域 RGB] 最小={patch.reshape(-1, 3).min(axis=0).tolist()}，最大={patch.reshape(-1, 3).max(axis=0).tolist()}")
        red, green, blue = frame.rgb[..., 0], frame.rgb[..., 1], frame.rgb[..., 2]
        marker_mask = (red < 90) & (green > 120) & (blue < 150) & (green > red * 1.8)
        rows, columns = np.nonzero(
            (red > 120) & (green > 75) & (blue < 38) & (red > green * 1.15) & (green > blue * 2.2)
        )
        print(f"[杯身像素框] x=[{columns.min()}, {columns.max()}] y=[{rows.min()}, {rows.max()}]")
        for index, pixels in enumerate(sorted(components(marker_mask), key=len, reverse=True)[:20], start=1):
            candidate = np.zeros_like(marker_mask)
            candidate[pixels[:, 0], pixels[:, 1]] = True
            points = _points_from_mask(frame, candidate)
            print(
                f"[绿色候选 {index}] 像素={len(pixels)}，"
                f"框=({pixels[:, 1].min()}, {pixels[:, 0].min()})-({pixels[:, 1].max()}, {pixels[:, 0].max()})，"
                f"三维均值={np.round(points.mean(axis=0), 4).tolist()}"
            )
    finally:
        for renderer in simulation._renderers.values():
            renderer.close()


if __name__ == "__main__":
    main()
