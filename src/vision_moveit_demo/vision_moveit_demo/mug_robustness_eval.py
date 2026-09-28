"""固定 RGB-D 下 SKU 级马克杯匹配器的现实退化回归。

此模块只在离线评测阶段读取由 MuJoCo 随机生成的 pose 标签来计算误差；标签绝不
传入分割、CAD 匹配或抓取位姿生成。目的不是声称仿真等同真实相机，而是把理想
渲染中的隐含假设显式变成可复现的噪声、遮挡和标定误差测试。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path

import mujoco
import numpy as np

from .cad_matching import CadMatcherDispatcher, ColorThresholdSegmenter, ObjectCatalog, RuleVlmAdapter, SegmentationResult
from .unified_scene import UnifiedCameraFrame, UnifiedPandaCupSimulation


@dataclass(frozen=True)
class DegradationProfile:
    name: str
    depth_noise_std_m: float
    depth_dropout_ratio: float
    mask_dropout_ratio: float
    mask_erosion_px: int
    occlusion_ratio: float
    calibration_translation_std_m: float
    calibration_yaw_std_rad: float
    yaw_span_rad: float


PROFILES = (
    DegradationProfile("受控基线", 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, np.deg2rad(15.0)),
    DegradationProfile("现实轻度", 0.0015, 0.05, 0.03, 1, 0.10, 0.0015, np.deg2rad(0.5), np.pi),
    DegradationProfile("现实中度", 0.0030, 0.14, 0.08, 2, 0.22, 0.0030, np.deg2rad(1.0), np.pi),
)


def _yaw_error_rad(estimate: float | None, truth: float) -> float:
    if estimate is None:
        return float("inf")
    return float(abs(np.arctan2(np.sin(estimate - truth), np.cos(estimate - truth))))


def _erode(mask: np.ndarray, iterations: int) -> np.ndarray:
    result = mask.copy()
    for _ in range(iterations):
        result = (
            result
            & np.roll(result, 1, axis=0)
            & np.roll(result, -1, axis=0)
            & np.roll(result, 1, axis=1)
            & np.roll(result, -1, axis=1)
        )
        result[0, :] = result[-1, :] = result[:, 0] = result[:, -1] = False
    return result


def _segmentation_from_mask(mask: np.ndarray, reference: SegmentationResult) -> SegmentationResult:
    rows, columns = np.nonzero(mask)
    if len(rows) < 120:
        raise RuntimeError("现实退化后目标 mask 有效像素不足")
    return replace(
        reference,
        mask=mask,
        pixel_count=int(len(rows)),
        bounding_box_xyxy=(int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1),
        confidence=float(reference.confidence * len(rows) / max(1, reference.pixel_count)),
        source="simulated_realistic_segmentation_degradation",
    )


def _degrade_frame_and_mask(
    frame: UnifiedCameraFrame,
    segmentation: SegmentationResult,
    profile: DegradationProfile,
    generator: np.random.Generator,
) -> tuple[UnifiedCameraFrame, SegmentationResult]:
    mask = _erode(segmentation.mask, profile.mask_erosion_px)
    target_rows, target_columns = np.nonzero(mask)
    keep = generator.random(len(target_rows)) >= profile.mask_dropout_ratio
    mask[target_rows[~keep], target_columns[~keep]] = False
    rows, columns = np.nonzero(mask)
    if profile.occlusion_ratio > 0.0 and len(rows):
        # 遮挡使用前景包围框内一条随机矩形带近似；它会保留场景渲染本身的几何，
        # 只模拟该区域没有可靠分割/深度这一真实传感器现象。
        width = max(1, int((columns.max() - columns.min() + 1) * profile.occlusion_ratio))
        start = int(generator.integers(columns.min(), columns.max() + 1))
        end = min(mask.shape[1], start + width)
        mask[:, start:end] = False
    degraded = _segmentation_from_mask(mask, segmentation)
    depth = frame.depth.copy()
    rows, columns = np.nonzero(degraded.mask)
    if profile.depth_noise_std_m:
        depth[rows, columns] += generator.normal(0.0, profile.depth_noise_std_m, len(rows))
    dropout = generator.random(len(rows)) < profile.depth_dropout_ratio
    depth[rows[dropout], columns[dropout]] = np.nan
    yaw_error = generator.normal(0.0, profile.calibration_yaw_std_rad)
    cosine, sine = np.cos(yaw_error), np.sin(yaw_error)
    calibration_error = np.eye(4)
    calibration_error[:3, :3] = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    calibration_error[:3, 3] = generator.normal(0.0, profile.calibration_translation_std_m, 3)
    return (
        replace(frame, depth=depth, base_from_camera=calibration_error @ frame.base_from_camera),
        degraded,
    )


def _set_mug_pose(simulation: UnifiedPandaCupSimulation, position: np.ndarray, yaw_rad: float) -> None:
    joint_id = simulation.model.joint("yellow_mug_freejoint").id
    qpos_address = simulation.model.jnt_qposadr[joint_id]
    dof_address = simulation.model.jnt_dofadr[joint_id]
    simulation.data.qpos[qpos_address : qpos_address + 3] = position
    simulation.data.qpos[qpos_address + 3 : qpos_address + 7] = np.array(
        [np.cos(yaw_rad / 2.0), 0.0, 0.0, np.sin(yaw_rad / 2.0)]
    )
    simulation.data.qvel[dof_address : dof_address + 6] = 0.0
    mujoco.mj_forward(simulation.model, simulation.data)


def _summarize(records: list[dict[str, object]]) -> dict[str, object]:
    matched = [record for record in records if record["status"] == "matched"]
    ready = [record for record in records if record.get("grasp_ready")]
    if not matched:
        return {"trials": len(records), "matched": 0, "grasp_ready": 0, "grasp_ready_rate": 0.0}
    position_errors = np.asarray([record["position_error_m"] for record in matched], dtype=float)
    yaw_errors = np.asarray([record["yaw_error_deg"] for record in matched], dtype=float)
    return {
        "trials": len(records),
        "matched": len(matched),
        "grasp_ready": len(ready),
        "grasp_ready_rate": len(ready) / len(records),
        "position_error_median_m": float(np.median(position_errors)),
        "position_error_p90_m": float(np.quantile(position_errors, 0.90)),
        "yaw_error_median_deg": float(np.median(yaw_errors)),
        "yaw_error_p90_deg": float(np.quantile(yaw_errors, 0.90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SKU 马克杯固定相机现实退化回归")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--trials", type=int, default=20, help="每种退化档的回合数，默认 20")
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--output", type=Path, default=Path("logs/robustness/mug_sku_fixed_camera.json"))
    arguments = parser.parse_args()
    if arguments.trials <= 0:
        raise ValueError("--trials 必须为正数")

    generator = np.random.default_rng(arguments.seed)
    simulation = UnifiedPandaCupSimulation(arguments.root)
    try:
        intent = RuleVlmAdapter().infer("抓取黄色马克杯把手并放到托盘")
        model = ObjectCatalog(arguments.root).find(intent)
        matcher = CadMatcherDispatcher()
        results: dict[str, object] = {
            "purpose": "离线评测；真值只用于误差统计，不作为匹配输入",
            "grasp_ready_threshold": {"position_error_m": 0.008, "yaw_error_deg": 12.0},
            "profiles": {},
        }
        for profile in PROFILES:
            records: list[dict[str, object]] = []
            for trial_index in range(arguments.trials):
                # 每回合都改变桌面内小范围位置；轻度档只改变可见把手附近的 yaw，
                # 两个现实档则覆盖完整桌面 yaw，包含把手被杯身遮挡的困难情况。
                position = np.array([0.45, -0.12, 0.445]) + np.array(
                    [generator.uniform(-0.012, 0.012), generator.uniform(-0.012, 0.012), 0.0]
                )
                yaw = np.pi + generator.uniform(-profile.yaw_span_rad, profile.yaw_span_rad)
                _set_mug_pose(simulation, position, yaw)
                try:
                    frame = simulation.cameras(("fixed",))["fixed"]
                    segmentation = ColorThresholdSegmenter().segment(frame.rgb, intent)
                    degraded_frame, degraded_segmentation = _degrade_frame_and_mask(frame, segmentation, profile, generator)
                    estimate = matcher.match(model, degraded_segmentation, degraded_frame)
                    position_error = float(np.linalg.norm(estimate.position_base_m - position))
                    yaw_error_deg = float(np.rad2deg(_yaw_error_rad(estimate.object_yaw_rad, yaw)))
                    records.append({
                        "trial": trial_index,
                        "status": "matched",
                        "position_error_m": position_error,
                        "yaw_error_deg": yaw_error_deg,
                        "grasp_ready": position_error <= 0.008 and yaw_error_deg <= 12.0,
                        "geometry_confidence": estimate.geometry_confidence,
                    })
                except RuntimeError as error:
                    records.append({"trial": trial_index, "status": "rejected", "reason": str(error), "grasp_ready": False})
            summary = _summarize(records)
            results["profiles"][profile.name] = {"summary": summary, "records": records}
            print(f"[{profile.name}] {json.dumps(summary, ensure_ascii=False)}", flush=True)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
        print(f"[已保存] {arguments.output}", flush=True)
    finally:
        simulation.close()


if __name__ == "__main__":
    main()
