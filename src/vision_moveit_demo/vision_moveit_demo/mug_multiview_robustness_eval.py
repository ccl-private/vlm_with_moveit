"""保存的主动腕部观测的现实退化回归。

本评测和控制路径严格分离：输入仅为一次真实执行前保存的 RGB-D、实例掩码、
相机外参与由固定相机产生的粗 yaw；MuJoCo pose 只在最后计算误差，不传入
分割、CAD 匹配或候选选择。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cad_matching import CadMatcherDispatcher, ObjectCatalog, RuleVlmAdapter, SegmentationResult
from .mug_robustness_eval import PROFILES, _degrade_frame_and_mask, _yaw_error_rad
from .unified_scene import UnifiedCameraFrame


def _load_observations(directory: Path) -> list[tuple[SegmentationResult, UnifiedCameraFrame, float]]:
    paths = sorted(directory.glob("*.npz"))
    if len(paths) < 2:
        raise ValueError(f"主动观察目录至少需要两帧 .npz：{directory}")
    observations = []
    for path in paths:
        data = np.load(path)
        required = {"rgb", "depth", "mask", "intrinsic", "base_from_camera"}
        if not required.issubset(data.files):
            raise ValueError(f"主动观察文件字段不完整：{path}")
        mask = data["mask"].astype(bool)
        rows, columns = np.nonzero(mask)
        if len(rows) < 120:
            raise ValueError(f"主动观察掩码像素不足：{path}")
        segmentation = SegmentationResult(
            mask=mask,
            confidence=1.0,
            pixel_count=int(len(rows)),
            bounding_box_xyxy=(int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1),
            source="saved_active_observation",
        )
        frame = UnifiedCameraFrame(
            name=path.stem,
            rgb=data["rgb"],
            depth=data["depth"],
            intrinsic=data["intrinsic"],
            world_from_camera=data["base_from_camera"],
            base_from_camera=data["base_from_camera"],
            timestamp=0.0,
        )
        # 旧日志没有该字段时不臆造可见性，退化为所有视角等权；新日志会保存它。
        visibility = float(data["expected_contact_visibility"]) if "expected_contact_visibility" in data.files else 1.0
        observations.append((segmentation, frame, visibility))
    return observations


def main() -> None:
    parser = argparse.ArgumentParser(description="保存的腕部 RGB-D 多视角马克杯现实退化回归")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--observation-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--truth-position-m", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--truth-yaw-rad", type=float, required=True)
    parser.add_argument(
        "--yaw-prior-rad",
        type=float,
        required=True,
        help="该回合固定相机粗匹配输出的 yaw；它是控制时已有的证据，不是 MuJoCo 真值。",
    )
    parser.add_argument("--output", type=Path, default=Path("logs/robustness/mug_sku_multiview.json"))
    arguments = parser.parse_args()
    if arguments.trials <= 0:
        raise ValueError("--trials 必须为正数")

    intent = RuleVlmAdapter().infer("抓取黄色马克杯把手并放到托盘")
    model = ObjectCatalog(arguments.root).find(intent)
    source = _load_observations(arguments.observation_dir)
    generator = np.random.default_rng(arguments.seed)
    truth_position = np.asarray(arguments.truth_position_m, dtype=np.float64)
    matcher = CadMatcherDispatcher()
    results: dict[str, object] = {
        "purpose": "离线评测；真值只用于误差统计，不作为匹配输入",
        "observation_dir": str(arguments.observation_dir),
        "yaw_prior_rad": arguments.yaw_prior_rad,
        "grasp_ready_threshold": {"position_error_m": 0.008, "yaw_error_deg": 12.0},
        "profiles": {},
    }
    for profile in PROFILES:
        records: list[dict[str, object]] = []
        for trial in range(arguments.trials):
            observations = []
            for segmentation, frame, visibility in source:
                degraded_frame, degraded_segmentation = _degrade_frame_and_mask(frame, segmentation, profile, generator)
                observations.append((degraded_segmentation, degraded_frame, visibility))
            try:
                pose = matcher.match_multiview(model, observations, yaw_prior_rad=arguments.yaw_prior_rad)
                position_error = float(np.linalg.norm(pose.position_base_m - truth_position))
                yaw_error_deg = float(np.rad2deg(_yaw_error_rad(pose.object_yaw_rad, arguments.truth_yaw_rad)))
                records.append(
                    {
                        "trial": trial,
                        "status": "matched",
                        "position_error_m": position_error,
                        "yaw_error_deg": yaw_error_deg,
                        "grasp_ready": position_error <= 0.008 and yaw_error_deg <= 12.0,
                        "critical_region_support": pose.critical_region_support,
                        "critical_region_margin": pose.critical_region_margin,
                    }
                )
            except RuntimeError as error:
                records.append({"trial": trial, "status": "rejected", "reason": str(error), "grasp_ready": False})
        matched = [record for record in records if record["status"] == "matched"]
        summary: dict[str, object] = {
            "trials": len(records),
            "matched": len(matched),
            "grasp_ready": sum(bool(record["grasp_ready"]) for record in records),
        }
        if matched:
            summary.update(
                position_error_median_m=float(np.median([record["position_error_m"] for record in matched])),
                yaw_error_median_deg=float(np.median([record["yaw_error_deg"] for record in matched])),
            )
        results["profiles"][profile.name] = {"summary": summary, "records": records}
        print(f"[{profile.name}] {json.dumps(summary, ensure_ascii=False)}", flush=True)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(f"[已保存] {arguments.output}", flush=True)


if __name__ == "__main__":
    main()
