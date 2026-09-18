"""可离线运行的端到端视觉任务演示。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .perception import ColorSemanticAdapter
from .simulator import CupTabletopSimulation
from .task import PickPlaceStateMachine


def _save_ppm(path: Path, rgb: np.ndarray) -> None:
    with path.open("wb") as output:
        output.write(f"P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n".encode())
        output.write(rgb.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="模拟相机到 MoveIt 技能目标的离线演示")
    parser.add_argument("--instruction", default="抓取红色杯子并放到托盘")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--root", type=Path, default=Path(os.environ["MOVEIT_EXPERIMENT_ROOT"]))
    arguments = parser.parse_args()
    simulation = CupTabletopSimulation(arguments.root / "assets" / "colored_cup_scene.xml")
    simulation.randomize_cups(arguments.seed)
    frame = simulation.render_camera()
    estimate = ColorSemanticAdapter().estimate(arguments.instruction, frame)
    skills = PickPlaceStateMachine().build_plan(estimate)
    log_dir = arguments.root / "logs" / "latest"
    log_dir.mkdir(parents=True, exist_ok=True)
    _save_ppm(log_dir / "camera.ppm", frame.rgb)
    truth = simulation.evaluation_truth(estimate.object_id)
    payload = {
        "instruction": arguments.instruction,
        "estimate": {"object_id": estimate.object_id, "position_base_m": estimate.position_base.tolist(), "confidence": estimate.confidence},
        "evaluation_only_truth_position_base_m": truth.tolist(),
        "translation_error_m": float(np.linalg.norm(estimate.position_base - truth)),
        "skills": [{"name": item.skill, "position_base_m": item.position_base.tolist()} for item in skills],
    }
    (log_dir / "episode.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
