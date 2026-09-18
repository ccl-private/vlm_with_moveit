"""文本任务到传统技能序列，不产生关节级命令。"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .perception import ObjectEstimate


@dataclass(frozen=True)
class SkillTarget:
    skill: str
    position_base: np.ndarray


class PickPlaceStateMachine:
    def __init__(self, minimum_confidence: float = 0.45) -> None:
        self.minimum_confidence = minimum_confidence

    def build_plan(self, estimate: ObjectEstimate) -> list[SkillTarget]:
        if estimate.confidence < self.minimum_confidence:
            raise RuntimeError("感知置信度不足，拒绝向 MoveIt 发送目标")
        if np.linalg.norm(estimate.position_base[:2]) > 0.55:
            raise RuntimeError("目标超出工作空间，拒绝规划")
        grasp = estimate.position_base.copy()
        return [
            SkillTarget("预抓取", grasp + np.array([0.0, 0.0, 0.12])),
            SkillTarget("接近", grasp), SkillTarget("闭合夹爪", grasp),
            SkillTarget("抬升", grasp + np.array([0.0, 0.0, 0.16])),
            SkillTarget("托盘上方", np.array([0.25, 0.18, 0.55])),
            SkillTarget("打开夹爪", np.array([0.25, 0.18, 0.55])),
        ]
