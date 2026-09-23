"""按任务阶段生成 MoveIt 碰撞场景策略。"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class PlanningPhase(StrEnum):
    """任务状态机对应的规划阶段。"""

    PREGRASP = "pregrasp"
    APPROACH = "approach"
    TRANSPORT = "transport"
    PLACE = "place"


@dataclass(frozen=True)
class TaskPlanningScenePolicy:
    """根据目标和阶段决定哪些物体留在世界碰撞场景中。

    ``compatibility_exclusions`` 是当前简化 Panda URDF 与 MoveIt 工装近似模型
    的已知假相交。它们在 MuJoCo 中仍参与真实物理接触；但若把它们加入 MoveIt
    场景，抓取重试时的起始状态会被错误判为碰撞，规划器甚至不会开始搜索。因此
    在整条抓取轨迹中都排除它们，而不是只在 home/pregrasp 阶段排除。
    """

    target_object: str
    destination_object: str = "tray"
    phase: PlanningPhase = PlanningPhase.PREGRASP
    compatibility_exclusions: frozenset[str] = frozenset(
        {"green_cylinder"}
    )

    def world_exclusions(self) -> frozenset[str]:
        exclusions = set(self.compatibility_exclusions)
        if self.phase in {
            PlanningPhase.APPROACH,
            PlanningPhase.TRANSPORT,
            PlanningPhase.PLACE,
        }:
            exclusions.add(self.target_object)
        # MoveIt 中的 tray 是实心近似盒，不能表达 MuJoCo 托盘的内腔。末段下降
        # 若保留它会被当作碰撞；实际接触和“落入托盘”判定由 MuJoCo 完整场景负责。
        if self.phase == PlanningPhase.PLACE:
            exclusions.add(self.destination_object)
        return frozenset(exclusions)

    def attached_object(self) -> str | None:
        if self.phase in {PlanningPhase.TRANSPORT, PlanningPhase.PLACE}:
            return self.target_object
        return None
