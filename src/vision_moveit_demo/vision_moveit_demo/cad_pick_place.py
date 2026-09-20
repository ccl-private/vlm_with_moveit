"""阶段 3 首个闭环：圆柱 OBJ 模型匹配驱动的抓取放置。

控制目标只来自规则语义、RGB 分割、深度点云和预载 CAD 模型。MuJoCo 真值仅在
回合末尾做评测，不参与 CAD 配准、抓取目标生成或 MoveIt 场景坐标。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import rclpy

from .cad_matching import (
    ColorThresholdSegmenter,
    CylinderCadMatcher,
    ObjectCatalog,
    RuleVlmAdapter,
    grasp_pose_from_template,
)
from .stage2_executor import MujocoTaskExecutor
from .truth_baseline import MoveItTrajectoryClient, _execute_trajectory
from .unified_scene import UnifiedPandaCupSimulation


def _write_pgm(path: Path, mask: np.ndarray) -> None:
    image = np.where(mask, 255, 0).astype(np.uint8)
    with path.open("wb") as output:
        output.write(f"P5\n{image.shape[1]} {image.shape[0]}\n255\n".encode())
        output.write(image.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段 3：圆柱 CAD 模型匹配抓取基线")
    parser.add_argument("--instruction", default="抓取红色杯子并放到托盘")
    parser.add_argument("--root", type=Path, default=Path(os.environ["MOVEIT_EXPERIMENT_ROOT"]))
    parser.add_argument("--vnc", action="store_true", help="在 VNC 中显示 CAD 匹配驱动的执行过程。")
    arguments = parser.parse_args()

    simulation = UnifiedPandaCupSimulation(arguments.root, create_renderers=False)
    viewer = None
    if arguments.vnc:
        from .unified_preview import VncOverlayViewer

        viewer = VncOverlayViewer(simulation, title="MoveIt 阶段 3 CAD 模型匹配")
        simulation.initialize_renderers()
        viewer.glfw.make_context_current(viewer.window)
        last_frame_time = 0.0

        def render_frame(state: str) -> bool:
            nonlocal last_frame_time
            remaining = 1.0 / 30.0 - (time.monotonic() - last_frame_time)
            if remaining > 0.0:
                time.sleep(remaining)
            last_frame_time = time.monotonic()
            return viewer.render_once(state)

    else:
        simulation.initialize_renderers()
        render_frame = None

    try:
        intent = RuleVlmAdapter().infer(arguments.instruction)
        catalog = ObjectCatalog(arguments.root)
        model = catalog.find(intent)
        fixed_frame = simulation.cameras()["fixed"]
        segmentation = ColorThresholdSegmenter().segment(fixed_frame.rgb, intent)
        pose = CylinderCadMatcher().match(model, segmentation, fixed_frame)
        template = model.grasp_templates[0]
        grasp_position, grasp_orientation = grasp_pose_from_template(pose, template)
        # 当前 Panda 顶抓模板使用向下夹爪；任何目录模板方向不一致时应明确拒绝，
        # 而不是静默忽略其 CAD 抓取姿态。
        if not np.allclose(grasp_orientation, np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-6):
            raise RuntimeError("当前 MoveIt 桥接仅支持向下顶抓模板方向")
        tray_center = catalog.tray_center_base_m()
        targets = [
            ("pregrasp", grasp_position + np.array([0.0, 0.0, template.approach_clearance_m])),
            ("approach", grasp_position),
            ("lift", grasp_position + np.array([0.0, 0.0, template.lift_clearance_m])),
            # 托盘位于 Panda 有效工作空间边缘；该安全预位由静态工装配置推导，
            # 不是 MuJoCo 托盘真值读取。
            ("place", tray_center + np.array([-0.20, 0.02, 0.30])),
        ]
        executor = MujocoTaskExecutor(simulation, frame_callback=render_frame)
        executor.set_display_state("CAD match complete: red_cylindrical_cup_v1")
        rclpy.init()
        client = MoveItTrajectoryClient()
        try:
            # 当前工装位置来自对象目录；目标红杯位置仅来自本次 CAD 配准。
            time.sleep(3.0)
            client.publish_scene_positions(catalog.fixture_collision_positions(), grasp_target="red_cup")
            time.sleep(1.0)
            executor.set_gripper(opened=True)
            for stage, target in targets[:2]:
                executor.set_display_state(f"CAD grasp / MoveIt: {stage}")
                client.publish_joint_state(executor.joint_positions())
                _execute_trajectory(executor, client.request(target))
                executor._event(f"moveit_{stage}_complete")
            executor.set_display_state("Gripper: close and stable attach CAD target")
            executor.set_gripper(opened=False)
            executor.attach("red_cup")
            for stage, target in targets[2:]:
                executor.set_display_state(f"CAD grasp / MoveIt: {stage} with red cup")
                client.publish_joint_state(executor.joint_positions())
                _execute_trajectory(executor, client.request(target))
                executor._event(f"moveit_{stage}_complete")
            executor.set_display_state("Gripper: release CAD target into tray")
            executor.set_gripper(opened=True)
            executor.release_to_tray(tray_center)
            # 仅评测：不会将这个真值结果反馈给任何下一步控制决策。
            evaluation_only_success = executor.object_in_tray("red_cup", tray_center)
            output = arguments.root / "logs" / "episodes" / "cad_matching_latest"
            output.mkdir(parents=True, exist_ok=True)
            _write_pgm(output / "target_mask.pgm", segmentation.mask)
            payload = {
                "instruction": arguments.instruction,
                "stage": "cad_model_matching_pick_place_baseline",
                "semantic_result": intent.as_dict(),
                "segmentation": segmentation.as_dict(),
                "cad_model": {
                    "model_id": model.model_id,
                    "mesh": str(model.mesh_path.relative_to(arguments.root)),
                    "symmetry": model.symmetry_type,
                    "selected_grasp_template": template.template_id,
                },
                "pose_estimate": pose.as_dict(),
                "grasp_position_base_m": grasp_position.tolist(),
                "fixture_tray_center_base_m": tray_center.tolist(),
                "events": [
                    {"name": event.name, "timestamp_s": event.timestamp, "details": event.details}
                    for event in executor.events
                ],
                "evaluation_only_success": evaluation_only_success,
            }
            (output / "episode.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            if not evaluation_only_success:
                raise RuntimeError("评测发现红杯未落入托盘")
            if viewer is not None:
                print("CAD 模型匹配回合成功。VNC 主窗口按 Esc 退出。", flush=True)
                while viewer.render_once("Success: CAD matched red cup in tray"):
                    time.sleep(1.0 / 30.0)
        finally:
            client.destroy_node()
            rclpy.shutdown()
    finally:
        simulation.close()


if __name__ == "__main__":
    main()
